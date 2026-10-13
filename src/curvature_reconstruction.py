#!/usr/bin/env python3
"""
Experiment IV, Step 2B: solver-independent numerical curvature reconstruction.

Applies the exact Hermite reconstruction operators R^H_Q of Step 2A
(src/curvature_reconstruction_exact.py) to LBM population data, for D3Q15,
D3Q19 and D3Q27:

    d_rec = (1 / C0) R^H_Q g_K,    C0 = 2 rho0 tau (tau - 1/2) c_s^4 = (2/9) rho0 tau (tau - 1/2),

where g_K = P_K,Q f^neq is the kinetic part of the non-equilibrium populations.

Interfaces (populations always on the LAST axis, shape (..., Q))
----------------------------------------------------------------
    reconstruct_from_normalized_kinetic(g_normalized, lattice)      d = R^H g            (no C0)
    reconstruct_from_kinetic(g_kinetic, lattice, tau, rho0=1.0)     d = R^H g / C0
    reconstruct_from_nonequilibrium(f_noneq, lattice, tau, rho0)    d = R^H P_K f_neq / C0
    reconstruct_from_populations(f, lattice, tau, rho0)             f_neq = f - f_eq(rho(f), u(f)), then as above

Helpers: observable_names, num_observables, num_populations, reconstruction_matrix,
kinetic_projector, ce2_prefactor, project_kinetic, macroscopic, equilibrium,
nonequilibrium_populations.

All four return shape (..., r_Q), with r_15 = 4, r_19 = 6, r_27 = 7, in the
order given by observable_names(lattice):

    D3Q15: (S_1, S_2, S_3, T)
    D3Q19: (A_x, B_x, B_y, A_z, A_y, B_z)
    D3Q27: (A_x, B_x, B_y, A_z, A_y, B_z, T)

Scope
-----
These functions only apply fixed linear maps to population data. They never use
a wavevector, velocity amplitude, analytical curvature, or the forward CE2
operator Lhat_Q. reconstruct_from_populations is a CE2-based curvature
*estimator*: it is asymptotically valid under the linearized, long-wavelength
(Chapman-Enskog second-order) assumptions of the manuscript, and it is not an
exact derivative operator for arbitrary nonlinear or finite-wavelength flows.

Implementation notes
--------------------
* R^H_Q is taken from the verified exact module and converted once to float64
  (each exact rational is rounded once); P_K,Q is built exactly from the
  canonical velocities/weights with the same definition and Hermite column
  ordering as load_lattice() in src/experiment_II_3D_shear.py, then converted
  once. SymPy is used only inside these cached one-time builders; repeated
  reconstruction calls are pure NumPy matrix products.
* Real inputs are computed in float64, complex inputs in complex128. Input
  arrays are never modified.
* Experiments II/III store populations as f[i, z, y, x]; pass
  np.moveaxis(f, 0, -1) to use them here.
"""

from __future__ import annotations

import importlib.util
import math
import numbers
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

_SRC = Path(__file__).resolve().parent
_EXACT_NAME = "curvature_reconstruction_exact"


def _load_exact_module():
    """Import the Step 2A exact module from this directory (src/ is not a package)."""
    if _EXACT_NAME in sys.modules:
        return sys.modules[_EXACT_NAME]
    spec = importlib.util.spec_from_file_location(_EXACT_NAME, _SRC / f"{_EXACT_NAME}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_EXACT_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_exact = _load_exact_module()

LATTICES: Tuple[str, ...] = tuple(_exact.LATTICES)
CS2 = 1.0 / 3.0
CS4 = CS2 * CS2


# -----------------------------------------------------------------------------
# Cached numerical lattice data (built once per lattice)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericalLattice:
    """Read-only float data for one lattice; population order = canonical CSV row order."""
    name: str
    c: np.ndarray             # (Q, 3) integer velocities
    w: np.ndarray             # (Q,) weights
    RH: np.ndarray            # (r_Q, Q) Hermite reconstruction matrix R^H_Q
    PK: np.ndarray            # (Q, Q) weighted kinetic projector I - P_<=2
    observables: Tuple[str, ...]

    @property
    def Q(self) -> int:
        return self.c.shape[0]

    @property
    def r(self) -> int:
        return self.RH.shape[0]


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _exact_kinetic_projector(velocities, weights):
    """
    Exact P_K = I - Phi (Phi^T W^-1 Phi)^-1 Phi^T W^-1 with the weighted Hermite
    columns w_i H(c_i), H in (1, cx, cy, cz, cx^2-cs2, cy^2-cs2, cz^2-cs2,
    cx cy, cx cz, cy cz): the definition and column order of load_lattice()
    in src/experiment_II_3D_shear.py, evaluated in rational arithmetic.
    """
    import sympy as sp

    cs2 = sp.Rational(1, 3)
    Q = len(velocities)
    Winv = sp.diag(*[1 / x for x in weights])
    rows = []
    for c, w in zip(velocities, weights):
        cx, cy, cz = (sp.Integer(v) for v in c)
        h = [1, cx, cy, cz, cx * cx - cs2, cy * cy - cs2, cz * cz - cs2, cx * cy, cx * cz, cy * cz]
        rows.append([w * x for x in h])
    Phi = sp.Matrix(rows)
    gram = Phi.T * Winv * Phi
    return sp.eye(Q) - Phi * gram.inv() * Phi.T * Winv


@lru_cache(maxsize=None)
def get_lattice(lattice: str) -> NumericalLattice:
    """Build (once) and return the cached numerical data for a lattice."""
    _check_lattice(lattice)
    op = _exact.load_operator(lattice)
    RH_exact = _exact.hermite_reconstruction(op)
    PK_exact = _exact_kinetic_projector(op.velocities, op.w)
    to_float = lambda M: np.array([[float(x) for x in M.row(i)] for i in range(M.rows)], dtype=np.float64)  # noqa: E731
    return NumericalLattice(
        name=lattice,
        c=_readonly(np.array(op.velocities, dtype=np.int64)),
        w=_readonly(np.array([float(x) for x in op.w], dtype=np.float64)),
        RH=_readonly(to_float(RH_exact)),
        PK=_readonly(to_float(PK_exact)),
        observables=tuple(_exact.OUTPUT_ORDER[lattice]),
    )


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------

def observable_names(lattice: str) -> Tuple[str, ...]:
    """Names of the reconstructed functionals, in output order (last axis)."""
    _check_lattice(lattice)
    return tuple(_exact.OUTPUT_ORDER[lattice])


def num_observables(lattice: str) -> int:
    return len(observable_names(lattice))


def num_populations(lattice: str) -> int:
    _check_lattice(lattice)
    return int(lattice[3:])


def reconstruction_matrix(lattice: str) -> np.ndarray:
    """Read-only float64 R^H_Q, shape (r_Q, Q)."""
    return get_lattice(lattice).RH


def kinetic_projector(lattice: str) -> np.ndarray:
    """Read-only float64 P_K,Q, shape (Q, Q)."""
    return get_lattice(lattice).PK


def ce2_prefactor(tau: float, rho0: float = 1.0) -> float:
    """C0 = 2 rho0 tau (tau - 1/2) c_s^4 (same expression as Experiment II)."""
    tau, rho0 = _check_tau(tau), _check_rho0(rho0)
    return 2.0 * rho0 * tau * (tau - 0.5) * CS4


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _check_lattice(lattice) -> None:
    if not isinstance(lattice, str) or lattice not in _exact.LATTICES:
        raise ValueError(f"Unsupported lattice {lattice!r}; expected one of {tuple(_exact.LATTICES)}.")


def _real_scalar(x, name: str) -> float:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, got {type(x).__name__}.")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}.")
    return x


def _check_tau(tau) -> float:
    tau = _real_scalar(tau, "tau")
    if not tau > 0.5:
        raise ValueError(f"tau must satisfy tau > 1/2 (C0 vanishes at tau = 1/2), got {tau}.")
    return tau


def _check_rho0(rho0) -> float:
    rho0 = _real_scalar(rho0, "rho0")
    if not rho0 > 0.0:
        raise ValueError(f"rho0 must be positive, got {rho0}.")
    return rho0


def _as_population_array(x, lattice: str, name: str, allow_complex: bool = True) -> np.ndarray:
    """Validate shape (..., Q) and finiteness; return float64/complex128 array (a new view or copy)."""
    _check_lattice(lattice)
    a = np.asarray(x)
    if a.dtype == object or not (np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_):
        raise TypeError(f"{name} must be a numeric array, got dtype {a.dtype}.")
    if a.dtype == np.bool_:
        raise TypeError(f"{name} must be numeric, not boolean.")
    if np.iscomplexobj(a):
        if not allow_complex:
            raise TypeError(f"{name} must be real-valued for this interface.")
        a = a.astype(np.complex128, copy=False)
    else:
        a = a.astype(np.float64, copy=False)
    Q = num_populations(lattice)
    if a.ndim == 0 or a.shape[-1] != Q:
        raise ValueError(f"{name} must have shape (..., {Q}) for {lattice}, got {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values.")
    return a


# -----------------------------------------------------------------------------
# Reconstruction interfaces
# -----------------------------------------------------------------------------

def reconstruct_from_normalized_kinetic(g_normalized, lattice: str) -> np.ndarray:
    """
    d_Q = R^H_Q g_normalized for already normalized kinetic populations
    (g_normalized = Lhat_Q q in the CE2 limit). No C0 is applied.
    """
    g = _as_population_array(g_normalized, lattice, "g_normalized")
    return g @ get_lattice(lattice).RH.T


def reconstruct_from_kinetic(g_kinetic, lattice: str, tau: float, rho0: float = 1.0) -> np.ndarray:
    """
    d_Q = (1/C0) R^H_Q g_kinetic for dimensional kinetic populations
    (g_kinetic = P_K f_neq; in the CE2 limit g_kinetic = C0 Lhat_Q q).
    The factor 1/C0 is applied here and only here.
    """
    c0 = ce2_prefactor(tau, rho0)
    return reconstruct_from_normalized_kinetic(g_kinetic, lattice) / c0


def project_kinetic(f_noneq, lattice: str) -> np.ndarray:
    """g_K = P_K,Q f_neq (populations on the last axis)."""
    f = _as_population_array(f_noneq, lattice, "f_noneq")
    return f @ get_lattice(lattice).PK.T


def reconstruct_from_nonequilibrium(f_noneq, lattice: str, tau: float, rho0: float = 1.0) -> np.ndarray:
    """d_Q = (1/C0) R^H_Q P_K,Q f_neq."""
    ce2_prefactor(tau, rho0)  # validate before any work
    return reconstruct_from_kinetic(project_kinetic(f_noneq, lattice), lattice, tau, rho0)


def macroscopic(f, lattice: str) -> Tuple[np.ndarray, np.ndarray]:
    """Local density (...,) and velocity (..., 3) from real populations (..., Q)."""
    f = _as_population_array(f, lattice, "f", allow_complex=False)
    lat = get_lattice(lattice)
    rho = np.sum(f, axis=-1)
    if not np.all(rho > 0.0):
        raise ValueError("Local density must be positive at every node.")
    j = f @ lat.c.astype(np.float64)
    return rho, j / rho[..., None]


def equilibrium(rho, u, lattice: str) -> np.ndarray:
    """
    Second-order isothermal equilibrium of Experiment II, populations on the last axis:
        f_eq_i = w_i rho (1 + c.u/cs2 + (c.u)^2/(2 cs4) - u.u/(2 cs2)).
    rho: (...,), u: (..., 3). Returns (..., Q).
    """
    lat = get_lattice(lattice)
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if u.shape[-1:] != (3,) or u.shape[:-1] != rho.shape:
        raise ValueError(f"Expected rho (...,) and u (..., 3) with matching leading shape, got {rho.shape}, {u.shape}.")
    cu = u @ lat.c.T.astype(np.float64)                      # (..., Q)
    usq = np.sum(u * u, axis=-1)[..., None]                   # (..., 1)
    return lat.w * rho[..., None] * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * usq / CS2)


def nonequilibrium_populations(f, lattice: str) -> np.ndarray:
    """f_neq = f - f_eq(rho, u), with rho and u computed locally (per node) from f."""
    rho, u = macroscopic(f, lattice)
    f_arr = _as_population_array(f, lattice, "f", allow_complex=False)
    return f_arr - equilibrium(rho, u, lattice)


def reconstruct_from_populations(f, lattice: str, tau: float, rho0: float = 1.0) -> np.ndarray:
    """
    CE2-based curvature estimator from full (real) populations f (..., Q):

        rho = sum_i f_i,  u = sum_i c_i f_i / rho        (local, per node)
        f_neq = f - f_eq(rho, u)                          (Experiment II equilibrium)
        d = (1/C0) R^H_Q P_K,Q f_neq,  C0 = C0(tau, rho0)

    The local density enters the equilibrium; rho0 is only the uniform
    reference density of the CE2 normalization. Asymptotically valid under the
    linearized long-wavelength assumptions of the manuscript; not an exact
    derivative operator for arbitrary nonlinear flows.

    Note: the second-order equilibrium lies in the Hermite <=2 space, which
    R^H_Q annihilates exactly (R^H_Q P_K,Q = R^H_Q). In exact arithmetic the
    result therefore equals R^H_Q f / C0; the equilibrium subtraction and the
    projection are kept so that the intermediate g_K is the physical kinetic
    signal and so that large equilibrium contributions are removed before the
    final contraction.
    """
    ce2_prefactor(tau, rho0)  # validate before any work
    return reconstruct_from_nonequilibrium(nonequilibrium_populations(f, lattice), lattice, tau, rho0)
