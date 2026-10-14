#!/usr/bin/env python3
"""
Experiment IV, Step 4: curvature reconstruction from spatially evolved BGK populations.

For D3Q15, D3Q19, D3Q27 and both transverse shear branches along k = (2 pi/N)(1,2,3):

1. Initialize a periodic N^3 field with the exact discrete shear eigenvector of
   Step 3 (same selection, tracking and ||U||_2 = 1 normalization), using the
   Experiment II initialization formula
       f_i(x) = w_i rho0 + U0 Re[v_i exp(i k.x)].
   Experiment II's own initialize_exact_mode is not called because its
   select_shear_mode finds only one branch on [123] for D3Q15/D3Q27 (Step 3).
2. Evolve with Experiment II's unmodified nonlinear-equilibrium BGK
   collide_and_stream (populations stored f[i, z, y, x]).
3. Reconstruct pointwise with Step 2B: reconstruct_from_populations (local rho,
   u, f_eq, f_neq, P_K, R^H / C0), node by node in vectorized z-slabs. No
   wavevector, Fourier transform, finite difference or analytical velocity
   enters the reconstruction.
4. Reference: U(t) = (1/rho0) sum_i c_i fhat_i(k, t) from the evolved field
   (spatial-average Fourier normalization, as Experiment II), J = -k k U,
   dhat_ref = D_Q q_F, d_ref(x) = 2 Re[dhat_ref exp(i k.x)].
5. Errors: spatial relative RMS, direct estimator R^H f / C0, per-observable,
   T, max-abs, modal / non-modal split, Fourier contamination.

For a real single-mode field the grid average of |Re(a e^{ik.x})|^2 is |a|^2/2
(2k is not aliased to 0 for the N used), so in the linear limit E_RMS equals
the complex-norm error of Step 3. This is checked, not assumed.

Scope: small-amplitude transverse Fourier modes on a periodic grid. This is not
a validation for arbitrary nonlinear flows.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _import(path: Path, name: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lin = _import(SRC / "experiment_IV_linearized_reconstruction.py", "experiment_IV_linearized_reconstruction")
exp2, recon = lin.exp2, lin.recon

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LATTICES = lin.LATTICES
TAU = 1.0
RHO0 = 1.0
U0 = 1e-6
N_VALUES: Tuple[int, ...] = (24, 32, 48, 64, 80)
TIMES: Tuple[int, ...] = (1, 2, 4, 8)            # t = 0 is also evaluated as a control
AMPLITUDES: Tuple[float, ...] = (1e-4, 1e-5, 1e-6, 1e-7)
AMPLITUDE_N: Tuple[int, ...] = (32, 48)
AMPLITUDE_TIMES: Tuple[int, ...] = (1, 8)
FIT_N_MIN = 48
TRACKING_N: Tuple[int, ...] = tuple(sorted(set(lin.N_VALUES) | set(N_VALUES)))
SLAB_NODES = 1 << 16                            # nodes per reconstruction chunk
EPS = np.finfo(float).eps

# Flag thresholds (reported, not applied silently)
CONTAMINATION_MAX = 1e-3       # non-modal / modal momentum amplitude
ROUNDOFF_FRACTION_MAX = 1e-3   # estimated round-off floor / E_RMS
PRE_ASYMPTOTIC_N = 48          # Step 3 local slopes are pre-asymptotic below this N
T_RELATIVE_FLOOR = lin.T_RELATIVE_FLOOR
T_NONLINEAR_TOL = 1e-2         # |T_rms - T_modal| / T_modal above this -> T limited by non-modal content


# -----------------------------------------------------------------------------
# Modes, initialization, Fourier helpers
# -----------------------------------------------------------------------------

def shear_modes(name: str, Ns: Iterable[int]) -> Dict[int, Dict[str, "lin.ShearMode"]]:
    """Step 3 shear modes, tracked over TRACKING_N so branch labels match Step 3."""
    lat = lin.load_lattice(name)
    allN = sorted(set(TRACKING_N) | set(Ns))
    per_N = {N: lin.find_shear_modes(lat, name, N, TAU, RHO0) for N in allN}
    lin.track_branches(per_N)
    return {N: {m.branch: m for m in per_N[N]} for N in Ns}


def phase_field(kvec: np.ndarray, N: int) -> np.ndarray:
    """exp(i k.x) on the (z, y, x) grid used by Experiment II."""
    x = np.arange(N)[None, None, :]
    y = np.arange(N)[None, :, None]
    z = np.arange(N)[:, None, None]
    return np.exp(1j * (kvec[0] * x + kvec[1] * y + kvec[2] * z))


def initialize(lat, v: np.ndarray, phase: np.ndarray, amplitude: float, rho0: float = RHO0) -> np.ndarray:
    """f_i = w_i rho0 + amplitude Re[v_i exp(i k.x)]  (Experiment II formula; ||U(v)||_2 = 1)."""
    Q = len(lat.w)
    N = phase.shape[0]
    f = np.empty((Q, N, N, N), dtype=float)
    for i in range(Q):
        f[i] = lat.w[i] * rho0 + amplitude * np.real(v[i] * phase)
    return f


def fourier_coefficient(field_: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Spatial-average Fourier coefficient at +k over the last three axes (Experiment II convention)."""
    return np.tensordot(field_, np.conj(phase), axes=((-3, -2, -1), (0, 1, 2))) / phase.size


def momentum_field(lat, f: np.ndarray) -> np.ndarray:
    """j = sum_i c_i f_i, shape (3, N, N, N)."""
    return np.tensordot(lat.c.T.astype(float), f, axes=(1, 0))


def mode_index(N: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """FFT bins (z, y, x) of +k and -k for k = (2 pi/N)(1,2,3)."""
    nx, ny, nz = (int(d) for d in lin.DIRECTION)
    return (nz % N, ny % N, nx % N), ((-nz) % N, (-ny) % N, (-nx) % N)


def contamination(j: np.ndarray) -> float:
    """sqrt(power of the momentum field outside the +-k bins / power in them)."""
    N = j.shape[1]
    J = np.fft.fftn(j, axes=(1, 2, 3))
    p = np.sum(np.abs(J) ** 2, axis=0)
    a, b = mode_index(N)
    modal = p[a] + p[b]
    return float(math.sqrt(max(p.sum() - modal, 0.0) / modal)) if modal > 0 else float("inf")


def reconstruct_slabs(f: np.ndarray, name: str, tau: float, rho0: float) -> np.ndarray:
    """Pointwise Step 2B reconstruction, vectorized over z-slabs. Returns (N, N, N, r)."""
    Q, Nz, Ny, Nx = f.shape
    r = recon.num_observables(name)
    out = np.empty((Nz, Ny, Nx, r), dtype=float)
    dz = max(1, SLAB_NODES // (Ny * Nx))
    for z0 in range(0, Nz, dz):
        z1 = min(Nz, z0 + dz)
        out[z0:z1] = recon.reconstruct_from_populations(np.moveaxis(f[:, z0:z1], 0, -1), name, tau, rho0)
    return out


def direct_estimate(f: np.ndarray, name: str, tau: float, rho0: float) -> np.ndarray:
    """R^H f / C0 applied to the raw populations, shape (N, N, N, r)."""
    Q = f.shape[0]
    RH = recon.reconstruction_matrix(name)
    d = RH @ f.reshape(Q, -1) / recon.ce2_prefactor(tau, rho0)          # (r, N^3), no population copy
    return np.moveaxis(d.reshape((RH.shape[0],) + f.shape[1:]), 0, -1)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

@dataclass
class Record:
    lattice: str
    N: int
    k_norm: float
    branch: str
    U0: float
    t: int
    tau: float
    rho0: float
    eigenvalue_real: float
    E_RMS: float
    E_RMS_direct: float
    direct_vs_projected: float          # RMS(d_direct - d_rec) / RMS(d_ref)
    max_abs_error_over_max_ref: float
    E_modal: float                      # ||dhat_rec - dhat_ref|| / ||dhat_ref|| at +k
    nonmodal_fraction: float            # RMS(d_rec - single-mode part) / RMS(d_ref)
    E_linearized_step3: float
    rel_diff_vs_step3: float
    T_rms_rel_error: float              # single-run spatial RMS (includes non-modal content)
    T_modal_rel_error: float            # from the +k Fourier component of the reconstructed field
    T_linearized_step3: float
    T_ref_over_d: float
    velocity_decay_deviation: float     # | |Uhat(t)| / |Uhat(0)| - lambda^t | / lambda^t
    contamination: float
    roundoff_estimate: float            # eps * RMS|f| / RMS|g_K|, relative floor for E
    reference_consistency: float        # re-extracted dhat_ref vs dhat_ref
    mass_drift: float
    momentum_total: float               # |sum_x j| / (N^3 U0)
    flags: str
    per_observable: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name -> (abs RMS / RMS(d_ref), own rel)


def diagnose(lat, name: str, f: np.ndarray, mode, phase: np.ndarray, t: int, U0_: float,
             E_lin: float, Uhat0_norm: float, mass0: float, tau: float = TAU, rho0: float = RHO0,
             T_lin: float = float("nan")) -> Record:
    N = f.shape[1]
    D = lin.observable_matrix(name)
    names = recon.observable_names(name)

    # Reference from the evolved field's hydrodynamic moments only.
    j = momentum_field(lat, f)
    Uhat = fourier_coefficient(j, phase) / rho0
    dhat_ref = D @ exp2.q_fourier(mode.kvec, Uhat)
    d_ref = 2.0 * np.real(dhat_ref[None, None, None, :] * phase[..., None])          # (N,N,N,r)
    ref_consistency = float(np.linalg.norm(fourier_coefficient(np.moveaxis(d_ref, -1, 0), phase) - dhat_ref)
                            / np.linalg.norm(dhat_ref))

    # Local reconstruction (populations only) and direct estimator.
    d_rec = reconstruct_slabs(f, name, tau, rho0)
    d_dir = direct_estimate(f, name, tau, rho0)

    rms = lambda a: float(np.sqrt(np.mean(np.sum(np.abs(a) ** 2, axis=-1))))   # noqa: E731
    R_ref = rms(d_ref)
    err = d_rec - d_ref
    E = rms(err) / R_ref
    E_dir = rms(d_dir - d_ref) / R_ref
    dvp = rms(d_dir - d_rec) / R_ref
    maxerr = float(np.max(np.abs(err)) / np.max(np.abs(d_ref)))

    dhat_rec = fourier_coefficient(np.moveaxis(d_rec, -1, 0), phase)
    E_modal = float(np.linalg.norm(dhat_rec - dhat_ref) / np.linalg.norm(dhat_ref))
    single = 2.0 * np.real(dhat_rec[None, None, None, :] * phase[..., None])
    nonmodal = rms(d_rec - single) / R_ref

    per_obs = {}
    for a, nm in enumerate(names):
        ra = float(np.sqrt(np.mean(err[..., a] ** 2)))
        ref_a = float(np.sqrt(np.mean(d_ref[..., a] ** 2)))
        per_obs[nm] = (ra / R_ref, ra / ref_a if ref_a > 0 else float("nan"))
    flags = []
    T_rel = T_modal = T_ratio = float("nan")
    if "T" in names:
        iT = names.index("T")
        refT = float(np.sqrt(np.mean(d_ref[..., iT] ** 2)))
        T_ratio = refT / R_ref
        if T_ratio >= T_RELATIVE_FLOOR:
            T_rel = per_obs["T"][1]
            T_modal = float(abs(dhat_rec[iT] - dhat_ref[iT]) / abs(dhat_ref[iT]))
            if abs(T_rel - T_modal) > T_NONLINEAR_TOL * T_modal:
                flags.append("T_nonlinear_limited")
        else:
            flags.append("T_reference_small")

    # Round-off floor estimate: f - f_eq cancels O(w) numbers, so the relative error of the kinetic
    # signal is about eps * ||w||_2 / RMS(g_K), with RMS(g_K) = U0 |lambda|^t ||P_K N v||_2 / sqrt(2).
    gK_rms = U0_ * float(np.linalg.norm(recon.project_kinetic(lat.N @ mode.v, name))) \
        * abs(mode.eigenvalue) ** t / math.sqrt(2.0)
    roundoff = EPS * float(np.linalg.norm(lat.w)) * rho0 / gK_rms if gK_rms > 0 else float("inf")

    cont = contamination(j)
    lam_t = abs(mode.eigenvalue) ** t
    decay_dev = abs(np.linalg.norm(Uhat) / Uhat0_norm - lam_t) / lam_t
    mass_drift = abs(float(f.sum()) - mass0) / mass0
    mom = float(np.linalg.norm(j.reshape(3, -1).sum(axis=1))) / (N ** 3 * U0_)

    if cont > CONTAMINATION_MAX:
        flags.append("nonlinear_contamination")
    if roundoff > ROUNDOFF_FRACTION_MAX * E:
        flags.append("roundoff_limited")
    if N < PRE_ASYMPTOTIC_N:
        flags.append("pre_asymptotic_k")
    if t == 0:
        flags.append("initial_state_control")

    return Record(
        lattice=name, N=N, k_norm=float(np.linalg.norm(mode.kvec)), branch=mode.branch, U0=U0_, t=t,
        tau=tau, rho0=rho0, eigenvalue_real=float(mode.eigenvalue.real),
        E_RMS=E, E_RMS_direct=E_dir, direct_vs_projected=dvp, max_abs_error_over_max_ref=maxerr,
        E_modal=E_modal, nonmodal_fraction=nonmodal, E_linearized_step3=E_lin,
        rel_diff_vs_step3=abs(E - E_lin) / E_lin, T_rms_rel_error=T_rel, T_modal_rel_error=T_modal,
        T_linearized_step3=T_lin, T_ref_over_d=T_ratio,
        velocity_decay_deviation=float(decay_dev), contamination=cont, roundoff_estimate=float(roundoff),
        reference_consistency=ref_consistency, mass_drift=mass_drift, momentum_total=mom,
        flags=";".join(flags), per_observable=per_obs,
    )


# -----------------------------------------------------------------------------
# Simulations
# -----------------------------------------------------------------------------

def simulate(name: str, N: int, mode, amplitude: float, times: Sequence[int], E_lin: float,
             tau: float = TAU, rho0: float = RHO0, lat=None, include_t0: bool = True,
             keep_fields_at: Optional[int] = None):
    """Evolve one configuration; return records (and optionally (d_rec, d_ref) at one time)."""
    lat = lin.load_lattice(name) if lat is None else lat
    E_lin, T_lin = (E_lin if isinstance(E_lin, tuple) else (E_lin, float("nan")))
    phase = phase_field(mode.kvec, N)
    f = initialize(lat, mode.v, phase, amplitude, rho0)
    mass0 = float(f.sum())
    Uhat0 = np.linalg.norm(fourier_coefficient(momentum_field(lat, f), phase)) / rho0
    eval_t = sorted(set(times) | ({0} if include_t0 else set()))
    out, kept = [], None
    for t in range(max(eval_t) + 1):
        if t in eval_t:
            out.append(diagnose(lat, name, f, mode, phase, t, amplitude, E_lin, Uhat0, mass0, tau, rho0, T_lin))
            if keep_fields_at is not None and t == keep_fields_at:
                j = momentum_field(lat, f)
                Uhat = fourier_coefficient(j, phase) / rho0
                dhat_ref = lin.observable_matrix(name) @ exp2.q_fourier(mode.kvec, Uhat)
                kept = (reconstruct_slabs(f, name, tau, rho0),
                        2.0 * np.real(dhat_ref[None, None, None, :] * phase[..., None]))
        if t < max(eval_t):
            f = exp2.collide_and_stream(lat, f, tau)
    return (out, kept) if keep_fields_at is not None else out


def linearized_errors(name: str, modes_by_N) -> Dict[Tuple[int, str], Tuple[float, float]]:
    """Step 3 (E_rec, T_rel) for each (N, branch); T_rel is nan where Step 3 does not report it."""
    lat = lin.load_lattice(name)
    out = {}
    for N, d in modes_by_N.items():
        for b, m in d.items():
            ev = lin.evaluate_mode(lat, name, m)
            out[(N, b)] = (ev.E_rec, ev.T_rel)
    return out


def run_primary(lattices=LATTICES, N_values=N_VALUES, times=TIMES, amplitude=U0, log=print) -> List[Record]:
    records = []
    for name in lattices:
        lat = lin.load_lattice(name)
        modes = shear_modes(name, N_values)
        E_lin = linearized_errors(name, modes)
        for N in N_values:
            for b in lin.BRANCHES:
                recs = simulate(name, N, modes[N][b], amplitude, times, E_lin[(N, b)], lat=lat)
                records.extend(recs)
                if log:
                    last = recs[-1]
                    log(f"{name} {b} N={N:3d}: E_RMS(t={last.t})={last.E_RMS:.6e}  E_lin={last.E_linearized_step3:.6e}"
                        f"  rel.diff={last.rel_diff_vs_step3:.1e}  contam={last.contamination:.1e}")
    return records


@dataclass
class AmplitudeRecord:
    lattice: str
    N: int
    branch: str
    U0: float
    t: int
    E_RMS_plus: float
    E_RMS_minus: float
    E_RMS_odd: float               # from (d(+U0) - d(-U0))/2 against (ref(+) - ref(-))/2
    E_linearized_step3: float
    rel_diff_plus: float
    rel_diff_odd: float
    contamination_plus: float
    roundoff_estimate: float


def run_amplitude_study(lattices=LATTICES, N_values=AMPLITUDE_N, amplitudes=AMPLITUDES,
                        times=AMPLITUDE_TIMES, log=print) -> List[AmplitudeRecord]:
    """Separate diagnostic: amplitude dependence and paired +-U0 odd-amplitude response."""
    out = []
    for name in lattices:
        lat = lin.load_lattice(name)
        modes = shear_modes(name, N_values)
        E_lin = linearized_errors(name, modes)
        D = lin.observable_matrix(name)
        for N in N_values:
            for b in lin.BRANCHES:
                m = modes[N][b]
                phase = phase_field(m.kvec, N)
                for A in amplitudes:
                    fp = initialize(lat, m.v, phase, +A)
                    fm = initialize(lat, m.v, phase, -A)
                    mass0 = float(fp.sum())
                    Uh0 = np.linalg.norm(fourier_coefficient(momentum_field(lat, fp), phase))
                    for t in range(max(times) + 1):
                        if t in times:
                            rp = diagnose(lat, name, fp, m, phase, t, A, E_lin[(N, b)][0], Uh0, mass0)
                            rm = diagnose(lat, name, fm, m, phase, t, A, E_lin[(N, b)][0], Uh0, mass0)
                            dp, dm = reconstruct_slabs(fp, name, TAU, RHO0), reconstruct_slabs(fm, name, TAU, RHO0)
                            refs = []
                            for f_ in (fp, fm):
                                Uh = fourier_coefficient(momentum_field(lat, f_), phase) / RHO0
                                dh = D @ exp2.q_fourier(m.kvec, Uh)
                                refs.append(2.0 * np.real(dh[None, None, None, :] * phase[..., None]))
                            d_odd, r_odd = 0.5 * (dp - dm), 0.5 * (refs[0] - refs[1])
                            E_odd = float(np.sqrt(np.sum((d_odd - r_odd) ** 2) / np.sum(r_odd ** 2)))
                            out.append(AmplitudeRecord(
                                lattice=name, N=N, branch=b, U0=A, t=t, E_RMS_plus=rp.E_RMS, E_RMS_minus=rm.E_RMS,
                                E_RMS_odd=E_odd, E_linearized_step3=E_lin[(N, b)][0],
                                rel_diff_plus=rp.rel_diff_vs_step3, rel_diff_odd=abs(E_odd - E_lin[(N, b)][0]) / E_lin[(N, b)][0],
                                contamination_plus=rp.contamination, roundoff_estimate=rp.roundoff_estimate))
                        if t < max(times):
                            fp = exp2.collide_and_stream(lat, fp, TAU)
                            fm = exp2.collide_and_stream(lat, fm, TAU)
                if log:
                    last = [r for r in out if (r.lattice, r.N, r.branch) == (name, N, b) and r.t == max(times)]
                    log(f"{name} {b} N={N}: " + "  ".join(
                        f"U0={r.U0:.0e}: +{r.rel_diff_plus:.1e}/odd {r.rel_diff_odd:.1e}" for r in last))
    return out


# -----------------------------------------------------------------------------
# Fits and outputs
# -----------------------------------------------------------------------------

def full_bgk_fits(records: Sequence[Record], step3_all: Dict[Tuple[str, str], float],
                  N_min: int = FIT_N_MIN) -> List[Dict]:
    """
    Fits ~ C k^p per lattice/branch/time for E_RMS, and for T (single-run RMS and modal),
    each alongside the Step 3 value on the same N set. Excluded points are listed.
    """
    quantities = (
        ("E_RMS", lambda r: r.E_RMS, lambda r: r.E_linearized_step3,
         ("nonlinear_contamination", "roundoff_limited")),
        ("T_rms_rel", lambda r: r.T_rms_rel_error, lambda r: r.T_linearized_step3,
         ("nonlinear_contamination", "roundoff_limited", "T_reference_small", "T_nonlinear_limited")),
        ("T_modal_rel", lambda r: r.T_modal_rel_error, lambda r: r.T_linearized_step3,
         ("nonlinear_contamination", "roundoff_limited", "T_reference_small")),
    )
    rows = []
    keys = sorted({(r.lattice, r.branch, r.t) for r in records if r.t > 0})
    for lattice, branch, t in keys:
        g = sorted([r for r in records if (r.lattice, r.branch, r.t) == (lattice, branch, t)], key=lambda r: r.N)
        for qname, get, get_lin, blocking in quantities:
            if qname.startswith("T_") and "T" not in recon.observable_names(lattice):
                continue                     # T is not an observable on this lattice (D3Q19)
            for label, nmin in (("asymptotic", N_min), ("N>=32", 32), ("all", 0)):
                pts = [r for r in g if r.N >= nmin]
                bad = [r.N for r in pts if any(fl in r.flags for fl in blocking) or not np.isfinite(get(r))]
                pts = [r for r in pts if r.N not in bad]
                if len(pts) < 3:
                    if bad and nmin == N_min:
                        rows.append({"lattice": lattice, "branch": branch, "t": t, "quantity": qname,
                                     "interval": label, "N_min": nmin, "N_max": max(r.N for r in g),
                                     "k_min": float("nan"), "k_max": float("nan"), "n_points": len(pts),
                                     "excluded_N": " ".join(map(str, bad)), "p_full_bgk": float("nan"),
                                     "R2_full_bgk": float("nan"), "C_full_bgk": float("nan"),
                                     "p_linearized_same_N": float("nan"), "R2_linearized_same_N": float("nan"),
                                     "p_difference": float("nan"),
                                     "p_linearized_step3_N48_128": step3_all.get((lattice, branch), float("nan"))
                                     if qname == "E_RMS" else float("nan"),
                                     "note": "fewer than 3 usable points"})
                    continue
                k = np.array([r.k_norm for r in pts])
                fit = lin.loglog_fit(k, np.array([get(r) for r in pts]))
                fit_lin = lin.loglog_fit(k, np.array([get_lin(r) for r in pts]))
                rows.append({
                    "lattice": lattice, "branch": branch, "t": t, "quantity": qname, "interval": label,
                    "N_min": pts[0].N, "N_max": pts[-1].N, "k_min": float(k.min()), "k_max": float(k.max()),
                    "n_points": len(pts), "excluded_N": " ".join(map(str, bad)),
                    "p_full_bgk": fit["p"], "R2_full_bgk": fit["R2"], "C_full_bgk": fit["C"],
                    "p_linearized_same_N": fit_lin["p"], "R2_linearized_same_N": fit_lin["R2"],
                    "p_difference": fit["p"] - fit_lin["p"],
                    "p_linearized_step3_N48_128": step3_all.get((lattice, branch), float("nan"))
                    if qname == "E_RMS" else float("nan"),
                    "note": "",
                })
    return rows


def step3_reference_fits() -> Dict[Tuple[str, str], float]:
    """Step 3 asymptotic exponents (N >= 48 over the full Step 3 N set), recomputed here."""
    res = lin.run_experiment()
    out = {}
    for r in lin.convergence_fits(res.evaluations):
        if r["quantity"] == "reconstruction_error" and r["interval"] == "asymptotic":
            out[(r["lattice"], r["branch"])] = r["p"]
    return out


def record_rows(records: Sequence[Record]) -> List[Dict]:
    rows = []
    for r in records:
        d = asdict(r)
        d.pop("per_observable")
        rows.append(d)
    return rows


def observable_rows(records: Sequence[Record]) -> List[Dict]:
    rows = []
    for r in records:
        for nm, (a, b) in r.per_observable.items():
            rows.append({"lattice": r.lattice, "N": r.N, "k_norm": r.k_norm, "branch": r.branch, "U0": r.U0, "t": r.t,
                         "observable": nm, "rms_error_over_rms_d_ref": a, "rms_error_over_own_rms": b})
    return rows


def make_comparison_figure(records: Sequence[Record], amp: Sequence[AmplitudeRecord], path: Path, t_plot: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin_res = lin.run_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    for lattice in LATTICES:
        for branch in lin.BRANCHES:
            st, bs = lin.LATTICE_STYLE[lattice], lin.BRANCH_STYLE[branch]
            g = sorted([e for e in lin_res.evaluations if e.mode.lattice == lattice and e.mode.branch == branch],
                       key=lambda e: np.linalg.norm(e.mode.kvec))
            ax.loglog([np.linalg.norm(e.mode.kvec) for e in g], [e.E_rec for e in g], color=st["color"],
                      linestyle=bs["linestyle"], linewidth=1.2, alpha=0.55)
            r = sorted([x for x in records if x.lattice == lattice and x.branch == branch and x.t == t_plot],
                       key=lambda x: x.k_norm)
            ax.loglog([x.k_norm for x in r], [x.E_RMS for x in r], linestyle="none", marker=st["marker"], markersize=7,
                      color=st["color"], markerfacecolor=st["color"] if bs["fill"] else "white",
                      markeredgewidth=1.4, label=f"{lattice} {branch}")
    ax.set_xlabel(r"$|\mathbf{k}|$  (direction [123])")
    ax.set_ylabel(r"relative error")
    ax.set_title(f"markers: full BGK $E^{{RMS}}_Q$ ($t={t_plot}$, $U_0=10^{{-6}}$)\nlines: exact linearized $E^{{rec}}_Q$ (Step 3)",
                 fontsize=9.5, color="#2b2b28")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")

    ax = axes[1]
    for lattice in LATTICES:
        st = lin.LATTICE_STYLE[lattice]
        for branch in lin.BRANCHES:
            bs = lin.BRANCH_STYLE[branch]
            g = sorted([a for a in amp if a.lattice == lattice and a.branch == branch and a.N == max(AMPLITUDE_N)
                        and a.t == max(AMPLITUDE_TIMES)], key=lambda a: a.U0)
            if not g:
                continue
            ax.loglog([a.U0 for a in g], [max(a.rel_diff_plus, 1e-17) for a in g], color=st["color"],
                      linestyle=bs["linestyle"], marker=st["marker"], markersize=6,
                      markerfacecolor=st["color"] if bs["fill"] else "white", linewidth=1.4,
                      label=f"{lattice} {branch}")
            ax.loglog([a.U0 for a in g], [max(a.rel_diff_odd, 1e-17) for a in g], color=st["color"],
                      linestyle=":", marker="x", markersize=5, linewidth=1.0)
    ax.set_xlabel(r"perturbation amplitude $U_0$")
    ax.set_ylabel(r"$|E^{RMS}_Q - E^{rec,lin}_Q| \,/\, E^{rec,lin}_Q$")
    ax.set_title(f"$N={max(AMPLITUDE_N)}$, $t={max(AMPLITUDE_TIMES)}$: single run (lines+markers), "
                 "paired $\\pm U_0$ odd part (x, dotted)", fontsize=9.5, color="#2b2b28")
    for a_ in axes:
        a_.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_spatial_figure(name: str, N: int, branch: str, t: int, path: Path, observable: str = "A_x") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = shear_modes(name, [N])[N][branch]
    lat = lin.load_lattice(name)
    E_lin = lin.evaluate_mode(lat, name, mode).E_rec
    _, (d_rec, d_ref) = simulate(name, N, mode, U0, [t], E_lin, lat=lat, include_t0=False, keep_fields_at=t)
    a = recon.observable_names(name).index(observable)
    x = np.arange(N)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.plot(x, d_ref[0, 0, :, a] / U0, color="#55554f", linewidth=1.6, label="reference $D_Q\\mathbf{q}_F$")
    ax.plot(x, d_rec[0, 0, :, a] / U0, linestyle="none", marker="o", markersize=4, color=lin.LATTICE_STYLE[name]["color"],
            label="local reconstruction")
    ax.set_xlabel("x (lattice units), y = z = 0")
    ax.set_ylabel(f"{observable} / $U_0$")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{name} {branch}, N={N}, t={t}", fontsize=9.5, color="#2b2b28")
    ax = axes[1]
    ax.plot(x, (d_rec[0, 0, :, a] - d_ref[0, 0, :, a]) / U0, color=lin.LATTICE_STYLE[name]["color"], linewidth=1.4)
    ax.axhline(0, color="#8a8a85", linewidth=0.8)
    ax.set_xlabel("x (lattice units), y = z = 0")
    ax.set_ylabel(f"({observable}$_{{rec}}$ - {observable}$_{{ref}}$) / $U_0$")
    ax.set_title("pointwise difference", fontsize=9.5, color="#2b2b28")
    for a_ in axes:
        a_.grid(True, color="#e4e3df", linewidth=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(records, amp, fits, outdir: Path, spatial: bool = True) -> Dict[str, Path]:
    paths = {
        "results": outdir / "experiment_IV_full_bgk_results.csv",
        "observables": outdir / "experiment_IV_full_bgk_observables.csv",
        "amplitude": outdir / "experiment_IV_full_bgk_amplitude.csv",
        "fits": outdir / "experiment_IV_full_bgk_fits.csv",
        "figure": outdir / "figure_IV_4_full_bgk_vs_linearized.png",
        "spatial": outdir / "figure_IV_5_spatial_comparison.png",
    }
    lin.write_csv(record_rows(records), paths["results"])
    lin.write_csv(observable_rows(records), paths["observables"])
    lin.write_csv([asdict(a) for a in amp], paths["amplitude"])
    lin.write_csv(fits, paths["fits"])
    make_comparison_figure(records, amp, paths["figure"])
    if spatial:
        make_spatial_figure("D3Q27", 32, "s1", 8, paths["spatial"])
    return paths


def print_summary(records, amp, fits) -> None:
    print("\nSummary (U0 = %.0e, tau = %g, rho0 = %g)" % (U0, TAU, RHO0))
    prim = [r for r in records if r.t > 0]
    print(f"  max |E_RMS - E_lin| / E_lin (t >= 1)           {max(r.rel_diff_vs_step3 for r in prim):.2e}")
    print(f"  max |E_RMS - E_lin| / E_lin (t = 0 control)    {max(r.rel_diff_vs_step3 for r in records if r.t == 0):.2e}")
    print(f"  max |E_direct - E_lin| / E_lin                 {max(abs(r.E_RMS_direct - r.E_linearized_step3) / r.E_linearized_step3 for r in prim):.2e}")
    print(f"  max RMS(direct - projected) / RMS(ref)         {max(r.direct_vs_projected for r in records):.2e}")
    print(f"  max non-modal fraction of d_rec                {max(r.nonmodal_fraction for r in records):.2e}")
    print(f"  max Fourier contamination (momentum)           {max(r.contamination for r in records):.2e}")
    print(f"  max estimated round-off floor                  {max(r.roundoff_estimate for r in records):.2e}")
    print(f"  max velocity-decay deviation from lambda^t     {max(r.velocity_decay_deviation for r in records):.2e}")
    print(f"  max reference re-extraction residual           {max(r.reference_consistency for r in records):.2e}")
    print(f"  max mass drift / max |total momentum|/(N^3 U0) {max(r.mass_drift for r in records):.2e} / "
          f"{max(r.momentum_total for r in records):.2e}")
    flagged = {}
    for r in records:
        for fl in filter(None, r.flags.split(";")):
            flagged[fl] = flagged.get(fl, 0) + 1
    print(f"  flags: {flagged}")
    print("\nConvergence fits of E_RMS at t = 8:")
    print(f"{'lattice':<7}{'br':<4}{'interval':<12}{'N':>9}{'p_full':>9}{'R2':>10}{'p_lin(same N)':>15}{'dp':>10}")
    for f_ in fits:
        if f_["t"] == 8 and f_["quantity"] == "E_RMS":
            print(f"{f_['lattice']:<7}{f_['branch']:<4}{f_['interval']:<12}{str(f_['N_min']) + '-' + str(f_['N_max']):>9}"
                  f"{f_['p_full_bgk']:>9.3f}{f_['R2_full_bgk']:>10.6f}{f_['p_linearized_same_N']:>15.3f}{f_['p_difference']:>10.1e}")
    print("\nT convergence at t = 8 (asymptotic interval, D3Q15/D3Q27):")
    for f_ in fits:
        if f_["t"] == 8 and f_["quantity"] != "E_RMS" and f_["interval"] == "asymptotic":
            print(f"  {f_['lattice']} {f_['branch']} {f_['quantity']:<12} p={f_['p_full_bgk']:.3f} "
                  f"(Step 3 same N: {f_['p_linearized_same_N']:.3f}) n={f_['n_points']} excluded N: {f_['excluded_N'] or '-'} {f_['note']}")
    if amp:
        print("\nAmplitude study (t = %d): rel. deviation from Step 3, single run / paired odd part" % max(AMPLITUDE_TIMES))
        for N in AMPLITUDE_N:
            for A in AMPLITUDES:
                g = [a for a in amp if a.N == N and a.U0 == A and a.t == max(AMPLITUDE_TIMES)]
                print(f"  N={N} U0={A:.0e}: max single {max(a.rel_diff_plus for a in g):.1e}, "
                      f"max odd {max(a.rel_diff_odd for a in g):.1e}, max contamination {max(a.contamination_plus for a in g):.1e}")


if __name__ == "__main__":
    recs = run_primary(N_values=(24, 32))
    print_summary(recs, [], full_bgk_fits(recs, {}))
