#!/usr/bin/env python3
"""
Experiment IV, Step 3: curvature reconstruction from exact linearized BGK shear eigenmodes.

For D3Q15, D3Q19 and D3Q27, and k_N = (2 pi / N) (1, 2, 3):

1. Build the exact Fourier amplification matrix A_Q(k, tau) = S_Q(k) C_Q(tau)
   (src/experiment_II_3D_shear.py; no N^3 grid is allocated).
2. Identify the two transverse shear eigenvectors by hydrodynamic character
   (momentum content, transverse fraction, density), not by eigenvalue magnitude.
3. Extract the actual kinetic response g_K = P_K N v (Experiment II's
   linearized equilibrium removal N, Step 2B's kinetic projector P_K).
4. Reconstruct d_rec = R^H g_K / C0 with Step 2B's reconstruct_from_kinetic,
   which receives only g_K, the lattice name, tau and rho0.
5. Compare with the independent analytical reference d_exact = D_Q q_F, where
   q_F holds the independent components of J_abg = -k_b k_g U_a, and U is the
   eigenvector's own momentum amplitude.
6. Separately, evaluate the population-space CE2 error against C0 Lhat_Q q_F.

Scope: this tests finite-wavenumber consistency of the reconstruction with
the CE2 asymptotics on exact eigenvectors of the *linearized* discrete BGK
scheme. It is not a reconstruction from spatially evolved nonlinear BGK
populations (Step 4), and recovering the observable functionals of transverse
Fourier modes does not imply recovery of the full incompressible curvature
tensor.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OPERATOR_DIR = ROOT / "operators"


def _import(path: Path, name: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp2 = _import(SRC / "experiment_II_3D_shear.py", "experiment_II_3D_shear")
recon = _import(SRC / "curvature_reconstruction.py", "curvature_reconstruction")
exact = _import(SRC / "curvature_reconstruction_exact.py", "curvature_reconstruction_exact")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LATTICES: Tuple[str, ...] = ("D3Q15", "D3Q19", "D3Q27")
TAU = 1.0
RHO0 = 1.0
DIRECTION = np.array([1, 2, 3], dtype=float)
N_VALUES: Tuple[int, ...] = (24, 32, 40, 48, 64, 80, 96, 128)
E1 = np.array([2.0, -1.0, 0.0]) / math.sqrt(5.0)
E2 = np.array([3.0, 0.0, -1.0]) / math.sqrt(10.0)
BRANCHES: Tuple[str, str] = ("s1", "s2")

# Identification / reliability thresholds (reported, never applied silently).
SELECTION_MARGIN_MAX = 0.5      # 3rd-best shear score / 2nd-best must stay below this
LONGITUDINAL_MAX = 0.1          # a shear candidate must be predominantly transverse
DEGENERACY_TOL = 1e-9           # |lambda_s1 - lambda_s2| below this -> degenerate pair
TRACKING_OVERLAP_MIN = 0.9      # |U_prev^H U_next| for continuity across N
REFERENCE_FLOOR = 1e-12         # ||d_exact|| below this -> relative error not reported
T_RELATIVE_FLOOR = 1e-2         # |T_exact| / ||d_exact|| below this -> T relative error not reported (flagged)

# Default convergence-fit interval (chosen after inspecting the data; see docs).
FIT_N_MIN = 48


def k_vector(N: int) -> np.ndarray:
    return 2.0 * np.pi * DIRECTION / N


def load_lattice(name: str):
    """Experiment II lattice object (silences the loader's pre-existing unclosed-file warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return exp2.load_lattice(OPERATOR_DIR / f"Lhat_{name}.csv")


def observable_matrix(name: str) -> np.ndarray:
    """D_Q from the exact Step 2A module, converted once to float."""
    return np.array(exact.observable_matrix(name).tolist(), dtype=float)


# -----------------------------------------------------------------------------
# Shear-mode identification
# -----------------------------------------------------------------------------

@dataclass
class EigenCandidate:
    index: int
    eigenvalue: complex
    hydro_content: float          # |j| / ||v||_w
    longitudinal_fraction: float  # |n . j| / |j|
    density_fraction: float       # |drho| / |j|
    score: float                  # transverse hydrodynamic score


@dataclass
class ShearMode:
    lattice: str
    N: int
    kvec: np.ndarray
    eigenvalue: complex
    v: np.ndarray                 # population eigenvector, normalized to ||U||_2 = 1
    U: np.ndarray                 # momentum amplitude / rho0, ||U||_2 = 1
    drho: complex
    longitudinal_fraction: float
    align_e1: float
    align_e2: float
    eig_residual: float           # ||A v - lambda v||_2 / ||v||_2
    shear_gap: float              # |lambda_this - lambda_other_shear|
    nonshear_separation: float    # min |lambda_this - lambda_j| over non-shear eigenvalues
    selection_score: float
    selection_margin: float       # 3rd-best score / 2nd-best score (pair-level)
    branch: str = ""
    tracking_overlap: float = float("nan")
    flags: List[str] = field(default_factory=list)


def classify_eigenpairs(lat, kvec: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray, List[EigenCandidate]]:
    """Eigen-decompose A and score every eigenvector by transverse hydrodynamic character."""
    A = exp2.amplification_matrix(lat, kvec, tau)
    lam, V = np.linalg.eig(A)
    n = kvec / np.linalg.norm(kvec)
    cands = []
    for j in range(len(lam)):
        v = V[:, j]
        drho, dj = exp2.hydrodynamic_moments(lat, v)
        jm = float(np.linalg.norm(dj))
        nv = exp2.weighted_norm(lat, v)
        hydro = jm / nv if nv > 0 else 0.0
        if jm > 0:
            long = float(abs(n @ dj) / jm)
            dens = float(abs(drho) / jm)
        else:
            long, dens = 1.0, float("inf")
        score = hydro * (1.0 - long) / (1.0 + dens)
        cands.append(EigenCandidate(j, complex(lam[j]), hydro, long, dens, score))
    return A, V, cands


def normalize_mode(lat, v: np.ndarray, rho0: float = RHO0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale v so that ||U||_2 = 1 with U = sum_i c_i v_i / rho0, and fix the
    global phase deterministically: U is rotated to be maximally real
    (phase from sum_a U_a^2) and its largest-magnitude real component is made
    positive. The phase convention cannot affect any reported error.
    """
    U = (lat.c.T @ v) / rho0
    s = 1.0 / np.linalg.norm(U)
    v, U = v * s, U * s
    z = np.sum(U * U)
    phase = np.exp(-0.5j * np.angle(z)) if abs(z) > 1e-14 else np.exp(-1j * np.angle(U[np.argmax(np.abs(U))]))
    v, U = v * phase, U * phase
    a = int(np.argmax(np.abs(U.real)))
    if U.real[a] < 0:
        v, U = -v, -U
    return v, U


def find_shear_modes(lat, name: str, N: int, tau: float = TAU, rho0: float = RHO0) -> List[ShearMode]:
    """Return the two transverse shear eigenmodes (unlabelled) with full diagnostics."""
    kvec = k_vector(N)
    n = kvec / np.linalg.norm(kvec)
    A, V, cands = classify_eigenpairs(lat, kvec, tau)
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    top, third = ranked[:2], ranked[2]
    margin = third.score / top[1].score if top[1].score > 0 else float("inf")
    lam_all = np.array([c.eigenvalue for c in cands])
    shear_idx = {top[0].index, top[1].index}

    modes = []
    for c, other in ((top[0], top[1]), (top[1], top[0])):
        v, U = normalize_mode(lat, V[:, c.index], rho0)
        drho, _ = exp2.hydrodynamic_moments(lat, v)
        lam = c.eigenvalue
        res = float(np.linalg.norm(A @ v - lam * v) / np.linalg.norm(v))
        others = [abs(lam - lam_all[j]) for j in range(len(lam_all)) if j not in shear_idx]
        flags = []
        if margin >= SELECTION_MARGIN_MAX:
            flags.append("ambiguous_selection")
        if c.longitudinal_fraction > LONGITUDINAL_MAX:
            flags.append("not_transverse")
        gap = abs(lam - other.eigenvalue)
        if gap < DEGENERACY_TOL:
            flags.append("degenerate_shear_pair")
        modes.append(ShearMode(
            lattice=name, N=N, kvec=kvec, eigenvalue=lam, v=v, U=U, drho=complex(drho),
            longitudinal_fraction=float(abs(n @ U) / np.linalg.norm(U)),
            align_e1=float(abs(E1 @ U)), align_e2=float(abs(E2 @ U)),
            eig_residual=res, shear_gap=float(gap), nonshear_separation=float(min(others)),
            selection_score=c.score, selection_margin=float(margin), flags=flags,
        ))
    return modes


def track_branches(per_N: Dict[int, List[ShearMode]]) -> None:
    """
    Label the two branches consistently across N. At the smallest k (largest N)
    's1' is the branch with the larger real eigenvalue (slower decay). Moving to
    larger k, each branch is continued by maximum polarization overlap
    |U_prev^H U_next|; overlaps below TRACKING_OVERLAP_MIN, or a change of
    eigenvalue ordering, are flagged rather than resolved silently.
    """
    Ns = sorted(per_N, reverse=True)
    first = sorted(per_N[Ns[0]], key=lambda m: -m.eigenvalue.real)
    for lab, m in zip(BRANCHES, first):
        m.branch, m.tracking_overlap = lab, 1.0
    prev = {m.branch: m for m in first}
    for N in Ns[1:]:
        a, b = per_N[N]
        o = lambda p, q: abs(np.vdot(p.U, q.U))  # noqa: E731
        straight = o(prev["s1"], a) + o(prev["s2"], b)
        swapped = o(prev["s1"], b) + o(prev["s2"], a)
        s1, s2 = (a, b) if straight >= swapped else (b, a)
        s1.branch, s2.branch = "s1", "s2"
        s1.tracking_overlap, s2.tracking_overlap = o(prev["s1"], s1), o(prev["s2"], s2)
        for m in (s1, s2):
            if m.tracking_overlap < TRACKING_OVERLAP_MIN:
                m.flags.append("tracking_ambiguous")
        if s1.eigenvalue.real < s2.eigenvalue.real:
            s1.flags.append("eigenvalue_order_swapped")
            s2.flags.append("eigenvalue_order_swapped")
        prev = {"s1": s1, "s2": s2}


def experiment_ii_selector_check(lat, name: str, N: int, modes: Sequence[ShearMode], tau: float = TAU) -> Dict:
    """Does Experiment II's select_shear_mode (targets e1, e2) recover both branches?"""
    kvec = k_vector(N)
    picks = [exp2.select_shear_mode(lat, kvec, e, tau) for e in (E1, E2)]

    def which(p):
        vp = p.eigenvector / np.linalg.norm(p.eigenvector)
        ov = [abs(np.vdot(m.v / np.linalg.norm(m.v), vp)) for m in modes]
        j = int(np.argmax(ov))
        return modes[j].branch, ov[j]

    (b1, o1), (b2, o2) = which(picks[0]), which(picks[1])
    return {
        "lattice": name, "N": N,
        "expII_e1_eigenvalue_real": picks[0].eigenvalue.real, "expII_e1_branch": b1, "expII_e1_overlap": o1,
        "expII_e2_eigenvalue_real": picks[1].eigenvalue.real, "expII_e2_branch": b2, "expII_e2_overlap": o2,
        "expII_finds_both_branches": b1 != b2,
    }


# -----------------------------------------------------------------------------
# Reference and reconstruction
# -----------------------------------------------------------------------------

def curvature_tensor(kvec: np.ndarray, U: np.ndarray) -> np.ndarray:
    """J[a, b, g] = d_b d_g u_a = -k_b k_g U_a for u = U exp(i k.x)."""
    return -np.einsum("b,g,a->abg", kvec, kvec, U)


def curvature_coordinates(J: np.ndarray) -> np.ndarray:
    """The 15 independent coordinates of J in the canonical order (Q_NAMES)."""
    ax = "xyz"
    return np.array([J[ax.index(nm[1]), ax.index(nm[3]), ax.index(nm[4])] for nm in exact.Q_NAMES])


@dataclass
class ModeEvaluation:
    mode: ShearMode
    g_K: np.ndarray
    d_rec: np.ndarray
    d_exact: np.ndarray
    q_F: np.ndarray
    g_ce2: np.ndarray
    E_rec: float
    E_pop: float
    E_pop_hermite3: float
    E_pop_complement: float
    T_abs: float
    T_rel: float
    T_exact_over_d: float           # |T_exact| / ||d_exact||
    direct_vs_projected: float      # ||R^H v / C0 - d_rec|| / ||d_rec||
    PKN_vs_PK: float                # ||P_K N v - P_K v||_w / ||P_K v||_w
    qF_vs_expII: float              # ||q_F - q_fourier(k, U)|| / ||q_F||
    ce2_control: float              # ||reconstruct(C0 Lhat q_F) - d_exact|| / ||d_exact||
    flags: List[str]


def hermite3_projector(lat) -> np.ndarray:
    """W^-1-orthogonal projector onto span{w_i H3_k(c_i)} (10 third-order Hermite columns)."""
    H = np.array(exact.hermite3_matrix([tuple(int(x) for x in c) for c in lat.c]).tolist(), dtype=float)  # (10, Q)
    Phi = (lat.w[:, None] * H.T)                                                                         # (Q, 10)
    r = np.linalg.matrix_rank(Phi)
    U_, s, _ = np.linalg.svd(np.diag(1.0 / np.sqrt(lat.w)) @ Phi, full_matrices=False)
    B = U_[:, :r]                                                       # orthonormal basis in W^-1/2-scaled space
    return np.diag(np.sqrt(lat.w)) @ B @ B.T @ np.diag(1.0 / np.sqrt(lat.w))


def evaluate_mode(lat, name: str, mode: ShearMode, tau: float = TAU, rho0: float = RHO0,
                  D: Optional[np.ndarray] = None, P3: Optional[np.ndarray] = None) -> ModeEvaluation:
    D = observable_matrix(name) if D is None else D
    P3 = hermite3_projector(lat) if P3 is None else P3
    v = mode.v.astype(np.complex128)
    flags = list(mode.flags)

    # Actual kinetic response of the eigenvector (no k, U or q is passed on).
    f_neq = lat.N @ v
    g_K = recon.project_kinetic(f_neq, name)
    d_rec = recon.reconstruct_from_kinetic(g_K, name, tau, rho0)

    # Independent analytical reference from the eigenvector's own momentum.
    q_F = curvature_coordinates(curvature_tensor(mode.kvec, mode.U))
    d_exact = D @ q_F
    nd = float(np.linalg.norm(d_exact))
    if nd < REFERENCE_FLOOR:
        flags.append("reference_near_zero")
        E_rec = float("nan")
    else:
        E_rec = float(np.linalg.norm(d_rec - d_exact) / nd)

    # Population-space CE2 diagnostic (not used for reconstruction).
    C0 = exp2.ce2_prefactor(tau, rho0)
    g_ce2 = C0 * (lat.lhat @ q_F)
    n_ce2 = exp2.weighted_norm(lat, g_ce2)
    diff = g_K - g_ce2
    E_pop = exp2.weighted_norm(lat, diff) / n_ce2
    E_pop3 = exp2.weighted_norm(lat, P3 @ diff) / n_ce2
    E_popc = exp2.weighted_norm(lat, diff - P3 @ diff) / n_ce2

    # T where it is reconstructed.
    names = recon.observable_names(name)
    T_abs = T_rel = T_ratio = float("nan")
    if "T" in names:
        iT = names.index("T")
        T_abs = float(abs(d_rec[iT] - d_exact[iT]))
        T_ratio = float(abs(d_exact[iT])) / max(nd, REFERENCE_FLOOR)
        if T_ratio >= T_RELATIVE_FLOOR:
            T_rel = T_abs / float(abs(d_exact[iT]))
        else:
            flags.append("T_reference_small")

    # Consistency checks.
    direct = (recon.reconstruction_matrix(name) @ v) / recon.ce2_prefactor(tau, rho0)
    pk_v = recon.project_kinetic(v, name)
    return ModeEvaluation(
        mode=mode, g_K=g_K, d_rec=d_rec, d_exact=d_exact, q_F=q_F, g_ce2=g_ce2,
        E_rec=E_rec, E_pop=float(E_pop), E_pop_hermite3=float(E_pop3), E_pop_complement=float(E_popc),
        T_abs=T_abs, T_rel=T_rel, T_exact_over_d=T_ratio,
        direct_vs_projected=float(np.linalg.norm(direct - d_rec) / np.linalg.norm(d_rec)),
        PKN_vs_PK=float(exp2.weighted_norm(lat, g_K - pk_v) / exp2.weighted_norm(lat, pk_v)),
        qF_vs_expII=float(np.linalg.norm(q_F - exp2.q_fourier(mode.kvec, mode.U)) / np.linalg.norm(q_F)),
        ce2_control=float(np.linalg.norm(recon.reconstruct_from_kinetic(g_ce2, name, tau, rho0) - d_exact) / nd),
        flags=sorted(set(flags)),
    )


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    evaluations: List[ModeEvaluation]
    selector_checks: List[Dict]


def run_experiment(lattices: Iterable[str] = LATTICES, N_values: Iterable[int] = N_VALUES,
                   tau: float = TAU, rho0: float = RHO0) -> ExperimentResult:
    evals, checks = [], []
    for name in lattices:
        lat = load_lattice(name)
        D, P3 = observable_matrix(name), hermite3_projector(lat)
        per_N = {N: find_shear_modes(lat, name, N, tau, rho0) for N in N_values}
        track_branches(per_N)
        for N in sorted(per_N):
            for m in sorted(per_N[N], key=lambda m: m.branch):
                evals.append(evaluate_mode(lat, name, m, tau, rho0, D, P3))
            checks.append(experiment_ii_selector_check(lat, name, N, per_N[N], tau))
    return ExperimentResult(evals, checks)


# -----------------------------------------------------------------------------
# Convergence fits
# -----------------------------------------------------------------------------

def loglog_fit(k: np.ndarray, E: np.ndarray) -> Dict[str, float]:
    x, y = np.log(k), np.log(E)
    p, logC = np.polyfit(x, y, 1)
    yhat = p * x + logC
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"p": float(p), "C": float(np.exp(logC)),
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "max_abs_log_residual": float(np.max(np.abs(y - yhat)))}


def usable(e: ModeEvaluation) -> bool:
    bad = {"ambiguous_selection", "not_transverse", "degenerate_shear_pair", "tracking_ambiguous",
           "reference_near_zero"}
    return not (bad & set(e.flags))


def convergence_fits(evals: Sequence[ModeEvaluation], N_min: int = FIT_N_MIN) -> List[Dict]:
    """Fits E ~ C k^p per lattice/branch for E_rec, E_pop, the Hermite-3 split and T_rel."""
    rows = []
    quantities = (("reconstruction_error", lambda e: e.E_rec),
                  ("population_error", lambda e: e.E_pop),
                  ("population_error_hermite3", lambda e: e.E_pop_hermite3),
                  ("population_error_complement", lambda e: e.E_pop_complement),
                  ("T_relative_error", lambda e: e.T_rel),
                  ("T_error_over_d_norm", lambda e: e.T_abs / float(np.linalg.norm(e.d_exact))))
    keys = sorted({(e.mode.lattice, e.mode.branch) for e in evals})
    for lattice, branch in keys:
        group = sorted([e for e in evals if (e.mode.lattice, e.mode.branch) == (lattice, branch)],
                       key=lambda e: e.mode.N)
        for qname, get in quantities:
            for label, nmin in (("asymptotic", N_min), ("all", 0)):
                pts = [e for e in group if e.mode.N >= nmin]
                excluded = [e.mode.N for e in pts if not usable(e) or not np.isfinite(get(e)) or get(e) <= 0]
                pts = [e for e in pts if e.mode.N not in excluded]
                if len(pts) < 3:
                    continue
                k = np.array([np.linalg.norm(e.mode.kvec) for e in pts])
                E = np.array([get(e) for e in pts])
                fit = loglog_fit(k, E)
                rows.append({
                    "lattice": lattice, "branch": branch, "quantity": qname, "interval": label,
                    "N_min": min(e.mode.N for e in pts), "N_max": max(e.mode.N for e in pts),
                    "k_min": float(k.min()), "k_max": float(k.max()), "n_points": len(pts),
                    "excluded_N": " ".join(str(n) for n in excluded),
                    "p": fit["p"], "C": fit["C"], "R2": fit["R2"],
                    "max_abs_log_residual": fit["max_abs_log_residual"],
                })
    return rows


def local_slopes(evals: Sequence[ModeEvaluation]) -> List[Dict]:
    """Consecutive-point exponents d log E / d log k (for choosing the fit interval)."""
    rows = []
    for lattice, branch in sorted({(e.mode.lattice, e.mode.branch) for e in evals}):
        g = sorted([e for e in evals if (e.mode.lattice, e.mode.branch) == (lattice, branch)],
                   key=lambda e: -e.mode.N)
        for a, b in zip(g[:-1], g[1:]):
            ka, kb = np.linalg.norm(a.mode.kvec), np.linalg.norm(b.mode.kvec)
            rows.append({
                "lattice": lattice, "branch": branch, "N_from": a.mode.N, "N_to": b.mode.N,
                "slope_reconstruction": math.log(b.E_rec / a.E_rec) / math.log(kb / ka),
                "slope_population": math.log(b.E_pop / a.E_pop) / math.log(kb / ka),
            })
    return rows


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def result_rows(evals: Sequence[ModeEvaluation], tau: float = TAU, rho0: float = RHO0) -> List[Dict]:
    rows = []
    for e in evals:
        m = e.mode
        rows.append({
            "lattice": m.lattice, "N": m.N,
            "kx": m.kvec[0], "ky": m.kvec[1], "kz": m.kvec[2], "k_norm": float(np.linalg.norm(m.kvec)),
            "tau": tau, "rho0": rho0, "branch": m.branch,
            "eigenvalue_real": m.eigenvalue.real, "eigenvalue_imag": m.eigenvalue.imag,
            "eig_residual": m.eig_residual,
            "density_contamination": abs(m.drho), "longitudinal_fraction": m.longitudinal_fraction,
            "align_e1": m.align_e1, "align_e2": m.align_e2,
            "shear_gap": m.shear_gap, "nonshear_separation": m.nonshear_separation,
            "selection_score": m.selection_score, "selection_margin": m.selection_margin,
            "tracking_overlap": m.tracking_overlap,
            "U_x_real": m.U[0].real, "U_x_imag": m.U[0].imag, "U_y_real": m.U[1].real,
            "U_y_imag": m.U[1].imag, "U_z_real": m.U[2].real, "U_z_imag": m.U[2].imag,
            "d_exact_norm": float(np.linalg.norm(e.d_exact)),
            "reconstruction_error": e.E_rec, "population_error": e.E_pop,
            "population_error_hermite3": e.E_pop_hermite3,
            "population_error_complement": e.E_pop_complement,
            "T_absolute_error": e.T_abs, "T_relative_error": e.T_rel,
            "T_exact_over_d_norm": e.T_exact_over_d,
            "T_absolute_error_over_d_norm": e.T_abs / float(np.linalg.norm(e.d_exact)),
            "direct_vs_projected": e.direct_vs_projected, "PKN_vs_PK": e.PKN_vs_PK,
            "qF_vs_expII": e.qF_vs_expII, "ce2_control_error": e.ce2_control,
            "flags": ";".join(e.flags),
        })
    return rows


def observable_rows(evals: Sequence[ModeEvaluation]) -> List[Dict]:
    rows = []
    for e in evals:
        m = e.mode
        nd = float(np.linalg.norm(e.d_exact))
        for name, dr, de in zip(recon.observable_names(m.lattice), e.d_rec, e.d_exact):
            rows.append({
                "lattice": m.lattice, "N": m.N, "k_norm": float(np.linalg.norm(m.kvec)), "branch": m.branch,
                "observable": name,
                "exact_real": de.real, "exact_imag": de.imag, "reconstructed_real": dr.real,
                "reconstructed_imag": dr.imag, "absolute_error": float(abs(dr - de)),
                "absolute_error_over_d_norm": float(abs(dr - de)) / nd,
            })
    return rows


def write_csv(rows: Sequence[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _csv_value(v) for k, v in r.items()})


def _csv_value(v):
    """Plain, round-trippable text: numpy scalars are converted; floats use repr (shortest exact)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return v


# Validated categorical slots (dataviz reference palette, light surface).
LATTICE_STYLE = {
    "D3Q15": {"color": "#2a78d6", "marker": "o"},
    "D3Q19": {"color": "#eb6834", "marker": "s"},
    "D3Q27": {"color": "#1baf7a", "marker": "^"},
}
BRANCH_STYLE = {"s1": {"linestyle": "-", "fill": True}, "s2": {"linestyle": "--", "fill": False}}


def make_figure(evals: Sequence[ModeEvaluation], quantity: str, ylabel: str, path: Path,
                ref_slopes: Sequence[int] = (1, 2)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    allk, allE = [], []
    for lattice in LATTICES:
        for branch in BRANCHES:
            g = sorted([e for e in evals if e.mode.lattice == lattice and e.mode.branch == branch],
                       key=lambda e: np.linalg.norm(e.mode.kvec))
            if not g:
                continue
            k = np.array([np.linalg.norm(e.mode.kvec) for e in g])
            E = np.array([getattr(e, quantity) for e in g])
            st, bs = LATTICE_STYLE[lattice], BRANCH_STYLE[branch]
            ax.loglog(k, E, linestyle=bs["linestyle"], linewidth=1.6, color=st["color"],
                      marker=st["marker"], markersize=6,
                      markerfacecolor=st["color"] if bs["fill"] else "white",
                      markeredgecolor=st["color"], markeredgewidth=1.4,
                      label=f"{lattice} {branch}")
            allk += list(k)
            allE += list(E)
    # Slope guides from a common point below the lowest curve at the smallest k.
    allk, allE = np.array(allk), np.array(allE)
    k0, k1 = allk.min(), allk.max()
    anchor = 0.45 * allE[np.isclose(allk, k0)].min()
    kk = np.geomspace(k0, k1, 2)
    for p in ref_slopes:
        ax.loglog(kk, anchor * (kk / k0) ** p, color="#8a8a85", linewidth=1.0, linestyle=":", zorder=0)
        ax.annotate(f"$\\propto k^{p}$", xy=(k1, anchor * (k1 / k0) ** p), xytext=(4, 0),
                    textcoords="offset points", color="#55554f", fontsize=9, va="center")
    ax.set_xlabel(r"$|\mathbf{k}|$  (direction [123])")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    ax.grid(False, which="minor")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(result: ExperimentResult, outdir: Path) -> Dict[str, Path]:
    paths = {
        "results": outdir / "experiment_IV_linearized_results.csv",
        "observables": outdir / "experiment_IV_linearized_observables.csv",
        "fits": outdir / "experiment_IV_linearized_convergence_fits.csv",
        "slopes": outdir / "experiment_IV_linearized_local_slopes.csv",
        "selector": outdir / "experiment_IV_linearized_selector_check.csv",
        "fig_rec": outdir / "figure_IV_1_reconstruction_error.png",
        "fig_pop": outdir / "figure_IV_2_population_error.png",
    }
    write_csv(result_rows(result.evaluations), paths["results"])
    write_csv(observable_rows(result.evaluations), paths["observables"])
    write_csv(convergence_fits(result.evaluations), paths["fits"])
    write_csv(local_slopes(result.evaluations), paths["slopes"])
    write_csv(result.selector_checks, paths["selector"])
    make_figure(result.evaluations, "E_rec", r"$E_Q^{rec}$ (relative, complex 2-norm)", paths["fig_rec"])
    make_figure(result.evaluations, "E_pop", r"$E_Q^{pop}$ (relative, weighted norm)", paths["fig_pop"])
    return paths


def print_summary(result: ExperimentResult) -> None:
    ev = result.evaluations
    print("Experiment IV, Step 3 - reconstruction from exact linearized BGK shear eigenmodes")
    print(f"tau = {TAU}, rho0 = {RHO0}, k = (2 pi / N)(1,2,3), N = {list(N_VALUES)}\n")
    print(f"{'lattice':<7}{'br':<4}{'N':>5}{'|k|':>8}{'lambda':>12}{'|drho|':>10}{'long':>10}"
          f"{'E_rec':>11}{'E_pop':>11}{'E_pop,H3':>11}{'T_rel':>11}  flags")
    for e in ev:
        m = e.mode
        print(f"{m.lattice:<7}{m.branch:<4}{m.N:>5}{np.linalg.norm(m.kvec):>8.4f}{m.eigenvalue.real:>12.8f}"
              f"{abs(m.drho):>10.2e}{m.longitudinal_fraction:>10.2e}{e.E_rec:>11.3e}{e.E_pop:>11.3e}"
              f"{e.E_pop_hermite3:>11.3e}{e.T_rel:>11.3e}  {';'.join(e.flags)}")
    print("\nMaxima over all cases:")
    print(f"  eigenpair residual ||Av - lam v||/||v||   {max(e.mode.eig_residual for e in ev):.2e}")
    print(f"  ||P_K N v - P_K v||_w / ||P_K v||_w       {max(e.PKN_vs_PK for e in ev):.2e}")
    print(f"  projected vs direct R^H reconstruction    {max(e.direct_vs_projected for e in ev):.2e}")
    print(f"  q_F vs Experiment II q_fourier            {max(e.qF_vs_expII for e in ev):.2e}")
    print(f"  synthetic CE2 control error               {max(e.ce2_control for e in ev):.2e}")
    print(f"  selection margin (3rd/2nd score)          {max(e.mode.selection_margin for e in ev):.2e}")
    print(f"  min shear-pair gap |lam1 - lam2|          {min(e.mode.shear_gap for e in ev):.2e}")
    print(f"  min separation from non-shear eigenvalues {min(e.mode.nonshear_separation for e in ev):.2e}")
    print(f"  min tracking overlap                      {min(e.mode.tracking_overlap for e in ev):.6f}")
    blocking = sum(not usable(e) for e in ev)
    tsmall = sum("T_reference_small" in e.flags for e in ev)
    print(f"  cases excluded from fits (identification/reference flags) {blocking}")
    print(f"  cases with |T_exact| < {T_RELATIVE_FLOOR:g} ||d_exact|| (T_rel not reported) {tsmall}")
    found = sum(c["expII_finds_both_branches"] for c in result.selector_checks)
    print(f"  Experiment II selector finds both branches in {found}/{len(result.selector_checks)} (lattice, N) cases")

    print("\nConvergence fits, asymptotic interval (N >= %d):" % FIT_N_MIN)
    print(f"{'lattice':<7}{'br':<4}{'quantity':<30}{'p':>8}{'R2':>10}{'n':>4}  k-range")
    for r in convergence_fits(ev):
        if r["interval"] == "asymptotic":
            print(f"{r['lattice']:<7}{r['branch']:<4}{r['quantity']:<30}{r['p']:>8.3f}{r['R2']:>10.6f}"
                  f"{r['n_points']:>4}  [{r['k_min']:.4f}, {r['k_max']:.4f}]")


if __name__ == "__main__":
    res = run_experiment()
    print_summary(res)
