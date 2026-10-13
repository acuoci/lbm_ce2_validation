#!/usr/bin/env python3
"""
Experiment IV, Step 3B: velocity-parity analysis of finite-wavenumber errors.

Working hypothesis (tested here, not assumed): the leading finite-wavenumber
population error of the exact linearized shear eigenmodes has even velocity
parity and is annihilated by the odd third-order Hermite reconstruction R^H_Q.

Velocity reversal (P g)_i = g_ibar with c_ibar = -c_i. P_+ = (I+P)/2, P_- = (I-P)/2.
For each Step 3 eigenmode (same eigenvectors, normalization, CE2 prediction and
weighted norm):

    e      = g_K^LBM - C0 Lhat q_F,   e^pm = P_pm e
    E^pm   = ||e^pm||_w / ||C0 Lhat q_F||_w
    E^rec  = ||R^H e / C0||_2 / ||D q_F||_2         (compared with the Step 3 value)
    r      = ||e^-||_w / ||e^+||_w

Nothing in Step 3 (or earlier) is modified; this module only reads the Step 3
evaluations.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
exp2, recon, exact = lin.exp2, lin.recon, lin.exact

LATTICES = lin.LATTICES
N_VALUES = lin.N_VALUES
FIT_N_MIN = lin.FIT_N_MIN


# -----------------------------------------------------------------------------
# Parity operators
# -----------------------------------------------------------------------------

def reversal_index(c: np.ndarray) -> np.ndarray:
    """ibar for every i, with c[ibar] = -c[i]; raises if the set is not closed under reversal."""
    lookup = {tuple(int(x) for x in ci): i for i, ci in enumerate(c)}
    idx = []
    for ci in c:
        key = tuple(-int(x) for x in ci)
        if key not in lookup:
            raise ValueError(f"Velocity set not closed under reversal: {tuple(ci)}")
        idx.append(lookup[key])
    return np.array(idx, dtype=int)


def reversal_matrix(c: np.ndarray) -> np.ndarray:
    """Permutation matrix P with (P g)_i = g_ibar."""
    Q = len(c)
    P = np.zeros((Q, Q))
    P[np.arange(Q), reversal_index(c)] = 1.0
    return P


def parity_projectors(c: np.ndarray):
    P = reversal_matrix(c)
    Ident = np.eye(len(c))
    return P, 0.5 * (Ident + P), 0.5 * (Ident - P)


def identity_residuals(name: str, N_values: Iterable[int] = N_VALUES, tau: float = lin.TAU) -> Dict[str, float]:
    """Max-abs residuals of all parity identities used in the analysis (float64)."""
    lat = lin.load_lattice(name)
    P, Pp, Pm = parity_projectors(lat.c)
    Q = len(lat.w)
    RH, PK = recon.reconstruction_matrix(name), recon.kinetic_projector(name)
    C = exp2.collision_matrix(lat, tau)
    mx = lambda M: float(np.max(np.abs(M)))  # noqa: E731
    out = {
        "P_involution": mx(P @ P - np.eye(Q)),
        "Pplus_idempotent": mx(Pp @ Pp - Pp),
        "Pminus_idempotent": mx(Pm @ Pm - Pm),
        "Pplus_Pminus": mx(Pp @ Pm),
        "Pplus_plus_Pminus": mx(Pp + Pm - np.eye(Q)),
        "velocities_reversed": mx(lat.c[reversal_index(lat.c)] + lat.c),
        "weights_symmetric": mx(lat.w[reversal_index(lat.c)] - lat.w),
        "RH_Pplus": mx(RH @ Pp),
        "Pplus_Lhat": mx(Pp @ lat.lhat),
        "commutator_P_PK": mx(P @ PK - PK @ P),
        "commutator_P_N": mx(P @ lat.N - lat.N @ P),
        "commutator_P_C": mx(P @ C - C @ P),
        "P_A_P_minus_A_of_minus_k": max(
            mx(P @ exp2.amplification_matrix(lat, lin.k_vector(N), tau) @ P
               - exp2.amplification_matrix(lat, -lin.k_vector(N), tau)) for N in N_values),
    }
    return out


def exact_RH_parity(name: str) -> bool:
    """Exact (SymPy) check that R^H P = -R^H, i.e. R^H P_+ = 0, on the canonical velocity set."""
    import sympy as sp
    op = exact.load_operator(name)
    RH = exact.hermite_reconstruction(op)
    idx = reversal_index(np.array(op.velocities))
    P = sp.zeros(op.Q, op.Q)
    for i, j in enumerate(idx):
        P[i, int(j)] = 1
    return RH * P == -RH and RH * (sp.eye(op.Q) + P) / 2 == sp.zeros(RH.rows, op.Q)


# -----------------------------------------------------------------------------
# Branch parity check: v(-k) = -P v(k) for the U-normalized branch
# -----------------------------------------------------------------------------

def mirrored_branch_residual(lat, name: str, mode, tau: float = lin.TAU) -> float:
    """
    Compute the shear eigenvectors at -k with the Step 3 selection and
    normalization, pick the one whose U matches mode.U, and return
    min_phi ||e^{i phi} v(-k) + P v(k)|| / ||v(k)||. A value at round-off level
    means the computed branch satisfies v(-k) = -P v(k) (U is preserved because
    U(P v) = -U(v)).
    """
    P = reversal_matrix(lat.c)
    _, V, cands = lin.classify_eigenpairs(lat, -mode.kvec, tau)
    top = sorted(cands, key=lambda c: c.score, reverse=True)[:2]
    best = None
    for c in top:
        v, U = lin.normalize_mode(lat, V[:, c.index])
        ov = abs(np.vdot(U, mode.U))
        if best is None or ov > best[0]:
            best = (ov, v)
    v_minus = best[1]
    target = -P @ mode.v
    z = np.vdot(v_minus, target)
    phase = z / abs(z) if abs(z) > 0 else 1.0
    return float(np.linalg.norm(phase * v_minus - target) / np.linalg.norm(mode.v))


# -----------------------------------------------------------------------------
# Error decomposition
# -----------------------------------------------------------------------------

@dataclass
class ParityRecord:
    lattice: str
    N: int
    k: float
    branch: str
    E_plus: float
    E_minus: float
    E_rec_parity: float
    E_rec_step3: float
    ratio_r: float
    abs_e_plus: float
    abs_e_minus: float
    reconstruction_odd_only: float     # ||R^H e - R^H e^-|| / ||R^H e||
    ce2_even_part: float               # ||P_+ C0 Lhat q_F||_w / ||C0 Lhat q_F||_w
    mirrored_branch_residual: float
    flags: str


def decompose(evaluation, lat, Pp: np.ndarray, Pm: np.ndarray, tau: float = lin.TAU,
              rho0: float = lin.RHO0) -> ParityRecord:
    m = evaluation.mode
    name = m.lattice
    e = evaluation.g_K - evaluation.g_ce2
    ep, em = Pp @ e, Pm @ e
    n_ce2 = exp2.weighted_norm(lat, evaluation.g_ce2)
    c0 = exp2.ce2_prefactor(tau, rho0)
    RH = recon.reconstruction_matrix(name)
    Re = RH @ e
    E_rec = float(np.linalg.norm(Re / c0) / np.linalg.norm(evaluation.d_exact))
    a_p, a_m = exp2.weighted_norm(lat, ep), exp2.weighted_norm(lat, em)
    return ParityRecord(
        lattice=name, N=m.N, k=float(np.linalg.norm(m.kvec)), branch=m.branch,
        E_plus=a_p / n_ce2, E_minus=a_m / n_ce2, E_rec_parity=E_rec, E_rec_step3=evaluation.E_rec,
        ratio_r=a_m / a_p, abs_e_plus=a_p, abs_e_minus=a_m,
        reconstruction_odd_only=float(np.linalg.norm(Re - RH @ em) / np.linalg.norm(Re)),
        ce2_even_part=exp2.weighted_norm(lat, Pp @ evaluation.g_ce2) / n_ce2,
        mirrored_branch_residual=mirrored_branch_residual(lat, name, m, tau),
        flags=";".join(evaluation.flags),
    )


def run_analysis(lattices: Iterable[str] = LATTICES, N_values: Iterable[int] = N_VALUES) -> List[ParityRecord]:
    result = lin.run_experiment(lattices, N_values)
    lats = {n: lin.load_lattice(n) for n in lattices}
    proj = {n: parity_projectors(lats[n].c) for n in lattices}
    return [decompose(e, lats[e.mode.lattice], proj[e.mode.lattice][1], proj[e.mode.lattice][2])
            for e in result.evaluations]


# -----------------------------------------------------------------------------
# Fits and output
# -----------------------------------------------------------------------------

FIT_QUANTITIES = (
    ("E_plus", "relative even error E^+"),
    ("E_minus", "relative odd error E^-"),
    ("E_rec_parity", "reconstruction error from R^H e"),
    ("ratio_r", "r = ||e^-|| / ||e^+||"),
    ("abs_e_plus", "absolute ||e^+||_w"),
    ("abs_e_minus", "absolute ||e^-||_w"),
)
PREDICTED = {"E_plus": 1, "E_minus": 2, "E_rec_parity": 2, "ratio_r": 1, "abs_e_plus": 3, "abs_e_minus": 4}


def parity_fits(records: Sequence[ParityRecord], N_min: int = FIT_N_MIN) -> List[Dict]:
    rows = []
    for lattice, branch in sorted({(r.lattice, r.branch) for r in records}):
        g = sorted([r for r in records if (r.lattice, r.branch) == (lattice, branch)], key=lambda r: r.N)
        for qname, _ in FIT_QUANTITIES:
            for label, nmin in (("asymptotic", N_min), ("all", 0)):
                pts = [r for r in g if r.N >= nmin]
                excluded = [r.N for r in pts if r.flags and any(f in r.flags for f in (
                    "ambiguous_selection", "not_transverse", "degenerate_shear_pair", "tracking_ambiguous",
                    "reference_near_zero"))]
                pts = [r for r in pts if r.N not in excluded]
                if len(pts) < 3:
                    continue
                fit = lin.loglog_fit(np.array([r.k for r in pts]), np.array([getattr(r, qname) for r in pts]))
                rows.append({
                    "lattice": lattice, "branch": branch, "quantity": qname, "interval": label,
                    "N_min": min(r.N for r in pts), "N_max": max(r.N for r in pts),
                    "k_min": min(r.k for r in pts), "k_max": max(r.k for r in pts), "n_points": len(pts),
                    "excluded_N": " ".join(map(str, excluded)),
                    "p": fit["p"], "predicted_p": PREDICTED[qname], "p_minus_predicted": fit["p"] - PREDICTED[qname],
                    "C": fit["C"], "R2": fit["R2"], "max_abs_log_residual": fit["max_abs_log_residual"],
                })
    return rows


def record_rows(records: Sequence[ParityRecord]) -> List[Dict]:
    rows = []
    for r in records:
        rows.append({
            "lattice": r.lattice, "N": r.N, "k_norm": r.k, "branch": r.branch,
            "E_plus": r.E_plus, "E_minus": r.E_minus, "ratio_r": r.ratio_r,
            "E_rec_parity": r.E_rec_parity, "E_rec_step3": r.E_rec_step3,
            "E_rec_relative_difference": abs(r.E_rec_parity - r.E_rec_step3) / r.E_rec_step3,
            "abs_e_plus": r.abs_e_plus, "abs_e_minus": r.abs_e_minus,
            "abs_e_plus_over_k3": r.abs_e_plus / r.k ** 3, "abs_e_minus_over_k4": r.abs_e_minus / r.k ** 4,
            "reconstruction_odd_only": r.reconstruction_odd_only, "ce2_even_part": r.ce2_even_part,
            "mirrored_branch_residual": r.mirrored_branch_residual, "flags": r.flags,
        })
    return rows


def identity_rows(N_values: Iterable[int] = N_VALUES) -> List[Dict]:
    rows = []
    for name in LATTICES:
        res = identity_residuals(name, N_values)
        rows.append({"lattice": name, **res, "RH_Pplus_exact_sympy": exact_RH_parity(name)})
    return rows


def make_parity_figure(records: Sequence[ParityRecord], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    panels = (("E_plus", r"even part  $E_Q^{+}$", 1), ("E_minus", r"odd part  $E_Q^{-}$", 2))
    for ax, (q, title, p_pred) in zip(axes, panels):
        allk, allE = [], []
        for lattice in LATTICES:
            for branch in lin.BRANCHES:
                g = sorted([r for r in records if r.lattice == lattice and r.branch == branch], key=lambda r: r.k)
                k = np.array([r.k for r in g])
                E = np.array([getattr(r, q) for r in g])
                st, bs = lin.LATTICE_STYLE[lattice], lin.BRANCH_STYLE[branch]
                ax.loglog(k, E, linestyle=bs["linestyle"], linewidth=1.6, color=st["color"], marker=st["marker"],
                          markersize=6, markerfacecolor=st["color"] if bs["fill"] else "white",
                          markeredgecolor=st["color"], markeredgewidth=1.4, label=f"{lattice} {branch}")
                allk += list(k)
                allE += list(E)
        allk, allE = np.array(allk), np.array(allE)
        k0, k1 = allk.min(), allk.max()
        anchor = 0.45 * allE[np.isclose(allk, k0)].min()
        kk = np.geomspace(k0, k1, 2)
        for p in (1, 2):
            ax.loglog(kk, anchor * (kk / k0) ** p, color="#8a8a85", linewidth=1.0, linestyle=":", zorder=0)
            ax.annotate(f"$\\propto k^{p}$", xy=(k1, anchor * (k1 / k0) ** p), xytext=(4, 0),
                        textcoords="offset points", color="#55554f", fontsize=9, va="center")
        ax.set_title(f"{title}   (predicted $O(k^{p_pred})$)", fontsize=10, color="#2b2b28")
        ax.set_xlabel(r"$|\mathbf{k}|$  (direction [123])")
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$\|\mathbf{e}^{\pm}\|_{w}\,/\,\|C_0\widehat{L}_Q\mathbf{q}_F\|_{w}$")
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(records: Sequence[ParityRecord], outdir: Path) -> Dict[str, Path]:
    paths = {
        "records": outdir / "experiment_IV_parity.csv",
        "fits": outdir / "experiment_IV_parity_fits.csv",
        "identities": outdir / "experiment_IV_parity_identities.csv",
        "figure": outdir / "figure_IV_3_parity_error.png",
    }
    lin.write_csv(record_rows(records), paths["records"])
    lin.write_csv(parity_fits(records), paths["fits"])
    lin.write_csv(identity_rows(), paths["identities"])
    make_parity_figure(records, paths["figure"])
    return paths


def print_summary(records: Sequence[ParityRecord]) -> None:
    print("Experiment IV, Step 3B - velocity-parity analysis of finite-wavenumber errors\n")
    print("Parity identities (max-abs residuals, float64; exact SymPy check of R^H P_+ = 0):")
    for row in identity_rows():
        print(f"  {row['lattice']}: " + ", ".join(
            f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "lattice"))
    print(f"\n{'lattice':<7}{'br':<4}{'N':>5}{'k':>8}{'E+':>11}{'E-':>11}{'r':>10}{'E_rec':>11}"
          f"{'|dE_rec|':>10}{'|e+|/k^3':>10}{'|e-|/k^4':>10}{'mirror':>9}")
    for r in records:
        print(f"{r.lattice:<7}{r.branch:<4}{r.N:>5}{r.k:>8.4f}{r.E_plus:>11.3e}{r.E_minus:>11.3e}{r.ratio_r:>10.3e}"
              f"{r.E_rec_parity:>11.3e}{abs(r.E_rec_parity - r.E_rec_step3) / r.E_rec_step3:>10.1e}"
              f"{r.abs_e_plus / r.k ** 3:>10.4f}{r.abs_e_minus / r.k ** 4:>10.4f}{r.mirrored_branch_residual:>9.1e}")
    print("\nMaxima: |E_rec(parity) - E_rec(Step 3)| / E_rec = %.1e,  ||R^H e - R^H e^-|| / ||R^H e|| = %.1e,"
          "  even part of CE2 prediction = %.1e,  mirrored-branch residual = %.1e" % (
              max(abs(r.E_rec_parity - r.E_rec_step3) / r.E_rec_step3 for r in records),
              max(r.reconstruction_odd_only for r in records), max(r.ce2_even_part for r in records),
              max(r.mirrored_branch_residual for r in records)))
    print(f"\nFitted exponents, N >= {FIT_N_MIN} (predicted in brackets):")
    fits = [f for f in parity_fits(records) if f["interval"] == "asymptotic"]
    qs = [q for q, _ in FIT_QUANTITIES]
    print(f"{'lattice':<7}{'br':<4}" + "".join(f"{q + ' [' + str(PREDICTED[q]) + ']':>19}" for q in qs) + f"{'min R2':>10}")
    for lattice, branch in sorted({(f['lattice'], f['branch']) for f in fits}):
        row = {f["quantity"]: f for f in fits if (f["lattice"], f["branch"]) == (lattice, branch)}
        print(f"{lattice:<7}{branch:<4}" + "".join(f"{row[q]['p']:>19.3f}" for q in qs)
              + f"{min(row[q]['R2'] for q in qs):>10.6f}")


if __name__ == "__main__":
    print_summary(run_analysis())
