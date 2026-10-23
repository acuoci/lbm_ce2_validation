#!/usr/bin/env python3
"""
Experiment III: consistency audit of the amplitude tables and the U0 = 0.02 time history.

Uses the unmodified Experiment III functions (src/experiment_III_taylor_green.py):
initialize_taylor_green, collide_and_stream, diagnose_state, macroscopic. Nothing in
the solver, operators or E_Q definition is changed; this script only chooses the
timesteps at which diagnostics are recorded.

1. Audit: recomputes every table of docs/experiment_III_production_protocol_refined.md
   (startup pilot, grid study, amplitude sweep at N = 48, time window) at the stated
   timesteps and compares with the tabulated 4-decimal values.
2. Time history: D3Q19 and D3Q27, N = 48, tau = 0.8, U0 = 0.02, diagnostics at
   t = 10, 20, 30, 40, 48 (t* = U0 t / N), with numerator and denominator of E_Q:
       E_Q = ||g_LBM - g_CE2||_rms / ||g_CE2||_rms ,
   and the RMS velocity of the decaying field, so that viscous amplitude decay can be
   separated from relaxation of the kinetic discrepancy.
3. Figure: reference_results/experiment_III/time_history/figure_III_1_global_CE2_error.{png,pdf}
   (first plotted point = timestep 10).

Usage (from the repository root):
    python scripts/reproduce_experiment_III_time_history.py            # audit + time history + figure
    python scripts/reproduce_experiment_III_time_history.py --skip-audit
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("exp3", ROOT / "src" / "experiment_III_taylor_green.py")
exp3 = importlib.util.module_from_spec(_spec)
sys.modules["exp3"] = exp3
_spec.loader.exec_module(exp3)

OPERATORS = ROOT / "operators"
OUT = ROOT / "reference_results" / "experiment_III" / "time_history"

N_PROD, TAU_PROD, U0_PROD = 48, 0.8, 0.02
HISTORY_STEPS = (10, 20, 30, 40, 48)
STARTUP_DISCARD = 10

# Tables of docs/experiment_III_production_protocol_refined.md:
# (section, lattice, N, U0, step) -> (E, R, X) as printed (4 decimals).
DOC_TABLES: Dict[Tuple[str, str, int, float, int], Tuple[float, float, float]] = {}
for step, E, R, X in ((0, 1.0000, 0.0000, 0.0368), (1, 1.1248, 2.1114, 0.9929), (2, 0.4791, 0.5902, 0.9478),
                      (3, 0.2422, 1.1745, 0.9880), (5, 0.1709, 1.0373, 0.9866), (8, 0.1639, 1.0264, 0.9873),
                      (10, 0.1604, 1.0261, 0.9878), (15, 0.1517, 1.0243, 0.9891), (20, 0.1429, 1.0219, 0.9902)):
    DOC_TABLES[("2 startup", "D3Q19", 32, 0.02, step)] = (E, R, X)
for N, E, R, X in ((32, 0.0811, 1.0161, 0.9969), (48, 0.1272, 1.0138, 0.9921), (64, 0.1729, 1.0180, 0.9855)):
    DOC_TABLES[("3 grid", "D3Q19", N, 0.01, 10)] = (E, R, X)
for N, E, R, X in ((32, 0.3199, 1.0652, 0.9540), (48, 0.5081, 1.1290, 0.8930), (64, 0.6917, 1.2196, 0.8236)):
    DOC_TABLES[("3 grid", "D3Q19", N, 0.04, 10)] = (E, R, X)
for U0, E, R, X in ((0.005, 0.0638, 1.0077, 0.9980), (0.010, 0.1272, 1.0138, 0.9921),
                    (0.020, 0.2541, 1.0378, 0.9696), (0.040, 0.5081, 1.1290, 0.8930)):
    DOC_TABLES[("4 amplitude", "D3Q19", 48, U0, 10)] = (E, R, X)
for U0, E, R, X in ((0.005, 0.0814, 1.0089, 0.9968), (0.010, 0.1624, 1.0188, 0.9872),
                    (0.020, 0.3246, 1.0573, 0.9517), (0.040, 0.6491, 1.1991, 0.8409)):
    DOC_TABLES[("4 amplitude", "D3Q27", 48, U0, 10)] = (E, R, X)
for step, E, R, X in ((10, 0.1272, 1.0138, 0.9921), (20, 0.1209, 1.0131, 0.9929), (40, 0.1090, 1.0113, 0.9942),
                      (60, 0.0984, 1.0105, 0.9953), (80, 0.0889, 1.0096, 0.9961), (96, 0.0818, 1.0091, 0.9967)):
    DOC_TABLES[("6 time window", "D3Q19", 48, 0.01, step)] = (E, R, X)
for step, E, R, X in ((10, 0.1624, 1.0188, 0.9872), (40, 0.1392, 1.0149, 0.9906), (96, 0.1044, 1.0110, 0.9947)):
    DOC_TABLES[("6 time window", "D3Q27", 48, 0.01, step)] = (E, R, X)


def load_lattice(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)    # pre-existing unclosed file in load_lattice
        return exp3.load_lattice(OPERATORS / f"Lhat_{name}.csv")


def run_series(lat, N: int, U0: float, tau: float, steps: Sequence[int], rho0: float = 1.0) -> List[Dict]:
    """Same sequence as exp3.run_case (init -> diagnose/evolve), plus numerator, denominator and u_rms."""
    steps = sorted(set(int(s) for s in steps))
    f = exp3.initialize_taylor_green(lat, N, U0, rho0)
    out = []
    for step in range(steps[-1] + 1):
        if step in steps:
            rec = exp3.diagnose_state(lat, f, tau, U0, step, rho0)
            _, u = exp3.macroscopic(lat, f)
            u_rms = float(np.sqrt(np.mean(np.sum(u * u, axis=0))))
            out.append({
                "lattice": lat.name, "N": N, "tau": tau, "U0": U0, "step": step,
                "t_star": U0 * step / N, "t_star_record": rec.t_convective,
                "E": rec.E, "R": rec.R, "X": rec.X,
                "numerator_rms": rec.E * rec.norm_ce2_rms,       # ||g_LBM - g_CE2||_rms
                "denominator_rms": rec.norm_ce2_rms,             # ||g_CE2||_rms
                "norm_lbm_rms": rec.norm_lbm_rms,
                "u_rms": u_rms, "max_mach": rec.max_mach, "mass": rec.mass,
            })
        if step < steps[-1]:
            f = exp3.collide_and_stream(lat, f, tau)
    return out


def write_csv(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def audit() -> List[Dict]:
    """Recompute every tabulated value at its stated timestep."""
    runs: Dict[Tuple[str, int, float], List[int]] = {}
    for (_, L, N, U0, step) in DOC_TABLES:
        runs.setdefault((L, N, U0), []).append(step)
    computed = {}
    for (L, N, U0), steps in sorted(runs.items()):
        lat = load_lattice(L)
        for r in run_series(lat, N, U0, TAU_PROD, steps):
            computed[(L, N, U0, r["step"])] = r
    rows = []
    for (sec, L, N, U0, step), (E, R, X) in sorted(DOC_TABLES.items()):
        c = computed[(L, N, U0, step)]
        ok = all(abs(round(c[k], 4) - v) <= 1e-4 + 1e-12 for k, v in (("E", E), ("R", R), ("X", X)))
        rows.append({"section": sec, "lattice": L, "N": N, "U0": U0, "step": step, "t_star": U0 * step / N,
                     "E_doc": E, "E_computed": c["E"], "R_doc": R, "R_computed": c["R"],
                     "X_doc": X, "X_computed": c["X"], "agrees_to_4_decimals": ok})
    return rows


def decay_rates(rows: Sequence[Dict]) -> Dict[str, float]:
    """Least-squares exponential decay rates (per timestep) for t >= STARTUP_DISCARD."""
    pts = [r for r in rows if r["step"] >= STARTUP_DISCARD]
    t = np.array([r["step"] for r in pts], dtype=float)
    out = {}
    for key in ("u_rms", "denominator_rms", "numerator_rms", "E"):
        out[key] = float(-np.polyfit(t, np.log([r[key] for r in pts]), 1)[0])
    return out


def make_figure(history: Sequence[Dict], path_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    style = {"D3Q19": ("#eb6834", "s"), "D3Q27": ("#1baf7a", "^")}
    fig, ax = plt.subplots(figsize=(8.8 / 2.54, 6.4 / 2.54))
    for L in ("D3Q19", "D3Q27"):
        g = sorted([r for r in history if r["lattice"] == L and r["step"] in HISTORY_STEPS], key=lambda r: r["step"])
        c, m = style[L]
        ax.plot([r["t_star"] for r in g], [r["E"] for r in g], color=c, marker=m, markersize=4, linewidth=1.0,
                label=f"{L}")
    ax.set_xlabel(r"$t^* = U_0 t / N$")
    ax.set_ylabel(r"$\mathcal{E}_Q$")
    ax.set_xlim(0.0, 0.021)
    ax.set_xticks([0.0, 0.005, 0.010, 0.015, 0.020])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.05, r"$N=48$, $\tau=0.8$, $U_0=0.02$" + "\n" + r"$t = 10, 20, 30, 40, 48$",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left", color="#3a3a36")
    fig.tight_layout(pad=0.3)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix("." + ext), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    ok = True

    if not args.skip_audit:
        rows = audit()
        write_csv(rows, args.outdir / "experiment_III_table_audit.csv")
        bad = [r for r in rows if not r["agrees_to_4_decimals"]]
        ok &= not bad
        print(f"Audit of protocol tables: {len(rows) - len(bad)}/{len(rows)} entries reproduced to 4 decimals")
        for r in bad:
            print("  MISMATCH", r)

    # Time history at U0 = 0.02 (startup steps 0, 1, 2, 5 recorded for the transient analysis, not plotted).
    history = []
    for L in ("D3Q19", "D3Q27"):
        history += run_series(load_lattice(L), N_PROD, U0_PROD, TAU_PROD, (0, 1, 2, 5) + HISTORY_STEPS)
    write_csv(history, args.outdir / "experiment_III_time_history_U0_0.02.csv")
    make_figure(history, args.outdir / "figure_III_1_global_CE2_error")

    nu = exp3.CS2 * (TAU_PROD - 0.5)
    k2 = 3.0 * (2.0 * math.pi / N_PROD) ** 2          # |k|^2 of the (1,1,1) Taylor-Green modes
    print(f"\nTime history, N={N_PROD}, tau={TAU_PROD}, U0={U0_PROD}  (nu k^2 = {nu * k2:.6f} per step; "
          f"kinetic transient factor |1-1/tau|^t = {abs(1 - 1 / TAU_PROD):.2f}^t)")
    print(f"{'lattice':<7}{'t':>4}{'t*':>10}{'E':>9}{'numerator':>12}{'denominator':>13}{'u_rms':>11}"
          f"{'E*u0/u':>9}{'num/u^2':>10}{'den/u':>10}")
    for L in ("D3Q19", "D3Q27"):
        g = [r for r in history if r["lattice"] == L]
        ref = next(r for r in g if r["step"] == 10)
        for r in g:
            print(f"{L:<7}{r['step']:>4}{r['t_star']:>10.5f}{r['E']:>9.4f}{r['numerator_rms']:>12.4e}"
                  f"{r['denominator_rms']:>13.4e}{r['u_rms']:>11.4e}{r['E'] * ref['u_rms'] / r['u_rms']:>9.4f}"
                  f"{r['numerator_rms'] / r['u_rms'] ** 2:>10.4f}{r['denominator_rms'] / r['u_rms']:>10.5f}")
        rates = decay_rates(g)
        print(f"  decay rates per step (t >= 10): u_rms {rates['u_rms']:.6f}, denominator {rates['denominator_rms']:.6f},"
              f" numerator {rates['numerator_rms']:.6f}, E {rates['E']:.6f}  (nu k^2 = {nu * k2:.6f})")
    print(f"\nOutputs in {args.outdir.relative_to(ROOT) if args.outdir.is_relative_to(ROOT) else args.outdir}"
          f"  ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
