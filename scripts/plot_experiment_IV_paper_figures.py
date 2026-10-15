#!/usr/bin/env python3
"""
Publication figures for Experiment IV (presentation only; no new numerics).

Figure A  Figures/figure_IV_population_vs_reconstruction.{pdf,png}
          (a) E_pop vs |k| and (b) E_rec vs |k|, exact linearized shear eigenmodes
          (same data as figure_IV_2_population_error.png and figure_IV_1_reconstruction_error.png).
Figure B  Figures/figure_IV_full_bgk_vs_linearized.{pdf,png}
          E_rec vs |k|: exact linearized (lines) and full BGK, U0 = 1e-6, t = 8 (symbols)
          (same data as the left panel of figure_IV_4_full_bgk_vs_linearized.png).

Data are read from the stored validation results (no simulation is rerun):
    reference_results/experiment_IV_linearized/experiment_IV_linearized_results.csv
    reference_results/experiment_IV_linearized/experiment_IV_linearized_convergence_fits.csv
    reference_results/experiment_IV_full_bgk/experiment_IV_full_bgk_results.csv
    reference_results/experiment_IV_full_bgk/experiment_IV_full_bgk_fits.csv

Before plotting, the data are verified (lattices/branches present, k = 2 pi sqrt(14)/N,
U0 and t of the BGK subset, consistency of the linearized reference between the two
result files, no excluded points) and, unless --skip-recompute is given, the
exact-linearized values are recomputed from the Step 3 eigenmode analysis (a few
seconds, no BGK simulation) and compared with the stored values. After plotting, the
data drawn on every axis are read back and compared with the stored values exactly.
Any failed check aborts without writing figures.

Usage (from anywhere):
    python scripts/plot_experiment_IV_paper_figures.py
    python scripts/plot_experiment_IV_paper_figures.py --outdir Figures --results-dir reference_results
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reuse the lattice/branch style definitions of the Experiment IV figures.
_spec = importlib.util.spec_from_file_location(
    "experiment_IV_linearized_reconstruction", ROOT / "src" / "experiment_IV_linearized_reconstruction.py")
lin = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lin
_spec.loader.exec_module(lin)

LATTICES: Tuple[str, ...] = ("D3Q15", "D3Q19", "D3Q27")
BRANCHES: Tuple[str, ...] = ("s1", "s2")
N_LINEARIZED: Tuple[int, ...] = (24, 32, 40, 48, 64, 80, 96, 128)
N_FULL_BGK: Tuple[int, ...] = (24, 32, 48, 64, 80)
U0_FULL_BGK = 1e-6
T_FULL_BGK = 8
TAU, RHO0 = 1.0, 1.0

CM = 1.0 / 2.54
FIG_A_SIZE = (17.5 * CM, 6.6 * CM)
FIG_B_SIZE = (8.8 * CM, 7.4 * CM)
GUIDE_COLOR = "#7a7a75"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


Series = Dict[Tuple[str, str], List[Tuple[int, float, float]]]   # (lattice, branch) -> [(N, k, value)]


def series(rows: Sequence[Dict[str, str]], column: str) -> Series:
    out: Series = {}
    for r in rows:
        out.setdefault((r["lattice"], r["branch"]), []).append((int(r["N"]), float(r["k_norm"]), float(r[column])))
    for v in out.values():
        v.sort(key=lambda x: x[1])
    return out


def load(results_dir: Path):
    lin_rows = read_csv(results_dir / "experiment_IV_linearized" / "experiment_IV_linearized_results.csv")
    lin_fits = read_csv(results_dir / "experiment_IV_linearized" / "experiment_IV_linearized_convergence_fits.csv")
    bgk_all = read_csv(results_dir / "experiment_IV_full_bgk" / "experiment_IV_full_bgk_results.csv")
    bgk_fits = read_csv(results_dir / "experiment_IV_full_bgk" / "experiment_IV_full_bgk_fits.csv")
    bgk_rows = [r for r in bgk_all if float(r["U0"]) == U0_FULL_BGK and int(r["t"]) == T_FULL_BGK]
    return lin_rows, lin_fits, bgk_all, bgk_rows, bgk_fits


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.rows: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(r[1] for r in self.rows)

    def text(self) -> str:
        return "\n".join(f"[{'PASS' if ok else 'FAIL'}] {n}" + (f"  ({d})" if d else "") for n, ok, d in self.rows)


def k_of_N(N: int) -> float:
    return 2.0 * math.pi * math.sqrt(14.0) / N


def verify_data(lin_rows, bgk_all, bgk_rows, recompute: bool) -> Checks:
    c = Checks()
    pop, rec = series(lin_rows, "population_error"), series(lin_rows, "reconstruction_error")
    bgk = series(bgk_rows, "E_RMS")
    keys = {(L, b) for L in LATTICES for b in BRANCHES}

    # 1. All six lattice/branch combinations, complete N sets.
    c.add("1. six lattice/branch combinations in linearized data", set(pop) == keys and set(rec) == keys)
    c.add("1. six lattice/branch combinations in full-BGK data (U0=1e-6, t=8)", set(bgk) == keys)
    c.add("1. linearized N set per series = 24,32,40,48,64,80,96,128",
          all(sorted(n for n, _, _ in pop[kk]) == sorted(N_LINEARIZED) and
              sorted(n for n, _, _ in rec[kk]) == sorted(N_LINEARIZED) for kk in keys),
          f"{len(lin_rows)} rows")
    c.add("1. full-BGK N set per series = 24,32,48,64,80",
          all(sorted(n for n, _, _ in bgk[kk]) == sorted(N_FULL_BGK) for kk in keys), f"{len(bgk_rows)} rows")

    # 2. Wavenumbers: k = 2 pi sqrt(14) / N, identical across datasets.
    dk = max(abs(k - k_of_N(n)) / k_of_N(n) for s in (pop, bgk) for v in s.values() for n, k, _ in v)
    c.add("2. k_norm = 2*pi*sqrt(14)/N in both datasets", dk < 1e-14, f"max rel. dev. {dk:.1e}")
    k_lin = {int(r["N"]): float(r["k_norm"]) for r in lin_rows}
    same_k = all(float(r["k_norm"]) == k_lin[int(r["N"])] for r in bgk_rows)
    c.add("2. full-BGK k_norm identical to linearized k_norm at equal N", same_k)
    comps = all(abs(float(r["kx"]) - 2 * math.pi / int(r["N"])) < 1e-15 and
                abs(float(r["ky"]) - 4 * math.pi / int(r["N"])) < 1e-15 and
                abs(float(r["kz"]) - 6 * math.pi / int(r["N"])) < 1e-15 for r in lin_rows)
    c.add("2. stored k components = (2 pi/N)(1,2,3)", comps)

    # 4. BGK subset parameters.
    c.add("4. full-BGK subset: U0 = 1e-6, t = 8, tau = 1, rho0 = 1",
          all(float(r["U0"]) == U0_FULL_BGK and int(r["t"]) == T_FULL_BGK and float(r["tau"]) == TAU
              and float(r["rho0"]) == RHO0 for r in bgk_rows) and len(bgk_rows) == 30,
          f"{len(bgk_rows)} of {len(bgk_all)} stored full-BGK records")
    c.add("4. linearized data: tau = 1, rho0 = 1",
          all(float(r["tau"]) == TAU and float(r["rho0"]) == RHO0 for r in lin_rows))

    # 5. Same reference definition / normalization: the full-BGK file stores, per case, the Step 3
    #    linearized error computed with the same code; it must equal the Step 3 file's value.
    lin_E = {(r["lattice"], r["branch"], int(r["N"])): float(r["reconstruction_error"]) for r in lin_rows}
    d5 = max(abs(float(r["E_linearized_step3"]) - lin_E[(r["lattice"], r["branch"], int(r["N"]))])
             / lin_E[(r["lattice"], r["branch"], int(r["N"]))] for r in bgk_rows)
    c.add("5. E_linearized_step3 (full-BGK file) = reconstruction_error (Step 3 file)", d5 < 1e-12,
          f"max rel. diff. {d5:.1e}")
    d5b = max(float(r["rel_diff_vs_step3"]) for r in bgk_rows)
    c.add("5. full-BGK E_RMS vs exact linearized (stored diagnostic)", d5b < 1e-5, f"max rel. diff. {d5b:.2e}")

    # 6. No silent exclusion: report all flags; none that would exclude a point from the validation.
    blocking_lin = {"ambiguous_selection", "not_transverse", "degenerate_shear_pair", "tracking_ambiguous",
                    "reference_near_zero"}
    blocking_bgk = {"nonlinear_contamination", "roundoff_limited"}
    fl_lin = {f for r in lin_rows for f in r["flags"].split(";") if f}
    fl_bgk = {f for r in bgk_rows for f in r["flags"].split(";") if f}
    c.add("6. no excluding flags in linearized data", not (fl_lin & blocking_lin),
          f"flags present: {sorted(fl_lin) or 'none'} (T-only flags do not affect E_pop/E_rec)")
    c.add("6. no excluding flags in full-BGK data", not (fl_bgk & blocking_bgk),
          f"flags present: {sorted(fl_bgk) or 'none'} (informative only)")
    finite = all(math.isfinite(v) and v > 0 for s in (pop, rec, bgk) for vv in s.values() for _, _, v in vv)
    c.add("6. all plotted values finite and positive (log axes)", finite)

    # 3. Recompute the exact linearized values (eigenmode analysis only; no BGK simulation).
    if recompute:
        res = lin.run_experiment(LATTICES, N_LINEARIZED, TAU, RHO0)
        d_rec = d_pop = d_k = 0.0
        for e in res.evaluations:
            key = (e.mode.lattice, e.mode.branch)
            row = {n: (k, v) for n, k, v in rec[key]}[e.mode.N]
            prow = {n: v for n, _, v in pop[key]}[e.mode.N]
            d_rec = max(d_rec, abs(e.E_rec - row[1]) / row[1])
            d_pop = max(d_pop, abs(e.E_pop - prow) / prow)
            d_k = max(d_k, abs(float(sum(x * x for x in e.mode.kvec)) ** 0.5 - row[0]) / row[0])
        tol = 1e-9   # same machine: identical; another LAPACK build: last digits only
        c.add("3. recomputed E_rec (exact linearized) = stored values", d_rec < tol, f"max rel. diff. {d_rec:.1e}")
        c.add("3. recomputed E_pop (exact linearized) = stored values", d_pop < tol, f"max rel. diff. {d_pop:.1e}")
        c.add("3. recomputed k = stored k", d_k < 1e-15, f"max rel. diff. {d_k:.1e}")
    else:
        c.add("3. recomputation of linearized values", True, "skipped (--skip-recompute)")
    return c


def verify_plotted(fig_axes, expected: Dict[str, Series], c: Checks) -> None:
    """Read back every data series drawn and compare with the stored values (exact)."""
    for label, ax in fig_axes.items():
        drawn = {}
        for ln in ax.get_lines():
            gid = ln.get_gid()
            if gid:
                drawn[gid] = (list(map(float, ln.get_xdata())), list(map(float, ln.get_ydata())))
        exp = expected[label]
        ok = True
        n_pts = 0
        for (L, b), pts in exp.items():
            gid = f"{L}|{b}"
            xs, ys = drawn.get(gid, ([], []))
            ok &= xs == [k for _, k, _ in pts] and ys == [v for _, _, v in pts]
            n_pts += len(xs)
        c.add(f"3/6. plotted data identical to stored data: {label}", ok and len(drawn) == 6, f"{n_pts} points")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,          # embed TrueType (editable text, journal-compliant)
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def style_of(L: str, b: str):
    st, bs = lin.LATTICE_STYLE[L], lin.BRANCH_STYLE[b]
    return st["color"], st["marker"], bs["linestyle"], bs["fill"]


def format_k_axis(ax, kmin: float, kmax: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(kmin, kmax)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4, 0.6, 0.8, 1.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[0.5, 0.7, 0.9]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$|\mathbf{k}|$")


def format_log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())


def draw_series(ax, data: Series, markersize: float = 3.6, lines: bool = True, markers: bool = True) -> None:
    for L in LATTICES:
        for b in BRANCHES:
            pts = data[(L, b)]
            color, marker, ls, fill = style_of(L, b)
            ax.plot([k for _, k, _ in pts], [v for _, _, v in pts], gid=f"{L}|{b}",
                    color=color, linestyle=ls if lines else "none", linewidth=1.0,
                    marker=marker if markers else None, markersize=markersize, markeredgewidth=0.8,
                    markerfacecolor=color if fill else "white", markeredgecolor=color,
                    zorder=3 if markers else 2, clip_on=False)


def slope_guide(ax, data: Series, p: int, label: str, kmin: float, kmax: float, factor: float = 0.45) -> None:
    """Thin dotted visual guide ~ k^p, placed below the lowest series; not a fit."""
    lowest_at_kmin = min(pts[0][2] for pts in data.values())
    k0 = min(pts[0][1] for pts in data.values())
    k1 = max(pts[-1][1] for pts in data.values())
    y0 = factor * lowest_at_kmin
    y1 = y0 * (k1 / k0) ** p
    ax.plot([k0, k1], [y0, y1], color=GUIDE_COLOR, linestyle=(0, (1, 1.6)), linewidth=0.8, zorder=1)
    # Label just below the right end of the guide, clear of the dotted line.
    ax.text(k1, y1 * 0.70, label, color="#4d4d48", fontsize=7.5, ha="right", va="top")
    return y0


def set_log_ylim(ax, data: Sequence[Series], floor: float = None, top_pad: float = 1.35,
                 bottom_pad: float = 0.75) -> None:
    """Axis limits only (data untouched): small headroom so no marker touches the frame."""
    vals = [v for s in data for pts in s.values() for _, _, v in pts]
    lo = min(vals) if floor is None else min(min(vals), floor)
    ax.set_ylim(lo * bottom_pad, max(vals) * top_pad)


def legend_handles(markersize: float = 3.6) -> List[Line2D]:
    hs = []
    for L in LATTICES:
        for b in BRANCHES:
            color, marker, ls, fill = style_of(L, b)
            hs.append(Line2D([], [], color=color, linestyle=ls, linewidth=1.0, marker=marker, markersize=markersize,
                             markeredgewidth=0.8, markerfacecolor=color if fill else "white", markeredgecolor=color,
                             label=f"{L} {b}"))
    return hs


def panel_label(ax, text: str) -> None:
    ax.text(0.03, 0.96, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def figure_A(pop: Series, rec: Series, kmin: float, kmax: float):
    fig, axes = plt.subplots(1, 2, figsize=FIG_A_SIZE)
    for ax, data, ylabel, p, glabel, tag in (
            (axes[0], pop, r"$E_Q^{\mathrm{pop}}$", 1, r"$O(k)$", "(a)"),
            (axes[1], rec, r"$E_Q^{\mathrm{rec}}$", 2, r"$O(k^2)$", "(b)")):
        draw_series(ax, data)
        y_guide = slope_guide(ax, data, p, glabel, kmin, kmax)
        set_log_ylim(ax, [data], floor=y_guide, bottom_pad=0.8)
        format_k_axis(ax, kmin, kmax)
        format_log_y(ax)
        ax.set_ylabel(ylabel)
        panel_label(ax, tag)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    return fig, {"Figure A (a) E_pop": axes[0], "Figure A (b) E_rec": axes[1]}


def figure_B(rec_lin: Series, rec_bgk: Series, kmin: float, kmax: float):
    fig, ax = plt.subplots(figsize=FIG_B_SIZE)
    draw_series(ax, rec_lin, lines=True, markers=False)        # exact linearized: lines only
    draw_series(ax, rec_bgk, lines=False, markers=True, markersize=4.0)   # full BGK: symbols only
    # Rename gids so the read-back check can tell the two datasets apart.
    for ln in ax.get_lines():
        gid = ln.get_gid()
        ln.set_gid(("lin:" if ln.get_linestyle() != "None" else "bgk:") + gid)
    format_k_axis(ax, kmin, kmax)
    format_log_y(ax)
    set_log_ylim(ax, [rec_lin, rec_bgk], top_pad=1.9, bottom_pad=0.55)   # room for legend and note
    ax.set_ylabel(r"$E_Q^{\mathrm{rec}}$")
    ax.legend(handles=legend_handles(4.0), loc="upper left", ncol=2, frameon=False, handlelength=2.4,
              columnspacing=0.9, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    ax.text(0.97, 0.04, "Lines: exact linearized BGK\n"
                        "Symbols: full BGK ($U_0=10^{-6}$, $t=8$)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#3a3a36", linespacing=1.35)
    fig.tight_layout(pad=0.3)
    return fig, ax


def readback_B(ax, rec_lin: Series, rec_bgk: Series, c: Checks) -> None:
    drawn = {ln.get_gid(): (list(map(float, ln.get_xdata())), list(map(float, ln.get_ydata())))
             for ln in ax.get_lines() if ln.get_gid() and ln.get_gid().startswith(("lin:", "bgk:"))}
    for prefix, data, label in (("lin:", rec_lin, "Figure B lines (exact linearized)"),
                                ("bgk:", rec_bgk, "Figure B symbols (full BGK)")):
        ok, n = True, 0
        for (L, b), pts in data.items():
            xs, ys = drawn.get(f"{prefix}{L}|{b}", ([], []))
            ok &= xs == [k for _, k, _ in pts] and ys == [v for _, _, v in pts]
            n += len(xs)
        c.add(f"3/6. plotted data identical to stored data: {label}", ok, f"{n} points")


# -----------------------------------------------------------------------------
# Fitted exponents (reported from the stored fit files; no new fitting)
# -----------------------------------------------------------------------------

def stored_exponents(lin_fits, bgk_fits) -> str:
    lines = ["Stored convergence exponents (fits produced by the validation scripts; N >= 48):",
             f"{'lattice':<7} {'branch':<6} {'p(E_pop)':>9} {'p(E_rec)':>9} {'p(E_RMS, full BGK)':>19}"]
    for L in LATTICES:
        for b in BRANCHES:
            pp = [r for r in lin_fits if r["lattice"] == L and r["branch"] == b and r["interval"] == "asymptotic"
                  and r["quantity"] == "population_error"][0]
            pr = [r for r in lin_fits if r["lattice"] == L and r["branch"] == b and r["interval"] == "asymptotic"
                  and r["quantity"] == "reconstruction_error"][0]
            pb = [r for r in bgk_fits if r["lattice"] == L and r["branch"] == b and r["interval"] == "asymptotic"
                  and r.get("quantity", "E_RMS") == "E_RMS" and int(r["t"]) == T_FULL_BGK][0]
            lines.append(f"{L:<7} {b:<6} {float(pp['p']):>9.3f} {float(pr['p']):>9.3f} {float(pb['p_full_bgk']):>19.3f}")
    lines.append("(linearized: N = 48..128, 5 points; full BGK: N = 48..80, 3 points)")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reference_results")
    ap.add_argument("--outdir", type=Path, default=ROOT / "Figures")
    ap.add_argument("--skip-recompute", action="store_true",
                    help="do not recompute the exact linearized values (verification check 3)")
    args = ap.parse_args(argv)

    lin_rows, lin_fits, bgk_all, bgk_rows, bgk_fits = load(args.results_dir)
    checks = verify_data(lin_rows, bgk_all, bgk_rows, recompute=not args.skip_recompute)
    if not checks.ok:
        print(checks.text())
        print("\nData verification FAILED; no figures written.")
        return 1

    pop = series(lin_rows, "population_error")
    rec = series(lin_rows, "reconstruction_error")
    rec_bgk = series(bgk_rows, "E_RMS")
    all_k = [k for s in (pop, rec_bgk) for v in s.values() for _, k, _ in v]
    kmin, kmax = min(all_k) / 1.06, max(all_k) * 1.06

    setup_style()
    figA, axesA = figure_A(pop, rec, kmin, kmax)
    verify_plotted(axesA, {"Figure A (a) E_pop": pop, "Figure A (b) E_rec": rec}, checks)
    figB, axB = figure_B(rec, rec_bgk, kmin, kmax)
    readback_B(axB, rec, rec_bgk, checks)
    xl = {tuple(ax.get_xlim()) for ax in list(axesA.values()) + [axB]}
    checks.add("shared horizontal limits in all panels", len(xl) == 1, f"[{kmin:.4f}, {kmax:.4f}]")

    if not checks.ok:
        print(checks.text())
        print("\nPlotted-data verification FAILED; no figures written.")
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fig, stem in ((figA, "figure_IV_population_vs_reconstruction"), (figB, "figure_IV_full_bgk_vs_linearized")):
        for ext in ("pdf", "png"):
            path = args.outdir / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                        metadata={"Creator": "scripts/plot_experiment_IV_paper_figures.py"} if ext == "pdf" else None)
            outputs.append(path)
        plt.close(fig)

    report = ["Experiment IV publication figures - verification report", "",
              checks.text(), "", stored_exponents(lin_fits, bgk_fits), "", "Outputs:"]
    report += [f"  {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}" for p in outputs]
    text = "\n".join(report)
    (args.outdir / "figure_IV_paper_verification.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
