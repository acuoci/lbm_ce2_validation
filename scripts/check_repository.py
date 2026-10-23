#!/usr/bin/env python3
"""
Repository-integrity and reproducibility checks.

Default (fast, a few seconds): Python syntax, required operators and archived
reproducibility material, release metadata, exact reconstruction identities
R^H_Q Lhat_Q = D_Q from the archived files, Experiment III step-10 values against
the manuscript tables, and the stored Experiment IV figure-verification report.
No expensive production sweep is run.

--full additionally runs the complete validation suite (about one minute):
the unittest suite, the exact D3Q15 operator and reconstruction-operator
generators in --check mode, the exact reconstruction report, the Experiment IV
figure verification (written to a temporary directory), and a rerun of the
Experiment III table audit and time history compared with the stored CSVs.

    python scripts/check_repository.py
    python scripts/check_repository.py --full

Exit status is nonzero if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")

PYTHON_SOURCES = [
    "src/experiment_I_d2q9_shear.py",
    "src/experiment_II_3D_shear.py",
    "src/experiment_III_taylor_green.py",
    "src/curvature_reconstruction_exact.py",
    "src/curvature_reconstruction.py",
    "src/experiment_IV_linearized_reconstruction.py",
    "src/experiment_IV_parity_analysis.py",
    "src/experiment_IV_full_bgk_reconstruction.py",
    "scripts/reproduce_experiment_I.py",
    "scripts/reproduce_experiment_II.py",
    "scripts/reproduce_experiment_III.py",
    "scripts/reproduce_experiment_III_time_history.py",
    "scripts/generate_d3q15_operator.py",
    "scripts/generate_reconstruction_operators.py",
    "scripts/reproduce_experiment_IV_linearized.py",
    "scripts/reproduce_experiment_IV_parity.py",
    "scripts/reproduce_experiment_IV_full_bgk.py",
    "scripts/plot_experiment_IV_paper_figures.py",
    "scripts/check_repository.py",
    "tests/test_d3q15_operator.py",
    "tests/test_curvature_reconstruction_exact.py",
    "tests/test_curvature_reconstruction.py",
    "tests/test_reconstruction_operator_files.py",
    "tests/test_experiment_IV_linearized.py",
    "tests/test_experiment_IV_parity.py",
    "tests/test_experiment_IV_full_bgk.py",
]

REQUIRED_FILES = {
    "metadata": ["README.md", "REPRODUCIBILITY.md", "CHANGELOG.md", "CITATION.cff", "LICENSE", "requirements.txt",
                 ".gitattributes"],
    "observability operators": [f"operators/Lhat_{q}.csv" for q in ("D3Q15", "D3Q19", "D3Q27")],
    "reconstruction operators": ["operators/reconstruction/observable_functionals.csv",
                                 "operators/reconstruction/sensitivity_factors.csv"]
                                + [f"operators/reconstruction/{k}_{q}.csv" for k in ("D", "RH")
                                   for q in ("D3Q15", "D3Q19", "D3Q27")],
    "Experiment I-III production outputs": [
        "reference_results/experiment_I/" + f for f in (
            "d2q9_shear_results.csv", "d2q9_fitted_coefficients.csv", "figure1_transfer_relation.png",
            "figure2a_B_recovery.png", "figure2b_cubic_error.png")] + [
        "reference_results/experiment_II/" + f for f in (
            "experiment_II_raw_results.csv", "experiment_II_convergence_fits.csv",
            "figure_population_vector_error.png", "figure_amplitude_alignment.png")] + [
        "reference_results/experiment_III/" + f for f in (
            "experiment_III_results.csv", "figure_III_1_global_CE2_error.png",
            "figure_III_2_amplitude_alignment.png", "figure_III_3_spectral_centroid.png")],
    "Experiment III time history": [
        "reference_results/experiment_III/time_history/" + f for f in (
            "experiment_III_time_history_U0_0.02.csv", "experiment_III_table_audit.csv",
            "figure_III_1_global_CE2_error.png", "figure_III_1_global_CE2_error.pdf")],
    "Experiment IV reference results": [
        "reference_results/experiment_IV_linearized/" + f for f in (
            "experiment_IV_linearized_results.csv", "experiment_IV_linearized_observables.csv",
            "experiment_IV_linearized_convergence_fits.csv", "experiment_IV_linearized_local_slopes.csv",
            "experiment_IV_linearized_selector_check.csv", "figure_IV_1_reconstruction_error.png",
            "figure_IV_2_population_error.png")] + [
        "reference_results/experiment_IV_parity/" + f for f in (
            "experiment_IV_parity.csv", "experiment_IV_parity_fits.csv", "experiment_IV_parity_identities.csv",
            "figure_IV_3_parity_error.png")] + [
        "reference_results/experiment_IV_full_bgk/" + f for f in (
            "experiment_IV_full_bgk_results.csv", "experiment_IV_full_bgk_observables.csv",
            "experiment_IV_full_bgk_amplitude.csv", "experiment_IV_full_bgk_fits.csv",
            "figure_IV_4_full_bgk_vs_linearized.png", "figure_IV_5_spatial_comparison.png")],
    "Experiment IV publication figures": [
        "Figures/" + f for f in (
            "figure_IV_population_vs_reconstruction.pdf", "figure_IV_population_vs_reconstruction.png",
            "figure_IV_full_bgk_vs_linearized.pdf", "figure_IV_full_bgk_vs_linearized.png",
            "figure_IV_paper_verification.txt")],
}

# Manuscript amplitude-sweep values (N = 48, tau = 0.8, U0 = 0.02, timestep 10), 4 decimals,
# as tabulated in docs/experiment_III_production_protocol_refined.md, Section 4.
EXP3_STEP10_TABLE = {"D3Q19": (0.2541, 1.0378, 0.9696), "D3Q27": (0.3246, 1.0573, 0.9517)}
EXP3_HISTORY_STEPS = (10, 20, 30, 40, 48)
# Section 4 amplitude sweep (N = 48, tau = 0.8, timestep 10): E for the archived production amplitudes.
EXP3_SWEEP_STEP10 = {("D3Q19", 0.005): 0.0638, ("D3Q19", 0.01): 0.1272, ("D3Q19", 0.02): 0.2541,
                     ("D3Q27", 0.005): 0.0814, ("D3Q27", 0.01): 0.1624, ("D3Q27", 0.02): 0.3246}


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.passed += ok
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        return ok

    def section(self, title: str) -> None:
        print(f"\n== {title}")


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_csv(rel: str) -> List[Dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# -----------------------------------------------------------------------------
# Fast checks
# -----------------------------------------------------------------------------

def check_syntax(rep: Report) -> None:
    rep.section("Python sources")
    listed = set(PYTHON_SOURCES)
    on_disk = {str(p.relative_to(ROOT)) for d in ("src", "scripts", "tests") for p in (ROOT / d).glob("*.py")}
    rep.add("every Python file in src/, scripts/, tests/ is listed", on_disk <= listed,
            ", ".join(sorted(on_disk - listed)) or f"{len(on_disk)} files")
    bad = []
    for rel in PYTHON_SOURCES:
        try:
            compile((ROOT / rel).read_text(encoding="utf-8"), rel, "exec")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{rel}: {exc}")
    rep.add(f"syntax of {len(PYTHON_SOURCES)} listed sources", not bad, "; ".join(bad))


def check_files(rep: Report) -> None:
    rep.section("Required files")
    for group, files in REQUIRED_FILES.items():
        missing = [f for f in files if not (ROOT / f).is_file() or (ROOT / f).stat().st_size == 0]
        rep.add(f"{group}: {len(files) - len(missing)}/{len(files)} present", not missing, ", ".join(missing))


def _cff_field(text: str, key: str, indent: int = 0) -> str | None:
    m = re.search(rf"^{' ' * indent}{key}:\s*\"?([^\"\n]*)\"?\s*$", text, re.M)
    return m.group(1).strip() if m else None


def check_metadata(rep: Report) -> None:
    rep.section("Release metadata")
    if not all((ROOT / f).is_file() for f in ("CITATION.cff", "CHANGELOG.md", "README.md", "REPRODUCIBILITY.md")):
        rep.add("metadata files readable", False, "see Required files")
        return
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = _cff_field(cff, "version")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(r"^## \[?v?(\d+\.\d+\.\d+)\]?", changelog, re.M)
    latest = m.group(1) if m else None
    rep.add("CITATION.cff version = latest CHANGELOG.md entry", version is not None and version == latest,
            f"CITATION {version}, CHANGELOG {latest}")
    title = _cff_field(cff, "title")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rep.add("README states the CITATION.cff title", title is not None and title in readme, title or "")
    article = cff.split("preferred-citation:", 1)[1] if "preferred-citation:" in cff else ""
    art_doi = _cff_field(article, "doi", 2)
    rep.add("article DOI (if any) is not a Zenodo software DOI",
            art_doi is None or not art_doi.startswith("10.5281/zenodo"), art_doi or "no article DOI recorded")
    zen = ROOT / ".zenodo.json"
    if zen.exists():
        z = json.loads(zen.read_text(encoding="utf-8"))
        rep.add(".zenodo.json version and title match CITATION.cff",
                z.get("version") == version and z.get("title") == title)
    else:
        rep.add(".zenodo.json absent (Zenodo will read CITATION.cff)", True)
    for doc in ("README.md", "REPRODUCIBILITY.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        names = ["Experiment I", "Experiment II", "Experiment III", "Experiment IV", "D3Q15", "RH_D3Q"]
        missing = [n for n in names if n not in text]
        rep.add(f"{doc} covers Experiments I-IV and the reconstruction operators", not missing, ", ".join(missing))


def check_reconstruction_files(rep: Report) -> None:
    rep.section("Reconstruction operators (exact, from archived files)")
    gen = _load("generate_reconstruction_operators", "scripts/generate_reconstruction_operators.py")
    for name, ok, detail in gen.verify_files():
        rep.add(name, ok, detail)


def check_experiment_III(rep: Report) -> None:
    rep.section("Experiment III Taylor-Green time history")
    rows = _read_csv("reference_results/experiment_III/time_history/experiment_III_time_history_U0_0.02.csv")
    for lat, (E, R, X) in EXP3_STEP10_TABLE.items():
        g = {int(r["step"]): r for r in rows if r["lattice"] == lat}
        ok_steps = set(EXP3_HISTORY_STEPS) <= set(g)
        rep.add(f"{lat}: diagnostics stored at t = {EXP3_HISTORY_STEPS}", ok_steps)
        if not ok_steps:
            continue
        r10 = g[10]
        got = tuple(round(float(r10[k]), 4) for k in ("E", "R", "X"))
        rep.add(f"{lat}: timestep-10 (E, R, X) = manuscript table {E, R, X}", got == (E, R, X), f"stored {got}")
        tstar = all(abs(float(g[s]["t_star"]) - 0.02 * s / 48) < 1e-15 for s in EXP3_HISTORY_STEPS)
        rep.add(f"{lat}: t* = U0 t / N for all plotted points", tstar)
        params = all(r["N"] == "48" and float(r["tau"]) == 0.8 and float(r["U0"]) == 0.02 for r in g.values())
        rep.add(f"{lat}: N = 48, tau = 0.8, U0 = 0.02", params)
    audit = _read_csv("reference_results/experiment_III/time_history/experiment_III_table_audit.csv")
    bad = [f"{r['section']} {r['lattice']} N={r['N']} U0={r['U0']} t={r['step']}" for r in audit
           if r["agrees_to_4_decimals"] != "True"]
    rep.add(f"protocol-table audit: {len(audit) - len(bad)}/{len(audit)} entries reproduced to 4 decimals",
            not bad and len(audit) == 32, "; ".join(bad))
    sweep = _read_csv("reference_results/experiment_III/experiment_III_results.csv")
    got = {(r["lattice"], float(r["U0"])): round(float(r["E"]), 4) for r in sweep if int(r["step"]) == 10}
    bad = [f"{k}: {got.get(k)} vs {v}" for k, v in EXP3_SWEEP_STEP10.items() if got.get(k) != v]
    rep.add("archived amplitude sweep: timestep-10 E = manuscript table (6 entries)", not bad, "; ".join(bad))
    src = (ROOT / "scripts/reproduce_experiment_III_time_history.py").read_text(encoding="utf-8")
    rep.add("figure script plots exactly t = 10, 20, 30, 40, 48",
            "HISTORY_STEPS = (10, 20, 30, 40, 48)" in src)


def check_experiment_IV(rep: Report) -> None:
    rep.section("Experiment IV stored verification")
    text = (ROOT / "Figures/figure_IV_paper_verification.txt").read_text(encoding="utf-8")
    n_pass, n_fail = text.count("[PASS]"), text.count("[FAIL]")
    rep.add("Figures/figure_IV_paper_verification.txt: all checks PASS", n_fail == 0 and n_pass >= 20,
            f"{n_pass} PASS, {n_fail} FAIL")
    ident = _read_csv("reference_results/experiment_IV_parity/experiment_IV_parity_identities.csv")
    rep.add("parity identities file non-empty", len(ident) > 0, f"{len(ident)} rows")


# -----------------------------------------------------------------------------
# Full validation suite
# -----------------------------------------------------------------------------

def _run(rep: Report, name: str, args: List[str], timeout: int = 900) -> Tuple[bool, str]:
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    out = proc.stdout + proc.stderr
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
    rep.add(name, proc.returncode == 0, tail[0][:120])
    return proc.returncode == 0, out


def run_full(rep: Report) -> None:
    rep.section("Full validation suite")
    ok, out = _run(rep, "unittest suite", ["-m", "unittest", "discover", "-s", "tests"])
    m = re.search(r"Ran (\d+) tests", out)
    if m:
        print(f"       {m.group(1)} tests")
    _run(rep, "exact D3Q15 operator generator --check", ["scripts/generate_d3q15_operator.py", "--check"])
    _run(rep, "exact reconstruction operators report", ["src/curvature_reconstruction_exact.py"])
    _run(rep, "archived reconstruction operators --check", ["scripts/generate_reconstruction_operators.py", "--check"])
    with tempfile.TemporaryDirectory() as td:
        _run(rep, "Experiment IV figure data verification (temporary output)",
             ["scripts/plot_experiment_IV_paper_figures.py", "--outdir", td])
        ok, _ = _run(rep, "Experiment III table audit + time history rerun (temporary output)",
                     ["scripts/reproduce_experiment_III_time_history.py", "--outdir", td])
        if ok:
            stored = "reference_results/experiment_III/time_history/"
            for fname, keys in (("experiment_III_time_history_U0_0.02.csv", ("E", "R", "X", "numerator_rms",
                                                                              "denominator_rms", "u_rms")),
                                ("experiment_III_table_audit.csv", ("E_computed", "R_computed", "X_computed"))):
                a = _read_csv(stored + fname)
                with open(Path(td) / fname, newline="", encoding="utf-8") as fh:
                    b = list(csv.DictReader(fh))
                dev = max(abs(float(x[k]) - float(y[k])) / max(abs(float(x[k])), 1e-300)
                          for x, y in zip(a, b) for k in keys) if len(a) == len(b) else float("inf")
                rep.add(f"rerun reproduces stored {fname}", dev <= 1e-10, f"max rel. dev. {dev:.1e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also run the complete validation suite (~1 min)")
    args = ap.parse_args(argv)
    sys.dont_write_bytecode = True
    rep = Report()
    check_syntax(rep)
    check_files(rep)
    check_metadata(rep)
    check_reconstruction_files(rep)
    check_experiment_III(rep)
    check_experiment_IV(rep)
    if args.full:
        run_full(rep)
    print(f"\n{rep.passed} passed, {rep.failed} failed")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
