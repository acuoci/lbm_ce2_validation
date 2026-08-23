#!/usr/bin/env python3
"""Reproduce the Section V D3Q19/D3Q27 CE2 population-operator sweep."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
solver = ROOT / "src" / "experiment_II_3D_shear.py"
spec = importlib.util.spec_from_file_location("exp2", solver)
exp2 = importlib.util.module_from_spec(spec)
sys.modules["exp2"] = exp2
spec.loader.exec_module(exp2)

NVALUES = [24, 32, 40, 48, 64, 80]
OPERATORS = ROOT / "operators"
OUT = ROOT / "reference_results" / "experiment_II"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [OPERATORS / f"Lhat_D3Q{q}.csv" for q in (19, 27) if not (OPERATORS / f"Lhat_D3Q{q}.csv").exists()]
    if missing:
        raise FileNotFoundError("Missing machine-readable operator file(s): " + ", ".join(str(p) for p in missing))

    all_results = []
    for q in (19, 27):
        lat = exp2.load_lattice(OPERATORS / f"Lhat_D3Q{q}.csv")
        for direction in ("100", "110", "111"):
            for N in NVALUES:
                _, pols = exp2.direction_definition(direction, 1)
                for pname, pol in pols:
                    r = exp2.run_case(
                        lat=lat,
                        N=N,
                        direction=direction,
                        polarization_name=pname,
                        polarization=pol,
                        m=1,
                        tau=1.0,
                        epsilon=1e-6,
                        nsteps=1,
                    )
                    all_results.append(r)
                    exp2.print_case(r)

    exp2.write_csv(all_results, OUT / "experiment_II_raw_results.csv")
    fits = exp2.fit_convergence(all_results)
    exp2.write_fit_csv(fits, OUT / "experiment_II_convergence_fits.csv")
    exp2.make_plots(all_results, OUT)
