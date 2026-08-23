#!/usr/bin/env python3
"""Reproduce the final Section V Taylor-Green production protocol."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
solver = ROOT / "src" / "experiment_III_taylor_green.py"
spec = importlib.util.spec_from_file_location("exp3", solver)
exp3 = importlib.util.module_from_spec(spec)
sys.modules["exp3"] = exp3
spec.loader.exec_module(exp3)

OPERATORS = ROOT / "operators"
OUT = ROOT / "reference_results" / "experiment_III"
N = 48
TAU = 0.8
U0_VALUES = (0.005, 0.01, 0.02)
STARTUP_DISCARD = 10
TSTAR_OUTPUTS = (0.002, 0.005, 0.010, 0.020)


def output_steps(U0: float):
    steps = {0, 1, 2, 5, 10}
    steps.update(int(round(tstar * N / U0)) for tstar in TSTAR_OUTPUTS)
    return sorted(steps)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [OPERATORS / f"Lhat_D3Q{q}.csv" for q in (19, 27) if not (OPERATORS / f"Lhat_D3Q{q}.csv").exists()]
    if missing:
        raise FileNotFoundError("Missing machine-readable operator file(s): " + ", ".join(str(p) for p in missing))

    all_records = []
    for q in (19, 27):
        lat = exp3.load_lattice(OPERATORS / f"Lhat_D3Q{q}.csv")
        for U0 in U0_VALUES:
            all_records.extend(
                exp3.run_case(
                    lat=lat,
                    N=N,
                    U0=U0,
                    tau=TAU,
                    max_tstar=0.02,
                    output_steps=output_steps(U0),
                    startup_discard=STARTUP_DISCARD,
                )
            )

    exp3.write_records(all_records, OUT / "experiment_III_results.csv")
    exp3.make_case_plots(all_records, OUT, startup_discard=STARTUP_DISCARD)
