#!/usr/bin/env python3
"""Reproduce Experiment IV, Step 3B: velocity-parity analysis of the Step 3 finite-wavenumber errors.

Uses the Step 3 configuration (D3Q15/D3Q19/D3Q27, k = (2 pi/N)(1,2,3), N = 24..128,
tau = 1, rho0 = 1, both shear branches) and writes CSV files and the parity figure to
reference_results/experiment_IV_parity/.
"""
from pathlib import Path
import importlib.util
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_IV_parity_analysis",
                                              ROOT / "src" / "experiment_IV_parity_analysis.py")
parity = importlib.util.module_from_spec(spec)
sys.modules["experiment_IV_parity_analysis"] = parity
spec.loader.exec_module(parity)

OUT = ROOT / "reference_results" / "experiment_IV_parity"

if __name__ == "__main__":
    records = parity.run_analysis(lattices=("D3Q15", "D3Q19", "D3Q27"),
                                  N_values=(24, 32, 40, 48, 64, 80, 96, 128))
    parity.print_summary(records)
    paths = parity.write_outputs(records, OUT)
    print("\nOutputs:")
    for p in paths.values():
        print(f"  {p.relative_to(ROOT)}")
