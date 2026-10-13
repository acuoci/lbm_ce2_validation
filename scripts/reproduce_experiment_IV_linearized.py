#!/usr/bin/env python3
"""Reproduce Experiment IV, Step 3: reconstruction from exact linearized BGK shear eigenmodes.

Writes CSV results and two diagnostic figures to reference_results/experiment_IV_linearized/.
Only Fourier amplification matrices are formed (Q x Q); no spatial grid is allocated.
"""
from pathlib import Path
import importlib.util
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "experiment_IV_linearized_reconstruction", ROOT / "src" / "experiment_IV_linearized_reconstruction.py")
exp4 = importlib.util.module_from_spec(spec)
sys.modules["experiment_IV_linearized_reconstruction"] = exp4
spec.loader.exec_module(exp4)

OUT = ROOT / "reference_results" / "experiment_IV_linearized"

if __name__ == "__main__":
    result = exp4.run_experiment(lattices=("D3Q15", "D3Q19", "D3Q27"),
                                 N_values=(24, 32, 40, 48, 64, 80, 96, 128),
                                 tau=1.0, rho0=1.0)
    exp4.print_summary(result)
    paths = exp4.write_outputs(result, OUT)
    print("\nOutputs:")
    for p in paths.values():
        print(f"  {p.relative_to(ROOT)}")
