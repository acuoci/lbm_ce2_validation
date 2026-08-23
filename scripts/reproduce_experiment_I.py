#!/usr/bin/env python3
"""Reproduce the Section V D2Q9 paper-production sweep."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
solver = ROOT / "src" / "experiment_I_d2q9_shear.py"
spec = importlib.util.spec_from_file_location("d2q9exp", solver)
mod = importlib.util.module_from_spec(spec)
sys.modules["d2q9exp"] = mod
spec.loader.exec_module(mod)
OUT = ROOT / "reference_results" / "experiment_I"

if __name__ == "__main__":
    mod.run_sweep(
        outdir=OUT,
        nx=256,
        nsteps=8,
        epsilon=1e-6,
        taus=(0.9, 1.0, 1.1),
        directions=("axis", "oblique", "diag"),
        mvalues=range(1, 9),
    )
