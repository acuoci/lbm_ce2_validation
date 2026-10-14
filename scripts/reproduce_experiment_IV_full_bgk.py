#!/usr/bin/env python3
"""Reproduce Experiment IV, Step 4: reconstruction from spatially evolved BGK populations.

Primary runs: D3Q15/D3Q19/D3Q27, both shear branches, k = (2 pi/N)(1,2,3), N = 24, 32, 48, 64, 80,
U0 = 1e-6, tau = 1, rho0 = 1, evaluated at t = 0 (control), 1, 2, 4, 8.
Amplitude study (separate): N = 32, 48; U0 = 1e-4 ... 1e-7; paired +-U0 runs; t = 1, 8.
Outputs go to reference_results/experiment_IV_full_bgk/. Peak memory is about 0.6 GB (N = 80, D3Q27).
"""
from pathlib import Path
import importlib.util
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_IV_full_bgk_reconstruction",
                                              ROOT / "src" / "experiment_IV_full_bgk_reconstruction.py")
exp4 = importlib.util.module_from_spec(spec)
sys.modules["experiment_IV_full_bgk_reconstruction"] = exp4
spec.loader.exec_module(exp4)

OUT = ROOT / "reference_results" / "experiment_IV_full_bgk"

if __name__ == "__main__":
    t0 = time.time()
    print("Primary runs (U0 = 1e-6)")
    records = exp4.run_primary(lattices=("D3Q15", "D3Q19", "D3Q27"), N_values=(24, 32, 48, 64, 80),
                               times=(1, 2, 4, 8), amplitude=1e-6, log=lambda s: print("  " + s, flush=True))
    print("\nAmplitude study (paired +-U0)")
    amp = exp4.run_amplitude_study(N_values=(32, 48), amplitudes=(1e-4, 1e-5, 1e-6, 1e-7), times=(1, 8),
                                   log=lambda s: print("  " + s, flush=True))
    fits = exp4.full_bgk_fits(records, exp4.step3_reference_fits())
    exp4.print_summary(records, amp, fits)
    paths = exp4.write_outputs(records, amp, fits, OUT)
    print("\nOutputs:")
    for p in paths.values():
        print(f"  {p.relative_to(ROOT)}")
    print(f"\nTotal time {time.time() - t0:.0f} s")
