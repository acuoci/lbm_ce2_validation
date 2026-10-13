#!/usr/bin/env python3
"""Fast repository-integrity checks; does not run the expensive production sweeps."""
from pathlib import Path
import py_compile
import sys

ROOT = Path(__file__).resolve().parents[1]
python_sources = [
    ROOT / "src/experiment_I_d2q9_shear.py",
    ROOT / "src/experiment_II_3D_shear.py",
    ROOT / "src/experiment_III_taylor_green.py",
    ROOT / "src/curvature_reconstruction_exact.py",
    ROOT / "src/curvature_reconstruction.py",
    ROOT / "src/experiment_IV_linearized_reconstruction.py",
    ROOT / "src/experiment_IV_parity_analysis.py",
    ROOT / "scripts/generate_d3q15_operator.py",
    ROOT / "scripts/reproduce_experiment_IV_linearized.py",
    ROOT / "scripts/reproduce_experiment_IV_parity.py",
    ROOT / "tests/test_d3q15_operator.py",
    ROOT / "tests/test_curvature_reconstruction_exact.py",
    ROOT / "tests/test_curvature_reconstruction.py",
    ROOT / "tests/test_experiment_IV_linearized.py",
    ROOT / "tests/test_experiment_IV_parity.py",
]
required_inputs = [
    ROOT / "operators/Lhat_D3Q15.csv",
    ROOT / "operators/Lhat_D3Q19.csv",
    ROOT / "operators/Lhat_D3Q27.csv",
]

failed = False
for path in python_sources:
    try:
        py_compile.compile(str(path), doraise=True)
        print(f"OK syntax: {path.relative_to(ROOT)}")
    except Exception as exc:
        print(f"FAIL syntax: {path.relative_to(ROOT)}: {exc}")
        failed = True

for path in required_inputs:
    if path.exists():
        print(f"OK input:  {path.relative_to(ROOT)}")
    else:
        print(f"MISSING:   {path.relative_to(ROOT)}")
        failed = True

# The exact D3Q15 operator verification (SymPy) is run separately:
#   python scripts/generate_d3q15_operator.py --check
sys.exit(1 if failed else 0)
