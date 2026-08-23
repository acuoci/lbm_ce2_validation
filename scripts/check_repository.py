#!/usr/bin/env python3
"""Fast repository-integrity checks; does not run the expensive production sweeps."""
from pathlib import Path
import py_compile
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "src/experiment_I_d2q9_shear.py",
    ROOT / "src/experiment_II_3D_shear.py",
    ROOT / "src/experiment_III_taylor_green.py",
    ROOT / "operators/Lhat_D3Q19.csv",
    ROOT / "operators/Lhat_D3Q27.csv",
]

failed = False
for path in required[:3]:
    try:
        py_compile.compile(str(path), doraise=True)
        print(f"OK syntax: {path.relative_to(ROOT)}")
    except Exception as exc:
        print(f"FAIL syntax: {path.relative_to(ROOT)}: {exc}")
        failed = True

for path in required[3:]:
    if path.exists():
        print(f"OK input:  {path.relative_to(ROOT)}")
    else:
        print(f"MISSING:   {path.relative_to(ROOT)}")
        failed = True

sys.exit(1 if failed else 0)
