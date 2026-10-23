"""Tests for the archived reconstruction operators in operators/reconstruction/."""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate_reconstruction_operators", ROOT / "scripts" / "generate_reconstruction_operators.py")


class ReconstructionOperatorFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = gen.build_files()
        cls.checks = gen.verify_files()

    def test_files_present_and_identical_to_exact_rebuild(self):
        for name, text in self.files.items():
            p = gen.RECON_DIR / name
            self.assertTrue(p.exists(), p)
            self.assertEqual(gen.read_text_exact(p), text, name)

    def test_all_file_level_identities(self):
        failed = [c for c, ok, _ in self.checks if not ok]
        self.assertEqual(failed, [])

    def test_identity_checked_for_every_lattice(self):
        for lat in gen.LATTICES:
            self.assertTrue(any(c.startswith(f"{lat}: R^H_Q Lhat_Q = D_Q") and ok for c, ok, _ in self.checks), lat)

    def test_numerical_interface_uses_archived_operator(self):
        os.environ.setdefault("MPLBACKEND", "Agg")
        import numpy as np
        from fractions import Fraction
        cr = _load("curvature_reconstruction", ROOT / "src" / "curvature_reconstruction.py")
        for lat in gen.LATTICES:
            rows = gen._read(gen.RECON_DIR / f"RH_{lat}.csv")
            RT = np.array([[float(Fraction(x)) for x in r[4:]] for r in rows[1:]])
            self.assertEqual(tuple(cr.observable_names(lat)), tuple(rows[0][4:]))
            np.testing.assert_allclose(cr.reconstruction_matrix(lat), RT.T, rtol=0, atol=1e-15)

    def test_detects_corruption(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shutil.copytree(gen.RECON_DIR, td / "reconstruction")
            p = td / "reconstruction" / "RH_D3Q19.csv"
            lines = gen.read_text_exact(p).split("\r\n")
            lines[2] = lines[2].replace("-2/3", "-1/3", 1)
            with p.open("w", encoding="utf-8", newline="") as fh:
                fh.write("\r\n".join(lines))
            checks = gen.verify_files(td / "reconstruction")
            self.assertFalse(all(ok for _, ok, _ in checks))


if __name__ == "__main__":
    unittest.main()
