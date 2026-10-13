#!/usr/bin/env python3
"""
Tests for Experiment IV, Step 3B (src/experiment_IV_parity_analysis.py).

Run from the repository root:  python -m unittest discover -s tests

Reduced wavenumber set (N = 24, 48, 128). The tests check identities and
consistency; they do not impose the hypothesised convergence exponents.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _import(path: Path, name: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


par = _import(ROOT / "src" / "experiment_IV_parity_analysis.py", "experiment_IV_parity_analysis")
lin, exp2, recon = par.lin, par.exp2, par.recon

LATTICES = ("D3Q15", "D3Q19", "D3Q27")
N_TEST = (24, 48, 128)
LAT = {n: lin.load_lattice(n) for n in LATTICES}
STEP3 = lin.run_experiment(LATTICES, N_TEST)
RECORDS = [par.decompose(e, LAT[e.mode.lattice], *par.parity_projectors(LAT[e.mode.lattice].c)[1:])
           for e in STEP3.evaluations]


class TestParityProjectors(unittest.TestCase):
    def test_reversal_permutation(self):
        for n in LATTICES:
            c = LAT[n].c
            idx = par.reversal_index(c)
            self.assertTrue(np.array_equal(c[idx], -c))
            self.assertTrue(np.array_equal(idx[idx], np.arange(len(c))))
            self.assertTrue(np.array_equal(LAT[n].w[idx], LAT[n].w))
            P = par.reversal_matrix(c)
            g = np.arange(len(c), dtype=float)
            self.assertTrue(np.array_equal(P @ g, g[idx]))

    def test_projector_algebra(self):
        for n in LATTICES:
            P, Pp, Pm = par.parity_projectors(LAT[n].c)
            Q = len(P)
            self.assertTrue(np.array_equal(Pp @ Pp, Pp))
            self.assertTrue(np.array_equal(Pm @ Pm, Pm))
            self.assertTrue(np.array_equal(Pp @ Pm, np.zeros((Q, Q))))
            self.assertTrue(np.array_equal(Pp + Pm, np.eye(Q)))

    def test_not_closed_under_reversal_raises(self):
        with self.assertRaises(ValueError):
            par.reversal_index(np.array([[0, 0, 0], [1, 0, 0]]))

    def test_all_identity_residuals_vanish(self):
        for n in LATTICES:
            for key, val in par.identity_residuals(n, N_TEST).items():
                self.assertLessEqual(val, 1e-15, (n, key))


class TestReconstructionAnnihilatesEvenParity(unittest.TestCase):
    def test_numerical(self):
        for n in LATTICES:
            _, Pp, _ = par.parity_projectors(LAT[n].c)
            self.assertLessEqual(np.max(np.abs(recon.reconstruction_matrix(n) @ Pp)), 1e-15)

    def test_exact(self):
        for n in LATTICES:
            self.assertTrue(par.exact_RH_parity(n), n)

    def test_ce2_prediction_is_odd(self):
        for n in LATTICES:
            _, Pp, _ = par.parity_projectors(LAT[n].c)
            self.assertLessEqual(np.max(np.abs(Pp @ LAT[n].lhat)), 0.0)


class TestErrorDecomposition(unittest.TestCase):
    def test_components_sum_and_pythagoras(self):
        for e in STEP3.evaluations:
            lat = LAT[e.mode.lattice]
            _, Pp, Pm = par.parity_projectors(lat.c)
            err = e.g_K - e.g_ce2
            np.testing.assert_allclose(Pp @ err + Pm @ err, err, atol=1e-15)
            # P is W^-1-self-adjoint (symmetric weights), so the split is orthogonal.
            n2 = exp2.weighted_norm(lat, Pp @ err) ** 2 + exp2.weighted_norm(lat, Pm @ err) ** 2
            self.assertAlmostEqual(np.sqrt(n2) / exp2.weighted_norm(lat, e.g_ce2), e.E_pop, delta=1e-13 * e.E_pop)

    def test_odd_part_carries_reconstruction(self):
        for r in RECORDS:
            self.assertLess(r.reconstruction_odd_only, 1e-13)
            self.assertEqual(r.ce2_even_part, 0.0)

    def test_reconstruction_error_matches_step3(self):
        for r in RECORDS:
            self.assertLess(abs(r.E_rec_parity - r.E_rec_step3), 1e-12 * r.E_rec_step3, (r.lattice, r.N, r.branch))

    def test_record_fields_consistent(self):
        for r in RECORDS:
            self.assertAlmostEqual(r.ratio_r, r.abs_e_minus / r.abs_e_plus, delta=1e-14)
            self.assertAlmostEqual(r.E_minus / r.E_plus, r.ratio_r, delta=1e-12)


class TestMirroredBranch(unittest.TestCase):
    def test_v_minus_k_equals_minus_P_v(self):
        for r in RECORDS:
            self.assertLess(r.mirrored_branch_residual, 1e-9, (r.lattice, r.N, r.branch))


class TestOutputs(unittest.TestCase):
    def test_fits_have_metadata_and_predictions(self):
        rows = par.parity_fits(RECORDS, N_min=48)
        self.assertTrue(rows)
        # Reduced test set: only the all-points interval has >= 3 points (N >= 48 has 2 here).
        self.assertFalse([r for r in rows if r["interval"] == "asymptotic"])
        keys = {(r["lattice"], r["branch"], r["quantity"]) for r in rows if r["interval"] == "all"}
        for n in LATTICES:
            for b in ("s1", "s2"):
                for q, _ in par.FIT_QUANTITIES:
                    self.assertIn((n, b, q), keys)
        for r in rows:
            for k in ("p", "predicted_p", "R2", "n_points", "k_min", "k_max", "excluded_N"):
                self.assertIn(k, r)

    def test_csv_rows(self):
        rows = par.record_rows(RECORDS)
        self.assertEqual(len(rows), len(LATTICES) * len(N_TEST) * 2)
        for k in ("E_plus", "E_minus", "ratio_r", "E_rec_parity", "E_rec_step3",
                  "abs_e_plus_over_k3", "abs_e_minus_over_k4", "mirrored_branch_residual"):
            self.assertIn(k, rows[0])


if __name__ == "__main__":
    unittest.main()
