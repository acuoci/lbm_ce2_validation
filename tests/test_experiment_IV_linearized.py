#!/usr/bin/env python3
"""
Tests for Experiment IV, Step 3 (src/experiment_IV_linearized_reconstruction.py).

Run from the repository root:

    python -m unittest discover -s tests

A reduced wavenumber set (N = 24, 48, 128) keeps the suite fast; the production
script uses N = 24, 32, 40, 48, 64, 80, 96, 128.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _import(path: Path, name: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e4 = _import(SRC / "experiment_IV_linearized_reconstruction.py", "experiment_IV_linearized_reconstruction")
recon = e4.recon

LATTICES = ("D3Q15", "D3Q19", "D3Q27")
N_TEST = (24, 48, 128)
LAT = {n: e4.load_lattice(n) for n in LATTICES}
RESULT = e4.run_experiment(LATTICES, N_TEST)
EVALS = RESULT.evaluations


def modes_for(name, N):
    return [e.mode for e in EVALS if e.mode.lattice == name and e.mode.N == N]


class Test01_AmplificationMatrix(unittest.TestCase):
    def test_dimensions(self):
        for n, Q in (("D3Q15", 15), ("D3Q19", 19), ("D3Q27", 27)):
            A = e4.exp2.amplification_matrix(LAT[n], e4.k_vector(48), e4.TAU)
            self.assertEqual(A.shape, (Q, Q))
            self.assertEqual(A.dtype, np.complex128)

    def test_wavevector(self):
        k = e4.k_vector(32)
        np.testing.assert_allclose(k, 2 * np.pi / 32 * np.array([1, 2, 3]), rtol=0, atol=0)
        self.assertAlmostEqual(float(e4.E1 @ e4.DIRECTION), 0.0, places=15)
        self.assertAlmostEqual(float(e4.E2 @ e4.DIRECTION), 0.0, places=15)


class Test02_EigenpairResiduals(unittest.TestCase):
    def test_residuals_near_machine_precision(self):
        for e in EVALS:
            m = e.mode
            A = e4.exp2.amplification_matrix(LAT[m.lattice], m.kvec, e4.TAU)
            res = np.linalg.norm(A @ m.v - m.eigenvalue * m.v) / np.linalg.norm(m.v)
            self.assertLess(res, 1e-13, (m.lattice, m.N, m.branch))
            self.assertAlmostEqual(res, m.eig_residual, delta=1e-15)


class Test03_VelocityNormalization(unittest.TestCase):
    def test_unit_momentum_amplitude(self):
        for e in EVALS:
            m = e.mode
            U = (LAT[m.lattice].c.T @ m.v) / e4.RHO0
            self.assertAlmostEqual(float(np.linalg.norm(U)), 1.0, places=13)
            np.testing.assert_allclose(U, m.U, atol=1e-14)


class Test04_ShearModeIdentification(unittest.TestCase):
    def test_two_distinct_transverse_hydrodynamic_modes(self):
        for n in LATTICES:
            for N in N_TEST:
                a, b = modes_for(n, N)
                with self.subTest(lattice=n, N=N):
                    self.assertEqual({a.branch, b.branch}, {"s1", "s2"})
                    self.assertGreater(abs(a.eigenvalue - b.eigenvalue), 1e-7)       # non-degenerate here
                    for m in (a, b):
                        self.assertLess(abs(m.eigenvalue.imag), 1e-12)                # real shear eigenvalues
                        self.assertLess(m.longitudinal_fraction, 1e-2)
                        self.assertLess(abs(m.drho), 0.1)
                        self.assertLess(m.selection_margin, 1e-3)
                        self.assertGreater(m.nonshear_separation, 0.05)
                        self.assertEqual(m.flags, [])
                    # The two polarizations span the transverse plane: e1 and e2 lie in span{U_a, U_b}
                    # up to the (small) longitudinal content.
                    B = np.column_stack([a.U, b.U])
                    for e in (e4.E1, e4.E2):
                        coef, *_ = np.linalg.lstsq(B, e.astype(complex), rcond=None)
                        self.assertLess(np.linalg.norm(B @ coef - e), 1e-2)

    def test_density_contamination_vanishes_as_k_decreases(self):
        for n in LATTICES:
            for br in ("s1", "s2"):
                d = [next(m for m in modes_for(n, N) if m.branch == br).drho for N in N_TEST]
                self.assertTrue(abs(d[2]) <= abs(d[1]) <= abs(d[0]) or abs(d[0]) < 1e-14, (n, br))

    def test_branch_tracking_continuity(self):
        for e in EVALS:
            self.assertGreater(e.mode.tracking_overlap, 0.999)

    def test_tracking_independent_of_input_order(self):
        lat = LAT["D3Q15"]
        per_N = {N: e4.find_shear_modes(lat, "D3Q15", N) for N in N_TEST}
        per_N_rev = {N: list(reversed(copy.deepcopy(per_N[N]))) for N in N_TEST}
        e4.track_branches(per_N)
        e4.track_branches(per_N_rev)
        for N in N_TEST:
            for br in ("s1", "s2"):
                m1 = next(m for m in per_N[N] if m.branch == br)
                m2 = next(m for m in per_N_rev[N] if m.branch == br)
                self.assertEqual(m1.eigenvalue, m2.eigenvalue)

    def test_experiment_ii_selector_comparison_recorded(self):
        self.assertEqual(len(RESULT.selector_checks), len(LATTICES) * len(N_TEST))
        for c in RESULT.selector_checks:
            self.assertGreater(min(c["expII_e1_overlap"], c["expII_e2_overlap"]), 0.999)


class Test05_ComplexReconstruction(unittest.TestCase):
    def test_complex_and_linear(self):
        for e in EVALS:
            n = e.mode.lattice
            self.assertEqual(e.d_rec.dtype, np.complex128)
            re = recon.reconstruct_from_kinetic(e.g_K.real, n, e4.TAU, e4.RHO0)
            im = recon.reconstruct_from_kinetic(e.g_K.imag, n, e4.TAU, e4.RHO0)
            np.testing.assert_allclose(e.d_rec, re + 1j * im, atol=1e-14 * np.linalg.norm(e.d_rec))

    def test_reconstruction_receives_only_population_data(self):
        calls = []
        original = recon.reconstruct_from_kinetic

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        recon.reconstruct_from_kinetic = spy
        try:
            lat = LAT["D3Q27"]
            m = e4.find_shear_modes(lat, "D3Q27", 48)[0]
            e4.evaluate_mode(lat, "D3Q27", m)
        finally:
            recon.reconstruct_from_kinetic = original
        args, kwargs = calls[0]
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 4)
        self.assertEqual(args[0].shape, (27,))
        self.assertEqual(args[1:], ("D3Q27", e4.TAU, e4.RHO0))


class Test06_ProjectedVersusDirect(unittest.TestCase):
    def test_equalities(self):
        for e in EVALS:
            self.assertLess(e.direct_vs_projected, 1e-12)
            self.assertLess(e.PKN_vs_PK, 1e-12)


class Test07_AnalyticalCurvature(unittest.TestCase):
    def test_simple_case_by_hand(self):
        k = np.array([1.0, 0.0, 0.0])
        U = np.array([0.0, 1.0, 0.0])
        q = e4.curvature_coordinates(e4.curvature_tensor(k, U))
        expected = np.zeros(15)
        expected[e4.exact.Q_NAMES.index("uy_xx")] = -1.0
        np.testing.assert_array_equal(q, expected)

    def test_mixed_case_by_hand(self):
        k = np.array([1.0, 2.0, 3.0])
        U = np.array([0.5, -1.0, 2.0])
        J = e4.curvature_tensor(k, U)
        self.assertEqual(J[0, 1, 2], -2.0 * 3.0 * 0.5)
        self.assertTrue(np.array_equal(J, np.transpose(J, (0, 2, 1))))
        q = e4.curvature_coordinates(J)
        self.assertEqual(q[e4.exact.Q_NAMES.index("uz_xy")], -1.0 * 2.0 * 2.0)

    def test_matches_experiment_ii_q_fourier(self):
        for e in EVALS:
            self.assertEqual(e.qF_vs_expII, 0.0)
            np.testing.assert_array_equal(e.q_F, e4.exp2.q_fourier(e.mode.kvec, e.mode.U))

    def test_reference_uses_exact_D(self):
        for n in LATTICES:
            D = e4.observable_matrix(n)
            self.assertEqual(D.shape, (recon.num_observables(n), 15))
            self.assertTrue(np.array_equal(D, np.array(e4.exact.observable_matrix(n).tolist(), dtype=float)))


class Test08_SyntheticCE2Control(unittest.TestCase):
    def test_control_reconstructs_reference(self):
        for e in EVALS:
            self.assertLess(e.ce2_control, 1e-13, (e.mode.lattice, e.mode.N))


class Test09_PhaseInvariance(unittest.TestCase):
    def test_errors_independent_of_eigenvector_phase(self):
        for n in LATTICES:
            lat = LAT[n]
            for m in e4.find_shear_modes(lat, n, 48):
                base = e4.evaluate_mode(lat, n, m)
                for theta in (0.7, 2.3, -1.1):
                    ph = np.exp(1j * theta)
                    m2 = copy.deepcopy(m)
                    m2.v, m2.U = m.v * ph, m.U * ph
                    ev = e4.evaluate_mode(lat, n, m2)
                    self.assertAlmostEqual(ev.E_rec, base.E_rec, delta=1e-13 * base.E_rec)
                    self.assertAlmostEqual(ev.E_pop, base.E_pop, delta=1e-13 * base.E_pop)
                    np.testing.assert_allclose(ev.d_rec, ph * base.d_rec, atol=1e-14)

    def test_normalization_is_deterministic(self):
        lat = LAT["D3Q19"]
        _, V, cands = e4.classify_eigenpairs(lat, e4.k_vector(48), e4.TAU)
        j = max(cands, key=lambda c: c.score).index
        v1, U1 = e4.normalize_mode(lat, V[:, j])
        v2, U2 = e4.normalize_mode(lat, V[:, j] * np.exp(1.9j) * 3.7)
        np.testing.assert_allclose(v1, v2, atol=1e-13)
        np.testing.assert_allclose(U1, U2, atol=1e-14)


class Test10_ConvergenceAndOutputs(unittest.TestCase):
    def test_errors_decrease_with_k(self):
        for n in LATTICES:
            for br in ("s1", "s2"):
                g = sorted([e for e in EVALS if e.mode.lattice == n and e.mode.branch == br], key=lambda e: e.mode.N)
                self.assertTrue(g[0].E_rec > g[1].E_rec > g[2].E_rec, (n, br))
                self.assertTrue(g[0].E_pop > g[1].E_pop > g[2].E_pop, (n, br))

    def test_fits_are_reported_with_metadata(self):
        rows = e4.convergence_fits(EVALS, N_min=48)
        self.assertTrue(rows)
        for r in rows:
            for key in ("p", "R2", "n_points", "k_min", "k_max", "N_min", "N_max", "excluded_N"):
                self.assertIn(key, r)
            self.assertGreaterEqual(r["n_points"], 3)

    def test_T_only_where_reconstructed(self):
        for e in EVALS:
            if e.mode.lattice == "D3Q19":
                self.assertTrue(np.isnan(e.T_abs) and np.isnan(e.T_rel))
            else:
                self.assertTrue(np.isfinite(e.T_abs))
                if "T_reference_small" in e.flags:
                    self.assertTrue(np.isnan(e.T_rel))
                    self.assertLess(e.T_exact_over_d, e4.T_RELATIVE_FLOOR)
                else:
                    self.assertTrue(np.isfinite(e.T_rel))

    def test_csv_writer_round_trip(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.csv"
            e4.write_csv(e4.result_rows(EVALS[:2]), path)
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertNotIn("np.", path.read_text())
            self.assertEqual(float(rows[0]["reconstruction_error"]), EVALS[0].E_rec)
            for col in ("lattice", "N", "kx", "ky", "kz", "k_norm", "tau", "branch", "eigenvalue_real",
                        "eigenvalue_imag", "density_contamination", "longitudinal_fraction",
                        "reconstruction_error", "population_error", "T_absolute_error", "T_relative_error"):
                self.assertIn(col, rows[0])


class Test11_ExistingFilesUnchanged(unittest.TestCase):
    DIGESTS = {
        "operators/Lhat_D3Q15.csv": "aca891677b26ae89e217d8fa22a6be020c226257e491c285d236aa8acb2ce6fc",
        "operators/Lhat_D3Q19.csv": "258b5dcd2ec4b4f8e735e4f180fe7fd91eb927bcb7471db12c1812dcd05c8a05",
        "operators/Lhat_D3Q27.csv": "b749b00ff3315789e1b1443d9d0568eed8b75a5ec49dddda61c9f721bcc1b5ca",
        "src/experiment_I_d2q9_shear.py": "a422fd024564324d18dbf679517d4129e103f43a10826557773e4c34c13348e9",
        "src/experiment_II_3D_shear.py": "2de52e02fdf8b303823f81713c044326f6e1156403ec75b465a5682269d08f92",
        "src/experiment_III_taylor_green.py": "2bf9260143f558fe473f3ac1aeb7e3e98ff27f0f3c1941bf097c442bef66c8ed",
    }

    def test_digests(self):
        for rel, digest in self.DIGESTS.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)


if __name__ == "__main__":
    unittest.main()
