#!/usr/bin/env python3
"""
Tests for Experiment IV, Step 4 (src/experiment_IV_full_bgk_reconstruction.py).

Run from the repository root:  python -m unittest discover -s tests

Small grids (N = 24) keep the suite fast; the production script runs N = 24 ... 80.
"""

from __future__ import annotations

import hashlib
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


fb = _import(ROOT / "src" / "experiment_IV_full_bgk_reconstruction.py", "experiment_IV_full_bgk_reconstruction")
lin, exp2, recon = fb.lin, fb.exp2, fb.recon

LATTICES = ("D3Q15", "D3Q19", "D3Q27")
N = 24
LAT = {n: lin.load_lattice(n) for n in LATTICES}
MODES = {n: fb.shear_modes(n, [N])[N] for n in LATTICES}
E_LIN = {(n, b): (lin.evaluate_mode(LAT[n], n, MODES[n][b]).E_rec, lin.evaluate_mode(LAT[n], n, MODES[n][b]).T_rel)
         for n in LATTICES for b in ("s1", "s2")}
RECORDS = {(n, b): fb.simulate(n, N, MODES[n][b], fb.U0, (1, 2), E_LIN[(n, b)], lat=LAT[n])
           for n in LATTICES for b in ("s1", "s2")}


class TestInitialization(unittest.TestCase):
    def test_exact_eigenmode_initialization(self):
        for n in LATTICES:
            for b, m in MODES[n].items():
                lat = LAT[n]
                phase = fb.phase_field(m.kvec, N)
                f = fb.initialize(lat, m.v, phase, fb.U0)
                self.assertEqual(f.shape, (len(lat.w), N, N, N))
                np.testing.assert_allclose(f.mean(axis=(1, 2, 3)), lat.w * fb.RHO0, atol=1e-15)
                fhat = exp2.extract_fourier_mode(lat, f, m.kvec, fb.RHO0)        # Experiment II convention
                np.testing.assert_allclose(fhat, 0.5 * fb.U0 * m.v, atol=1e-12 * fb.U0)
                Uhat = fb.fourier_coefficient(fb.momentum_field(lat, f), phase) / fb.RHO0
                np.testing.assert_allclose(Uhat, 0.5 * fb.U0 * m.U, atol=1e-12 * fb.U0)
                self.assertAlmostEqual(np.linalg.norm(m.U), 1.0, places=13)

    def test_branch_labels_match_step3(self):
        step3 = lin.run_experiment(LATTICES, (N,))
        for e in step3.evaluations:
            self.assertEqual(MODES[e.mode.lattice][e.mode.branch].eigenvalue, e.mode.eigenvalue)


class TestCollisionStreaming(unittest.TestCase):
    def test_one_step_reproduces_linear_eigenvalue(self):
        for n in LATTICES:
            lat = LAT[n]
            for m in MODES[n].values():
                f0 = fb.initialize(lat, m.v, fb.phase_field(m.kvec, N), fb.U0)
                f1 = exp2.collide_and_stream(lat, f0, fb.TAU)
                h0 = exp2.extract_fourier_mode(lat, f0, m.kvec, fb.RHO0)
                h1 = exp2.extract_fourier_mode(lat, f1, m.kvec, fb.RHO0)
                self.assertLess(np.linalg.norm(h1 - m.eigenvalue * h0) / np.linalg.norm(h0), 1e-8, n)

    def test_conservation(self):
        for (n, b), recs in RECORDS.items():
            for r in recs:
                self.assertLess(r.mass_drift, 1e-14, (n, b, r.t))
                self.assertLess(r.momentum_total, 1e-9, (n, b, r.t))

    def test_velocity_decays_as_eigenvalue_power(self):
        for recs in RECORDS.values():
            for r in recs:
                self.assertLess(r.velocity_decay_deviation, 1e-9)


class TestLocalReconstruction(unittest.TestCase):
    def test_shapes_and_chunking(self):
        n = "D3Q19"
        lat, m = LAT[n], MODES[n]["s1"]
        f = fb.initialize(lat, m.v, fb.phase_field(m.kvec, N), fb.U0)
        d = fb.reconstruct_slabs(f, n, fb.TAU, fb.RHO0)
        self.assertEqual(d.shape, (N, N, N, 6))
        saved = fb.SLAB_NODES
        try:
            fb.SLAB_NODES = N * N * 5          # forces several uneven slabs
            d2 = fb.reconstruct_slabs(f, n, fb.TAU, fb.RHO0)
        finally:
            fb.SLAB_NODES = saved
        np.testing.assert_array_equal(d, d2)
        self.assertEqual(fb.direct_estimate(f, n, fb.TAU, fb.RHO0).shape, (N, N, N, 6))

    def test_reconstruction_receives_only_populations(self):
        calls = []
        original = recon.reconstruct_from_populations

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        recon.reconstruct_from_populations = spy
        try:
            n = "D3Q27"
            m = MODES[n]["s2"]
            fb.reconstruct_slabs(fb.initialize(LAT[n], m.v, fb.phase_field(m.kvec, N), fb.U0), n, fb.TAU, fb.RHO0)
        finally:
            recon.reconstruct_from_populations = original
        for args, kwargs in calls:
            self.assertEqual(kwargs, {})
            self.assertEqual(len(args), 4)
            self.assertEqual(args[0].shape[-1], 27)
            self.assertEqual(args[0].dtype, np.float64)
            self.assertEqual(args[1:], ("D3Q27", fb.TAU, fb.RHO0))

    def test_projected_and_direct_agree(self):
        for recs in RECORDS.values():
            for r in recs:
                self.assertLess(r.direct_vs_projected, 1e-6)
                self.assertLess(abs(r.E_RMS_direct - r.E_RMS), 1e-6 * r.E_RMS)


class TestReference(unittest.TestCase):
    def test_reference_normalization_and_consistency(self):
        for (n, b), recs in RECORDS.items():
            for r in recs:
                self.assertLess(r.reference_consistency, 1e-11)

    def test_reference_at_t0_matches_step3_qF(self):
        for n in LATTICES:
            lat = LAT[n]
            for m in MODES[n].values():
                phase = fb.phase_field(m.kvec, N)
                f = fb.initialize(lat, m.v, phase, fb.U0)
                Uhat = fb.fourier_coefficient(fb.momentum_field(lat, f), phase) / fb.RHO0
                D = lin.observable_matrix(n)
                dhat = D @ exp2.q_fourier(m.kvec, Uhat)
                ev = lin.evaluate_mode(lat, n, m)
                np.testing.assert_allclose(dhat, 0.5 * fb.U0 * ev.d_exact, rtol=1e-11, atol=1e-11 * fb.U0)
                field_ = 2.0 * np.real(dhat[None, None, None, :] * phase[..., None])
                rms = np.sqrt(np.mean(np.sum(field_ ** 2, axis=-1)))
                self.assertAlmostEqual(rms / (np.sqrt(2.0) * np.linalg.norm(dhat)), 1.0, places=12)

    def test_T_handling(self):
        for (n, b), recs in RECORDS.items():
            for r in recs:
                if n == "D3Q19":
                    self.assertTrue(np.isnan(r.T_rms_rel_error) and np.isnan(r.T_modal_rel_error))
                    self.assertNotIn("T", r.per_observable)
                elif "T_reference_small" in r.flags:
                    self.assertTrue(np.isnan(r.T_rms_rel_error) and np.isnan(r.T_modal_rel_error))
                else:
                    self.assertTrue(np.isfinite(r.T_rms_rel_error))
                    # Modal T error reproduces the Step 3 linearized value.
                    self.assertLess(abs(r.T_modal_rel_error - r.T_linearized_step3), 1e-5 * r.T_linearized_step3)


class TestSmallAmplitudeConsistencyWithStep3(unittest.TestCase):
    def test_matches_linearized_error(self):
        for (n, b), recs in RECORDS.items():
            for r in recs:
                self.assertLess(r.rel_diff_vs_step3, 1e-6, (n, b, r.t))
                self.assertLess(abs(r.E_modal - r.E_linearized_step3), 1e-6 * r.E_linearized_step3)

    def test_all_lattices_and_observable_counts(self):
        for (n, b), recs in RECORDS.items():
            self.assertEqual([r.t for r in recs], [0, 1, 2])
            self.assertEqual(len(recs[0].per_observable), recon.num_observables(n))

    def test_fits_and_rows(self):
        recs = [r for v in RECORDS.values() for r in v]
        self.assertEqual(len(fb.record_rows(recs)), len(recs))
        self.assertTrue(fb.observable_rows(recs))
        self.assertEqual(fb.full_bgk_fits(recs, {}), [])   # one N only: no fit is possible, and none is invented


class TestExistingFilesUnchanged(unittest.TestCase):
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
