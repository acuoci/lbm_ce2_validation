#!/usr/bin/env python3
"""
Tests for the Step 2B numerical reconstruction interface
(src/curvature_reconstruction.py).

Run from the repository root:

    python -m unittest discover -s tests

Synthetic CE2 test data g = C0 Lhat_Q q are built here, in the tests only; the
production reconstruction functions never use Lhat_Q (checked in Test 0).
Reference values D_Q q are computed exactly (SymPy) and compared in float64.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import sympy as sp

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


num = _import(SRC / "curvature_reconstruction.py", "curvature_reconstruction")
ex = _import(SRC / "curvature_reconstruction_exact.py", "curvature_reconstruction_exact")

LATTICES = ("D3Q15", "D3Q19", "D3Q27")
R_DIM = {"D3Q15": 4, "D3Q19": 6, "D3Q27": 7}
ORDER = {
    "D3Q15": ("S_1", "S_2", "S_3", "T"),
    "D3Q19": ("A_x", "B_x", "B_y", "A_z", "A_y", "B_z"),
    "D3Q27": ("A_x", "B_x", "B_y", "A_z", "A_y", "B_z", "T"),
}
RTOL = 1e-12

# Exact data used only to manufacture synthetic inputs and reference outputs.
OPS = {n: ex.load_operator(n) for n in LATTICES}
DMAT = {n: ex.observable_matrix(n) for n in LATTICES}
L_FLOAT = {n: np.array(OPS[n].Lhat.tolist(), dtype=float) for n in LATTICES}

Q_VECTORS = [
    [sp.Integer(k + 1) for k in range(15)],
    [(-1) ** k * sp.Rational(k + 1, k + 2) for k in range(15)],
    [sp.Rational(3, 7), -2, 0, sp.Rational(5, 11), 1, 0, -sp.Rational(1, 3), 4, 0,
     sp.Rational(2, 9), -1, 0, sp.Rational(7, 5), 3, -sp.Rational(4, 13)],
    [1 if k in (4, 9, 14) else 0 for k in range(15)],            # pure fully mixed curvature
    [sp.Rational(1, 1000) * (k % 4 - 1) for k in range(15)],     # small, with zeros
]


def exact_expected(n: str, qv) -> np.ndarray:
    return np.array([float(x) for x in DMAT[n] * sp.Matrix(qv)])


def ce2_kinetic(n: str, qv, c0: float) -> np.ndarray:
    """Synthetic dimensional kinetic populations g = C0 Lhat q (tests only)."""
    return c0 * (L_FLOAT[n] @ np.array([float(x) for x in qv]))


def hermite_le2_basis(n: str) -> np.ndarray:
    """(Q, 10) weighted Hermite columns w_i H(c_i) through second order."""
    c = num.get_lattice(n).c.astype(float)
    w = num.get_lattice(n).w
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    cs2 = 1.0 / 3.0
    H = np.stack([np.ones_like(cx), cx, cy, cz, cx * cx - cs2, cy * cy - cs2, cz * cz - cs2,
                  cx * cy, cx * cz, cy * cz], axis=1)
    return w[:, None] * H


def assert_close(tc: unittest.TestCase, actual, expected, rtol=RTOL, msg=""):
    expected = np.asarray(expected)
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale, err_msg=msg)


class Test0_ModuleStructure(unittest.TestCase):
    def test_metadata(self):
        for n in LATTICES:
            self.assertEqual(num.observable_names(n), ORDER[n])
            self.assertEqual(num.num_observables(n), R_DIM[n])
            self.assertEqual(num.num_populations(n), int(n[3:]))
            self.assertEqual(num.reconstruction_matrix(n).shape, (R_DIM[n], int(n[3:])))
            self.assertEqual(num.kinetic_projector(n).shape, (int(n[3:]),) * 2)

    def test_cached_matrices_are_readonly_and_from_step_2a(self):
        for n in LATTICES:
            RH = num.reconstruction_matrix(n)
            self.assertFalse(RH.flags.writeable)
            self.assertFalse(num.kinetic_projector(n).flags.writeable)
            exact = ex.hermite_reconstruction(OPS[n])
            self.assertTrue(np.array_equal(RH, np.array(exact.tolist(), dtype=float)))
            self.assertIs(num.get_lattice(n), num.get_lattice(n))      # cached

    def test_production_code_never_uses_forward_operator_or_solver(self):
        tree = ast.parse((SRC / "curvature_reconstruction.py").read_text())
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertNotIn("Lhat", attrs | names)
        self.assertFalse({"gram_matrix", "minimum_norm_reconstruction", "q_fourier",
                          "collide_and_stream", "taylor_green_velocity"} & (attrs | names))
        self.assertFalse(any("experiment" in m for m in imported))


class Test1_ExactCE2Reconstruction(unittest.TestCase):
    def test_all_interfaces(self):
        tau, rho0 = 0.8, 1.0
        c0 = num.ce2_prefactor(tau, rho0)
        for n in LATTICES:
            lat = num.get_lattice(n)
            for k, qv in enumerate(Q_VECTORS):
                expected = exact_expected(n, qv)
                g = ce2_kinetic(n, qv, c0)
                feq = num.equilibrium(np.array(1.0), np.zeros(3), n)
                with self.subTest(lattice=n, vector=k):
                    assert_close(self, num.reconstruct_from_normalized_kinetic(g / c0, n), expected)
                    assert_close(self, num.reconstruct_from_kinetic(g, n, tau, rho0), expected)
                    assert_close(self, num.reconstruct_from_nonequilibrium(g, n, tau, rho0), expected)
                    # f = f_eq(rho=1, u=0) + g; the CE2 vector carries no mass or momentum.
                    rtol_pop = RTOL if np.max(np.abs(g)) > 1e-3 else 1e-10
                    assert_close(self, num.reconstruct_from_populations(feq + g, n, tau, rho0), expected,
                                 rtol=rtol_pop)
                    self.assertEqual(num.reconstruct_from_kinetic(g, n, tau, rho0).shape, (lat.r,))


class Test2_InterfaceConsistency(unittest.TestCase):
    def test_consistent_normalization(self):
        rng = np.random.default_rng(2)
        for n in LATTICES:
            Q = num.num_populations(n)
            for tau, rho0 in ((0.6, 1.0), (1.3, 0.7)):
                c0 = num.ce2_prefactor(tau, rho0)
                f_neq = rng.standard_normal((5, Q))
                g = num.project_kinetic(f_neq, n)
                a = num.reconstruct_from_normalized_kinetic(g, n) / c0
                b = num.reconstruct_from_kinetic(g, n, tau, rho0)
                c = num.reconstruct_from_nonequilibrium(f_neq, n, tau, rho0)
                assert_close(self, b, a, rtol=1e-14)
                assert_close(self, c, a, rtol=1e-13)

    def test_populations_equals_nonequilibrium_of_f_minus_feq(self):
        rng = np.random.default_rng(3)
        for n in LATTICES:
            rho = 1.0 + 0.05 * rng.standard_normal(6)
            u = 0.03 * rng.standard_normal((6, 3))
            f = num.equilibrium(rho, u, n) + 1e-3 * rng.standard_normal((6, num.num_populations(n)))
            r2, u2 = num.macroscopic(f, n)
            f_neq = f - num.equilibrium(r2, u2, n)
            assert_close(self, num.reconstruct_from_populations(f, n, 0.9, 1.1),
                         num.reconstruct_from_nonequilibrium(f_neq, n, 0.9, 1.1), rtol=1e-13)


class Test3_ProjectionInvariance(unittest.TestCase):
    def test_RH_PK_equals_RH_numerically(self):
        rng = np.random.default_rng(4)
        for n in LATTICES:
            RH, PK = num.reconstruction_matrix(n), num.kinetic_projector(n)
            f = rng.standard_normal((20, num.num_populations(n)))
            assert_close(self, (f @ PK.T) @ RH.T, f @ RH.T, rtol=1e-13)
            self.assertLess(np.max(np.abs(RH @ PK - RH)), 1e-14)

    def test_RH_annihilates_hermite_le2_exactly(self):
        # Exact statement behind the numerical invariance: R^H Phi_<=2 = 0.
        for n in LATTICES:
            op = OPS[n]
            cs2 = sp.Rational(1, 3)
            Phi = sp.Matrix([[w * h for h in (1, cx, cy, cz, cx * cx - cs2, cy * cy - cs2, cz * cz - cs2,
                                              cx * cy, cx * cz, cy * cz)]
                             for (cx, cy, cz), w in zip(op.velocities, op.w)])
            self.assertEqual(ex.hermite_reconstruction(op) * Phi, sp.zeros(R_DIM[n], 10), n)

    def test_adding_hermite_le2_components_changes_nothing(self):
        rng = np.random.default_rng(5)
        for n in LATTICES:
            Phi = hermite_le2_basis(n)
            f = rng.standard_normal(num.num_populations(n))
            base_norm = num.reconstruct_from_normalized_kinetic(f, n)
            base_neq = num.reconstruct_from_nonequilibrium(f, n, 0.8)
            for _ in range(5):
                h = Phi @ rng.standard_normal(10)
                assert_close(self, num.reconstruct_from_normalized_kinetic(f + h, n), base_norm, rtol=1e-13)
                assert_close(self, num.reconstruct_from_nonequilibrium(f + h, n, 0.8), base_neq, rtol=1e-13)

    def test_projector_matches_experiment_ii(self):
        exp2 = _import(SRC / "experiment_II_3D_shear.py", "exp2_test_step2b")
        for n in LATTICES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)   # pre-existing unclosed file in load_lattice
                lat = exp2.load_lattice(ROOT / "operators" / f"Lhat_{n}.csv")
            self.assertLess(np.max(np.abs(lat.PK - num.kinetic_projector(n))), 1e-12, n)
            self.assertTrue(np.array_equal(lat.c, num.get_lattice(n).c))
            self.assertTrue(np.array_equal(lat.w, num.get_lattice(n).w))
            for tau, rho0 in ((0.8, 1.0), (1.25, 2.0)):
                self.assertAlmostEqual(exp2.ce2_prefactor(tau, rho0), num.ce2_prefactor(tau, rho0), places=15)


class Test3b_IntermediateSteps(unittest.TestCase):
    def test_project_kinetic_applies_PK(self):
        rng = np.random.default_rng(13)
        for n in LATTICES:
            f = rng.standard_normal((4, num.num_populations(n)))
            gK = num.project_kinetic(f, n)
            assert_close(self, gK, f @ num.kinetic_projector(n).T, rtol=1e-15)
            # g_K carries no Hermite <=2 content: zero mass, momentum and second moments.
            Phi = hermite_le2_basis(n)
            moments = gK @ (Phi / num.get_lattice(n).w[:, None])
            self.assertLess(np.max(np.abs(moments)), 1e-14)
            self.assertGreater(np.max(np.abs(gK - f)), 1e-3)

    def test_nonequilibrium_uses_local_density_and_velocity(self):
        rng = np.random.default_rng(14)
        for n in LATTICES:
            rho = 1.0 + 0.3 * rng.random(6)
            u = 0.05 * rng.standard_normal((6, 3))
            f_eq = num.equilibrium(rho, u, n)
            self.assertLess(np.max(np.abs(num.nonequilibrium_populations(f_eq, n))), 1e-15)
            g = 1e-3 * rng.standard_normal((6, num.num_populations(n)))
            g = num.project_kinetic(g, n)                        # no mass/momentum
            assert_close(self, num.nonequilibrium_populations(f_eq + g, n), g, rtol=1e-10)

    def test_equilibrium_is_annihilated_by_RH(self):
        # Documents why the estimator is insensitive to the equilibrium subtraction.
        rng = np.random.default_rng(15)
        for n in LATTICES:
            f_eq = num.equilibrium(1.0 + 0.1 * rng.random(5), 0.05 * rng.standard_normal((5, 3)), n)
            self.assertLess(np.max(np.abs(num.reconstruct_from_normalized_kinetic(f_eq, n))), 1e-15)


class Test4_Vectorization(unittest.TestCase):
    def test_shapes_against_pointwise(self):
        rng = np.random.default_rng(6)
        tau, rho0 = 0.9, 1.2
        for n in LATTICES:
            Q, r = num.num_populations(n), R_DIM[n]
            for shape in ((), (7,), (3, 4, 5)):
                g = rng.standard_normal(shape + (Q,))
                rho = 1.0 + 0.02 * rng.standard_normal(shape)
                u = 0.02 * rng.standard_normal(shape + (3,))
                f = num.equilibrium(rho, u, n) + 1e-4 * g
                outs = {
                    "normalized": num.reconstruct_from_normalized_kinetic(g, n),
                    "kinetic": num.reconstruct_from_kinetic(g, n, tau, rho0),
                    "noneq": num.reconstruct_from_nonequilibrium(g, n, tau, rho0),
                    "populations": num.reconstruct_from_populations(f, n, tau, rho0),
                }
                for key, out in outs.items():
                    self.assertEqual(out.shape, shape + (r,), (n, shape, key))
                for idx in np.ndindex(*shape):
                    with self.subTest(lattice=n, shape=shape, idx=idx):
                        assert_close(self, outs["normalized"][idx], num.reconstruct_from_normalized_kinetic(g[idx], n), 1e-14)
                        assert_close(self, outs["kinetic"][idx], num.reconstruct_from_kinetic(g[idx], n, tau, rho0), 1e-14)
                        assert_close(self, outs["noneq"][idx], num.reconstruct_from_nonequilibrium(g[idx], n, tau, rho0), 1e-14)
                        assert_close(self, outs["populations"][idx], num.reconstruct_from_populations(f[idx], n, tau, rho0), 1e-12)

    def test_solver_layout_via_moveaxis(self):
        exp2 = _import(SRC / "experiment_II_3D_shear.py", "exp2_test_step2b")
        rng = np.random.default_rng(7)
        for n in LATTICES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                lat = exp2.load_lattice(ROOT / "operators" / f"Lhat_{n}.csv")
            rho = 1.0 + 0.01 * rng.standard_normal((4, 3, 2))           # (Nz, Ny, Nx)
            u = 0.02 * rng.standard_normal((3, 4, 3, 2))                # (3, Nz, Ny, Nx)
            feq_solver = exp2.equilibrium(lat, rho, u)                  # (Q, Nz, Ny, Nx)
            feq_here = num.equilibrium(rho, np.moveaxis(u, 0, -1), n)   # (Nz, Ny, Nx, Q)
            assert_close(self, np.moveaxis(feq_solver, 0, -1), feq_here, rtol=1e-14)


class Test5_ComplexInputs(unittest.TestCase):
    def test_complex_preserved_and_linear(self):
        rng = np.random.default_rng(8)
        for n in LATTICES:
            Q = num.num_populations(n)
            g1 = rng.standard_normal((4, Q)) + 1j * rng.standard_normal((4, Q))
            g2 = rng.standard_normal((4, Q)) + 1j * rng.standard_normal((4, Q))
            a, b = 0.3 - 1.7j, -2.0 + 0.5j
            for fn in (lambda g: num.reconstruct_from_normalized_kinetic(g, n),
                       lambda g: num.reconstruct_from_kinetic(g, n, 0.75, 1.3),
                       lambda g: num.reconstruct_from_nonequilibrium(g, n, 0.75, 1.3)):
                out = fn(g1)
                self.assertEqual(out.dtype, np.complex128)
                assert_close(self, fn(a * g1 + b * g2), a * fn(g1) + b * fn(g2), rtol=1e-13)
                assert_close(self, out, fn(g1.real) + 1j * fn(g1.imag), rtol=1e-14)
            self.assertEqual(num.reconstruct_from_kinetic(g1.real, n, 0.75).dtype, np.float64)

    def test_complex_ce2_fourier_amplitude(self):
        c0 = num.ce2_prefactor(1.1, 1.0)
        for n in LATTICES:
            q = np.array([float(x) for x in Q_VECTORS[2]]) * (0.4 - 0.9j)
            g = c0 * (L_FLOAT[n] @ q)
            expected = (np.array(DMAT[n].tolist(), dtype=float) @ q)
            assert_close(self, num.reconstruct_from_kinetic(g, n, 1.1), expected)

    def test_integer_and_float32_inputs_promoted(self):
        for n in LATTICES:
            Q = num.num_populations(n)
            gi = np.arange(Q)
            self.assertEqual(num.reconstruct_from_normalized_kinetic(gi, n).dtype, np.float64)
            assert_close(self, num.reconstruct_from_normalized_kinetic(gi.astype(np.float32), n),
                         num.reconstruct_from_normalized_kinetic(gi.astype(float), n), rtol=1e-14)


class Test6_FullPopulations(unittest.TestCase):
    def test_equilibrium_plus_ce2_perturbation(self):
        rng = np.random.default_rng(9)
        shape = (3, 4, 2)
        for n in LATTICES:
            for tau, rho0 in ((0.7, 1.0), (1.5, 1.05)):
                c0 = num.ce2_prefactor(tau, rho0)
                rho = 1.0 + 0.05 * rng.standard_normal(shape)
                u = 0.03 * rng.standard_normal(shape + (3,))
                qf = 1e-3 * rng.standard_normal(shape + (15,))            # node-dependent curvature
                g = c0 * qf @ L_FLOAT[n].T                                # (..., Q) synthetic CE2
                c = num.get_lattice(n).c.astype(float)
                # The CE2 vector carries no mass or momentum, so rho and u are unchanged.
                self.assertLess(np.max(np.abs(g.sum(axis=-1))), 1e-15)
                self.assertLess(np.max(np.abs(g @ c)), 1e-15)
                f = num.equilibrium(rho, u, n) + g
                r2, u2 = num.macroscopic(f, n)
                assert_close(self, r2, rho, rtol=1e-14)
                assert_close(self, u2, u, rtol=1e-11)
                expected = qf @ np.array(DMAT[n].tolist(), dtype=float).T
                out = num.reconstruct_from_populations(f, n, tau, rho0)
                assert_close(self, out, expected, rtol=1e-9)
                # A second-order (viscous-stress-like) non-equilibrium part does not change the result.
                Phi2 = hermite_le2_basis(n)[:, 4:]
                f2 = f + 1e-3 * rng.standard_normal(shape + (6,)) @ Phi2.T
                assert_close(self, num.reconstruct_from_populations(f2, n, tau, rho0), out, rtol=1e-9)

    def test_local_density_used_in_equilibrium(self):
        # Pure equilibrium at arbitrary (rho, u): f_neq = 0, so nothing is reconstructed,
        # whatever rho0 is. Using rho0 instead of the local density would leave a residual.
        rng = np.random.default_rng(10)
        for n in LATTICES:
            rho = 1.0 + 0.2 * rng.standard_normal(8) ** 2
            u = 0.05 * rng.standard_normal((8, 3))
            out = num.reconstruct_from_populations(num.equilibrium(rho, u, n), n, 0.8, rho0=1.0)
            self.assertLess(np.max(np.abs(out)), 1e-12)


class Test7_Normalization(unittest.TestCase):
    def test_prefactor_closed_form(self):
        for tau in (0.51, 0.6, 0.8, 1.0, 1.7, 3.0):
            for rho0 in (0.25, 1.0, 2.5):
                exact = sp.Rational(2, 9) * sp.nsimplify(rho0) * sp.nsimplify(tau) * (sp.nsimplify(tau) - sp.Rational(1, 2))
                self.assertLessEqual(abs(num.ce2_prefactor(tau, rho0) - float(exact)), 1e-13 * float(exact))  # tau - 1/2 cancels near tau = 0.51

    def test_applied_exactly_once(self):
        qv = Q_VECTORS[2]
        for n in LATTICES:
            expected = exact_expected(n, qv)
            for tau in (0.55, 0.8, 1.0, 1.9):
                for rho0 in (0.5, 1.0, 3.0):
                    c0 = num.ce2_prefactor(tau, rho0)
                    self.assertNotAlmostEqual(c0, 1.0)
                    g = ce2_kinetic(n, qv, c0)
                    for out in (num.reconstruct_from_kinetic(g, n, tau, rho0),
                                num.reconstruct_from_nonequilibrium(g, n, tau, rho0),
                                num.reconstruct_from_populations(num.equilibrium(np.array(1.0), np.zeros(3), n) + g,
                                                                 n, tau, rho0)):
                        with self.subTest(lattice=n, tau=tau, rho0=rho0):
                            assert_close(self, out, expected, rtol=1e-11)
                            self.assertFalse(np.allclose(out, expected * c0))      # not applied zero times
                            self.assertFalse(np.allclose(out, expected / c0))      # not applied twice

    def test_reference_density_scaling(self):
        # C0 is linear in rho0: doubling rho0 must halve every interface's output (once, not quartered).
        rng = np.random.default_rng(11)
        for n in LATTICES:
            g = rng.standard_normal(num.num_populations(n))
            for fn in (num.reconstruct_from_kinetic, num.reconstruct_from_nonequilibrium):
                assert_close(self, fn(g, n, 0.9, 2.0), 0.5 * fn(g, n, 0.9, 1.0), rtol=1e-14)

    def test_normalized_interface_has_no_prefactor(self):
        for n in LATTICES:
            qv = Q_VECTORS[1]
            g_norm = L_FLOAT[n] @ np.array([float(x) for x in qv])
            assert_close(self, num.reconstruct_from_normalized_kinetic(g_norm, n), exact_expected(n, qv))


class Test8_InvalidInputs(unittest.TestCase):
    def test_lattice(self):
        for bad in ("D3Q13", "d3q19", "D2Q9", "", None, 19):
            with self.assertRaises(ValueError):
                num.reconstruct_from_normalized_kinetic(np.zeros(19), bad)
            with self.assertRaises(ValueError):
                num.observable_names(bad)

    def test_shapes(self):
        for n in LATTICES:
            Q = num.num_populations(n)
            for bad in (np.zeros(Q + 1), np.zeros((Q, 2)), np.zeros((3, Q - 1)), np.array(1.0)):
                for call in (lambda a: num.reconstruct_from_normalized_kinetic(a, n),
                             lambda a: num.reconstruct_from_kinetic(a, n, 0.8),
                             lambda a: num.reconstruct_from_nonequilibrium(a, n, 0.8),
                             lambda a: num.reconstruct_from_populations(a, n, 0.8)):
                    with self.assertRaises(ValueError):
                        call(bad)
            with self.assertRaises(ValueError):
                num.equilibrium(np.ones(3), np.zeros((4, 3)), n)

    def test_nonfinite(self):
        for n in LATTICES:
            Q = num.num_populations(n)
            for bad_val in (np.nan, np.inf, -np.inf, complex(np.nan, 0), complex(0, np.inf)):
                a = np.full((2, Q), 0.1, dtype=complex if isinstance(bad_val, complex) else float)
                a[1, 3] = bad_val
                with self.assertRaises(ValueError):
                    num.reconstruct_from_normalized_kinetic(a, n)
                with self.assertRaises(ValueError):
                    num.reconstruct_from_kinetic(a, n, 0.8)
                with self.assertRaises(ValueError):
                    num.reconstruct_from_nonequilibrium(a, n, 0.8)
            f = num.equilibrium(np.ones(2), np.zeros((2, 3)), n)
            f[0, 0] = np.nan
            with self.assertRaises(ValueError):
                num.reconstruct_from_populations(f, n, 0.8)

    def test_relaxation_and_reference_density(self):
        n = "D3Q19"
        g = np.zeros(19)
        for bad in (0.5, 0.49, 0.0, -1.0, np.nan, np.inf):
            for fn in (num.reconstruct_from_kinetic, num.reconstruct_from_nonequilibrium,
                       num.reconstruct_from_populations):
                with self.assertRaises(ValueError):
                    fn(num.equilibrium(np.array(1.0), np.zeros(3), n), n, bad)
        for bad in (True, 1 + 0j, "1.0", np.array([1.0]), None):
            with self.assertRaises(TypeError):
                num.reconstruct_from_kinetic(g, n, bad)
        for bad in (0.0, -1.0, np.nan):
            with self.assertRaises(ValueError):
                num.reconstruct_from_kinetic(g, n, 0.8, rho0=bad)
        self.assertGreater(num.ce2_prefactor(np.float32(0.8)), 0)   # numpy real scalars accepted

    def test_populations_specific(self):
        for n in LATTICES:
            Q = num.num_populations(n)
            with self.assertRaises(TypeError):
                num.reconstruct_from_populations(np.ones(Q) * (1 + 1j) / Q, n, 0.8)
            with self.assertRaises(ValueError):
                num.reconstruct_from_populations(-np.ones(Q) / Q, n, 0.8)
            with self.assertRaises(TypeError):
                num.reconstruct_from_normalized_kinetic(np.array(["a"] * Q), n)
            with self.assertRaises(TypeError):
                num.reconstruct_from_normalized_kinetic(np.ones(Q, dtype=bool), n)

    def test_inputs_not_modified_and_readonly_accepted(self):
        rng = np.random.default_rng(12)
        for n in LATTICES:
            Q = num.num_populations(n)
            g = rng.standard_normal((3, Q))
            f = num.equilibrium(np.ones(3), np.zeros((3, 3)), n) + 1e-4 * g
            g0, f0 = g.copy(), f.copy()
            g.setflags(write=False)
            f.setflags(write=False)
            num.reconstruct_from_normalized_kinetic(g, n)
            num.reconstruct_from_kinetic(g, n, 0.8)
            num.reconstruct_from_nonequilibrium(g, n, 0.8)
            num.reconstruct_from_populations(f, n, 0.8)
            self.assertTrue(np.array_equal(g, g0) and np.array_equal(f, f0))


class Test9_Regression(unittest.TestCase):
    """The Step 1 and Step 2A suites run in the same discovery; here: inputs untouched."""

    DIGESTS = {
        "Lhat_D3Q15.csv": "aca891677b26ae89e217d8fa22a6be020c226257e491c285d236aa8acb2ce6fc",
        "Lhat_D3Q19.csv": "258b5dcd2ec4b4f8e735e4f180fe7fd91eb927bcb7471db12c1812dcd05c8a05",
        "Lhat_D3Q27.csv": "b749b00ff3315789e1b1443d9d0568eed8b75a5ec49dddda61c9f721bcc1b5ca",
    }

    def test_canonical_operators_unchanged(self):
        for name, digest in self.DIGESTS.items():
            self.assertEqual(hashlib.sha256((ROOT / "operators" / name).read_bytes()).hexdigest(), digest, name)

    def test_step_2a_identity_still_holds(self):
        for n in LATTICES:
            self.assertEqual(ex.hermite_reconstruction(OPS[n]) * OPS[n].Lhat, DMAT[n])


if __name__ == "__main__":
    unittest.main()
