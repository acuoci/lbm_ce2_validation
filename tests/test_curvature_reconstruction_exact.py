#!/usr/bin/env python3
"""
Exact tests for the Experiment IV reconstruction operators
(src/curvature_reconstruction_exact.py).

Run from the repository root:

    python -m unittest discover -s tests

Every assertion uses exact rational arithmetic (SymPy). The functional
definitions, output orderings and reference values below are written out
independently of the module under test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "curvature_reconstruction_exact.py"


def _import(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _import(MODULE, "curvature_reconstruction_exact")

LATTICES = ("D3Q15", "D3Q19", "D3Q27")
Q = ("ux_xy ux_xz ux_yy ux_zz ux_yz uy_xy uy_yz uy_xx uy_zz uy_xz "
     "uz_xz uz_yz uz_xx uz_yy uz_xy").split()
R = sp.Rational

# Independent restatement of the functional definitions (Section 3 of the task).
q = sp.Matrix(sp.symbols(" ".join(Q)))
sym = dict(zip(Q, q))
F = {
    "A_x": 2 * sym["ux_xy"] + sym["uy_xx"],
    "A_y": 2 * sym["uy_yz"] + sym["uz_yy"],
    "A_z": 2 * sym["uz_xz"] + sym["ux_zz"],
    "B_x": 2 * sym["ux_xz"] + sym["uz_xx"],
    "B_y": 2 * sym["uy_xy"] + sym["ux_yy"],
    "B_z": 2 * sym["uz_yz"] + sym["uy_zz"],
    "T": sym["ux_yz"] + sym["uy_xz"] + sym["uz_xy"],
}
F["S_1"] = F["A_x"] + F["B_z"]
F["S_2"] = F["B_x"] + F["A_y"]
F["S_3"] = F["B_y"] + F["A_z"]
ORDER = {
    "D3Q15": ("S_1", "S_2", "S_3", "T"),
    "D3Q19": ("A_x", "B_x", "B_y", "A_z", "A_y", "B_z"),
    "D3Q27": ("A_x", "B_x", "B_y", "A_z", "A_y", "B_z", "T"),
}
RANKS = {"D3Q15": 4, "D3Q19": 6, "D3Q27": 7}
EXPECTED_CHI2 = {
    "D3Q15": [R(2, 27), R(2, 27), R(2, 27), R(1, 81)],
    "D3Q19": [R(8, 81)] * 6,
    "D3Q27": [R(2, 27)] * 6 + [R(1, 27)],
}


def row_of(expr) -> sp.Matrix:
    return sp.Matrix([[sp.expand(expr).coeff(s) for s in q]])


def D_reference(lattice: str) -> sp.Matrix:
    return sp.Matrix.vstack(*[row_of(F[n]) for n in ORDER[lattice]])


def rank(M: sp.Matrix) -> int:
    return len(M.rref()[1])


def same_row_space(A, B) -> bool:
    return rank(A) == rank(B) == rank(sp.Matrix.vstack(A, B))


OPS = {n: cr.load_operator(n) for n in LATTICES}
D = {n: cr.observable_matrix(n) for n in LATTICES}
RH = {n: cr.hermite_reconstruction(OPS[n]) for n in LATTICES}
RSTAR, GPLUS = {}, {}
for _n in LATTICES:
    RSTAR[_n], GPLUS[_n] = cr.minimum_norm_reconstruction(OPS[_n])


class Test0_InputsAndOrdering(unittest.TestCase):
    def test_canonical_operators_loaded_exactly(self):
        for n, Qn in (("D3Q15", 15), ("D3Q19", 19), ("D3Q27", 27)):
            op = OPS[n]
            self.assertEqual(op.Lhat.shape, (Qn, 15))
            self.assertTrue(all(isinstance(x, sp.Rational) for x in op.Lhat))
            self.assertEqual(sum(op.w), 1)

    def test_observable_matrices_match_definitions(self):
        for n in LATTICES:
            self.assertEqual(D[n], D_reference(n), n)
            self.assertEqual(D[n].shape, (len(ORDER[n]), 15))
            self.assertEqual(cr.OUTPUT_ORDER[n], ORDER[n])

    def test_module_does_not_modify_canonical_csvs(self):
        digests = {
            "D3Q15": "aca891677b26ae89e217d8fa22a6be020c226257e491c285d236aa8acb2ce6fc",
            "D3Q19": "258b5dcd2ec4b4f8e735e4f180fe7fd91eb927bcb7471db12c1812dcd05c8a05",
            "D3Q27": "b749b00ff3315789e1b1443d9d0568eed8b75a5ec49dddda61c9f721bcc1b5ca",
        }
        for n, h in digests.items():
            self.assertEqual(hashlib.sha256(cr.operator_path(n).read_bytes()).hexdigest(), h, n)


class Test1_RanksAndRowSpaces(unittest.TestCase):
    def test_ranks(self):
        for n in LATTICES:
            self.assertEqual(rank(D[n]), RANKS[n], n)
            self.assertEqual(rank(OPS[n].Lhat), RANKS[n], n)

    def test_row_space_equality(self):
        for n in LATTICES:
            self.assertTrue(same_row_space(D[n], OPS[n].Lhat), n)


class Test2_IntersectionAndSum(unittest.TestCase):
    def test_intersection(self):
        D15, D19 = D["D3Q15"], D["D3Q19"]
        S = sp.Matrix.vstack(*[row_of(F[k]) for k in ("S_1", "S_2", "S_3")])
        self.assertEqual(rank(S), 3)
        self.assertTrue(same_row_space(sp.Matrix.vstack(D15, S), D15))  # span S within row(D15)
        self.assertTrue(same_row_space(sp.Matrix.vstack(D19, S), D19))  # span S within row(D19)
        dim_cap = rank(D15) + rank(D19) - rank(sp.Matrix.vstack(D15, D19))
        self.assertEqual(dim_cap, 3)                                   # so the intersection is exactly span S

    def test_sum(self):
        self.assertTrue(same_row_space(sp.Matrix.vstack(D["D3Q15"], D["D3Q19"]), D["D3Q27"]))

    def test_module_cross_lattice_report(self):
        self.assertTrue(all(cr.verify_cross_lattice().values()))


class Test3_HermiteReconstruction(unittest.TestCase):
    def test_identity(self):
        for n in LATTICES:
            self.assertEqual(RH[n] * OPS[n].Lhat, D[n], n)
            self.assertEqual(RH[n].shape, (len(ORDER[n]), OPS[n].Q))

    def test_stated_mixing_inverse(self):
        M = sp.Matrix([[1, R(-1, 2)], [R(-1, 2), 1]])
        self.assertEqual(M.inv(), R(4, 3) * sp.Matrix([[1, R(1, 2)], [R(1, 2), 1]]))
        self.assertEqual(cr.MIXING.inv(), cr.MIXING_INVERSE_STATED)

    def test_d3q19_mixing_systems(self):
        op = OPS["D3Q19"]
        M = sp.Matrix([[1, R(-1, 2)], [R(-1, 2), 1]])
        for (ma, mb), (fa, fb) in ((("xxy", "yzz"), ("A_x", "B_z")),
                                   (("xxz", "yyz"), ("B_x", "A_y")),
                                   (("xyy", "xzz"), ("B_y", "A_z"))):
            lhs = cr.hermite3_matrix(op.velocities, (ma, mb)) * op.Lhat
            rhs = M * sp.Matrix.vstack(row_of(F[fa]), row_of(F[fb]))
            self.assertEqual(lhs, rhs, (ma, mb))

    def test_stated_direct_identities(self):
        cases = {
            "D3Q15": {"xxy": F["S_1"], "xxz": F["S_2"], "xyy": F["S_3"], "xyz": 3 * F["T"]},
            "D3Q27": {"xxy": F["A_x"], "xxz": F["B_x"], "xyy": F["B_y"], "xzz": F["A_z"],
                      "yyz": F["A_y"], "yzz": F["B_z"], "xyz": F["T"]},
        }
        for n, ids in cases.items():
            op = OPS[n]
            for lab, expr in ids.items():
                self.assertEqual(cr.hermite3_matrix(op.velocities, (lab,)) * op.Lhat, row_of(expr), (n, lab))

    def test_hermite_polynomial(self):
        self.assertEqual(cr.hermite3((1, 1, 0), "xxy"), R(2, 3))
        self.assertEqual(cr.hermite3((0, 1, 0), "xxy"), R(-1, 3))
        self.assertEqual(cr.hermite3((1, 0, 0), "xxx"), 0)
        self.assertEqual(cr.hermite3((1, -1, 1), "xyz"), -1)


class Test4_MinimumNormReconstruction(unittest.TestCase):
    def test_identity(self):
        for n in LATTICES:
            self.assertEqual(RSTAR[n] * OPS[n].Lhat, D[n], n)

    def test_formula(self):
        for n in LATTICES:
            op = OPS[n]
            self.assertEqual(RSTAR[n], D[n] * GPLUS[n] * op.Lhat.T * op.Winv, n)

    def test_exact_pseudoinverse(self):
        for n in LATTICES:
            G = cr.gram_matrix(OPS[n])
            P = GPLUS[n]
            self.assertEqual(G, G.T)
            self.assertEqual(rank(G), RANKS[n])
            self.assertEqual(G * P * G, G)
            self.assertEqual(P * G * P, P)
            self.assertEqual((G * P).T, G * P)
            self.assertEqual((P * G).T, P * G)
            self.assertTrue(all(isinstance(x, sp.Rational) for x in P))

    def test_pinv_against_sympy(self):
        for n in LATTICES:
            self.assertEqual(GPLUS[n], cr.gram_matrix(OPS[n]).pinv(), n)

    def test_pinv_edge_cases(self):
        Z = sp.zeros(3, 2)
        self.assertEqual(cr.exact_pinv(Z), sp.zeros(2, 3))
        M = sp.Matrix([[1, 2], [2, 4], [0, 0]])
        P = cr.exact_pinv(M)
        self.assertTrue(all(cr.penrose_conditions(M, P).values()))


class Test5_MinimumNormOrthogonality(unittest.TestCase):
    def test_weighted_orthogonality_to_ker_LT(self):
        for n in LATTICES:
            op = OPS[n]
            ker = op.Lhat.T.nullspace()
            self.assertEqual(len(ker), op.Q - RANKS[n])
            for j in range(RSTAR[n].rows):
                for z in ker:
                    self.assertEqual((RSTAR[n][j, :] * op.W * z)[0, 0], 0, (n, j))

    def test_minimality_against_hermite_rows(self):
        # Any b with Lhat^T b = d satisfies |b|_W^2 = |b*|_W^2 + |b - b*|_W^2.
        for n in LATTICES:
            W = OPS[n].W
            for j in range(D[n].rows):
                bh, bs = RH[n][j, :], RSTAR[n][j, :]
                nh, ns = (bh * W * bh.T)[0, 0], (bs * W * bs.T)[0, 0]
                diff = bh - bs
                self.assertEqual(nh, ns + (diff * W * diff.T)[0, 0])
                self.assertGreaterEqual(nh, ns)

    def test_perturbed_solution_is_not_shorter(self):
        # Adding any kernel direction of Lhat^T keeps Lhat^T b = d and increases the W-norm.
        for n in LATTICES:
            op = OPS[n]
            z = op.Lhat.T.nullspace()[0].T
            b = RSTAR[n][0, :]
            bp = b + R(1, 7) * z
            self.assertEqual(bp * op.Lhat, b * op.Lhat)
            self.assertGreater((bp * op.W * bp.T)[0, 0], (b * op.W * b.T)[0, 0])


class Test6_Sensitivities(unittest.TestCase):
    def test_exact_values_and_identity(self):
        for n in LATTICES:
            chi2 = cr.squared_sensitivities(D[n], GPLUS[n])
            norms = cr.weighted_row_norms_sq(RSTAR[n], OPS[n].W)
            self.assertEqual(chi2, EXPECTED_CHI2[n], n)
            self.assertEqual(chi2, norms, n)
            self.assertTrue(all(isinstance(x, sp.Rational) for x in chi2))

    def test_verify_lattice_summary(self):
        for n in LATTICES:
            r = cr.verify_lattice(n)
            self.assertTrue(r.ok, n)
            self.assertEqual(r.chi2, EXPECTED_CHI2[n])


class Test7_ReconstructionOfCurvatureVectors(unittest.TestCase):
    VECTORS = [
        sp.Matrix(list(range(1, 16))),
        sp.Matrix([(-1) ** k * R(k + 1, k + 2) for k in range(15)]),
        sp.Matrix([R(3, 7), -2, 0, R(5, 11), 1, 0, -R(1, 3), 4, 0, R(2, 9), -1, 0, R(7, 5), 3, -R(4, 13)]),
        sp.Matrix([1 if k in (4, 9, 14) else 0 for k in range(15)]),   # pure fully mixed curvature
    ]

    def test_three_way_agreement(self):
        for n in LATTICES:
            L = OPS[n].Lhat
            for k, v in enumerate(self.VECTORS):
                g = L * v                     # normalized kinetic population response
                expected = D[n] * v
                with self.subTest(lattice=n, vector=k):
                    self.assertEqual(RH[n] * g, expected)
                    self.assertEqual(RSTAR[n] * g, expected)

    def test_against_symbolic_definitions(self):
        for n in LATTICES:
            v = self.VECTORS[2]
            vals = dict(zip(q, v))
            expected = sp.Matrix([F[name].subs(vals) for name in ORDER[n]])
            self.assertEqual(RH[n] * OPS[n].Lhat * v, expected, n)


class Test8_Observability(unittest.TestCase):
    CASES = (
        ("T", "D3Q15", True), ("T", "D3Q19", False), ("T", "D3Q27", True),
        ("A_x", "D3Q15", False), ("A_x", "D3Q19", True), ("A_x", "D3Q27", True),
        ("ux_xy", "D3Q15", False), ("ux_xy", "D3Q19", False), ("ux_xy", "D3Q27", False),
    )

    def functional(self, name):
        return row_of(F[name]) if name in F else row_of(sym[name])

    def test_observability_and_witnesses(self):
        for fname, n, expected in self.CASES:
            L, d = OPS[n].Lhat, self.functional(fname)
            with self.subTest(functional=fname, lattice=n):
                self.assertEqual(cr.is_observable(L, d), expected)
                z = cr.nullspace_witness(L, d)
                if expected:
                    self.assertIsNone(z)
                else:
                    self.assertIsNotNone(z)
                    self.assertTrue(all(isinstance(x, sp.Integer) for x in z))
                    self.assertEqual(L * z, sp.zeros(L.rows, 1))
                    self.assertNotEqual((d * z)[0, 0], 0)

    def test_witnesses_explicit(self):
        e = lambda **kw: sp.Matrix([kw.get(k, 0) for k in Q])  # noqa: E731
        explicit = {
            ("T", "D3Q19"): e(ux_yz=1),
            ("A_x", "D3Q15"): e(ux_xy=1, uy_zz=-2),
            ("ux_xy", "D3Q15"): e(ux_xy=1, uy_xx=-2),
            ("ux_xy", "D3Q19"): e(ux_xy=1, uy_xx=-2),
            ("ux_xy", "D3Q27"): e(ux_xy=1, uy_xx=-2),
        }
        for (fname, n), z in explicit.items():
            L, d = OPS[n].Lhat, self.functional(fname)
            self.assertEqual(L * z, sp.zeros(L.rows, 1), (fname, n))
            self.assertNotEqual((d * z)[0, 0], 0, (fname, n))
            self.assertEqual(cr.nullspace_witness(L, d), z, (fname, n))

    def test_all_reconstructed_outputs_observable(self):
        for n in LATTICES:
            for j in range(D[n].rows):
                self.assertTrue(cr.is_observable(OPS[n].Lhat, D[n][j, :]))

    def test_rank_test_agrees_with_kernel_pairing(self):
        # Independent criterion: d observable iff d z = 0 for every z in ker(Lhat).
        for n in LATTICES:
            L = OPS[n].Lhat
            ker = L.nullspace()
            for name in list(F) + Q:
                d = self.functional(name)
                pairing_zero = all((d * z)[0, 0] == 0 for z in ker)
                self.assertEqual(cr.is_observable(L, d), pairing_zero, (n, name))


class Test9_NegativeControls(unittest.TestCase):
    """Corrupted operators must break the identities (the checks are not vacuous)."""

    def test_wrong_mixing_coefficient_detected(self):
        op = OPS["D3Q19"]
        K = cr.hermite_coefficient_matrix("D3Q19")
        K[0, cr.H3_LABELS.index("yzz")] = R(1, 2)   # correct value is 2/3
        self.assertNotEqual(K * cr.hermite3_matrix(op.velocities) * op.Lhat, D["D3Q19"])

    def test_wrong_t_scaling_detected(self):
        op = OPS["D3Q15"]
        K = cr.hermite_coefficient_matrix("D3Q15")
        K[3, cr.H3_LABELS.index("xyz")] = 1         # correct value is 1/3
        self.assertNotEqual(K * cr.hermite3_matrix(op.velocities) * op.Lhat, D["D3Q15"])

    def test_unobservable_row_cannot_be_reconstructed(self):
        # Asking D3Q19 for T: the min-norm formula returns something, but R* Lhat != D.
        op = OPS["D3Q19"]
        Dbad = sp.Matrix.vstack(D["D3Q19"], row_of(F["T"]))
        Rbad, _ = cr.minimum_norm_reconstruction(op, Dbad)
        self.assertNotEqual(Rbad * op.Lhat, Dbad)
        self.assertEqual((Rbad * op.Lhat)[6, :], sp.zeros(1, 15))


if __name__ == "__main__":
    unittest.main()
