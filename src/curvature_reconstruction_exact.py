#!/usr/bin/env python3
"""
Experiment IV, Step 2A: exact curvature-reconstruction operators.

For each canonical normalized CE2 operator Lhat_Q (Q x 15, operators/Lhat_DQ*.csv)
this module builds, in exact rational arithmetic (SymPy):

    D_Q   observable-functional matrix,          d_Q = D_Q q          (n_Q x 15)
    R^H_Q explicit third-order-Hermite reconstruction, R^H_Q Lhat_Q = D_Q (n_Q x Q)
    R*_Q  weighted minimum-norm reconstruction,
              R*_Q = D_Q G_Q^+ Lhat_Q^T W_Q^-1,   G_Q = Lhat_Q^T W_Q^-1 Lhat_Q
    chi^2 squared sensitivity factors, chi^2_{Q,j} = d_j^T G_Q^+ d_j

and provides an exact observability test with rational nullspace witnesses.
It is independent of the LBM solvers; nothing here uses floating point.

Output ordering (rows of D_Q, R^H_Q, R*_Q)
------------------------------------------
    D3Q15: (S_1, S_2, S_3, T)                          n = 4
    D3Q19: (A_x, B_x, B_y, A_z, A_y, B_z)              n = 6
    D3Q27: (A_x, B_x, B_y, A_z, A_y, B_z, T)           n = 7

Columns of D_Q follow the canonical curvature ordering (Q_NAMES); columns of
R^H_Q and R*_Q follow the population (row) ordering of the canonical CSV.

Usage
-----
    python src/curvature_reconstruction_exact.py      # print verification report

Exit status is nonzero if any identity fails.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_DIR = ROOT / "operators"

CS2 = sp.Rational(1, 3)
AXES = "xyz"
LATTICES: Tuple[str, ...] = ("D3Q15", "D3Q19", "D3Q27")

Q_NAMES: Tuple[str, ...] = (
    "ux_xy", "ux_xz", "ux_yy", "ux_zz", "ux_yz",
    "uy_xy", "uy_yz", "uy_xx", "uy_zz", "uy_xz",
    "uz_xz", "uz_yz", "uz_xx", "uz_yy", "uz_xy",
)
HEADER: Tuple[str, ...] = ("cx", "cy", "cz", "w") + Q_NAMES
NQ = len(Q_NAMES)

# -----------------------------------------------------------------------------
# Curvature functionals (coefficients on the canonical coordinates q)
# -----------------------------------------------------------------------------

BASE_FUNCTIONALS: Dict[str, Dict[str, int]] = {
    "A_x": {"ux_xy": 2, "uy_xx": 1},
    "A_y": {"uy_yz": 2, "uz_yy": 1},
    "A_z": {"uz_xz": 2, "ux_zz": 1},
    "B_x": {"ux_xz": 2, "uz_xx": 1},
    "B_y": {"uy_xy": 2, "ux_yy": 1},
    "B_z": {"uz_yz": 2, "uy_zz": 1},
    "T":   {"ux_yz": 1, "uy_xz": 1, "uz_xy": 1},
}
COMPOSITE_FUNCTIONALS: Dict[str, Dict[str, int]] = {
    "S_1": {"A_x": 1, "B_z": 1},
    "S_2": {"B_x": 1, "A_y": 1},
    "S_3": {"B_y": 1, "A_z": 1},
}
OUTPUT_ORDER: Dict[str, Tuple[str, ...]] = {
    "D3Q15": ("S_1", "S_2", "S_3", "T"),
    "D3Q19": ("A_x", "B_x", "B_y", "A_z", "A_y", "B_z"),
    "D3Q27": ("A_x", "B_x", "B_y", "A_z", "A_y", "B_z", "T"),
}
EXPECTED_RANK: Dict[str, int] = {"D3Q15": 4, "D3Q19": 6, "D3Q27": 7}

# -----------------------------------------------------------------------------
# Third-order Hermite moments and the stated reconstruction recipes
# -----------------------------------------------------------------------------

H3_LABELS: Tuple[str, ...] = ("xxx", "yyy", "zzz", "xxy", "xxz", "xyy", "xzz", "yyz", "yzz", "xyz")

# D3Q19 mixing: (m_a, m_b)^T = MIXING (F_a, F_b)^T for the three pairs below.
MIXING = sp.Matrix([[1, sp.Rational(-1, 2)], [sp.Rational(-1, 2), 1]])
MIXING_INVERSE_STATED = sp.Rational(4, 3) * sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 1]])
D3Q19_MIXING_PAIRS: Tuple[Tuple[Tuple[str, str], Tuple[str, str]], ...] = (
    (("xxy", "yzz"), ("A_x", "B_z")),
    (("xxz", "yyz"), ("B_x", "A_y")),
    (("xyy", "xzz"), ("B_y", "A_z")),
)

# Direct recipes: functional = sum_k coeff_k m_k.
DIRECT_RECIPES: Dict[str, Dict[str, Dict[str, sp.Rational]]] = {
    "D3Q15": {
        "S_1": {"xxy": sp.Integer(1)},
        "S_2": {"xxz": sp.Integer(1)},
        "S_3": {"xyy": sp.Integer(1)},
        "T":   {"xyz": sp.Rational(1, 3)},
    },
    "D3Q27": {
        "A_x": {"xxy": sp.Integer(1)},
        "B_x": {"xxz": sp.Integer(1)},
        "B_y": {"xyy": sp.Integer(1)},
        "A_z": {"xzz": sp.Integer(1)},
        "A_y": {"yyz": sp.Integer(1)},
        "B_z": {"yzz": sp.Integer(1)},
        "T":   {"xyz": sp.Integer(1)},
    },
}


# -----------------------------------------------------------------------------
# Exact loading
# -----------------------------------------------------------------------------

def _rational(s: str) -> sp.Rational:
    fr = Fraction(s.strip())
    return sp.Rational(fr.numerator, fr.denominator)


@dataclass
class ExactOperator:
    name: str
    velocities: List[Tuple[int, int, int]]
    w: List[sp.Rational]
    Lhat: sp.Matrix                     # Q x 15

    @property
    def Q(self) -> int:
        return len(self.velocities)

    @property
    def W(self) -> sp.Matrix:
        return sp.diag(*self.w)

    @property
    def Winv(self) -> sp.Matrix:
        return sp.diag(*[1 / x for x in self.w])


def operator_path(lattice: str) -> Path:
    if lattice not in LATTICES:
        raise ValueError(f"Unknown lattice {lattice!r}; expected one of {LATTICES}.")
    return OPERATOR_DIR / f"Lhat_{lattice}.csv"


def load_operator(lattice: str, path: Optional[Path] = None) -> ExactOperator:
    """Read a canonical operator CSV with exact rationals (read-only)."""
    path = Path(path) if path is not None else operator_path(lattice)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if tuple(rows[0]) != HEADER:
        raise ValueError(f"{path.name}: unexpected header {rows[0]}")
    body = rows[1:]
    vel = [(int(r[0]), int(r[1]), int(r[2])) for r in body]
    w = [_rational(r[3]) for r in body]
    lhat = sp.Matrix([[_rational(x) for x in r[4:]] for r in body])
    if lhat.shape != (len(vel), NQ):
        raise ValueError(f"{path.name}: Lhat has shape {lhat.shape}")
    if sum(w) != 1:
        raise ValueError(f"{path.name}: weights sum to {sum(w)}")
    if f"D3Q{len(vel)}" != lattice:
        raise ValueError(f"{path.name}: {len(vel)} populations, expected {lattice}")
    return ExactOperator(name=lattice, velocities=vel, w=w, Lhat=lhat)


# -----------------------------------------------------------------------------
# Functionals and D_Q
# -----------------------------------------------------------------------------

def functional_row(coeffs: Mapping[str, int]) -> sp.Matrix:
    """1 x 15 row vector from coefficients on curvature coordinate names."""
    row = sp.zeros(1, NQ)
    for name, c in coeffs.items():
        row[0, Q_NAMES.index(name)] += sp.Rational(c)
    return row


def named_functional(name: str) -> sp.Matrix:
    """Row vector for A_*, B_*, T, S_1..S_3, or a single coordinate such as 'ux_xy'."""
    if name in BASE_FUNCTIONALS:
        return functional_row(BASE_FUNCTIONALS[name])
    if name in COMPOSITE_FUNCTIONALS:
        out = sp.zeros(1, NQ)
        for part, c in COMPOSITE_FUNCTIONALS[name].items():
            out += c * named_functional(part)
        return out
    if name in Q_NAMES:
        return functional_row({name: 1})
    raise KeyError(f"Unknown functional {name!r}")


def observable_matrix(lattice: str) -> sp.Matrix:
    """D_Q, rows in OUTPUT_ORDER[lattice]."""
    return sp.Matrix.vstack(*[named_functional(n) for n in OUTPUT_ORDER[lattice]])


# -----------------------------------------------------------------------------
# Hermite reconstruction R^H_Q
# -----------------------------------------------------------------------------

def hermite3(c: Sequence[int], label: str) -> sp.Rational:
    """H3_abg(c) = c_a c_b c_g - c_s^2 (c_a d_bg + c_b d_ag + c_g d_ab), c_s^2 = 1/3."""
    a, b, g = (AXES.index(ch) for ch in label)
    d = lambda i, j: 1 if i == j else 0  # noqa: E731
    return sp.Integer(c[a] * c[b] * c[g]) - CS2 * (c[a] * d(b, g) + c[b] * d(a, g) + c[g] * d(a, b))


def hermite3_matrix(velocities: Sequence[Sequence[int]],
                    labels: Sequence[str] = H3_LABELS) -> sp.Matrix:
    """Moment-extraction matrix: row k gives m_k = sum_i H3_k(c_i) g_i."""
    return sp.Matrix([[hermite3(c, lab) for c in velocities] for lab in labels])


def hermite_recipe(lattice: str) -> Dict[str, Dict[str, sp.Rational]]:
    """Coefficients of each output functional on the Hermite moments m_k."""
    if lattice in DIRECT_RECIPES:
        return DIRECT_RECIPES[lattice]
    if lattice == "D3Q19":
        inv = MIXING.inv()
        recipe: Dict[str, Dict[str, sp.Rational]] = {}
        for (ma, mb), (fa, fb) in D3Q19_MIXING_PAIRS:
            recipe[fa] = {ma: inv[0, 0], mb: inv[0, 1]}
            recipe[fb] = {ma: inv[1, 0], mb: inv[1, 1]}
        return recipe
    raise ValueError(lattice)


def hermite_coefficient_matrix(lattice: str) -> sp.Matrix:
    """K_Q (n_Q x 10) with R^H_Q = K_Q * hermite3_matrix(c)."""
    recipe = hermite_recipe(lattice)
    K = sp.zeros(len(OUTPUT_ORDER[lattice]), len(H3_LABELS))
    for i, name in enumerate(OUTPUT_ORDER[lattice]):
        for lab, c in recipe[name].items():
            K[i, H3_LABELS.index(lab)] = c
    return K


def hermite_reconstruction(op: ExactOperator) -> sp.Matrix:
    """R^H_Q (n_Q x Q)."""
    return hermite_coefficient_matrix(op.name) * hermite3_matrix(op.velocities)


# -----------------------------------------------------------------------------
# Exact pseudoinverse and minimum-norm reconstruction R*_Q
# -----------------------------------------------------------------------------

def exact_rank(M: sp.Matrix) -> int:
    return len(M.rref()[1])


def exact_pinv(M: sp.Matrix) -> sp.Matrix:
    """
    Moore-Penrose pseudoinverse by exact full-rank factorization M = F C
    (F = pivot columns of M, C = nonzero rows of rref(M)):
        M^+ = C^T (C C^T)^-1 (F^T F)^-1 F^T.
    """
    R, piv = M.rref()
    r = len(piv)
    if r == 0:
        return sp.zeros(M.cols, M.rows)
    F = M[:, list(piv)]
    C = R[:r, :]
    return C.T * (C * C.T).inv() * (F.T * F).inv() * F.T


def penrose_conditions(M: sp.Matrix, P: sp.Matrix) -> Dict[str, bool]:
    """The four Moore-Penrose conditions for real matrices, checked exactly."""
    return {
        "MPM=M": M * P * M == M,
        "PMP=P": P * M * P == P,
        "(MP)^T=MP": (M * P).T == M * P,
        "(PM)^T=PM": (P * M).T == P * M,
    }


def gram_matrix(op: ExactOperator) -> sp.Matrix:
    """G_Q = Lhat^T W^-1 Lhat (15 x 15)."""
    return op.Lhat.T * op.Winv * op.Lhat


def minimum_norm_reconstruction(op: ExactOperator, D: Optional[sp.Matrix] = None
                                ) -> Tuple[sp.Matrix, sp.Matrix]:
    """Return (R*_Q, G_Q^+) with R*_Q = D G^+ Lhat^T W^-1."""
    D = observable_matrix(op.name) if D is None else D
    Gp = exact_pinv(gram_matrix(op))
    return D * Gp * op.Lhat.T * op.Winv, Gp


def squared_sensitivities(D: sp.Matrix, Gplus: sp.Matrix) -> List[sp.Rational]:
    """chi^2_j = d_j^T G^+ d_j for each row d_j of D."""
    return [(D[j, :] * Gplus * D[j, :].T)[0, 0] for j in range(D.rows)]


def weighted_row_norms_sq(R: sp.Matrix, W: sp.Matrix) -> List[sp.Rational]:
    """b_j^T W b_j for each row b_j of R."""
    return [(R[j, :] * W * R[j, :].T)[0, 0] for j in range(R.rows)]


# -----------------------------------------------------------------------------
# Observability and nullspace witnesses
# -----------------------------------------------------------------------------

def is_observable(Lhat: sp.Matrix, d: sp.Matrix) -> bool:
    """True iff the functional d (1 x 15) lies in row(Lhat)."""
    return exact_rank(sp.Matrix.vstack(Lhat, d)) == exact_rank(Lhat)


def _primitive_integer(v: sp.Matrix) -> sp.Matrix:
    """Scale a rational vector to coprime integers with a positive first nonzero entry."""
    den = math.lcm(*[int(sp.Rational(x).q) for x in v]) if len(v) else 1
    ints = [int(x * den) for x in v]
    g = math.gcd(*ints) if any(ints) else 1
    ints = [x // g for x in ints]
    first = next((x for x in ints if x != 0), 1)
    sign = 1 if first > 0 else -1
    return sp.Matrix([sign * x for x in ints])


def nullspace_witness(Lhat: sp.Matrix, d: sp.Matrix) -> Optional[sp.Matrix]:
    """
    For an unobservable functional d, return an exact integer curvature vector
    z (15 x 1) with Lhat z = 0 and d z != 0. Return None if d is observable.

    Since row(Lhat) = ker(Lhat)^perp, d is unobservable iff some basis vector
    of ker(Lhat) pairs nontrivially with d; the first such basis vector of the
    exact rref nullspace is returned (deterministic).
    """
    for z in Lhat.nullspace():
        if (d * z)[0, 0] != 0:
            z = _primitive_integer(z)
            assert Lhat * z == sp.zeros(Lhat.rows, 1) and (d * z)[0, 0] != 0
            return z
    return None


def describe_vector(z: sp.Matrix) -> str:
    return " + ".join(f"{z[k]}*{Q_NAMES[k]}" for k in range(len(z)) if z[k] != 0).replace("+ -", "- ")


# -----------------------------------------------------------------------------
# Row-space helpers
# -----------------------------------------------------------------------------

def same_row_space(A: sp.Matrix, B: sp.Matrix) -> bool:
    ra, rb = exact_rank(A), exact_rank(B)
    return ra == rb == exact_rank(sp.Matrix.vstack(A, B))


def intersection_dimension(A: sp.Matrix, B: sp.Matrix) -> int:
    return exact_rank(A) + exact_rank(B) - exact_rank(sp.Matrix.vstack(A, B))


def kernel_of_transpose(Lhat: sp.Matrix) -> List[sp.Matrix]:
    """Basis of ker(Lhat^T) in population space (Q x 1 vectors)."""
    return Lhat.T.nullspace()


# -----------------------------------------------------------------------------
# Full verification
# -----------------------------------------------------------------------------

@dataclass
class LatticeResult:
    name: str
    outputs: Tuple[str, ...]
    rank_D: int
    rank_L: int
    row_space_equal: bool
    hermite_ok: bool
    min_norm_ok: bool
    orthogonality_ok: bool
    penrose_ok: bool
    sensitivity_ok: bool
    chi2: List[sp.Rational]
    hermite_norm_sq: List[sp.Rational]
    D: sp.Matrix
    RH: sp.Matrix
    Rstar: sp.Matrix
    details: List[str] = field(default_factory=list)

    @property
    def hermite_equals_min_norm(self) -> bool:
        """Observation (not a requirement): is the Hermite reconstruction already minimum-norm?"""
        return self.RH == self.Rstar

    @property
    def ok(self) -> bool:
        return all((self.rank_D == EXPECTED_RANK[self.name], self.rank_L == EXPECTED_RANK[self.name],
                    self.row_space_equal, self.hermite_ok, self.min_norm_ok, self.orthogonality_ok,
                    self.penrose_ok, self.sensitivity_ok))


def verify_lattice(lattice: str) -> LatticeResult:
    op = load_operator(lattice)
    L, W = op.Lhat, op.W
    D = observable_matrix(lattice)
    details: List[str] = []

    rank_D, rank_L = exact_rank(D), exact_rank(L)
    row_eq = same_row_space(D, L)

    RH = hermite_reconstruction(op)
    hermite_ok = RH * L == D
    if not hermite_ok:
        details.append(f"R^H Lhat - D = {RH * L - D}")

    G = gram_matrix(op)
    Gp = exact_pinv(G)
    pen = penrose_conditions(G, Gp)
    penrose_ok = all(pen.values()) and G == G.T
    Rstar = D * Gp * L.T * op.Winv
    min_norm_ok = Rstar * L == D
    if not min_norm_ok:
        details.append(f"R* Lhat - D = {Rstar * L - D}")

    # Independent minimum-norm check: weighted orthogonality to ker(Lhat^T).
    kerLT = kernel_of_transpose(L)
    orth_ok = all((Rstar[j, :] * W * z)[0, 0] == 0 for j in range(Rstar.rows) for z in kerLT)
    orth_ok = orth_ok and len(kerLT) == op.Q - rank_L

    chi2 = squared_sensitivities(D, Gp)
    norms = weighted_row_norms_sq(Rstar, W)
    sens_ok = chi2 == norms and all(isinstance(x, sp.Rational) and x > 0 for x in chi2)

    return LatticeResult(
        name=lattice, outputs=OUTPUT_ORDER[lattice], rank_D=rank_D, rank_L=rank_L,
        row_space_equal=row_eq, hermite_ok=hermite_ok, min_norm_ok=min_norm_ok,
        orthogonality_ok=orth_ok, penrose_ok=penrose_ok, sensitivity_ok=sens_ok,
        chi2=chi2, hermite_norm_sq=weighted_row_norms_sq(RH, W), D=D, RH=RH, Rstar=Rstar,
        details=details,
    )


def verify_cross_lattice() -> Dict[str, bool]:
    D15, D19, D27 = (observable_matrix(n) for n in LATTICES)
    S = sp.Matrix.vstack(*[named_functional(n) for n in ("S_1", "S_2", "S_3")])
    in15 = same_row_space(sp.Matrix.vstack(D15, S), D15)
    in19 = same_row_space(sp.Matrix.vstack(D19, S), D19)
    return {
        "row(D15) ∩ row(D19) = span{S1,S2,S3}":
            in15 and in19 and exact_rank(S) == 3 and intersection_dimension(D15, D19) == 3,
        "row(D15) + row(D19) = row(D27)": same_row_space(sp.Matrix.vstack(D15, D19), D27),
    }


OBSERVABILITY_CASES: Tuple[Tuple[str, str, bool], ...] = (
    ("T", "D3Q15", True), ("T", "D3Q19", False), ("T", "D3Q27", True),
    ("A_x", "D3Q15", False), ("A_x", "D3Q19", True), ("A_x", "D3Q27", True),
    ("ux_xy", "D3Q15", False), ("ux_xy", "D3Q19", False), ("ux_xy", "D3Q27", False),
)


def verify_observability() -> List[Tuple[str, str, bool, bool, Optional[sp.Matrix]]]:
    """(functional, lattice, expected, observed, witness) for OBSERVABILITY_CASES."""
    out = []
    ops = {n: load_operator(n) for n in LATTICES}
    for fname, lat, expected in OBSERVABILITY_CASES:
        L, d = ops[lat].Lhat, named_functional(fname)
        obs = is_observable(L, d)
        out.append((fname, lat, expected, obs, None if obs else nullspace_witness(L, d)))
    return out


def main() -> int:
    ok = True
    results = [verify_lattice(n) for n in LATTICES]
    print("Experiment IV - exact curvature-reconstruction operators")
    print("--------------------------------------------------------")
    print(f"{'Lattice':<8}{'rank D':>7}{'rank L':>7}  {'row(D)=row(L)':<14}{'Hermite':<9}"
          f"{'Min-norm':<9}{'W-orth':<8}{'Penrose':<9}{'Sensitivity':<11}")
    pf = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    for r in results:
        ok &= r.ok
        print(f"{r.name:<8}{r.rank_D:>7}{r.rank_L:>7}  {pf(r.row_space_equal):<14}{pf(r.hermite_ok):<9}"
              f"{pf(r.min_norm_ok):<9}{pf(r.orthogonality_ok):<8}{pf(r.penrose_ok):<9}{pf(r.sensitivity_ok):<11}")
        print(f"{'':<8}R^H == R* exactly: {r.hermite_equals_min_norm}")
        for line in r.details:
            print("   ", line)

    print("\nSquared sensitivities chi^2 = d^T G^+ d = b*^T W b*   (Hermite row b^H^T W b^H)")
    for r in results:
        for name, c, h in zip(r.outputs, r.chi2, r.hermite_norm_sq):
            print(f"  {r.name}  {name:<4} chi^2 = {str(c):<8} Hermite-row norm^2 = {h}")

    print("\nCross-lattice row spaces")
    for k, v in verify_cross_lattice().items():
        ok &= v
        print(f"  {k:<42}{pf(v)}")

    print("\nObservability")
    for fname, lat, expected, obs, z in verify_observability():
        good = obs == expected
        ok &= good
        status = "observable" if obs else "UNobservable"
        print(f"  {fname:<6}{lat}: {status:<13} expected {'observable' if expected else 'unobservable':<13}{pf(good)}")
        if z is not None:
            print(f"        witness z = {describe_vector(z)}   (Lhat z = 0, d.z = {(named_functional(fname) * z)[0, 0]})")

    print("\nAll checks passed." if ok else "\nVERIFICATION FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
