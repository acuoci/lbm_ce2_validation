#!/usr/bin/env python3
"""
Generate and verify the normalized D3Q15 CE2 curvature-observability operator.

    Lhat_15 = 1/(2 c_s^6) * P_K,15 W_15 C3_15 E_inc = 27/2 * P_K,15 W_15 A

All construction steps use exact rational arithmetic (SymPy). No floating
point is used to build or verify the operator; floating point appears only in
the compatibility check against the existing NumPy loader of Experiment II.

Usage (from the repository root)
--------------------------------
    python scripts/generate_d3q15_operator.py --check   # verify; never writes
    python scripts/generate_d3q15_operator.py --write   # verify, write CSV, read back

Exit status is nonzero if any check fails. With --write the CSV is written
only if every algebraic check passes first.

Conventions
-----------
* Population ordering: rest; axis velocities (+x,-x,+y,-y,+z,-z); the eight
  body diagonals (+-1,+-1,+-1) in lexicographic order. This follows the
  ordering pattern of operators/Lhat_D3Q19.csv (rest, axis pairs, remaining
  shell in lexicographic order).
* Curvature-coordinate ordering and CSV header are identical to
  operators/Lhat_D3Q19.csv and operators/Lhat_D3Q27.csv.
* Hermite <=2 column ordering matches load_lattice() in
  src/experiment_II_3D_shear.py.
* CSV entries are canonical SymPy rational strings; the file is written with
  Python's csv.writer default line terminator (CRLF), as in the existing files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import os
import sys
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_DIR = ROOT / "operators"
DEFAULT_CSV = OPERATOR_DIR / "Lhat_D3Q15.csv"
EXP2_MODULE = ROOT / "src" / "experiment_II_3D_shear.py"

# -----------------------------------------------------------------------------
# Fixed conventions
# -----------------------------------------------------------------------------

CS2 = sp.Rational(1, 3)
NORMALIZATION = sp.Integer(1) / (2 * CS2**3)   # 1/(2 c_s^6) = 27/2
AXES = "xyz"

Q_NAMES: Tuple[str, ...] = (
    "ux_xy", "ux_xz", "ux_yy", "ux_zz", "ux_yz",
    "uy_xy", "uy_yz", "uy_xx", "uy_zz", "uy_xz",
    "uz_xz", "uz_yz", "uz_xx", "uz_yy", "uz_xy",
)
HEADER: Tuple[str, ...] = ("cx", "cy", "cz", "w") + Q_NAMES
NQ = len(Q_NAMES)

# Dependent curvature components, eliminated through differentiated
# incompressibility  sum_alpha d_alpha d_gamma u_alpha = 0:
#   u_x,xx = -u_y,xy - u_z,xz
#   u_y,yy = -u_x,xy - u_z,yz
#   u_z,zz = -u_x,xz - u_y,yz
DEPENDENT_COMPONENTS: Dict[str, Dict[str, int]] = {
    "ux_xx": {"uy_xy": -1, "uz_xz": -1},
    "uy_yy": {"ux_xy": -1, "uz_yz": -1},
    "uz_zz": {"ux_xz": -1, "uy_yz": -1},
}

HERMITE2_LABELS: Tuple[str, ...] = (
    "1", "cx", "cy", "cz",
    "cx^2-cs2", "cy^2-cs2", "cz^2-cs2",
    "cx*cy", "cx*cz", "cy*cz",
)
HERMITE3_LABELS: Tuple[str, ...] = (
    "xxx", "yyy", "zzz", "xxy", "xxz", "xyy", "xzz", "yyz", "yzz", "xyz",
)
INDEPENDENT_HERMITE3: Tuple[str, ...] = ("xxy", "xxz", "xyy", "xyz")

# Reference results (manuscript Appendix B.1 / B; task specification).
REFERENCE_M15 = sp.Matrix([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0],
    [2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3],
])
EXPECTED_PIVOTS: Tuple[int, ...] = (0, 1, 2, 4)
REFERENCE_COORDINATE_MATRIX = sp.diag(2, 2, 1, 3)
EXPECTED_COORDINATE_DET = 12
EXPECTED_RANK_PHI = 10
EXPECTED_RANK_PK = 5
EXPECTED_RANK_LHAT = 4
EXPECTED_NULLITY = 11
LOADER_TOL = 1e-12


# -----------------------------------------------------------------------------
# Lattice
# -----------------------------------------------------------------------------

def d3q15_velocities() -> List[Tuple[int, int, int]]:
    """Rest, axis pairs (+x,-x,+y,-y,+z,-z), body diagonals lexicographically."""
    rest = [(0, 0, 0)]
    axis = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    corners = [tuple(v) for v in product((-1, 1), repeat=3)]
    return rest + axis + corners


def d3q15_weight(v: Sequence[int]) -> sp.Rational:
    norm2 = sum(x * x for x in v)
    table = {0: sp.Rational(2, 9), 1: sp.Rational(1, 9), 3: sp.Rational(1, 72)}
    if norm2 not in table:
        raise ValueError(f"Velocity {v} is not a D3Q15 velocity.")
    return table[norm2]


# -----------------------------------------------------------------------------
# Incompressibility embedding E_inc : R^15 -> R^{3x3x3}
# -----------------------------------------------------------------------------

def _component_key(name: str) -> Tuple[int, Tuple[int, int]]:
    """'uy_xz' -> (alpha=1, (beta, gamma)=(0, 2)) with beta <= gamma."""
    alpha = AXES.index(name[1])
    b, g = sorted((AXES.index(name[3]), AXES.index(name[4])))
    return alpha, (b, g)


def _check_component_partition() -> None:
    """The 18 symmetric components must split into 15 independent + 3 dependent."""
    all_keys = {(a, (b, g)) for a in range(3) for b in range(3) for g in range(b, 3)}
    indep = {_component_key(n) for n in Q_NAMES}
    dep = {_component_key(n) for n in DEPENDENT_COMPONENTS}
    if len(all_keys) != 18 or len(indep) != 15 or len(dep) != 3:
        raise RuntimeError("Curvature component bookkeeping is inconsistent.")
    if indep & dep or (indep | dep) != all_keys:
        raise RuntimeError("Independent/dependent components do not partition the 18 components.")


def curvature_component(alpha: int, beta: int, gamma: int) -> Tuple[sp.Integer, ...]:
    """
    Exact integer coefficients of J_{alpha beta gamma} = d_beta d_gamma u_alpha
    in terms of the 15 independent coordinates q.

    Derivative symmetry is encoded by keying on the sorted pair (beta, gamma).
    The three dependent diagonal components are encoded explicitly from the
    differentiated incompressibility constraint (DEPENDENT_COMPONENTS); they
    are never inferred numerically.
    """
    key = (alpha, tuple(sorted((beta, gamma))))
    independent = {_component_key(n): j for j, n in enumerate(Q_NAMES)}
    dependent = {_component_key(n): rel for n, rel in DEPENDENT_COMPONENTS.items()}
    coeff = [sp.Integer(0)] * NQ
    if key in independent:
        coeff[independent[key]] = sp.Integer(1)
    elif key in dependent:
        for name, val in dependent[key].items():
            coeff[Q_NAMES.index(name)] = sp.Integer(val)
    else:  # unreachable once _check_component_partition() has passed
        raise KeyError(f"Curvature component {key} not covered by the embedding.")
    return tuple(coeff)


def incompressibility_embedding() -> List[List[List[Tuple[sp.Integer, ...]]]]:
    """E[alpha][beta][gamma] -> tuple of 15 exact integer coefficients."""
    _check_component_partition()
    return [[[curvature_component(a, b, g) for g in range(3)] for b in range(3)] for a in range(3)]


def embedding_matrix(E) -> sp.Matrix:
    """27 x 15 matrix; row index 9*alpha + 3*beta + gamma."""
    return sp.Matrix([list(E[a][b][g]) for a in range(3) for b in range(3) for g in range(3)])


# -----------------------------------------------------------------------------
# Hermite polynomials
# -----------------------------------------------------------------------------

def hermite_le2_values(c: Sequence[int]) -> List[sp.Rational]:
    """Ten Hermite polynomials through order 2, in load_lattice() column order."""
    cx, cy, cz = (sp.Integer(x) for x in c)
    return [
        sp.Integer(1), cx, cy, cz,
        cx * cx - CS2, cy * cy - CS2, cz * cz - CS2,
        cx * cy, cx * cz, cy * cz,
    ]


def hermite3(c: Sequence[int], idx: Tuple[int, int, int]) -> sp.Rational:
    """H3_{abg}(c) = c_a c_b c_g - c_s^2 (c_a d_bg + c_b d_ag + c_g d_ab)."""
    a, b, g = idx
    cv = [sp.Integer(x) for x in c]
    d = lambda i, j: sp.Integer(1) if i == j else sp.Integer(0)  # noqa: E731
    return cv[a] * cv[b] * cv[g] - CS2 * (cv[a] * d(b, g) + cv[b] * d(a, g) + cv[g] * d(a, b))


def label_to_idx(label: str) -> Tuple[int, int, int]:
    return tuple(AXES.index(ch) for ch in label)  # type: ignore[return-value]


def hermite3_moment_matrix(velocities: Sequence[Sequence[int]],
                           labels: Sequence[str] = HERMITE3_LABELS) -> sp.Matrix:
    """Rows: Hermite-3 moment functionals m_abg = sum_i H3_abg(c_i) g_i."""
    return sp.Matrix([[hermite3(c, label_to_idx(lab)) for c in velocities] for lab in labels])


# -----------------------------------------------------------------------------
# Exact construction
# -----------------------------------------------------------------------------

def weighted_projector_exact(phi: sp.Matrix, winv: sp.Matrix) -> sp.Matrix:
    """Phi (Phi^T W^-1 Phi)^-1 Phi^T W^-1 with an exact inverse."""
    gram = phi.T * winv * phi
    if gram.det() == 0:
        raise RuntimeError("Hermite Gram matrix is singular.")
    return phi * gram.inv() * phi.T * winv


@dataclass
class OperatorBuild:
    name: str
    velocities: List[Tuple[int, int, int]]
    w: List[sp.Rational]
    W: sp.Matrix
    Winv: sp.Matrix
    Phi: sp.Matrix          # Q x 10 weighted Hermite columns through order 2
    P_le1: sp.Matrix
    P_le2: sp.Matrix
    PK: sp.Matrix
    P_eq: sp.Matrix         # linearized equilibrium projector (loader comparison only)
    E: list                 # E[a][b][g] -> 15 coefficients
    E27: sp.Matrix          # 27 x 15
    C3: sp.Matrix           # Q x 27 raw cubic contraction
    A: sp.Matrix            # Q x 15 = C3 E27
    Lhat: sp.Matrix         # Q x 15


def build_operator(velocities: Sequence[Sequence[int]],
                   weights: Sequence[sp.Rational],
                   name: str | None = None) -> OperatorBuild:
    """Exact normalized CE2 operator for an arbitrary 3-D velocity set."""
    vel = [tuple(int(x) for x in v) for v in velocities]
    w = [sp.Rational(x) for x in weights]
    Q = len(vel)
    if len(w) != Q:
        raise ValueError("Velocity/weight length mismatch.")

    W = sp.diag(*w)
    Winv = sp.diag(*[1 / x for x in w])

    hvals = [hermite_le2_values(c) for c in vel]
    Phi = sp.Matrix([[w[i] * hvals[i][a] for a in range(10)] for i in range(Q)])
    P_le1 = weighted_projector_exact(Phi[:, 0:4], Winv)
    P_le2 = weighted_projector_exact(Phi, Winv)
    PK = sp.eye(Q) - P_le2
    P_eq = sp.Matrix(Q, Q, lambda i, j: w[i] * (1 + sum(vel[i][k] * vel[j][k] for k in range(3)) / CS2))

    E = incompressibility_embedding()
    E27 = embedding_matrix(E)
    # Full contraction over all (alpha, beta, gamma); no manual symmetry factors.
    C3 = sp.Matrix([[sp.Integer(c[a] * c[b] * c[g])
                     for a in range(3) for b in range(3) for g in range(3)] for c in vel])
    A = C3 * E27
    Lhat = NORMALIZATION * PK * W * A

    return OperatorBuild(
        name=name or f"D3Q{Q}", velocities=vel, w=w, W=W, Winv=Winv, Phi=Phi,
        P_le1=P_le1, P_le2=P_le2, PK=PK, P_eq=P_eq, E=E, E27=E27, C3=C3, A=A, Lhat=Lhat,
    )


def build_d3q15() -> OperatorBuild:
    vel = d3q15_velocities()
    return build_operator(vel, [d3q15_weight(v) for v in vel], name="D3Q15")


# -----------------------------------------------------------------------------
# Exact linear-algebra helpers
# -----------------------------------------------------------------------------

def exact_rank(M: sp.Matrix) -> int:
    """Rank from exact rational row reduction."""
    return len(M.rref()[1])


def pivot_columns(M: sp.Matrix) -> Tuple[int, ...]:
    return tuple(int(p) for p in M.rref()[1])


def is_zero(M: sp.Matrix) -> bool:
    return all(x == 0 for x in M)


def all_rational(M: sp.Matrix) -> bool:
    return all(isinstance(x, sp.Rational) for x in M)


def is_psd_by_charpoly(M: sp.Matrix) -> bool:
    """
    Exact PSD test for a real symmetric matrix: all (real) eigenvalues are
    nonnegative iff the coefficients of det(lambda I - M) alternate in sign
    (zeros allowed).
    """
    lam = sp.Symbol("lam")
    coeffs = sp.Poly(M.charpoly(lam).as_expr(), lam).all_coeffs()
    return all(((-1) ** k) * a >= 0 for k, a in enumerate(coeffs))


# -----------------------------------------------------------------------------
# CSV serialization
# -----------------------------------------------------------------------------

def canonical_csv_text(b: OperatorBuild) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf)  # default lineterminator '\r\n', as in the existing files
    wr.writerow(HEADER)
    for i, v in enumerate(b.velocities):
        wr.writerow([str(v[0]), str(v[1]), str(v[2]), str(b.w[i])]
                    + [str(b.Lhat[i, j]) for j in range(NQ)])
    return buf.getvalue()


def to_rational(s: str) -> sp.Rational:
    fr = Fraction(s.strip())
    return sp.Rational(fr.numerator, fr.denominator)


def read_operator_csv(path: Path):
    """Return (header, velocities, weights, Lhat) with exact rationals."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    vel = [(int(r[0]), int(r[1]), int(r[2])) for r in body]
    w = [to_rational(r[3]) for r in body]
    lhat = sp.Matrix([[to_rational(x) for x in r[4:]] for r in body])
    return header, vel, w, lhat


# -----------------------------------------------------------------------------
# Check reporting
# -----------------------------------------------------------------------------

@dataclass
class CheckResult:
    label: str
    value: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, label: str, ok: bool, value: str | None = None, detail: str = "") -> None:
        self.results.append(CheckResult(label, value if value is not None else ("PASS" if ok else "FAIL"),
                                        bool(ok), detail))

    def run(self, label: str, fn: Callable[[], Tuple[bool, str | None, str]]) -> None:
        try:
            ok, value, detail = fn()
        except Exception as exc:  # report every failure explicitly
            ok, value, detail = False, "ERROR", f"{type(exc).__name__}: {exc}"
        self.add(label, ok, value, detail)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def first_failure(self) -> CheckResult | None:
        return next((r for r in self.results if not r.ok), None)

    def print(self, title: str = "D3Q15 CE2 operator verification") -> None:
        print(title)
        print("-" * len(title))
        for r in self.results:
            print(f"{r.label + ':':<40}{r.value}{'' if r.ok else '   <-- FAIL'}")
            if not r.ok and r.detail:
                for line in r.detail.splitlines():
                    print(f"    {line}")
        first = self.first_failure()
        print("All checks passed." if first is None else f"VERIFICATION FAILED (first failure: {first.label}).")


def matrix_mismatch(expected: sp.Matrix, actual: sp.Matrix, max_items: int = 20) -> str:
    if expected.shape != actual.shape:
        return f"shape expected {expected.shape}, got {actual.shape}"
    lines = [f"[{i},{j}] expected {expected[i, j]}, got {actual[i, j]}"
             for i in range(expected.rows) for j in range(expected.cols)
             if expected[i, j] != actual[i, j]]
    extra = f"\n... {len(lines) - max_items} more" if len(lines) > max_items else ""
    return "\n".join(lines[:max_items]) + extra


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_lattice(b: OperatorBuild, rep: Report) -> None:
    rep.add("Number of populations", len(b.velocities) == 15, str(len(b.velocities)))
    rep.add("Number of curvature coordinates", b.Lhat.cols == 15 and NQ == 15, str(NQ))

    rep.run("Unique velocities", lambda: (len(set(b.velocities)) == 15, None, "duplicate velocities"))

    def weight_sum():
        s = sum(b.w)
        return s == 1, str(s), f"sum = {s}"
    rep.run("Sum of lattice weights", weight_sum)

    def iso2():
        bad = []
        for a in range(3):
            for bb in range(3):
                s = sum(b.w[i] * c[a] * c[bb] for i, c in enumerate(b.velocities))
                exp = CS2 if a == bb else 0
                if s != exp:
                    bad.append(f"({a},{bb}): expected {exp}, got {s}")
        first = [sum(b.w[i] * c[a] for i, c in enumerate(b.velocities)) for a in range(3)]
        if any(x != 0 for x in first):
            bad.append(f"first moments nonzero: {first}")
        return not bad, None, "\n".join(bad)
    rep.run("Second-order isotropy", iso2)

    def iso4():
        bad = []
        d = lambda i, j: 1 if i == j else 0  # noqa: E731
        for a, bb, g, e in product(range(3), repeat=4):
            s = sum(b.w[i] * c[a] * c[bb] * c[g] * c[e] for i, c in enumerate(b.velocities))
            exp = CS2**2 * (d(a, bb) * d(g, e) + d(a, g) * d(bb, e) + d(a, e) * d(bb, g))
            if s != exp:
                bad.append(f"({a},{bb},{g},{e}): expected {exp}, got {s}")
        for a, bb, g in product(range(3), repeat=3):
            s = sum(b.w[i] * c[a] * c[bb] * c[g] for i, c in enumerate(b.velocities))
            if s != 0:
                bad.append(f"third moment ({a},{bb},{g}) = {s}")
        return not bad, None, "\n".join(bad)
    rep.run("Fourth-order isotropy", iso4)


def check_embedding(b: OperatorBuild, rep: Report) -> None:
    E = b.E

    def over_basis(test):
        bad = []
        for j in range(NQ):  # J = E e_j for each canonical basis vector
            J = [[[E[a][bb][g][j] for g in range(3)] for bb in range(3)] for a in range(3)]
            bad.extend(f"e_{j} ({Q_NAMES[j]}): {m}" for m in test(J))
        return not bad, None, "\n".join(bad)

    def incompressible(J):
        return [f"sum_a J[a,a,{g}] = {sum(J[a][a][g] for a in range(3))}"
                for g in range(3) if sum(J[a][a][g] for a in range(3)) != 0]

    def symmetric(J):
        return [f"J[{a},{bb},{g}] != J[{a},{g},{bb}]"
                for a in range(3) for bb in range(3) for g in range(3) if J[a][bb][g] != J[a][g][bb]]

    def integer_readback():
        bad = []
        if not all(isinstance(x, sp.Integer) for x in b.E27):
            bad.append("embedding has non-integer coefficients")
        for j, nm in enumerate(Q_NAMES):
            a, (bb, g) = _component_key(nm)
            if list(E[a][bb][g]) != [sp.Integer(1 if k == j else 0) for k in range(NQ)]:
                bad.append(f"independent position {nm} does not return q[{j}]")
        if exact_rank(b.E27) != NQ:
            bad.append("embedding is not injective")
        return not bad, None, "\n".join(bad)

    rep.run("Incompressibility embedding", lambda: over_basis(incompressible))
    rep.run("Derivative-index symmetry", lambda: over_basis(symmetric))
    rep.run("Embedding integer/readback", integer_readback)


def check_projectors(b: OperatorBuild, rep: Report) -> None:
    Q = len(b.velocities)

    def rank_phi():
        r = exact_rank(b.Phi)
        return r == EXPECTED_RANK_PHI, str(r), f"expected {EXPECTED_RANK_PHI}, got {r}"

    def rank_pk():
        r, tr = exact_rank(b.PK), b.PK.trace()
        return (r == EXPECTED_RANK_PK and tr == EXPECTED_RANK_PK, str(r),
                f"expected {EXPECTED_RANK_PK}, got rank {r}, trace {tr}")

    def idempotence():
        bad = []
        if b.P_le2 * b.P_le2 != b.P_le2:
            bad.append("P_le2^2 != P_le2")
        if b.PK * b.PK != b.PK:
            bad.append("P_K^2 != P_K")
        if not is_zero(b.PK * b.P_le2):
            bad.append("P_K P_le2 != 0")
        if not is_zero(b.P_le2 * b.PK):
            bad.append("P_le2 P_K != 0")
        if b.P_le2 * b.Phi != b.Phi:
            bad.append("P_le2 does not fix the Hermite columns")
        return not bad, None, "\n".join(bad)

    def self_adjoint():
        bad = []
        if b.PK.T * b.Winv != b.Winv * b.PK:
            bad.append("P_K^T W^-1 != W^-1 P_K")
        if b.P_le2.T * b.Winv != b.Winv * b.P_le2:
            bad.append("P_le2^T W^-1 != W^-1 P_le2")
        return not bad, None, "\n".join(bad)

    rep.run("Hermite <=2 subspace rank", rank_phi)
    rep.run("Kinetic projector rank", rank_pk)
    rep.run("Projector idempotence/orthogonality", idempotence)
    rep.run("Weighted self-adjointness", self_adjoint)
    rep.run("Projectors exactly rational",
            lambda: (all_rational(b.P_le2) and all_rational(b.PK) and b.PK.shape == (Q, Q), None, ""))


def check_operator(b: OperatorBuild, rep: Report) -> None:
    L = b.Lhat

    def rank():
        r = exact_rank(L)
        return r == EXPECTED_RANK_LHAT, str(r), f"expected {EXPECTED_RANK_LHAT}, got {r}"

    def nullity():
        ns = L.nullspace()
        ok = len(ns) == EXPECTED_NULLITY and all(is_zero(L * v) for v in ns)
        return ok, str(len(ns)), f"expected {EXPECTED_NULLITY}, got {len(ns)}"

    def kinetic():
        bad = []
        if b.PK * L != L:
            bad.append("P_K Lhat != Lhat")
        if not is_zero(b.P_le2 * L):
            bad.append("P_le2 Lhat != 0")
        return not bad, None, "\n".join(bad)

    def gram():
        G = L.T * b.Winv * L
        bad = []
        if G != G.T:
            bad.append("G not symmetric")
        if not all_rational(G):
            bad.append("G has non-rational entries")
        r = exact_rank(G)
        if r != EXPECTED_RANK_LHAT:
            bad.append(f"rank G = {r}, expected {EXPECTED_RANK_LHAT}")
        if not is_psd_by_charpoly(G):
            bad.append("G fails exact PSD test (charpoly sign alternation)")
        if G.is_positive_semidefinite is not True:
            bad.append("SymPy is_positive_semidefinite is not True")
        return not bad, f"PASS (rank {r})" if not bad else None, "\n".join(bad)

    rep.run("Normalization 1/(2 c_s^6)",
            lambda: (NORMALIZATION == sp.Rational(27, 2), str(NORMALIZATION), f"got {NORMALIZATION}"))
    rep.run("Normalized CE2 operator shape",
            lambda: (L.shape == (15, 15) and all_rational(L), str(L.shape), f"shape {L.shape} / rational entries"))
    rep.run("Normalized CE2 operator rank", rank)
    rep.run("Nullspace dimension", nullity)
    rep.run("Kinetic projection invariance", kinetic)
    rep.run("Gram L^T W^-1 L sym./PSD/rank", gram)


def check_moments(b: OperatorBuild, rep: Report) -> None:
    L = b.Lhat
    Cmom = hermite3_moment_matrix(b.velocities)
    Cind = hermite3_moment_matrix(b.velocities, INDEPENDENT_HERMITE3)

    def reference():
        M = Cmom * L
        return M == REFERENCE_M15, None, matrix_mismatch(REFERENCE_M15, M)

    def rank_m():
        r = exact_rank(Cmom * L)
        return r == EXPECTED_RANK_LHAT, str(r), f"expected {EXPECTED_RANK_LHAT}, got {r}"

    def pivots():
        p = pivot_columns(L)
        return p == EXPECTED_PIVOTS, str(list(p)), f"expected {list(EXPECTED_PIVOTS)}, got {list(p)}"

    def coord_matrix():
        K = Cind * L[:, list(EXPECTED_PIVOTS)]
        return (K == REFERENCE_COORDINATE_MATRIX, ", ".join(INDEPENDENT_HERMITE3),
                matrix_mismatch(REFERENCE_COORDINATE_MATRIX, K))

    def det():
        d = (Cind * L[:, list(EXPECTED_PIVOTS)]).det()
        return d == EXPECTED_COORDINATE_DET, str(d), f"expected {EXPECTED_COORDINATE_DET}, got {d}"

    def aliases():
        bad = []
        for c in b.velocities:
            for p, q in (("xxy", "yzz"), ("xxz", "yyz"), ("xyy", "xzz")):
                hp, hq = hermite3(c, label_to_idx(p)), hermite3(c, label_to_idx(q))
                if hp != hq:
                    bad.append(f"c={c}: H_{p}={hp} != H_{q}={hq}")
            for z in ("xxx", "yyy", "zzz"):
                hz = hermite3(c, label_to_idx(z))
                if hz != 0:
                    bad.append(f"c={c}: H_{z}={hz} != 0")
        if all(hermite3(c, (0, 1, 2)) == 0 for c in b.velocities):
            bad.append("H_xyz vanishes identically (unexpected for D3Q15)")
        return not bad, None, "\n".join(bad)

    def mixed():
        q = sp.Matrix(sp.symbols(" ".join(Q_NAMES)))
        T = q[Q_NAMES.index("ux_yz")] + q[Q_NAMES.index("uy_xz")] + q[Q_NAMES.index("uz_xy")]
        m_xyz = (hermite3_moment_matrix(b.velocities, ("xyz",)) * L * q)[0]
        diff = sp.expand(m_xyz - 3 * T)
        return diff == 0, None, f"m_xyz - 3T = {diff}"

    rep.run("Reference moment matrix M15", reference)
    rep.run("Moment matrix rank", rank_m)
    rep.run("Pivot columns (zero-based)", pivots)
    rep.run("Independent Hermite coordinates", coord_matrix)
    rep.run("Coordinate matrix determinant", det)
    rep.run("D3Q15 cubic aliases", aliases)
    rep.run("Fully mixed curvature channel", mixed)


def check_committed_operators(rep: Report) -> None:
    """
    Supplementary, read-only: the same exact construction applied to the
    committed D3Q19/D3Q27 velocity sets must reproduce the committed operator
    CSVs entry by entry. This ties the D3Q15 construction (normalization,
    embedding, column ordering) to the conventions of the existing files.
    """
    for q in (19, 27):
        path = OPERATOR_DIR / f"Lhat_D3Q{q}.csv"

        def run(path=path):
            if not path.exists():
                return False, "MISSING", str(path)
            _, vel, w, lhat = read_operator_csv(path)
            return build_operator(vel, w).Lhat == lhat, None, matrix_mismatch(lhat, build_operator(vel, w).Lhat)
        rep.run(f"Same construction reproduces D3Q{q}", run)


def check_csv(b: OperatorBuild, path: Path, rep: Report) -> None:
    def exact():
        if not path.exists():
            return False, "MISSING", f"{path} does not exist"
        bad = []
        if path.read_bytes().decode("utf-8") != canonical_csv_text(b):
            bad.append("file text differs from canonical serialization")
        header, vel, w, lhat = read_operator_csv(path)
        if tuple(header) != HEADER:
            bad.append(f"header mismatch: {header}")
        if vel != b.velocities:
            bad.append("velocity ordering/content mismatch")
        if w != b.w:
            bad.append("weight mismatch")
        if lhat != b.Lhat:
            bad.append("operator mismatch:\n" + matrix_mismatch(b.Lhat, lhat))
        return not bad, None, "\n".join(bad)
    rep.run("CSV exact reproducibility", exact)


def load_experiment_ii():
    """Import src/experiment_II_3D_shear.py the same way the production scripts do."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    name = "exp2_d3q15_check"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXP2_MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def loader_compatibility(b: OperatorBuild, path: Path) -> Tuple[bool, str]:
    """Load the CSV with the unmodified Experiment II loader and compare with exact objects."""
    import numpy as np

    # The existing loader opens the CSV without closing it (ResourceWarning);
    # silence only that warning here, the loader itself is left unchanged.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        lat = load_experiment_ii().load_lattice(path)
    bad = []
    if lat.name != "D3Q15":
        bad.append(f"lat.name = {lat.name!r}")
    for attr, shape in (("c", (15, 3)), ("w", (15,)), ("lhat", (15, 15))):
        if getattr(lat, attr).shape != shape:
            bad.append(f"lat.{attr}.shape = {getattr(lat, attr).shape}, expected {shape}")
    if list(lat.q_names) != list(Q_NAMES):
        bad.append("q_names mismatch")
    if not np.array_equal(lat.c, np.array(b.velocities, dtype=int)):
        bad.append("velocity array mismatch")

    to_np = lambda M: np.array(M.tolist(), dtype=float)  # noqa: E731
    # Both sides round each exact rational once to the nearest double: bit-identical.
    if not np.array_equal(lat.w, np.array([float(x) for x in b.w])):
        bad.append("weights not bit-identical")
    if not np.array_equal(lat.lhat, to_np(b.Lhat)):
        bad.append("lhat not bit-identical")
    for attr, M in (("Winv", b.Winv), ("P_eq", b.P_eq), ("P_le1", b.P_le1), ("P_le2", b.P_le2),
                    ("P2", b.P_le2 - b.P_le1), ("PK", b.PK), ("N", sp.eye(15) - b.P_eq)):
        res = float(np.max(np.abs(getattr(lat, attr) - to_np(M))))
        if not res <= LOADER_TOL:
            bad.append(f"{attr}: max |float - exact| = {res:.3e} > {LOADER_TOL:g}")
    return not bad, "\n".join(bad)


def check_loader(b: OperatorBuild, path: Path, rep: Report) -> None:
    def run():
        if not path.exists():
            return False, "MISSING", f"{path} does not exist"
        ok, detail = loader_compatibility(b, path)
        return ok, None, detail
    rep.run("Experiment II loader compatibility", run)


def verify_algebra(b: OperatorBuild) -> Report:
    rep = Report()
    check_lattice(b, rep)
    check_embedding(b, rep)
    check_projectors(b, rep)
    check_operator(b, rep)
    check_moments(b, rep)
    check_committed_operators(rep)
    return rep


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify algebra and existing CSV; never writes")
    mode.add_argument("--write", action="store_true", help="verify algebra, write CSV, read back and verify")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="operator CSV path (default: operators/Lhat_D3Q15.csv)")
    args = p.parse_args(argv)

    b = build_d3q15()
    rep = verify_algebra(b)

    if args.write:
        if not rep.ok:
            rep.print()
            print(f"\nAlgebraic verification failed; {args.csv} NOT written.")
            return 1
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            fh.write(canonical_csv_text(b))
        print(f"Wrote {args.csv}\n")

    check_csv(b, args.csv, rep)
    check_loader(b, args.csv, rep)
    rep.print()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
