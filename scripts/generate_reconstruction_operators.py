#!/usr/bin/env python3
"""
Archive and verify the exact lattice-specific curvature-reconstruction operators.

The operators are built in exact rational arithmetic by the unmodified module
src/curvature_reconstruction_exact.py from the canonical normalized CE2
observability operators operators/Lhat_D3Q{15,19,27}.csv. This script only
serializes them, and verifies the files independently of SymPy:

    operators/reconstruction/observable_functionals.csv   definitions of A_*, B_*, T, S_1..S_3 on q
    operators/reconstruction/D_D3Q{15,19,27}.csv          D_Q   (n_Q x 15): d_Q = D_Q q
    operators/reconstruction/RH_D3Q{15,19,27}.csv         (R^H_Q)^T (Q x n_Q), one row per population
    operators/reconstruction/sensitivity_factors.csv      chi^2_{Q,j} = d_j^T G_Q^+ d_j

Identities verified from the files alone with Python Fractions:

    R^H_Q Lhat_Q = D_Q                         (Q = 15, 19, 27)
    D_Q rows = observable_functionals.csv rows (names in the R^H column header)
    cx, cy, cz, w columns of RH_D3Q*.csv identical to Lhat_D3Q*.csv (population ordering)
    R^H_Q W_Q (R^H_Q)^T has diagonal chi^2_Q  (Hermite reconstruction is minimum-norm)

Usage (from the repository root)
--------------------------------
    python scripts/generate_reconstruction_operators.py --check   # rebuild in memory, compare, verify files; never writes
    python scripts/generate_reconstruction_operators.py --write   # rebuild, verify, write, read back and verify

Exit status is nonzero on any failure. Output is byte-for-byte deterministic
(exact rational strings, csv.writer default CRLF terminator as in operators/*.csv).
The canonical Lhat_D3Q*.csv files are only read.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_DIR = ROOT / "operators"
RECON_DIR = OPERATOR_DIR / "reconstruction"
LATTICES = ("D3Q15", "D3Q19", "D3Q27")
FUNCTIONAL_ORDER = ("A_x", "A_y", "A_z", "B_x", "B_y", "B_z", "T", "S_1", "S_2", "S_3")


def _load_exact():
    name = "curvature_reconstruction_exact"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# Serialization (exact module -> CSV text)
# -----------------------------------------------------------------------------

def _csv_text(rows: Sequence[Sequence]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in rows:
        w.writerow([str(x) for x in r])
    return buf.getvalue()


def build_files() -> Dict[str, str]:
    """Return {relative file name: exact CSV text}; raises if any exact identity fails."""
    ex = _load_exact()
    qn = list(ex.Q_NAMES)
    files: Dict[str, str] = {}
    files["observable_functionals.csv"] = _csv_text(
        [["functional"] + qn] + [[n] + list(ex.named_functional(n)) for n in FUNCTIONAL_ORDER])
    chi_rows = [["lattice", "functional", "chi2"]]
    for lat in LATTICES:
        res = ex.verify_lattice(lat)
        if not res.ok:
            raise RuntimeError(f"{lat}: exact verification failed: {res.details}")
        op = ex.load_operator(lat)
        outs = list(ex.OUTPUT_ORDER[lat])
        files[f"D_{lat}.csv"] = _csv_text(
            [["functional"] + qn] + [[outs[j]] + list(res.D[j, :]) for j in range(len(outs))])
        rows = [["cx", "cy", "cz", "w"] + outs]
        for i, (c, w) in enumerate(zip(op.velocities, op.w)):
            rows.append(list(c) + [w] + [res.RH[j, i] for j in range(len(outs))])
        files[f"RH_{lat}.csv"] = _csv_text(rows)
        chi_rows += [[lat, n, x] for n, x in zip(outs, res.chi2)]
    files["sensitivity_factors.csv"] = _csv_text(chi_rows)
    return files


# -----------------------------------------------------------------------------
# Independent file verification (Fractions only, no SymPy)
# -----------------------------------------------------------------------------

def read_text_exact(path: Path) -> str:
    """File text with line terminators preserved (Path.read_text(newline=) needs Python >= 3.13)."""
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def _read(path: Path) -> List[List[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.reader(fh) if r]


def _frac(s: str) -> Fraction:
    if "." in s or "e" in s.lower():
        raise ValueError(f"non-exact entry {s!r}")
    return Fraction(s.strip())


def verify_files(recon_dir: Path = RECON_DIR, operator_dir: Path = OPERATOR_DIR) -> List[Tuple[str, bool, str]]:
    """List of (check, passed, detail) computed from the CSV files only."""
    out: List[Tuple[str, bool, str]] = []
    fun = _read(recon_dir / "observable_functionals.csv")
    qn = fun[0][1:]
    fdef = {r[0]: [_frac(x) for x in r[1:]] for r in fun[1:]}
    out.append(("observable_functionals.csv: 15 curvature columns, 10 functionals",
                len(qn) == 15 and tuple(fdef) == FUNCTIONAL_ORDER, ""))
    chi = {(r[0], r[1]): _frac(r[2]) for r in _read(recon_dir / "sensitivity_factors.csv")[1:]}
    for lat in LATTICES:
        L = _read(operator_dir / f"Lhat_{lat}.csv")
        RH = _read(recon_dir / f"RH_{lat}.csv")
        D = _read(recon_dir / f"D_{lat}.csv")
        Q = len(L) - 1
        out.append((f"{lat}: Lhat header = D header curvature ordering", L[0][4:] == qn == D[0][1:], ""))
        names = RH[0][4:]
        out.append((f"{lat}: R^H columns = D rows = {','.join(names)}", names == [r[0] for r in D[1:]], ""))
        same_cw = len(RH) == len(L) and all(a[:4] == b[:4] for a, b in zip(RH[1:], L[1:]))
        out.append((f"{lat}: population ordering (cx,cy,cz,w) identical to Lhat_{lat}.csv", same_cw, f"Q = {Q}"))
        Lm = [[_frac(x) for x in r[4:]] for r in L[1:]]
        w = [_frac(r[3]) for r in L[1:]]
        R = [[_frac(x) for x in r[4:]] for r in RH[1:]]            # Q x n  (= R^H transpose)
        Dm = [[_frac(x) for x in r[1:]] for r in D[1:]]            # n x 15
        n = len(names)
        prod = [[sum(R[i][j] * Lm[i][k] for i in range(Q)) for k in range(15)] for j in range(n)]
        out.append((f"{lat}: R^H_Q Lhat_Q = D_Q exactly ({n} x 15)", prod == Dm, ""))
        out.append((f"{lat}: D_Q rows match observable_functionals.csv",
                    all(Dm[j] == fdef[nm] for j, nm in enumerate(names)), ""))
        norms = [sum(R[i][j] ** 2 * w[i] for i in range(Q)) for j in range(n)]
        out.append((f"{lat}: Hermite row norms b^T W b = chi^2 (minimum-norm)",
                    norms == [chi[(lat, nm)] for nm in names], ", ".join(f"{nm}={c}" for nm, c in zip(names, norms))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="rebuild in memory, compare with files, verify (default)")
    mode.add_argument("--write", action="store_true", help="rebuild, verify, write files, read back and verify")
    args = ap.parse_args(argv)

    files = build_files()
    ok = True
    if args.write:
        RECON_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            with (RECON_DIR / name).open("w", newline="", encoding="utf-8") as fh:
                fh.write(text)
        print(f"Wrote {len(files)} files to {RECON_DIR.relative_to(ROOT)}/")
    for name, text in files.items():
        p = RECON_DIR / name
        same = p.exists() and read_text_exact(p) == text
        ok &= same
        print(f"[{'PASS' if same else 'FAIL'}] {p.relative_to(ROOT)} identical to exact rebuild")
    if ok:
        for check, passed, detail in verify_files():
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {check}" + (f"  ({detail})" if detail else ""))
    print("\nAll reconstruction-operator checks passed." if ok else "\nRECONSTRUCTION-OPERATOR CHECK FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
