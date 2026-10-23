# CE2 population operators

This directory contains the canonical machine-readable CE2 population operators (used by Experiments II, III and IV) and, in `reconstruction/`, the exact curvature-reconstruction operators derived from them (see the last section).

- `Lhat_D3Q19.csv`: D3Q19 normalized CE2 observability operator.
- `Lhat_D3Q27.csv`: D3Q27 normalized CE2 observability operator.
- `Lhat_D3Q15.csv`: D3Q15 normalized CE2 observability operator (input for Experiment IV; see below).
- `reconstruction/`: observable functionals $D_Q$, Hermite reconstruction operators $R^H_Q$ and sensitivity factors for D3Q15, D3Q19 and D3Q27.

All `Lhat_D3Q*.csv` files include the lattice velocity components, quadrature weights, and the 15 curvature-coordinate columns in the ordering expected by the numerical implementations.

## File format (all operators)

Header (one line):

```text
cx,cy,cz,w,ux_xy,ux_xz,ux_yy,ux_zz,ux_yz,uy_xy,uy_yz,uy_xx,uy_zz,uy_xz,uz_xz,uz_yz,uz_xx,uz_yy,uz_xy
```

- Columns 1–3: integer lattice velocity `c_i`; column 4: weight `w_i`.
- Columns 5–19: row `i` of the normalized operator `Lhat_Q` in the curvature coordinates

  ```text
  q = (u_x,xy, u_x,xz, u_x,yy, u_x,zz, u_x,yz,
       u_y,xy, u_y,yz, u_y,xx, u_y,zz, u_y,xz,
       u_z,xz, u_z,yz, u_z,xx, u_z,yy, u_z,xy)
  ```

  The remaining components are eliminated through differentiated incompressibility: `u_x,xx = -u_y,xy - u_z,xz`, `u_y,yy = -u_x,xy - u_z,yz`, `u_z,zz = -u_x,xz - u_y,yz`.
- All entries are exact integers or fractions (e.g. `0`, `1/9`, `-3/8`); never decimals.
- The row order defines the population ordering; velocities, weights and operator rows must never be reordered independently.
- The files are read by `load_lattice()` in `src/experiment_II_3D_shear.py` and `src/experiment_III_taylor_green.py`.

## D3Q15

### Velocity set and weights (row order)

| Rows | Velocities | Weight |
|---|---|---|
| 1 | `(0,0,0)` | `2/9` |
| 2–7 | `(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)` | `1/9` |
| 8–15 | `(±1,±1,±1)` in lexicographic order: `(-1,-1,-1), (-1,-1,1), (-1,1,-1), (-1,1,1), (1,-1,-1), (1,-1,1), (1,1,-1), (1,1,1)` | `1/72` |

This follows the ordering pattern of `Lhat_D3Q19.csv`: rest velocity, axis pairs `+x,-x,+y,-y,+z,-z`, then the remaining shell in lexicographic order. The lattice sound speed is `c_s^2 = 1/3`.

### Definition

```text
Lhat_15 = 1/(2 c_s^6) * P_K,15 W_15 C3_15 E_inc = (27/2) * P_K,15 W_15 C3_15 E_inc
```

- `W_15 = diag(w_i)`; weighted inner product `<f,g>_w = f^T W_15^{-1} g`.
- `P_K,15 = I - P_<=2,15`, where `P_<=2,15` is the `W^{-1}`-orthogonal projector onto the span of the weighted Hermite columns `w_i H(c_i)`, with `H` in {`1`, `c_x`, `c_y`, `c_z`, `c_x^2-c_s^2`, `c_y^2-c_s^2`, `c_z^2-c_s^2`, `c_x c_y`, `c_x c_z`, `c_y c_z`} (the column order of `load_lattice()`).
- `E_inc` (27 x 15, exact integers) maps `q` to the full curvature tensor `J_abg = d_b d_g u_a`, which is symmetric in `(b,g)` and satisfies `sum_a J_aag = 0`.
- `C3_15` (15 x 27) is the full cubic contraction `[C3 J]_i = sum_{a,b,g} c_ia c_ib c_ig J_abg`. All index permutations are included; no symmetry factors are added by hand.
- The dimensional prefactor `C0 = 2 rho0 tau (tau - 1/2) c_s^4` is **not** included. The tabulated operator is independent of `tau` and `rho0`.

### Structure

| Quantity | Value |
|---|---|
| Matrix shape | 15 x 15 (populations x curvature coordinates) |
| Rank of `Lhat_15` | 4 |
| Kernel dimension | 11 |
| Rank of the Hermite <=2 space | 10 |
| Rank of `P_K,15` (kinetic population subspace) | 5 |
| Pivot columns (zero-based) | `[0, 1, 2, 4]` (`u_x,xy`, `u_x,xz`, `u_x,yy`, `u_x,yz`) |

The five-dimensional kinetic population subspace (the range of `P_K,15`) and the four-dimensional CE2 curvature image (the range of `Lhat_15`) are different objects. The curvature image is a four-dimensional subspace of the kinetic space.

The third-order Hermite moments `m_abg = sum_i H3_abg(c_i) g_i` of `g = Lhat_15 q` are

```text
m_xxx = m_yyy = m_zzz = 0
m_xxy = m_yzz = 2 u_x,xy + u_y,xx + u_y,zz + 2 u_z,yz
m_xxz = m_yyz = 2 u_x,xz + 2 u_y,yz + u_z,xx + u_z,yy
m_xyy = m_xzz = u_x,yy + u_x,zz + 2 u_y,xy + 2 u_z,xz
m_xyz = 3 (u_x,yz + u_y,xz + u_z,xy)
```

So D3Q15 retains three aggregate planar curvature combinations (`m_xxy`, `m_xxz`, `m_xyy`) and one fully mixed curvature combination, `T = u_x,yz + u_y,xz + u_z,xy`. The fully mixed combination is detected through the body-diagonal channel `H3_xyz = c_x c_y c_z`, which vanishes identically on D3Q19. The duplicate and zero rows come from the D3Q15 aliases `H3_xxy = H3_yzz`, `H3_xxz = H3_yyz`, `H3_xyy = H3_xzz` and `H3_xxx = H3_yyy = H3_zzz = 0`, which hold on every D3Q15 velocity. The four moments `(m_xxy, m_xxz, m_xyy, m_xyz)` are faithful coordinates on the curvature image: applied to the pivot columns of `Lhat_15`, they give `diag(2, 2, 1, 3)`, whose determinant is 12.

### Regeneration and verification

`Lhat_D3Q15.csv` is generated programmatically with exact rational arithmetic (SymPy) by `scripts/generate_d3q15_operator.py`. From the repository root:

```bash
python scripts/generate_d3q15_operator.py --check   # rebuild in memory, run all exact checks, compare with the CSV; never writes
python scripts/generate_d3q15_operator.py --write   # rebuild, verify, write the CSV, read it back and compare
python -m unittest discover -s tests                # automated test suite (tests/test_d3q15_operator.py)
```

`--write` writes the file only if every algebraic check passes, and the output is byte-for-byte deterministic. As a supplementary consistency check, the generator applies the same exact construction to the velocity sets of `Lhat_D3Q19.csv` and `Lhat_D3Q27.csv` and confirms that it reproduces both committed operators exactly. Those files are only read, never modified.

## Curvature-reconstruction operators (`reconstruction/`)

Exact operators reconstructing the observable curvature functionals from normalized kinetic populations, $\mathbf d_Q=R^H_Q\widehat{\mathbf g}_Q$ with $\widehat{\mathbf g}_Q=\widehat L_Q\mathbf q$. Derivation and verified results: `docs/experiment_IV_reconstruction_operators.md`.

| File | Format |
|---|---|
| `observable_functionals.csv` | header `functional,ux_xy,...,uz_xy`; rows `A_x, A_y, A_z, B_x, B_y, B_z, T, S_1, S_2, S_3`; integer coefficients on the canonical curvature coordinates |
| `D_D3Q15.csv`, `D_D3Q19.csv`, `D_D3Q27.csv` | $D_Q$; same header; rows in the output order of the lattice: D3Q15 `(S_1,S_2,S_3,T)`, D3Q19 `(A_x,B_x,B_y,A_z,A_y,B_z)`, D3Q27 `(A_x,B_x,B_y,A_z,A_y,B_z,T)` |
| `RH_D3Q15.csv`, `RH_D3Q19.csv`, `RH_D3Q27.csv` | header `cx,cy,cz,w,<functionals>`; one row per population, in exactly the row order of `Lhat_D3Q*.csv`; column $j$ is row $j$ of $R^H_Q$, i.e. the file stores $(R^H_Q)^T$ |
| `sensitivity_factors.csv` | `lattice,functional,chi2` with $\chi^2_{Q,j}=\mathbf d_j^TG_Q^+\mathbf d_j$, $G_Q=\widehat L_Q^TW_Q^{-1}\widehat L_Q$ |

All entries are exact integers or fractions. $R^H_Q=K_Q\,\mathcal C_Q$ combines third-order Hermite moments $m_{\alpha\beta\gamma}=\sum_iH^{(3)}_{\alpha\beta\gamma}(\mathbf c_i)\widehat g_i$ (for example $T=\tfrac13m_{xyz}$ on D3Q15, $A_x=\tfrac43m_{xxy}+\tfrac23m_{yzz}$ on D3Q19, $A_x=m_{xxy}$ on D3Q27). It equals the weighted minimum-norm reconstruction $R^\star_Q=D_QG_Q^+\widehat L_Q^TW_Q^{-1}$ exactly, so each row has $\mathbf b^TW\mathbf b=\chi^2$. The defining identity

```text
R^H_Q Lhat_Q = D_Q,   Q = 15, 19, 27
```

is verified from the files alone (Python `fractions`, no SymPy) by:

```bash
python scripts/generate_reconstruction_operators.py --check   # rebuild exactly, compare byte-for-byte, verify identities; never writes
python scripts/generate_reconstruction_operators.py --write   # regenerate (deterministic), then verify
python scripts/check_repository.py
```

The numerical interface `src/curvature_reconstruction.py` builds the same $R^H_Q$ in memory; `tests/test_reconstruction_operator_files.py` checks that it agrees with these files.
