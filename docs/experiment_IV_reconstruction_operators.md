# Experiment IV — Step 2A: Exact Curvature-Reconstruction Operators

## 1. Scope

This step builds the exact linear operators that reconstruct observable velocity-curvature functionals from normalized CE2 kinetic populations

$$
\widehat{\mathbf g}_Q=\widehat L_Q\,\mathbf q ,\qquad \widehat L_Q\in\mathbb Q^{Q\times15},
$$

for D3Q15, D3Q19 and D3Q27. Everything is computed in exact rational arithmetic (SymPy) from the canonical, unmodified files `operators/Lhat_D3Q{15,19,27}.csv`. No LBM simulation and no floating point are involved. Reconstruction from time-evolved populations belongs to later steps.

- Implementation: `src/curvature_reconstruction_exact.py`
- Tests: `tests/test_curvature_reconstruction_exact.py`
- Report: `python src/curvature_reconstruction_exact.py`

## 2. Curvature coordinates and functionals

The canonical coordinate ordering (CSV columns 5–19) is

$$
\mathbf q=(u_{x,xy},u_{x,xz},u_{x,yy},u_{x,zz},u_{x,yz},u_{y,xy},u_{y,yz},u_{y,xx},u_{y,zz},u_{y,xz},u_{z,xz},u_{z,yz},u_{z,xx},u_{z,yy},u_{z,xy})^T .
$$

The functionals are

$$
\begin{aligned}
A_x&=2u_{x,xy}+u_{y,xx}, & B_x&=2u_{x,xz}+u_{z,xx},\\
A_y&=2u_{y,yz}+u_{z,yy}, & B_y&=2u_{y,xy}+u_{x,yy},\\
A_z&=2u_{z,xz}+u_{x,zz}, & B_z&=2u_{z,yz}+u_{y,zz},\\
T&=u_{x,yz}+u_{y,xz}+u_{z,xy},
\end{aligned}
\qquad
S_1=A_x+B_z,\; S_2=B_x+A_y,\; S_3=B_y+A_z .
$$

## 3. Matrices, ordering and dimensions

| Lattice | Output vector $\mathbf d_Q$ (row order) | $D_Q$ | $R^H_Q$, $R^\star_Q$ | $G_Q$, $G_Q^+$ |
|---|---|---|---|---|
| D3Q15 | $(S_1,S_2,S_3,T)$ | 4 × 15 | 4 × 15 | 15 × 15 |
| D3Q19 | $(A_x,B_x,B_y,A_z,A_y,B_z)$ | 6 × 15 | 6 × 19 | 15 × 15 |
| D3Q27 | $(A_x,B_x,B_y,A_z,A_y,B_z,T)$ | 7 × 15 | 7 × 27 | 15 × 15 |

- Columns of $D_Q$ follow $\mathbf q$.
- Columns of $R^H_Q$ and $R^\star_Q$ follow the population (row) order of the canonical CSV of that lattice.
- A reconstruction is applied as $\mathbf d_Q=R_Q\,\widehat{\mathbf g}_Q$.

### 3.1 Hermite reconstruction $R^H_Q$

$R^H_Q=K_Q\,\mathcal C_Q$. Here $\mathcal C_Q\in\mathbb Q^{10\times Q}$ extracts the moments $m_{\alpha\beta\gamma}=\sum_i H^{(3)}_{\alpha\beta\gamma}(\mathbf c_i)\,\widehat g_i$ with $c_s^2=1/3$, in the order `xxx, yyy, zzz, xxy, xxz, xyy, xzz, yyz, yzz, xyz`. $K_Q$ holds the coefficients below.

| D3Q15 | D3Q19 | D3Q27 |
|---|---|---|
| $S_1=m_{xxy}$ | $A_x=\tfrac43m_{xxy}+\tfrac23m_{yzz}$ | $A_x=m_{xxy}$ |
| $S_2=m_{xxz}$ | $B_x=\tfrac43m_{xxz}+\tfrac23m_{yyz}$ | $B_x=m_{xxz}$ |
| $S_3=m_{xyy}$ | $B_y=\tfrac43m_{xyy}+\tfrac23m_{xzz}$ | $B_y=m_{xyy}$ |
| $T=\tfrac13m_{xyz}$ | $A_z=\tfrac23m_{xyy}+\tfrac43m_{xzz}$ | $A_z=m_{xzz}$ |
| | $A_y=\tfrac23m_{xxz}+\tfrac43m_{yyz}$ | $A_y=m_{yyz}$ |
| | $B_z=\tfrac23m_{xxy}+\tfrac43m_{yzz}$ | $B_z=m_{yzz}$ |
| | | $T=m_{xyz}$ |

The D3Q19 coefficients come from inverting the three mixing systems

$$
\begin{pmatrix}m_{xxy}\\m_{yzz}\end{pmatrix}=M\begin{pmatrix}A_x\\B_z\end{pmatrix},\quad
\begin{pmatrix}m_{xxz}\\m_{yyz}\end{pmatrix}=M\begin{pmatrix}B_x\\A_y\end{pmatrix},\quad
\begin{pmatrix}m_{xyy}\\m_{xzz}\end{pmatrix}=M\begin{pmatrix}B_y\\A_z\end{pmatrix},\qquad
M=\begin{pmatrix}1&-\tfrac12\\-\tfrac12&1\end{pmatrix},\;
M^{-1}=\tfrac43\begin{pmatrix}1&\tfrac12\\\tfrac12&1\end{pmatrix}.
$$

All the stated moment identities were checked directly against the canonical operators, and no discrepancy was found.

### 3.2 Weighted minimum-norm reconstruction $R^\star_Q$

$$
G_Q=\widehat L_Q^TW_Q^{-1}\widehat L_Q,\qquad
R^\star_Q=D_Q\,G_Q^+\,\widehat L_Q^TW_Q^{-1},\qquad W_Q=\operatorname{diag}(w_i).
$$

Row $j$ of $R^\star_Q$ is $b_j=W_Q^{-1}\widehat L_QG_Q^+d_j$. It is the solution of $\widehat L_Q^Tb_j=d_j$ that minimizes $b_j^TW_Qb_j$.

$G_Q^+$ is computed exactly from the full-rank factorization $G=FC$ (with $F$ the pivot columns of $G$ and $C$ the nonzero rows of its reduced row echelon form):

$$
G^+=C^T(CC^T)^{-1}(F^TF)^{-1}F^T .
$$

All four Moore–Penrose conditions are checked exactly. The result is also cross-checked against SymPy's own `pinv`.

### 3.3 Sensitivity factors

$$
\chi^2_{Q,j}=d_j^TG_Q^+d_j=b_j^TW_Qb_j .
$$

## 4. Verified results (exact)

| Lattice | rank $D_Q$ | rank $\widehat L_Q$ | row$(D_Q)$ = row$(\widehat L_Q)$ | $R^H_Q\widehat L_Q=D_Q$ | $R^\star_Q\widehat L_Q=D_Q$ | $W$-orthogonality | $\chi^2$ identity |
|---|---|---|---|---|---|---|---|
| D3Q15 | 4 | 4 | PASS | PASS | PASS | PASS | PASS |
| D3Q19 | 6 | 6 | PASS | PASS | PASS | PASS | PASS |
| D3Q27 | 7 | 7 | PASS | PASS | PASS | PASS | PASS |

Cross-lattice row-space identities (exact):

- $\operatorname{row}(D_{15})\cap\operatorname{row}(D_{19})=\operatorname{span}\{S_1,S_2,S_3\}$: dimension $4+6-7=3$, and $S_1,S_2,S_3$ lie in both row spaces.
- $\operatorname{row}(D_{15})+\operatorname{row}(D_{19})=\operatorname{row}(D_{27})$.

### 4.1 Squared sensitivity factors

| Lattice | Functional | $\chi^2$ |
|---|---|---|
| D3Q15 | $S_1$, $S_2$, $S_3$ | $2/27$ |
| D3Q15 | $T$ | $1/81$ |
| D3Q19 | $A_x$, $B_x$, $B_y$, $A_z$, $A_y$, $B_z$ | $8/81$ |
| D3Q27 | $A_x$, $B_x$, $B_y$, $A_z$, $A_y$, $B_z$ | $2/27$ |
| D3Q27 | $T$ | $1/27$ |

### 4.2 Observation: the Hermite reconstruction is already minimum-norm

On all three lattices $R^H_Q=R^\star_Q$ exactly. Each Hermite row has $b^{H\,T}Wb^H=\chi^2$, and since both rows satisfy $\widehat L_Q^Tb=d$, the Pythagorean identity $\|b^H\|_W^2=\|b^\star\|_W^2+\|b^H-b^\star\|_W^2$ forces $b^H=b^\star$.

The tests assert the identity $R^H\widehat L=R^\star\widehat L=D$ and the Pythagorean inequality. They do not require $R^H=R^\star$, which is reported as an observation (`LatticeResult.hermite_equals_min_norm`).

### 4.3 Observability and nullspace witnesses

A functional $d$ is observable on lattice $Q$ iff $d\in\operatorname{row}(\widehat L_Q)$, tested by exact rank. When it is not observable, `nullspace_witness` returns a primitive integer vector $z$ with $\widehat L_Qz=0$ and $d^Tz\neq0$. Such a $z$ is a curvature state that produces no CE2 kinetic signal but changes $d$.

| Functional | D3Q15 | D3Q19 | D3Q27 |
|---|---|---|---|
| $T$ | observable | **unobservable**: $z=u_{x,yz}$, $T(z)=1$ | observable |
| $A_x$ | **unobservable**: $z=u_{x,xy}-2u_{y,zz}$, $A_x(z)=2$ | observable | observable |
| $u_{x,xy}$ | **unobservable**: $z=u_{x,xy}-2u_{y,xx}$, $u_{x,xy}(z)=1$ | **unobservable**: same $z$ | **unobservable**: same $z$ |

For example, the D3Q15 witness for $A_x$ has $A_x=2$ and $B_z=-2$, so $S_1=0$. The witness for $u_{x,xy}$ has $u_{x,xy}=1$ and $A_x=0$.

## 5. Module API (`src/curvature_reconstruction_exact.py`)

| Function | Purpose |
|---|---|
| `load_operator(lattice)` | Read a canonical CSV exactly (header, shape and weight-sum validated; read-only) |
| `named_functional(name)` / `functional_row(coeffs)` | 1 × 15 row for `A_x`…`T`, `S_1`…`S_3`, or any coordinate such as `ux_xy` |
| `observable_matrix(lattice)` | $D_Q$ |
| `hermite3_matrix(velocities, labels)` | Moment extraction $\mathcal C_Q$ |
| `hermite_coefficient_matrix(lattice)`, `hermite_reconstruction(op)` | $K_Q$, $R^H_Q$ |
| `gram_matrix(op)`, `exact_pinv(M)`, `penrose_conditions(M, P)` | $G_Q$, exact $G_Q^+$ and its checks |
| `minimum_norm_reconstruction(op, D=None)` | $(R^\star_Q, G_Q^+)$ |
| `squared_sensitivities(D, G_plus)`, `weighted_row_norms_sq(R, W)` | $\chi^2$, $b^TWb$ |
| `is_observable(Lhat, d)`, `nullspace_witness(Lhat, d)` | Exact observability test and witness |
| `verify_lattice`, `verify_cross_lattice`, `verify_observability`, `main` | Full verification and report |

## 6. Reproduction

```bash
python src/curvature_reconstruction_exact.py      # report; nonzero exit on failure
python -m unittest discover -s tests              # includes the Step 1 D3Q15 tests
python scripts/check_repository.py
```

The operators are archived as exact CSV files in `operators/reconstruction/` (`D_D3Q*.csv`, `RH_D3Q*.csv`, `observable_functionals.csv`, `sensitivity_factors.csv`; format in `operators/README.md`). `python scripts/generate_reconstruction_operators.py --check` rebuilds them from this module, compares them byte for byte and verifies $R^H_Q\widehat L_Q=D_Q$ from the files alone.
