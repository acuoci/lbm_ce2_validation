# Experiment IV — Step 3B: Velocity-Parity Analysis of Finite-Wavenumber Errors

## 1. Question

In Step 3 ([`experiment_IV_linearized_reconstruction.md`](experiment_IV_linearized_reconstruction.md)), the full kinetic population error converged as $O(k)$, while the reconstructed curvature observables converged as $O(k^2)$.

**Working hypothesis, tested here and not assumed:** the leading finite-wavenumber population error has even velocity parity, and is annihilated by the odd third-order Hermite reconstruction $R^H_Q$.

- Code: `src/experiment_IV_parity_analysis.py`. It only reads the Step 3 evaluations: same eigenvectors, normalization ($\lVert\mathbf U\rVert_2=1$), CE2 prediction and weighted norm. Nothing in Steps 1–3 or Experiments I–III is changed.
- Driver: `scripts/reproduce_experiment_IV_parity.py`
- Tests: `tests/test_experiment_IV_parity.py`
- Outputs: `reference_results/experiment_IV_parity/`, following the repository convention (Step 3 writes to `reference_results/experiment_IV_linearized/`)

| File | Contents |
|---|---|
| `experiment_IV_parity.csv` | Per-case values |
| `experiment_IV_parity_fits.csv` | Fits |
| `experiment_IV_parity_identities.csv` | Identity residuals |
| `figure_IV_3_parity_error.png` | Figure |

**Configuration (as in Step 3):** D3Q15, D3Q19, D3Q27; $\mathbf k=\frac{2\pi}{N}(1,2,3)$ with $N=24,\dots,128$; $\tau=1$, $\rho_0=1$; both shear branches. That gives 48 cases.

## 2. Definitions

$(\mathcal P_Q\mathbf g)_i=g_{\bar i}$ with $\mathbf c_{\bar i}=-\mathbf c_i$, and $P_\pm=\tfrac12(I\pm\mathcal P_Q)$.

$$
\mathbf e_Q=\mathbf g^{LBM}_{K,Q}-C_0\widehat L_Q\mathbf q_F,\quad
\mathbf e^\pm_Q=P_\pm\mathbf e_Q,\quad
E^\pm_Q=\frac{\lVert\mathbf e^\pm_Q\rVert_{w}}{\lVert C_0\widehat L_Q\mathbf q_F\rVert_{w}},\quad
E^{rec}_Q=\frac{\lVert R^H_Q\mathbf e_Q/C_0\rVert_2}{\lVert D_Q\mathbf q_F\rVert_2},\quad
r_Q=\frac{\lVert\mathbf e^-_Q\rVert_w}{\lVert\mathbf e^+_Q\rVert_w}.
$$

## 3. Results

### 3.1 Exact algebraic identities

The following hold with floating-point residual exactly 0 on all three lattices; $R^H_QP_+=0$ is also verified in exact rational arithmetic:

- $\mathcal P_Q^2=I$; $P_\pm^2=P_\pm$; $P_+P_-=0$; $P_++P_-=I$; $w_{\bar i}=w_i$.
- $\boxed{R^H_QP_+=0}$ (SymPy). Every row of $R^H_Q$ is a combination of $H^{(3)}_{\alpha\beta\gamma}(\mathbf c_i)$, which is odd in $\mathbf c$.
- $P_+\widehat L_Q=0$: the CE2 prediction is purely odd, so $\mathbf e^+=P_+\mathbf g^{LBM}_K$.
- $\mathcal P_Q$ commutes with $P_{K,Q}$, with the linearized equilibrium removal $N_Q$, and with the BGK collision matrix $C_Q(\tau)$.
- $\boxed{\mathcal P_Q\,\mathsf A_Q(\mathbf k)\,\mathcal P_Q=\mathsf A_Q(-\mathbf k)}$ at every $N$, since $\mathcal P\mathsf S(\mathbf k)\mathcal P=\mathsf S(-\mathbf k)$ and $\mathcal P\mathsf C\mathcal P=\mathsf C$.

Since the weights are symmetric, $\mathcal P_Q$ is self-adjoint in the $W^{-1}$ inner product. The parity split is therefore orthogonal, and $(E^+)^2+(E^-)^2=(E^{pop})^2$; this is checked against the Step 3 values.

### 3.2 Consistency with Step 3

Maxima over the 48 cases:

| Check | Value |
|---|---|
| $\lvert E^{rec}(R^H\mathbf e)-E^{rec}(\text{Step 3})\rvert/E^{rec}$ | 1.1e-13 |
| $\lVert R^H\mathbf e-R^H\mathbf e^-\rVert/\lVert R^H\mathbf e\rVert$ | 1.4e-15 |
| Even part of the CE2 prediction | 0 |

### 3.3 Numerical convergence ($N\ge48$, 5 points, $k\in[0.184,0.490]$)

Exponents $p$ in $E\sim Ck^p$; the hypothesis's predicted value is in brackets.

| Lattice | Branch | $E^+$ [1] | $E^-$ [2] | $E^{rec}$ [2] | $r$ [1] | $\lVert\mathbf e^+\rVert_w$ [3] | $\lVert\mathbf e^-\rVert_w$ [4] | min $R^2$ |
|---|---|---|---|---|---|---|---|---|
| D3Q15 | s1 | 1.002 | 2.038 | 2.037 | 1.036 | 3.001 | 4.038 | 0.99988 |
| D3Q15 | s2 | 1.019 | 2.025 | 2.025 | 1.007 | 3.019 | 4.025 | 0.99997 |
| D3Q19 | s1 | 1.017 | 2.006 | 2.011 | 0.989 | 3.017 | 4.006 | 0.99998 |
| D3Q19 | s2 | 1.026 | 2.024 | 2.024 | 0.998 | 3.026 | 4.024 | 0.99994 |
| D3Q27 | s1 | 1.011 | 2.008 | 2.003 | 0.997 | 3.011 | 4.008 | 0.99999 |
| D3Q27 | s2 | 1.010 | 2.020 | 2.023 | 1.010 | 3.010 | 4.020 | 0.99999 |

No case was flagged or excluded. All-points fits are also in the CSV.

The scaled norms settle to finite nonzero values. From $N=48$ to $N=128$:

| Lattice | Branch | $\lVert\mathbf e^+\rVert_w/k^3$ | $\lVert\mathbf e^-\rVert_w/k^4$ |
|---|---|---|---|
| D3Q15 | s1 | 0.0372 → 0.0372 | 0.0232 → 0.0223 |
| D3Q15 | s2 | 0.0290 → 0.0285 | 0.0174 → 0.0169 |
| D3Q19 | s1 | 0.0826 → 0.0812 | 0.0207 → 0.0206 |
| D3Q19 | s2 | 0.0241 → 0.0235 | 0.0223 → 0.0217 |
| D3Q27 | s1 | 0.1083 → 0.1071 | 0.0251 → 0.0249 |
| D3Q27 | s2 | 0.0690 → 0.0683 | 0.0231 → 0.0226 |

These are statements about norms only; convergence of the coefficient vectors themselves is not inferred.

The ratio $r_Q$ falls from 0.22–0.90 at $N=24$ to 0.04–0.17 at $N=128$. So at small $k$ the population error is dominated by its even part.

**Mirrored branch.** The shear eigenvectors computed at $-\mathbf k$, with the same selection and normalization, satisfy $\mathbf v(-\mathbf k)=-\mathcal P\mathbf v(\mathbf k)$. The residual is at most 2.5e-14 at $N=24$ and rises to at most 1.4e-11 at $N=128$, consistent with eigenvector conditioning of about machine epsilon divided by the shear-pair gap (5.6e-6 at $N=128$). The minus sign keeps $\mathbf U$ unchanged, because $\mathbf U(\mathcal P\mathbf v)=-\mathbf U(\mathbf v)$.

### 3.4 Figure

`figure_IV_3_parity_error.png` plots $E^+$ (left) and $E^-$ (right) against $k$ for both branches on all three lattices, with $\propto k$ and $\propto k^2$ guides. Colour and marker shape identify the lattice; filled markers with solid lines are branch s1, and open markers with dashed lines are branch s2.

## 4. Interpretation

**1. Exact algebraic identity.**

- $R^H_QP_+=0$ holds exactly: the reconstruction cannot see any even-parity population content.
- Consequently $R^H_Q\mathbf e=R^H_Q\mathbf e^-$, so $E^{rec}$ is controlled by the odd error alone.
- $\mathcal P\mathsf A(\mathbf k)\mathcal P=\mathsf A(-\mathbf k)$ also holds exactly, for any $\tau$; the numerical results here are for $\tau=1$ only.

**2. Numerically observed behaviour (48 cases, one direction, $\tau=1$).**

- $E^+=O(k^{1.00\text{–}1.03})$, $E^-=O(k^{2.01\text{–}2.04})$ and $r=O(k^{0.99\text{–}1.04})$, as the hypothesis predicts.
- The absolute even error behaves as $O(k^3)$ and the absolute odd error as $O(k^4)$, while $\lVert C_0\widehat L\mathbf q_F\rVert_w=O(k^2)$.
- $E^{rec}$ tracks $E^-$ in exponent. It is not equal to $E^-$ in general, since $R^H$ is not an isometry on the odd subspace.

**3. Analytical status.** These results are consistent with the parity-cancellation mechanism and support it strongly, but they do not prove it. Numerical convergence over $k\in[0.18,0.98]$ in one direction, at one $\tau$, is not an asymptotic theorem.

**What a proof needs.** The exact covariance $\mathcal P\mathsf A(k\hat{\mathbf n})\mathcal P=\mathsf A(-k\hat{\mathbf n})$ supplies the structure. Suppose that along the ray $k\hat{\mathbf n}$:

- each shear eigenvector branch, normalized so that $\mathbf U$ is fixed, is analytic in the scalar $k$ in a neighbourhood of $k=0$, including $k<0$; and
- its continuation satisfies $\mathbf v(-k)=-\mathcal P\mathbf v(k)$, the relation verified numerically above at each computed $k$.

Then the Taylor coefficients $\mathbf v(k)=\sum_n k^n\mathbf v_n$ obey $\mathcal P\mathbf v_n=(-1)^{n+1}\mathbf v_n$. Because $\mathcal P$ commutes with $P_KN$, the $O(k^n)$ term of $\mathbf g_K$ has parity $(-1)^{n+1}$. The same relation gives $\mathbf U(-k)=\mathbf U(k)$, so $\mathbf q_F$ and $C_0\widehat L\mathbf q_F$ contain only even powers of $k$ times odd vectors. With the established CE2 agreement at $O(k^2)$, the leading residual $\mathbf e=k^3\mathbf g_3+O(k^4)$ would be even and the odd remainder $O(k^4)$, which is the observed pattern.

**The steps not established here are:**

- (a) **Analyticity of the shear branches through the degenerate point $k=0$.** The two branches coincide at $k=0$, and here split only at about $O(k^4)$. $\mathsf A$ is not normal, so a Rellich-type analytic-perturbation result does not apply directly. A Kato reduction to the two-dimensional shear subspace, carried to the order at which the degeneracy is lifted, would be needed to exclude fractional-power (Puiseux) behaviour and to fix the branches.
- (b) **Compatibility of the normalization with the symmetry.** Fixing $\lVert\mathbf U\rVert=1$ and the phase convention must preserve analyticity and the relation $\mathbf v(-k)=-\mathcal P\mathbf v(k)$.
- (c) **Identification of the branch at $-k$** with the parity image of the branch at $+k$ by continuity through $k=0$, rather than only by matching polarizations.
- (d) **Nonvanishing of the even $O(k^3)$ coefficient $\mathbf g_3$.** This is needed for $E^+$ to be exactly first order rather than higher. Numerically, $\lVert\mathbf e^+\rVert_w/k^3$ settles at nonzero values.

The $O(k^4)$ behaviour of the mixed observable $T$ seen in Step 3 is outside this task and is not addressed here.

## 5. Reproduction

```bash
python scripts/reproduce_experiment_IV_parity.py     # about 1 s; writes reference_results/experiment_IV_parity/
python -m unittest discover -s tests
python scripts/check_repository.py
```
