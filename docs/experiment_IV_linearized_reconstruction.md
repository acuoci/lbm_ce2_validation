# Experiment IV — Step 3: Reconstruction from Exact Linearized BGK Shear Eigenmodes

## 1. Purpose and scope

This step tests whether the verified reconstruction operators recover the observable curvature functionals from the **actual kinetic populations of exact eigenvectors of the linearized discrete BGK scheme**, and how the error behaves at finite wavenumber. The reconstruction is

$$
\mathbf d^{rec}_Q=\frac1{C_0}R^H_Q\,\mathbf g^{LBM}_{K,Q},\qquad
\mathbf g^{LBM}_{K,Q}=P_{K,Q}N_Q\mathbf v_s,\qquad
C_0=2\rho_0\tau(\tau-\tfrac12)c_s^4 ,
$$

and the reference is $\mathbf d^{exact}_Q=D_Q\mathbf q_F$, where $\mathbf q_F$ holds the independent components of $J_{\alpha\beta\gamma}=-k_\beta k_\gamma U_\alpha$ built from the eigenvector's own momentum amplitude $\mathbf U$.

**What this does not show:**

- Reconstruction from spatially evolved populations with the nonlinear equilibrium (Step 4).
- Unique recovery of the full incompressible curvature tensor. Only the lattice's observable functionals are reconstructed, and only for transverse Fourier modes.

Here this is a test of finite-wavenumber consistency with the CE2 asymptotic theory.

- Code: `src/experiment_IV_linearized_reconstruction.py`
- Driver: `scripts/reproduce_experiment_IV_linearized.py`
- Tests: `tests/test_experiment_IV_linearized.py`
- Outputs: `reference_results/experiment_IV_linearized/`

## 2. Configuration

| Item | Value |
|---|---|
| Lattices | D3Q15, D3Q19, D3Q27 (canonical operator files) |
| $\tau$, $\rho_0$ | 1, 1 |
| Wavevector | $\mathbf k_N=\frac{2\pi}{N}(1,2,3)$, $N\in\{24,32,40,48,64,80,96,128\}$, $\lvert\mathbf k\rvert\in[0.184,0.980]$ |
| Nominal transverse directions (reference only) | $\mathbf e_1=(2,-1,0)/\sqrt5$, $\mathbf e_2=(3,0,-1)/\sqrt{10}$ |
| Arithmetic | complex128; only $Q\times Q$ Fourier matrices are formed, no $N^3$ grid |

Reused without modification:

- **Experiment II:** `load_lattice`, `amplification_matrix` ($\mathsf A=\mathsf S\mathsf C$), `hydrodynamic_moments`, `q_fourier`, `ce2_prefactor`, `weighted_norm`, and the linearized equilibrium-removal operator `N`.
- **Step 2B:** `project_kinetic` and `reconstruct_from_kinetic`.
- **Step 2A:** $D_Q$.

## 3. Shear-mode identification

At $\tau=1$ the collision matrix equals $P_{eq}$, which has rank 4. So $\mathsf A_Q$ has exactly four nonzero eigenvalues: two real shear eigenvalues and a complex-conjugate sound pair. All other eigenvalues are zero and carry no momentum.

Each eigenvector is scored by its transverse hydrodynamic character:

$$
s=\frac{\lvert\mathbf j\rvert}{\lVert\mathbf v\rVert_w}\cdot\frac{1-\lvert\hat{\mathbf k}\cdot\mathbf j\rvert/\lvert\mathbf j\rvert}{1+\lvert\delta\rho\rvert/\lvert\mathbf j\rvert} .
$$

The two highest scores are taken. The selection is flagged as ambiguous if the third-best score exceeds half the second-best. Eigenvalue magnitude is not used.

Each eigenvector is then normalized to $\lVert\mathbf U\rVert_2=1$ with a deterministic phase. Branches are labelled at the smallest $k$ ($N=128$): s1 has the larger, slower-decaying eigenvalue. They are continued to larger $k$ by maximum polarization overlap $\lvert\mathbf U_{prev}^\dagger\mathbf U\rvert$. Any overlap below 0.9, a swap in eigenvalue order, a degenerate pair ($\lvert\lambda_1-\lambda_2\rvert<10^{-9}$), or a longitudinal fraction above 0.1 would be flagged, and the case excluded from the fits.

**Experiment II's `select_shear_mode` is not sufficient for [123].** It selects by target polarization. On D3Q15 and D3Q27, targeting $\mathbf e_1$ and targeting $\mathbf e_2$ both return branch s1, at every $N$; only on D3Q19 does it find both branches (8 of 24 lattice–$N$ cases). This is recorded in `experiment_IV_linearized_selector_check.csv`. The selector is unchanged; it remains valid for the degenerate [100], [110] and [111] cases it was written for.

### Identification diagnostics (all 48 cases)

| Quantity | Value |
|---|---|
| Max eigenpair residual $\lVert\mathsf A\mathbf v-\lambda\mathbf v\rVert/\lVert\mathbf v\rVert$ | 1.5e-15 |
| Max selection margin (3rd / 2nd score) | 9.7e-05 |
| Min shear-pair gap $\lvert\lambda_{s1}-\lambda_{s2}\rvert$ | 5.6e-06 (D3Q15, $N=128$); the gap shrinks roughly as $k^4$ |
| Min distance to non-shear eigenvalues | 0.105 |
| Min tracking overlap | 0.999994 |
| Max longitudinal fraction / max $\lvert\delta\rho\rvert$ | 2.8e-03 / 4.6e-02 (both at $N=24$; they decrease as $k$ decreases) |
| Cases flagged as ambiguous, degenerate or near-zero reference | 0 |

On D3Q19, branch s2 is exactly transverse and density-free to round-off ($\lvert\delta\rho\rvert\sim10^{-15}$).

Consistency checks, maximum over all cases:

| Check | Value |
|---|---|
| $\lVert P_KN\mathbf v-P_K\mathbf v\rVert_w/\lVert P_K\mathbf v\rVert_w$ | 1.9e-14 |
| Projected vs direct $R^H\mathbf v/C_0$ | 1.2e-14 |
| $\mathbf q_F$ vs Experiment II `q_fourier` | 0 |
| Synthetic CE2 control, reconstruction of $C_0\widehat L\mathbf q_F$ | 3.8e-16 |

## 4. Error diagnostics

$$
E^{rec}_Q=\frac{\lVert\mathbf d^{rec}-\mathbf d^{exact}\rVert_2}{\lVert\mathbf d^{exact}\rVert_2},\qquad
E^{pop}_Q=\frac{\lVert\mathbf g^{LBM}_K-C_0\widehat L_Q\mathbf q_F\rVert_w}{\lVert C_0\widehat L_Q\mathbf q_F\rVert_w}.
$$

Both use complex norms. $\widehat L_Q$ appears only in $E^{pop}$ and the synthetic control, never in the reconstruction input.

$E^{pop}$ is additionally split into the part inside the third-order Hermite span, $\mathrm{span}\{w_iH^{(3)}_{\alpha\beta\gamma}(\mathbf c_i)\}$ (`population_error_hermite3`), and its $W^{-1}$-orthogonal complement (`population_error_complement`). $R^H$ reads only third-order Hermite moments.

For $T$ (D3Q15 and D3Q27), the relative error is reported only when $\lvert T^{exact}\rvert\ge10^{-2}\lVert\mathbf d^{exact}\rVert$; otherwise the case is flagged `T_reference_small`. The error relative to $\lVert\mathbf d^{exact}\rVert$ is always reported.

### Errors at the extreme wavenumbers

| Lattice | Branch | $E^{rec}$ ($N=24$) | $E^{rec}$ ($N=128$) | $E^{pop}$ ($N=24$) | $E^{pop}$ ($N=128$) |
|---|---|---|---|---|---|
| D3Q15 | s1 | 7.06e-02 | 2.09e-03 | 9.02e-02 | 1.40e-02 |
| D3Q15 | s2 | 8.96e-02 | 2.80e-03 | 1.74e-01 | 2.58e-02 |
| D3Q19 | s1 | 6.16e-02 | 2.05e-03 | 2.21e-01 | 3.74e-02 |
| D3Q19 | s2 | 1.14e-01 | 3.59e-03 | 1.71e-01 | 2.14e-02 |
| D3Q27 | s1 | 3.98e-02 | 1.39e-03 | 2.84e-01 | 4.95e-02 |
| D3Q27 | s2 | 7.83e-02 | 2.46e-03 | 2.81e-01 | 4.75e-02 |

Per-observable absolute errors are in `experiment_IV_linearized_observables.csv`.

## 5. Convergence

The fit interval was chosen after inspecting the consecutive-point slopes (`experiment_IV_linearized_local_slopes.csv`). Those slopes approach their limits monotonically:

- **Reconstruction:** 2.02–2.27 between $N=24$ and 32, falling to 2.00–2.02 between $N=96$ and 128.
- **Population:** 1.11–1.57 falling to 1.01–1.05 over the same range.

The primary fits use $N\ge48$, i.e. $k\in[0.184,0.490]$ with 5 points per fit. All-points fits ($N=24$–128, 8 points) are also in the CSV. No points were excluded.

**Exponents $p$ for $E\sim Ck^p$, interval $N\ge48$:**

| Lattice | Branch | $E^{rec}$ | $E^{pop}$ | $E^{pop}$, Hermite-3 part | $E^{pop}$, complement | $T$ rel. | $T$ err. / $\lVert\mathbf d\rVert$ |
|---|---|---|---|---|---|---|---|
| D3Q15 | s1 | 2.037 | 1.040 | 2.038 | 1.002 | 4.011 | 4.011 |
| D3Q15 | s2 | 2.025 | 1.054 | 2.025 | 1.019 | flagged | 3.976 |
| D3Q19 | s1 | 2.011 | 1.024 | 2.006 | 1.017 | n/a | n/a |
| D3Q19 | s2 | 2.024 | 1.106 | 2.024 | 1.026 | n/a | n/a |
| D3Q27 | s1 | 2.003 | 1.016 | 2.003 | 1.014 | 4.014 | 4.013 |
| D3Q27 | s2 | 2.023 | 1.022 | 2.023 | 1.014 | 4.034 | 4.037 |

Fit quality: $R^2\ge0.9992$ for every fit in the table; $R^2\ge0.99997$ for every $E^{rec}$ fit. All-points exponents for $E^{rec}$ are 2.006–2.092.

## 6. Discussion

- **Reconstruction converges as $k^2$, population error as $k^1$.** Over the asymptotic interval, $E^{rec}$ decreases as $k^{2.00}$–$k^{2.04}$ on every lattice and branch, while $E^{pop}$ decreases only as $k^{1.02}$–$k^{1.11}$.
- **The leading population error is invisible to the reconstruction.** The split shows that the $O(k)$ part of the population discrepancy lies entirely outside the third-order Hermite span (complement exponents 1.00–1.03). The part inside that span, which is all that $R^H$ reads, is $O(k^2)$ and closely tracks $E^{rec}$; for several branches the two coincide to the printed digits. This experiment identifies the error's location in population space, not its analytical origin.
- **The fully mixed functional $T$ is reconstructed with error $O(k^4)$ relative to $\lVert\mathbf d\rVert$** on D3Q15 and D3Q27, much more accurately than the observable vector as a whole.
- **On D3Q15 branch s2, $T^{exact}$ vanishes at leading order.** That branch's polarization converges to $\mathbf U\propto\mathbf k\times(6,3,2)\propto(-5,16,-9)/\sqrt{362}$. Because $T=-(2\pi/N)^2(6,3,2)\cdot\mathbf U$, $\lvert T^{exact}\rvert/\lVert\mathbf d\rVert$ falls from $3.7\times10^{-3}$ ($N=24$) to $9.9\times10^{-5}$ ($N=128$), roughly as $k^2$, so the relative $T$ error is flagged there rather than fitted.
- **Lattice ranking differs between the two errors.** Among the s1 branches, D3Q27 has the smallest $E^{rec}$ but the largest $E^{pop}$. A larger population mismatch therefore does not imply a larger reconstruction error, because most of it lies outside the Hermite-3 span.

## 7. Limitations and reproducibility

- **Exact linearized eigenvectors only, at a single $\tau$ and direction.** These are exact eigenvectors of the linearized discrete operator at $\tau=1$, along one generic direction [123]. Other $\tau$, directions, nonlinear equilibria and spatial simulations are not covered.
- **Branch identification is validated only for this configuration.** The criteria and thresholds were checked here at $\tau=1$, where the non-hydrodynamic eigenvalues are exactly zero. For $\tau\ne1$ the selection margin must be re-examined; it is reported per case.
- **Reconstructing the observable functionals is not reconstructing the full curvature tensor.** D3Q19 does not observe $T$, and D3Q15 observes only $S_1,S_2,S_3,T$.
- **Fit intervals and thresholds are documented choices.** Both fit intervals are written to the CSV.
- **Results may differ across machines in the last digits.** Eigenvectors come from LAPACK (`numpy.linalg.eig`). Phase normalization and branch tracking make the reported quantities independent of LAPACK's phase and ordering choices. Repeated runs on one machine are byte-identical.

## 8. Reproduction

```bash
python scripts/reproduce_experiment_IV_linearized.py   # about 3 s; writes reference_results/experiment_IV_linearized/
python -m unittest discover -s tests
python scripts/check_repository.py
```
