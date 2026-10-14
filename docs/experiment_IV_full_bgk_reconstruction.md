# Experiment IV — Step 4: Reconstruction from Spatially Evolved BGK Populations

## 1. Question and scope

Does pointwise curvature reconstruction from populations evolved by the full nonlinear-equilibrium BGK collision–streaming algorithm reproduce the exact linearized shear-mode results of Step 3 in the small-amplitude limit?

$$
\mathbf d^{rec}_Q(\mathbf x,t)=\frac1{C_0}R^H_Q\,\mathbf g^{LBM}_{K,Q}(\mathbf x,t)
\qquad\text{vs}\qquad
\mathbf d^{ref}_Q(\mathbf x,t)=D_Q\mathbf q_F(\mathbf x,t),\qquad C_0=\tfrac29\rho_0\tau(\tau-\tfrac12).
$$

This step covers small-amplitude transverse Fourier modes on a periodic grid, at $\tau=1$ and along the direction [123]. It is **not** a validation for arbitrary nonlinear flows.

- Code: `src/experiment_IV_full_bgk_reconstruction.py`
- Driver: `scripts/reproduce_experiment_IV_full_bgk.py` (about 1 min; peak memory about 0.6 GB at $N=80$ on D3Q27)
- Tests: `tests/test_experiment_IV_full_bgk.py`
- Outputs: `reference_results/experiment_IV_full_bgk/`

## 2. Method

| Item | Choice |
|---|---|
| Lattices / branches | D3Q15, D3Q19, D3Q27; shear branches s1 and s2, labelled as in Step 3 |
| Wavevector | $\mathbf k_N=\frac{2\pi}{N}(1,2,3)$ with $N=24,32,48,64,80$, so $\lvert\mathbf k\rvert\in[0.29,0.98]$ |
| Parameters | $\tau=1$, $\rho_0=1$, $U_0=10^{-6}$ (primary) |
| Times | $t=0$ (control), then 1, 2, 4, 8 |
| Initialization | $f_i=w_i\rho_0+U_0\,\mathrm{Re}[v_ie^{i\mathbf k\cdot\mathbf x}]$, with $\mathbf v$ the Step 3 exact eigenvector normalized to $\lVert\mathbf U\rVert_2=1$; this is Experiment II's formula |
| Dynamics | Experiment II `collide_and_stream`, unmodified: full second-order equilibrium, periodic `np.roll` streaming, layout `f[i,z,y,x]` |
| Reconstruction | Step 2B `reconstruct_from_populations` per node (local $\rho$, $\mathbf u$, $\mathbf f^{eq}$, $\mathbf f^{neq}$, $P_K$, $R^H/C_0$), vectorized over $z$-slabs of $2^{16}$ nodes |
| Direct estimator | $R^H\mathbf f/C_0$ on the raw populations |
| Reference | $\hat{\mathbf U}(t)=\rho_0^{-1}\langle\mathbf j\,e^{-i\mathbf k\cdot\mathbf x}\rangle$ from the evolved moments (spatial-average normalization, as Experiment II); $\hat J=-\mathbf k\mathbf k\hat{\mathbf U}$; $\hat{\mathbf d}^{ref}=D_Q\mathbf q_F$; $\mathbf d^{ref}(\mathbf x)=2\,\mathrm{Re}[\hat{\mathbf d}^{ref}e^{i\mathbf k\cdot\mathbf x}]$ |

**Why Step 3's eigenvectors rather than Experiment II's `initialize_exact_mode`.** The Experiment II routine selects the mode with `select_shear_mode`, which on [123] returns only branch s1 for D3Q15 and D3Q27 (Step 3). The same initialization formula is therefore applied to the Step 3 eigenvectors.

**Isolation of the reconstruction.** The reconstruction routine receives only the local population arrays, the lattice name, $\tau$ and $\rho_0$; a test intercepts its calls to confirm this. The wavevector, analytical velocity and Fourier transforms are used only for the reference and the diagnostics.

**Relation to Step 3.** For a real single-mode field, the grid average of $\lvert\mathrm{Re}(\mathbf a e^{i\mathbf k\cdot\mathbf x})\rvert^2$ is exactly $\lvert\mathbf a\rvert^2/2$, since $2\mathbf k$ is not aliased to 0 for these $N$. So in the linear limit the spatial RMS error equals Step 3's complex-norm error, with the same mode normalization and reference definition. This is checked rather than assumed.

## 3. Diagnostics

The following are computed per lattice, branch, $N$, $t$ and $U_0$; see `experiment_IV_full_bgk_results.csv` and `experiment_IV_full_bgk_observables.csv`.

| Diagnostic | Definition / purpose |
|---|---|
| $E^{RMS}_Q$ | Spatial relative RMS error over all nodes and observables |
| $E^{RMS}_{direct}$ and direct–projected difference | Same error for $R^H\mathbf f/C_0$, and the gap between the two estimators |
| Max-abs error | $\max\lvert\mathbf d^{rec}-\mathbf d^{ref}\rvert/\max\lvert\mathbf d^{ref}\rvert$ |
| Per-observable RMS errors | Observable ordering of Step 2A |
| $E_{modal}$ | Error of the $+\mathbf k$ Fourier component of the reconstructed field |
| Non-modal fraction | RMS of the reconstructed field outside its single-mode part, relative to RMS of $\mathbf d^{ref}$ |
| $T$ errors (D3Q15, D3Q27) | Single-run RMS and modal relative errors. Reported only if RMS$(T^{ref})\ge10^{-2}\,$RMS$(\mathbf d^{ref})$, otherwise flagged `T_reference_small`. Flagged `T_nonlinear_limited` if the two differ by more than 1 % |
| Fourier contamination | Momentum power outside the $\pm\mathbf k$ bins over power in them (FFT) |
| Round-off floor estimate | $\epsilon\lVert\mathbf w\rVert/\mathrm{RMS}(\mathbf g_K)$ |
| Velocity decay | Deviation of $\lvert\hat{\mathbf U}(t)\rvert/\lvert\hat{\mathbf U}(0)\rvert$ from $\lambda^t$ |
| Reference consistency | Re-extracting $\hat{\mathbf d}^{ref}$ from $\mathbf d^{ref}(\mathbf x)$ |
| Conservation | Mass drift and net momentum |

**Flags:** `nonlinear_contamination` (contamination above $10^{-3}$) and `roundoff_limited` (round-off floor above $10^{-3}E^{RMS}$) would exclude a point from the $E^{RMS}$ fits; neither occurred. `pre_asymptotic_k` marks $N<48$ and is informative only. `initial_state_control` marks $t=0$.

## 4. Results

Measured with Python 3.10.12 and NumPy 2.2.6.

### 4.1 Agreement with Step 3 ($U_0=10^{-6}$; 30 configurations at 5 times each)

| Quantity | Max over all cases |
|---|---|
| $\lvert E^{RMS}-E^{rec,lin}\rvert/E^{rec,lin}$, $t\ge1$ | 3.4e-06 |
| Same, $t=0$ control | 6.0e-08 |
| $\lvert E_{modal}-E^{rec,lin}\rvert/E^{rec,lin}$ | 8.5e-08 |
| $\lvert E^{RMS}_{direct}-E^{rec,lin}\rvert/E^{rec,lin}$ | 3.4e-06 |
| RMS(direct − projected) / RMS(ref) | 3.6e-09 |
| Estimated round-off floor | 3.8e-08 |
| Non-modal fraction of $\mathbf d^{rec}$ | 2.7e-09 at $t=0$; 1.3e-05 to 1.5e-05 for $t\ge1$ |
| Fourier contamination of the momentum field | ≤ 1.8e-07 |
| Velocity decay vs $\lambda^t$ | 7.4e-11 |
| Reference re-extraction residual | 4.5e-13 |
| Mass drift / net momentum $/(N^3U_0)$ | 2.4e-14 / 2.0e-11 |

No case was flagged `nonlinear_contamination` or `roundoff_limited`.

**$E^{RMS}$ at $t=8$:**

| Lattice | Branch | $N=24$ | 32 | 48 | 64 | 80 |
|---|---|---|---|---|---|---|
| D3Q15 | s1 | 7.056e-02 | 3.668e-02 | 1.543e-02 | 8.516e-03 | 5.403e-03 |
| D3Q15 | s2 | 8.963e-02 | 4.766e-02 | 2.040e-02 | 1.133e-02 | 7.207e-03 |
| D3Q19 | s1 | 6.160e-02 | 3.377e-02 | 1.476e-02 | 8.254e-03 | 5.269e-03 |
| D3Q19 | s2 | 1.143e-01 | 6.100e-02 | 2.616e-02 | 1.454e-02 | 9.252e-03 |
| D3Q27 | s1 | 3.982e-02 | 2.230e-02 | 9.877e-03 | 5.548e-03 | 3.549e-03 |
| D3Q27 | s2 | 7.826e-02 | 4.177e-02 | 1.792e-02 | 9.962e-03 | 6.341e-03 |

These agree with the Step 3 exact-linearized values to within 3.4e-6 relative (§4.1); differences are confined to the last printed digit at most.

**Mechanism of the residual difference.** At $t\ge1$ the reconstructed field carries a non-modal component of relative size $n\approx10^{-5}$. It appears in the first collision, from the nonlinear equilibrium, and then stays roughly constant. Because it is orthogonal to the $\pm\mathbf k$ mode, it raises $E^{RMS}$ in quadrature: $E^{RMS}\approx(E_{modal}^2+n^2)^{1/2}$. Case by case, $(E^{RMS}-E^{rec,lin})/E^{rec,lin}$ matches $n^2/2E^2$; for example D3Q27 s1 at $N=80$ gives 2.79e-6 measured against 2.77e-6 predicted. $E_{modal}$ itself agrees with Step 3 to the round-off level.

### 4.2 Evaluation time

There is no initialization transient. The extracted velocity decays as $\lambda^t$ to 7e-11, and the maximum deviation from Step 3 is about the same at every time: 3.4e-6, 3.3e-6, 3.1e-6 and 2.8e-6 at $t=1,2,4,8$.

### 4.3 Convergence ($t=8$)

Fits $E^{RMS}\sim Ck^p$, compared with Step 3 values on the same $N$. $N\ge48$ ($k\in[0.29,0.49]$, 3 points) is the asymptotic interval identified in Step 3.

| Lattice | Branch | $p$, $N\ge48$ | $R^2$ | $p$, $N\ge32$ (4 pts) | $p$, all (5 pts) | Step 3 $p$, same $N\ge48$ | Step 3 $p$, $N=48$–128 |
|---|---|---|---|---|---|---|---|
| D3Q15 | s1 | 2.055 | 0.999986 | 2.091 | 2.130 | 2.055 | 2.037 |
| D3Q15 | s2 | 2.037 | 0.999994 | 2.062 | 2.090 | 2.037 | 2.025 |
| D3Q19 | s1 | 2.016 | 0.999999 | 2.028 | 2.041 | 2.016 | 2.011 |
| D3Q19 | s2 | 2.035 | 0.999994 | 2.059 | 2.085 | 2.035 | 2.024 |
| D3Q27 | s1 | 2.004 | 1.000000 | 2.006 | 2.008 | 2.004 | 2.003 |
| D3Q27 | s2 | 2.034 | 0.999994 | 2.058 | 2.084 | 2.034 | 2.023 |

Full-BGK and linearized exponents agree to within 5.2e-6 on identical $N$ sets. The full-BGK fits do not extend below $k=0.29$ (no $N>80$), which is why their exponents sit slightly above Step 3's $N=48$–128 values: a pre-asymptotic drift that Step 3 already showed in its local slopes.

**$T$ (D3Q15, D3Q27), $N\ge48$.**

- *Modal $T$ error:* $p=4.016$ (D3Q15 s1), $4.021$ (D3Q27 s1) and $4.050$ (D3Q27 s2), equal to Step 3 on the same $N$ to $10^{-3}$.
- *Single-run RMS $T$ error:* usable only for D3Q15 s1 ($p=4.014$). On D3Q27, the cases $N=64$ (s1) and $N=80$ (s1, s2) are flagged `T_nonlinear_limited` and excluded; this is explained in §5.
- *D3Q15 s2:* $T$ is flagged `T_reference_small` at every $N$, as in Step 3.
- *D3Q19:* $T$ is not reconstructed.

### 4.4 Amplitude dependence (separate diagnostic; $N=32,48$; $t=1,8$; paired $\pm U_0$)

Maximum $\lvert E^{RMS}-E^{rec,lin}\rvert/E^{rec,lin}$ at $t=8$ over all lattices and branches (`experiment_IV_full_bgk_amplitude.csv`):

| $U_0$ | $N=32$ single | $N=32$ paired odd part | $N=48$ single | $N=48$ paired odd part |
|---|---|---|---|---|
| 1e-4 | 2.8e-05 | 4.7e-08 | 8.1e-04 | 1.3e-07 |
| 1e-5 | 2.8e-07 | 6.3e-10 | 8.1e-06 | 6.9e-10 |
| 1e-6 | 4.5e-09 | 5.8e-09 | 7.1e-08 | 1.2e-08 |
| 1e-7 | 6.5e-08 | 6.5e-08 | 1.5e-07 | 1.5e-07 |

- **Large amplitudes.** The single-run deviation scales as $U_0^2$: an $O(U_0)$ non-modal component entering in quadrature. Pairing $+U_0$ and $-U_0$ runs, which keeps only the odd-amplitude response, removes most of it.
- **Small amplitudes.** The deviation rises as $U_0$ decreases, because the cancellation in $\mathbf f-\mathbf f^{eq}$ becomes relatively larger. Single and paired results then coincide.
- **Choice of $U_0$.** $U_0=10^{-6}$ lies near the optimum at these $N$.
- **Dependence on $k$.** The nonlinear deviation grows rapidly as $k$ decreases (×29 from $N=32$ to 48 at $U_0=10^{-4}$), because $E^{rec}$ itself shrinks as $k^2$ while the non-modal content does not.

### 4.5 Figures

- `figure_IV_4_full_bgk_vs_linearized.png`
  - Left: full-BGK $E^{RMS}$ at $t=8$ (markers) on the Step 3 exact-linearized curves (lines).
  - Right: deviation from Step 3 against $U_0$ at $N=48$, $t=8$, for single runs (lines with markers) and the paired odd part (crosses on dotted lines).
- `figure_IV_5_spatial_comparison.png`: $A_x/U_0$ along $x$ at $y=z=0$ for D3Q27 s1, $N=32$, $t=8$, reference against local reconstruction, with the pointwise difference.

## 5. Anomalies, limitations, open issues

- **$T$ on D3Q27 at $N\ge64$.** $T$ is reconstructed so accurately (relative error about $10^{-5}$) that the single-run RMS error reaches the $O(U_0)$ non-modal floor. For s1 at $N=80$ the single run gives 1.774e-5 against Step 3's 1.575e-5. The modal $T$ error and a paired $\pm U_0$ run both reproduce Step 3 (1.5750e-5). These cases are flagged and excluded from the single-run $T$ fit, so $T$ convergence at $N\ge64$ relies on the modal (or paired) estimate.
- **Single direction, single $\tau$, linear regime.** All runs are at $\tau=1$, direction [123], and amplitudes of $10^{-7}$ to $10^{-4}$; in this regime the flow is linear to high accuracy. Agreement with Step 3 therefore tests the full pipeline — real-valued fields, nonlinear equilibrium, streaming, pointwise reconstruction and normalization — not nonlinear-flow reconstruction.
- **Limited small-$k$ coverage.** The asymptotic fit uses only 3 points ($N=48$–80). $N=96$–128 were not run, because the fits already coincide with Step 3 on identical $N$, and a single run at $N=128$ would cost about 4× the memory of $N=80$.
- **Round-off.** The cancellation in $\mathbf f-\mathbf f^{eq}$ sets a relative floor of at most 3.8e-8 here. It would grow as $1/(U_0k^2)$ for smaller amplitudes or wavenumbers.
- **Nothing upstream was modified.** Canonical operators, Experiments I–III and the Steps 1–3B code are unchanged.

## 6. Assessment

In the small-amplitude limit, local reconstruction from spatially evolved nonlinear-equilibrium BGK populations reproduces the exact linearized shear-mode reconstruction on all three lattices and both branches:

- **The errors agree.** $E^{RMS}$ matches Step 3 to a few parts in $10^6$, and the modal error to $10^{-7}$.
- **The convergence agrees.** The exponents ($p\approx2.00$–$2.06$ over $N=48$–80) are identical to Step 3 on the same $N$.
- **The residual difference is accounted for quantitatively.** It is explained by an $O(U_0)$ nonlinear non-modal component, which adds in quadrature, together with an $O(\epsilon/U_0)$ round-off floor.

The result establishes consistency of the full algorithmic pipeline with the linearized analysis for transverse Fourier modes. It does not establish reconstruction accuracy for arbitrary nonlinear flows.

## 7. Reproduction

```bash
python scripts/reproduce_experiment_IV_full_bgk.py
python -m unittest discover -s tests
python scripts/check_repository.py
```
