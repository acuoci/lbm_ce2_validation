# Experiment IV — Step 2B: Numerical Curvature-Reconstruction Interface

## 1. Scope

`src/curvature_reconstruction.py` applies the exact Hermite reconstruction operators $R^H_Q$ of Step 2A ([`experiment_IV_reconstruction_operators.md`](experiment_IV_reconstruction_operators.md)) to LBM population data in NumPy, for D3Q15, D3Q19 and D3Q27.

The module only applies fixed linear maps to population arrays. It never uses a wavevector, a velocity amplitude, an analytical curvature field, or the forward CE2 operator $\widehat L_Q$. It does not depend on the Experiment II/III solvers. Tests: `tests/test_curvature_reconstruction.py`.

Numerical convergence of the reconstruction on actual LBM data has **not** been assessed here. That belongs to the subsequent experiments.

## 2. Normalization

$$
\mathbf g^{CE2}_{K,Q}=C_0\,\widehat L_Q\,\mathbf q,\qquad
C_0=2\rho_0\tau\left(\tau-\tfrac12\right)c_s^4=\tfrac29\,\rho_0\,\tau\left(\tau-\tfrac12\right),\qquad
\mathbf d^{rec}_Q=\frac1{C_0}\,R^H_Q\,\mathbf g_{K,Q}.
$$

$C_0$ is the same expression as `ce2_prefactor` in Experiment II. It is applied in exactly one place, `reconstruct_from_kinetic`, and every dimensional interface passes through that function. The relaxation time must satisfy $\tau>1/2$, since $C_0$ vanishes at $\tau=1/2$. The reference density must satisfy $\rho_0>0$. $\rho_0$ is the uniform CE2 reference density; it is never used in place of the local density.

## 3. API

Populations are always on the **last** axis: shape `(..., Q)`. Every interface returns shape `(..., r_Q)`.

| Function | Computes | Input |
|---|---|---|
| `reconstruct_from_normalized_kinetic(g_normalized, lattice)` | $R^H_Q\,\mathbf g$ (no $C_0$) | normalized kinetic populations, e.g. $\widehat L_Q\mathbf q$ |
| `reconstruct_from_kinetic(g_kinetic, lattice, tau, rho0=1.0)` | $R^H_Q\,\mathbf g/C_0$ | dimensional kinetic populations $P_{K,Q}\mathbf f^{neq}$ |
| `reconstruct_from_nonequilibrium(f_noneq, lattice, tau, rho0=1.0)` | $R^H_Q P_{K,Q}\mathbf f^{neq}/C_0$ | non-equilibrium populations |
| `reconstruct_from_populations(f, lattice, tau, rho0=1.0)` | local $\rho,\mathbf u$ → $\mathbf f^{eq}$ → $\mathbf f^{neq}$ → as above | full populations (real) |

Helper functions:

| Function | Purpose |
|---|---|
| `observable_names(lattice)` | Output order |
| `num_observables(lattice)` | $r_Q$ |
| `num_populations(lattice)` | $Q$ |
| `reconstruction_matrix(lattice)` | $R^H_Q$, read-only, $r_Q\times Q$ |
| `kinetic_projector(lattice)` | $P_{K,Q}$, read-only, $Q\times Q$ |
| `ce2_prefactor(tau, rho0)` | $C_0$ |
| `project_kinetic(f_noneq, lattice)` | $P_{K,Q}\mathbf f^{neq}$ |
| `macroscopic(f, lattice)` | Local $\rho$ and $\mathbf u$ |
| `equilibrium(rho, u, lattice)` | Experiment II's second-order equilibrium, last-axis layout |
| `nonequilibrium_populations(f, lattice)` | $\mathbf f-\mathbf f^{eq}(\rho(\mathbf f),\mathbf u(\mathbf f))$ |

### Observable ordering (last output axis)

| Lattice | $Q$ | $r_Q$ | Output |
|---|---|---|---|
| D3Q15 | 15 | 4 | $(S_1,S_2,S_3,T)$ |
| D3Q19 | 19 | 6 | $(A_x,B_x,B_y,A_z,A_y,B_z)$ |
| D3Q27 | 27 | 7 | $(A_x,B_x,B_y,A_z,A_y,B_z,T)$ |

The population order is the row order of the canonical `operators/Lhat_DQ*.csv`, which is also the order used by Experiments II and III.

## 4. Accepted population representations

- **Shape `(Q,)`, `(M, Q)`, `(Nx, Ny, Nz, Q)`, or any `(..., Q)`.** The computation is fully vectorized, with no Python loop over nodes. The Experiment II/III solvers store `f[i, z, y, x]`; pass `np.moveaxis(f, 0, -1)`.
- **Real inputs** are computed in float64. Integer and float32 arrays are promoted.
- **Complex inputs**, such as Fourier amplitudes, are computed in complex128, and the result is complex-linear. They are accepted by the three kinetic and non-equilibrium interfaces. `reconstruct_from_populations` requires real populations, because density, velocity and equilibrium are nonlinear in $\mathbf f$.
- **Rejected inputs:** unsupported lattice names (case-sensitive), wrong last-axis length, 0-d arrays, nonfinite values, boolean/string/object arrays, non-real or non-finite `tau`/`rho0`, $\tau\le1/2$, $\rho_0\le0$, and nonpositive local density.
- **Inputs are never modified.** Read-only arrays are accepted.

## 5. Implementation

- $R^H_Q$ is built once per lattice from the verified exact module (`curvature_reconstruction_exact.hermite_reconstruction`) and converted to float64, with each exact rational rounded once. It is cached as a read-only array.
- $P_{K,Q}$ uses the definition and Hermite column order of `load_lattice()` in Experiment II. It is evaluated exactly from the canonical velocities and weights, then converted once. It agrees with Experiment II's floating-point projector to within $10^{-12}$.
- SymPy is used only in these one-time cached builders. Every reconstruction call is a NumPy matrix product.
- **Property used in the tests:** $R^H_Q\Phi_{\le2}=0$ exactly. Consequently $R^H_QP_{K,Q}=R^H_Q$, and $R^H_Q\mathbf f^{eq}=0$ for the second-order equilibrium, whose populations are $w_i$ times a polynomial of degree ≤ 2 in $\mathbf c_i$. In exact arithmetic `reconstruct_from_populations(f)` therefore equals $R^H_Q\mathbf f/C_0$. The equilibrium subtraction and the projection are kept so that $\mathbf g_K$ is the physical kinetic signal, and so that large equilibrium contributions are removed before the final contraction. `nonequilibrium_populations` and `project_kinetic` are tested separately, because their effect does not show up in the reconstructed output.

## 6. Exact CE2 reconstruction vs actual LBM populations

For synthetic data $\mathbf g=C_0\widehat L_Q\mathbf q$, all four interfaces return $D_Q\mathbf q$ to round-off. This is the identity $R^H_Q\widehat L_Q=D_Q$ of Step 2A, now in floating point.

Real LBM populations are not purely CE2. They contain higher-order Chapman–Enskog terms, finite-wavelength corrections, nonlinear contributions and transients. `reconstruct_from_populations` is therefore a **CE2-based curvature estimator**: it is asymptotically valid under the linearized, long-wavelength assumptions of the manuscript, and it is not an exact derivative operator for arbitrary nonlinear flows. How close it comes to the true curvature functionals on evolved populations is the subject of the following experiments.

**Floating-point note:** `reconstruct_from_populations` forms $\mathbf f-\mathbf f^{eq}$, a difference of $O(w_i)$ numbers. Its relative accuracy is roughly machine epsilon × $|\mathbf f|/|\mathbf g_K|$: about $10^{-13}$ for kinetic perturbations of order $10^{-4}$, and about $10^{-11}$ for perturbations of order $10^{-6}$. The kinetic and non-equilibrium interfaces do not have this cancellation.

## 7. Minimal example

```python
import importlib.util, sys
import numpy as np

spec = importlib.util.spec_from_file_location("curvature_reconstruction", "src/curvature_reconstruction.py")
cr = importlib.util.module_from_spec(spec); sys.modules["curvature_reconstruction"] = cr; spec.loader.exec_module(cr)

lattice, tau, rho0 = "D3Q19", 0.8, 1.0
# f_solver: populations from a solver, shape (Q, Nz, Ny, Nx) as in Experiments II/III
f = np.moveaxis(f_solver, 0, -1)                       # -> (Nz, Ny, Nx, 19)
d = cr.reconstruct_from_populations(f, lattice, tau, rho0)   # -> (Nz, Ny, Nx, 6)
A_x = d[..., cr.observable_names(lattice).index("A_x")]
```

## 8. Verification

```bash
python -m unittest discover -s tests     # Steps 1, 2A, 2B
python scripts/check_repository.py
```

Maximum relative residuals of $\mathbf d^{rec}$ against the exact $D_Q\mathbf q$, for synthetic CE2 data. The tests use five deterministic $\mathbf q$, $\tau\in\{0.55,0.8,1.0,1.9\}$ and $\rho_0\in\{0.5,1,3\}$. Measured with Python 3.10.12 and NumPy 2.2.6; the last digit can vary with the BLAS/NumPy build.

| Lattice | normalized | kinetic | non-equilibrium | populations |
|---|---|---|---|---|
| D3Q15 | 2.6e-16 | 3.9e-16 | 3.9e-16 | 2.8e-13 |
| D3Q19 | 3.2e-16 | 4.3e-16 | 4.3e-16 | 3.8e-13 |
| D3Q27 | 2.2e-16 | 4.3e-16 | 4.3e-16 | 7.0e-13 |

The population-path residuals reflect the cancellation described in §6. The smallest test vector has entries of order $10^{-3}$.
