# Numerical Experiment I — D2Q9 Exact Shear-Mode LBM

## 1. Objective

Experiment I verifies the finite-wavenumber transverse-shear relation predicted for the linearized D2Q9 BGK lattice Boltzmann scheme:

$$
\eta_K^{\mathrm{pop}}=
A(\theta,\tau)k
\left[
1+B(\theta,\tau)k^2+O(k^4)
\right].
$$

Here:

- $k=|𝐤|$ is the lattice wavenumber;
- $\theta$ is the propagation angle;
- $\tau$ is the BGK relaxation time;
- $A(\theta,\tau)$ is the leading CE2 transfer coefficient;
- $B(\theta,\tau)$ is the finite-wavenumber correction obtained from the exact D2Q9 amplification matrix.

The purpose of the experiment is to recover these coefficients from **time-evolved D2Q9 populations**, rather than from the analytical amplification matrix alone.

The calculation is deliberately minimal: a single periodic transverse shear eigenmode, no forcing, no boundaries, and a perturbation amplitude sufficiently small to remain in the linear regime.

---

## 2. D2Q9 BGK model

The lattice Boltzmann equation is

$$
f_i(𝐱+𝐜_i,t+1)=
f_i(𝐱,t)-
\frac{1}{\tau}
\left[
f_i(𝐱,t)-f_i^{eq}(𝐱,t)
\right].
$$

The D2Q9 velocities are

$$
𝐜_0=(0,0),
$$

$$
𝐜_{1-4}=
(1,0),\ (0,1),\ (-1,0),\ (0,-1),
$$

$$
𝐜_{5-8}=
(1,1),\ (-1,1),\ (-1,-1),\ (1,-1).
$$

The corresponding weights are

$$
w_0=\frac{4}{9},
\qquad
w_{1-4}=\frac{1}{9},
\qquad
w_{5-8}=\frac{1}{36},
$$

with

$$
c_s^2=\frac{1}{3}.
$$

The quadratic isothermal equilibrium is

$$
f_i^{eq}=
w_i\rho
\left[
1+
\frac{𝐜_i\cdot𝐮}{c_s^2}+
\frac{(𝐜_i\cdot𝐮)^2}{2c_s^4}-
\frac{|𝐮|^2}{2c_s^2}
\right].
$$

---

## 3. Exact Fourier amplification matrix

For a Fourier perturbation

$$
\delta f_i(𝐱,t)=
\widehat f_i(t)e^{i𝐤\cdot𝐱},
$$

the linearized dynamics satisfy

$$
\widehat{𝐟}(t+1)=
\mathsf A(𝐤;\tau)\widehat{𝐟}(t),
$$

with

$$
\mathsf A(𝐤;\tau)=
\mathsf S(𝐤)\mathsf C(\tau).
$$

The streaming matrix is diagonal:

$$
\mathsf S_{ii}=
e^{-i𝐤\cdot𝐜_i}.
$$

The BGK collision matrix is

$$
\mathsf C(\tau) =
\left(1-\frac{1}{\tau}\right)I
+
\frac{1}{\tau}P_{eq},
$$

where

$$
(P_{eq})_{ij} =
w_i
\left[
1+
\frac{𝐜_i\cdot𝐜_j}{c_s^2}
\right].
$$

This matrix is constructed directly for every $(𝐤,\tau)$ pair.

---

## 4. Transverse shear eigenmode

As $k\rightarrow0$, the transverse hydrodynamic eigenvector approaches

$$
[v_0]_i =
w_i
\frac{𝐜_i\cdot𝐞_\perp}{c_s^2},
$$

with

$$
𝐞_\perp =
(-\sin\theta,\cos\theta),
$$

and

$$
𝐤 =
k(\cos\theta,\sin\theta).
$$

At finite $k$, all nine eigenpairs of $\mathsf A(𝐤;\tau)$ are computed.

The transverse shear branch is selected from its hydrodynamic content. For each right eigenvector $v_j$, define the momentum perturbation

$$
\delta𝐣_j =
\sum_i 𝐜_i v_{j,i},
$$

and the normalized transverse score

$$
S_j =
\frac{
|\delta𝐣_j\cdot𝐞_\perp|
}{
\|\delta𝐣_j\|+\epsilon
}.
$$

The selected mode must have:

- large transverse score;
- negligible density perturbation;
- negligible longitudinal momentum;
- an eigenvalue belonging to the hydrodynamic shear branch.

The arbitrary phase and scalar normalization of the eigenvector do not affect the population-norm ratio used below.

---

## 5. Lattice-compatible wavevectors

On a periodic $N_x\times N_y$ lattice,

$$
k_x=\frac{2\pi m_x}{N_x},
\qquad
k_y=\frac{2\pi m_y}{N_y},
$$

so that

$$
k=\sqrt{k_x^2+k_y^2},
$$

and

$$
\theta=\mathrm{atan2}(k_y,k_x).
$$

The production calculations use three lattice-compatible direction families:

| Family | Integer mode | Propagation angle |
|---|---|---|
| $[10]$ | $(m,0)$ | $\theta=0$ |
| $[21]$ | $(2m,m)$ | $\theta=\arctan(1/2)$ |
| $[11]$ | $(m,m)$ | $\theta=\pi/4$ |

These directions provide three symmetry-distinct orientations while preserving exact periodicity.

---

## 6. Exact eigenmode initialization

To eliminate kinetic startup transients, the simulation is initialized directly with the exact discrete shear eigenmode:

$$
f_i(𝐱,0) =
w_i\rho_0
+
\varepsilon
\mathrm{Re}
\left[
v_{s,i}e^{i𝐤\cdot𝐱}
\right].
$$

The production values are

$$
\rho_0=1,
\qquad
\varepsilon=10^{-6}.
$$

In the linear regime, the initialized state evolves as a single discrete eigenmode:

$$
\widehat{𝐟}(t) =
\lambda_s^t\widehat{𝐟}(0).
$$

Consequently, normalized population diagnostics should remain constant in time up to floating-point and finite-amplitude effects.

---

## 7. Fourier extraction

At each timestep, the Fourier coefficient of population $i$ is evaluated as

$$
\widehat f_i(𝐤,t) =
\frac{1}{N_xN_y}
\sum_{𝐱}
\left[
f_i(𝐱,t)-w_i\rho_0
\right]
e^{-i𝐤\cdot𝐱}.
$$

The linear equilibrium contribution is

$$
\widehat{𝐟}^{eq} =
P_{eq}\widehat{𝐟},
$$

and therefore

$$
\widehat{𝐟}^{neq} =
(I-P_{eq})\widehat{𝐟}.
$$

Using Fourier-space populations avoids ambiguities associated with the population-dependent phases of the discrete eigenvector.

---

## 8. Hydrodynamic and kinetic projectors

The weighted D2Q9 population norm is

$$
\|g\|_{w,9}^2 =
g^\dagger W_9^{-1}g.
$$

Let $P_{\le1,9}$ be the weighted projector onto density and momentum, and let $P_{\le2,9}$ be the projector onto Hermite orders $0$, $1$, and $2$.

Define

$$
P_{2,9} =
P_{\le2,9}-P_{\le1,9},
$$

and

$$
P_{K,9} =
I-P_{\le2,9}.
$$

The hydrodynamic second-order and kinetic components are

$$
h_s =
P_{2,9}\widehat{𝐟}^{neq},
$$

and

$$
k_s =
P_{K,9}\widehat{𝐟}^{neq}.
$$

The primary numerical observable is

$$
\eta_K^{LBM} =
\frac{
\|k_s\|_{w,9}
}{
\|h_s\|_{w,9}
}.
$$

For an exact eigenmode, this ratio should be independent of time in the linear regime.

---

## 9. Analytical coefficients

Introduce

$$
q =
\cos^2\theta\sin^2\theta,
\qquad
\xi =
2\tau-1.
$$

The leading coefficient satisfies

$$
A^2(\theta,\tau) =
\frac{\xi^2}{6}(1-3q).
$$

For $\tau>1/2$,

$$
A(\theta,\tau) =
\frac{\xi}{\sqrt{6}}
\sqrt{1-3q}.
$$

The finite-wavenumber coefficient is

$$
B(\theta,\tau) =
\frac{
\xi^4
+
q(-21\xi^4+10\xi^2+1)
+
q^2(72\xi^4-42\xi^2-4)
}{
12\xi^2(1-3q)
}.
$$

The quantity tested numerically is therefore

$$
\eta_K^{LBM} =
Ak
\left[
1+Bk^2+O(k^4)
\right].
$$

---

## 10. Production parameters

The production configuration used for the manuscript is:

| Parameter | Value |
|---|---|
| Lattice | D2Q9 |
| Domain | $256\times256$ |
| Relaxation times | $\tau=0.9,\ 1.0,\ 1.1$ |
| Direction families | $[10],\ [21],\ [11]$ |
| Harmonics | 8 per direction |
| Perturbation amplitude | $\varepsilon=10^{-6}$ |
| Boundary conditions | Periodic |
| Collision model | BGK |
| Forcing | None |

The complete sweep contains

$$
3\times3\times8=72
$$

independent simulations.

The five smallest wavenumbers in each $(\theta,\tau)$ family are used for asymptotic coefficient recovery.

---

## 11. Primary diagnostics

### Leading-order coefficient

Define

$$
R_A(k) =
\frac{\eta_K^{LBM}}{k}.
$$

The expected limit is

$$
R_A(k)
\rightarrow
A(\theta,\tau)
\qquad
\text{as } k\rightarrow0.
$$

Equivalently, plotting $R_A$ against $k^2$ should approach an intercept equal to $A$.

### Finite-wavenumber coefficient

Define

$$
R_B(k) =
\frac{
\eta_K^{LBM}/[A(\theta,\tau)k]-1
}{
k^2
}.
$$

The asymptotic prediction is

$$
R_B(k) =
B(\theta,\tau)+O(k^2),
$$

so that

$$
R_B(k)\rightarrow B(\theta,\tau)
\qquad
\text{as } k\rightarrow0.
$$

This provides a direct numerical test of the analytical coefficient $B$.

### Cubic-approximation error

Define

$$
\eta_{AB} =
Ak(1+Bk^2),
$$

and

$$
E_{AB}(k) =
\frac{
|\eta_K^{LBM}-\eta_{AB}|
}{
|\eta_K^{LBM}|
}.
$$

Since

$$
\eta_K^{LBM} =
Ak
\left[
1+Bk^2+O(k^4)
\right],
$$

the expected relative truncation error is

$$
E_{AB}(k)=O(k^4).
$$

A log-log fit of $E_{AB}$ against $k$ should therefore approach a slope of four.

---

## 12. Coefficient fitting

For each $(\theta,\tau)$ family, fit the smallest five wavenumbers using

$$
\frac{\eta_K^{LBM}}{k} =
A_{\mathrm{fit}}
+
C_2 k^2
+
C_4 k^4.
$$

The recovered finite-wavenumber coefficient is then

$$
B_{\mathrm{fit}} =
\frac{C_2}{A_{\mathrm{fit}}}.
$$

The fitted values $A_{\mathrm{fit}}$ and $B_{\mathrm{fit}}$ are compared directly with their analytical counterparts.

Including the $k^4$ contribution reduces bias from the next neglected term in the asymptotic expansion.

---

## 13. Amplitude check

The main calculation uses

$$
\varepsilon=10^{-6},
$$

so that the dynamics remain in the tangent regime.

For a representative configuration, the calculation may also be repeated at

$$
\varepsilon =
10^{-6},\ 10^{-4},\ 10^{-2}.
$$

A convenient relative measure is

$$
E_\varepsilon =
\frac{
|\eta_K(\varepsilon)-\eta_K(10^{-6})|
}{
|\eta_K(10^{-6})|
}.
$$

The $10^{-6}$ and $10^{-4}$ results should be indistinguishable within numerical accuracy, while the largest amplitude may show finite-amplitude effects.

This check is auxiliary and is not part of the principal production sweep.

---

## 14. Recommended outputs

### Primary figure

Plot

$$
\frac{\eta_K^{LBM}}{Ak}
$$

against $k^2$ for the three propagation directions at $\tau=1$.

The analytical finite-wavenumber prediction is

$$
1+B(\theta,\tau)k^2.
$$

This representation simultaneously tests the leading normalization and the orientation-dependent finite-$k$ correction.

### Secondary diagnostics

Useful auxiliary outputs are:

- $R_B(k)$ versus $k^2$;
- $E_{AB}(k)$ versus $k$ on log-log axes;
- fitted $A$ and $B$ coefficients for all nine $(\theta,\tau)$ combinations;
- an optional amplitude-sensitivity check.

A compact coefficient table can use the structure:

| $\tau$ | Direction | $A_{\mathrm{exact}}$ | $A_{\mathrm{fit}}$ | Relative error | $B_{\mathrm{exact}}$ | $B_{\mathrm{fit}}$ | Relative error |
|---:|---|---:|---:|---:|---:|---:|---:|

---

## 15. Implementation checks

Before accepting the production results, verify automatically that:

1. the selected mode has negligible density perturbation;
2. its momentum is transverse to the wavevector:

$$
𝐤\cdot\delta𝐣\approx0;
$$

3. the measured modal decay agrees with the selected eigenvalue:

$$
\widehat{𝐟}(t+1)
\approx
\lambda_s\widehat{𝐟}(t);
$$

4. $\eta_K^{LBM}$ is constant in time for exact-eigenmode initialization;
5. reducing the perturbation amplitude does not change the normalized result;
6. the Fourier convention is consistent with

$$
\mathsf S_{ii} =
e^{-i𝐤\cdot𝐜_i}.
$$

These checks should be implemented as assertions or retained as diagnostic outputs.

---

## 16. Interpretation

Experiment I is successful if the numerical populations recover all three asymptotic properties:

$$
\frac{\eta_K^{LBM}}{k}
\rightarrow
A(\theta,\tau),
$$

$$
R_B(k)
\rightarrow
B(\theta,\tau),
$$

and

$$
E_{AB}(k)\sim k^4.
$$

Together with amplitude independence in the linear regime, these results establish that

$$
\eta_K^{LBM} =
A(\theta,\tau)k
\left[
1+B(\theta,\tau)k^2+O(k^4)
\right]
$$

is recovered directly from the full D2Q9 BGK collision-streaming dynamics.

The experiment does not address turbulence, boundaries, forcing, nonlinear finite-Mach populations, alternative collision models, or adaptive-resolution effects. Its purpose is deliberately narrow: **to verify the D2Q9 finite-wavenumber shear transfer relation and its analytical coefficients directly from evolved lattice populations.**
