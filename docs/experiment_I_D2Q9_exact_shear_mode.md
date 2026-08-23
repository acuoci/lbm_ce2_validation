# Numerical Experiment I — D2Q9 Exact Shear-Mode LBM

## 1. Context and purpose

The theoretical analysis predicts that, for the transverse shear branch of the linearized D2Q9 BGK lattice Boltzmann scheme, the ratio between the kinetic and hydrodynamic non-equilibrium components behaves at small wavenumber as

\[
\eta_K^{\mathrm{pop}}(k,\theta,\tau)
=
A(\theta,\tau)\,k
\left[
1+B(\theta,\tau)k^2+O(k^4)
\right].
\]

Here:

- \(k=|\mathbf k|\) is the physical lattice wavenumber;
- \(\theta\) is the propagation angle of \(\mathbf k\);
- \(\tau\) is the BGK relaxation time;
- \(A(\theta,\tau)\) is the leading CE2 transfer coefficient;
- \(B(\theta,\tau)\) is the finite-wavenumber correction derived from the exact D2Q9 amplification matrix.

The purpose of this experiment is to verify that the above relation is recovered from **actual time-evolved D2Q9 populations**, rather than only from symbolic perturbation theory.

The experiment is intentionally minimal. It uses a single periodic transverse shear eigenmode, no forcing, no boundaries, no nonlinear flow physics, and a perturbation amplitude small enough to remain in the linear regime.

The numerical test therefore addresses one precise question:

\[
\boxed{
\text{Does the kinetic-to-stress ratio measured from the full LBM evolution converge to the analytical }A\text{ and }B\text{ coefficients?}
}
\]

---

## 2. Governing discrete model

Use the standard D2Q9 BGK lattice Boltzmann equation in lattice units,

\[
f_i(\mathbf x+\mathbf c_i,t+1)
=
f_i(\mathbf x,t)
-
\frac{1}{\tau}
\left[
f_i(\mathbf x,t)-f_i^{eq}(\mathbf x,t)
\right].
\]

The D2Q9 velocities are

\[
\mathbf c_0=(0,0),
\]

\[
\mathbf c_{1-4}
=
(1,0),(0,1),(-1,0),(0,-1),
\]

\[
\mathbf c_{5-8}
=
(1,1),(-1,1),(-1,-1),(1,-1),
\]

with weights

\[
w_0=\frac49,
\qquad
w_{1-4}=\frac19,
\qquad
w_{5-8}=\frac1{36},
\]

and

\[
c_s^2=\frac13.
\]

The equilibrium distribution is

\[
f_i^{eq}
=
w_i\rho
\left[
1+
\frac{\mathbf c_i\cdot\mathbf u}{c_s^2}
+
\frac{(\mathbf c_i\cdot\mathbf u)^2}{2c_s^4}
-
\frac{|\mathbf u|^2}{2c_s^2}
\right].
\]

For the actual verification, the perturbation amplitude should be sufficiently small that the dynamics remain effectively linear.

---

## 3. Exact Fourier amplification matrix

For a Fourier mode

\[
\delta f_i(\mathbf x,t)
=
\widehat f_i(t)e^{i\mathbf k\cdot\mathbf x},
\]

the linearized D2Q9 dynamics are governed by

\[
\widehat{\mathbf f}(t+1)
=
\mathsf A(\mathbf k;\tau)
\widehat{\mathbf f}(t),
\]

with

\[
\boxed{
\mathsf A(\mathbf k;\tau)
=
\mathsf S(\mathbf k)\,
\mathsf C(\tau).
}
\]

The streaming matrix is diagonal,

\[
\mathsf S_{ii}
=
e^{-i\mathbf k\cdot\mathbf c_i},
\]

and the linearized BGK collision matrix is

\[
\mathsf C(\tau)
=
\left(1-\frac1\tau\right)I
+
\frac1\tau P_{eq},
\]

where

\[
(P_{eq})_{ij}
=
w_i
\left[
1+
\frac{\mathbf c_i\cdot\mathbf c_j}{c_s^2}
\right].
\]

The numerical implementation should construct this matrix directly for each \((\mathbf k,\tau)\).

---

## 4. Identification of the transverse shear eigenmode

At \(k\rightarrow0\), the transverse hydrodynamic eigenvector tends to

\[
[v_0]_i
=
w_i
\frac{\mathbf c_i\cdot\mathbf e_\perp}{c_s^2},
\]

where

\[
\mathbf e_\perp
=
(-\sin\theta,\cos\theta),
\]

and

\[
\mathbf k
=
k(\cos\theta,\sin\theta).
\]

For finite \(k\), compute all nine eigenpairs of

\[
\mathsf A(\mathbf k;\tau).
\]

The transverse shear eigenvector should be selected using its **hydrodynamic content**, rather than eigenvalue continuity alone.

A robust selection criterion is:

1. normalize each right eigenvector \(v_j\);
2. compute its momentum perturbation

\[
\delta\mathbf j_j
=
\sum_i \mathbf c_i v_{j,i};
\]

3. compute the normalized transverse score

\[
S_j
=
\frac{
|\delta\mathbf j_j\cdot\mathbf e_\perp|
}{
\|\delta\mathbf j_j\|+\epsilon
};
\]

4. among eigenvectors with eigenvalues close to the hydrodynamic branch, select the one with the largest transverse score and negligible density perturbation.

For sufficiently small \(k\), this branch is unambiguous.

The phase and scalar normalization of the eigenvector are arbitrary and do not affect the ratio \(\eta_K^{\mathrm{pop}}\).

---

## 5. Lattice-compatible wavevectors

Because the time-stepping experiment is performed on a periodic lattice of size \(N_x\times N_y\), use integer Fourier modes:

\[
k_x=\frac{2\pi m_x}{N_x},
\qquad
k_y=\frac{2\pi m_y}{N_y}.
\]

Thus

\[
k
=
\sqrt{k_x^2+k_y^2},
\qquad
\theta
=
\operatorname{atan2}(k_y,k_x).
\]

No interpolation or phase reconstruction is then required.

Recommended symmetry-distinct directions are:

\[
(m_x,m_y)=(m,0)
\]

for an axial direction,

\[
(m_x,m_y)=(2m,m)
\]

for

\[
\theta=\arctan(1/2),
\]

and

\[
(m_x,m_y)=(m,m)
\]

for the diagonal direction,

\[
\theta=\frac{\pi}{4}.
\]

By changing \(N\) and/or the integer multiplier \(m\), a sequence of small \(k\) values can be generated while preserving the same propagation angle.

---

## 6. Exact eigenmode initialization

The principal verification should avoid kinetic startup transients.

Let \(v_s(\mathbf k,\tau)\) be the selected complex shear eigenvector of the exact amplification matrix.

Initialize the real-space populations as

\[
\boxed{
f_i(\mathbf x,0)
=
w_i\rho_0
+
\varepsilon
\operatorname{Re}
\left[
v_{s,i}
e^{i\mathbf k\cdot\mathbf x}
\right].
}
\]

Recommended values are

\[
\rho_0=1,
\qquad
\varepsilon=10^{-6}
\]

for the principal linear test.

This initialization is preferable to setting

\[
f_i=f_i^{eq}(\rho,\mathbf u),
\]

because equilibrium initialization contains no initial non-equilibrium component and produces transient adjustment before the exact shear eigenstructure is established.

The exact eigenmode initialization should produce a single decaying discrete eigenmode:

\[
\widehat{\mathbf f}(t)
=
\lambda_s^t
\widehat{\mathbf f}(0).
\]

Consequently, the normalized kinetic ratio should remain constant in time to round-off accuracy.

This time invariance is itself a useful implementation check.

---

## 7. Time integration

Only a short run is needed.

A typical choice is

\[
N_t=10\text{--}20
\]

time steps.

At every step:

1. compute \(\rho\) and \(\mathbf u\);
2. evaluate the quadratic equilibrium;
3. perform BGK collision;
4. stream periodically;
5. extract the Fourier coefficient at the initialized wavevector;
6. evaluate the kinetic observability diagnostics.

Because the initial state is an exact linear eigenmode, no long-time statistical averaging is required.

The perturbation amplitude should be monitored to confirm exponential decay with the eigenvalue predicted by the amplification matrix.

---

## 8. Fourier-space extraction of the population mode

The diagnostic should be evaluated from the complex Fourier coefficient of each population.

For each \(i\),

\[
\widehat f_i(\mathbf k,t)
=
\frac{1}{N_xN_y}
\sum_{\mathbf x}
\left[
f_i(\mathbf x,t)-w_i\rho_0
\right]
e^{-i\mathbf k\cdot\mathbf x}.
\]

The use of Fourier coefficients is preferable to a pointwise real-space diagnostic because the discrete eigenvector is generally complex and different populations may carry different phases.

Define the Fourier-space equilibrium perturbation from the measured density and momentum mode, or equivalently use the linear equilibrium projector:

\[
\widehat{\mathbf f}^{eq}
=
P_{eq}\widehat{\mathbf f}.
\]

Then

\[
\boxed{
\widehat{\mathbf f}^{neq}
=
(I-P_{eq})\widehat{\mathbf f}.
}
\]

For the small-amplitude verification, this linear Fourier-space definition is the cleanest diagnostic.

---

## 9. Hydrodynamic and kinetic projectors

Define the D2Q9 weighted population metric

\[
\|g\|_{w,9}^2
=
g^\dagger W_9^{-1}g.
\]

Let

\[
P_{\le1,9}
\]

be the weighted projector onto density and momentum, and

\[
P_{\le2,9}
\]

the weighted projector onto Hermite orders \(0,1,2\).

Then

\[
P_{2,9}
=
P_{\le2,9}-P_{\le1,9},
\]

and

\[
P_{K,9}
=
I-P_{\le2,9}.
\]

For the Fourier non-equilibrium vector,

\[
\boxed{
h_s
=
P_{2,9}\widehat{\mathbf f}^{neq},
}
\]

\[
\boxed{
k_s
=
P_{K,9}\widehat{\mathbf f}^{neq}.
}
\]

The primary numerical observable is

\[
\boxed{
\eta_K^{LBM}
=
\frac{
\|k_s\|_{w,9}
}{
\|h_s\|_{w,9}
}.
}
\]

For an exact eigenmode this value should be independent of time, apart from floating-point and nonlinear-amplitude effects.

---

## 10. Analytical coefficients to be tested

Use the theoretical notation

\[
q
=
\cos^2\theta\sin^2\theta,
\qquad
\xi
=
2\tau-1.
\]

The leading coefficient is

\[
\boxed{
A^2(\theta,\tau)
=
\frac{\xi^2}{6}(1-3q).
}
\]

For \(\tau>1/2\),

\[
A(\theta,\tau)
=
\frac{\xi}{\sqrt6}
\sqrt{1-3q}.
\]

The next-order coefficient is

\[
\boxed{
B(\theta,\tau)
=
\frac{
\xi^4
+
q(-21\xi^4+10\xi^2+1)
+
q^2(72\xi^4-42\xi^2-4)
}{
12\xi^2(1-3q)
}.
}
\]

The numerical experiment should test

\[
\boxed{
\eta_K^{LBM}
=
Ak
\left[
1+Bk^2+O(k^4)
\right].
}
\]

---

## 11. Recommended parameter set

A compact but sufficient parameter sweep is:

### Relaxation times

\[
\tau
=
0.9,\quad1.0,\quad1.1.
\]

These values sample:

- a case inside the globally non-negative-\(B\) interval;
- the particularly convenient \(\tau=1\) case;
- a case outside the global \(B\ge0\) interval.

### Angles

Use three lattice-compatible directions:

\[
\theta=0,
\]

\[
\theta=\arctan(1/2),
\]

\[
\theta=\frac{\pi}{4}.
\]

### Wavenumbers

Use approximately 6–8 values satisfying

\[
0.05\lesssim k\lesssim0.5.
\]

The smallest values should establish the asymptotic limit; the larger values show where the cubic approximation begins to lose accuracy.

A practical way to preserve a fixed direction is to keep \((m_x:m_y)\) constant and vary the domain size \(N\).

### Perturbation amplitude

Principal tests:

\[
\varepsilon=10^{-6}.
\]

A minimal amplitude-sensitivity check can be performed for one representative case with

\[
\varepsilon
=
10^{-6},\quad
10^{-4},\quad
10^{-2}.
\]

No full Mach-number study is required.

---

## 12. Primary post-processing diagnostics

### 12.1 Leading-order coefficient

Define

\[
R_A(k)
=
\frac{\eta_K^{LBM}}{k}.
\]

The expected limit is

\[
\boxed{
R_A(k)\rightarrow A(\theta,\tau)
\qquad
k\rightarrow0.
}
\]

A plot of \(R_A\) against \(k^2\) should therefore have intercept \(A\).

---

### 12.2 Direct test of the finite-\(k\) coefficient

Define

\[
\boxed{
R_B(k)
=
\frac{
\eta_K^{LBM}/[A(\theta,\tau)k]-1
}{
k^2
}.
}
\]

The theoretical prediction is

\[
\boxed{
R_B(k)
=
B(\theta,\tau)+O(k^2).
}
\]

Therefore,

\[
R_B(k)\rightarrow B
\]

as \(k\rightarrow0\).

This is the most direct numerical verification of the symbolic coefficient \(B\).

---

### 12.3 Error of the cubic approximation

Define

\[
\eta_{AB}
=
Ak(1+Bk^2),
\]

and

\[
\boxed{
E_{AB}(k)
=
\frac{
|\eta_K^{LBM}-\eta_{AB}|
}{
|\eta_K^{LBM}|
}.
}
\]

Because

\[
\eta_K
=
Ak[1+Bk^2+O(k^4)],
\]

the relative error should behave as

\[
\boxed{
E_{AB}(k)=O(k^4)
}
\]

in the asymptotic regime.

A log-log plot should therefore approach a slope of four.

---

### 12.4 Exact-eigenmode temporal invariance

At fixed \((k,\theta,\tau)\), define

\[
E_t
=
\frac{
|\eta_K^{LBM}(t)-\eta_K^{LBM}(0)|
}{
|\eta_K^{LBM}(0)|
}.
\]

For a sufficiently small perturbation initialized with the exact shear eigenvector,

\[
E_t
\]

should remain close to machine precision over the short run.

This is primarily an implementation check and need not become a paper figure unless unexpected behavior occurs.

---

## 13. Minimal amplitude check

The theory is a linearized tangent result.

For one representative case, e.g.

\[
\tau=1,
\qquad
\theta=\arctan(1/2),
\qquad
k\approx0.2,
\]

repeat the simulation for

\[
\varepsilon
=
10^{-6},\quad10^{-4},\quad10^{-2}.
\]

Compute

\[
E_\varepsilon
=
\frac{
|\eta_K(\varepsilon)-\eta_K(10^{-6})|
}{
|\eta_K(10^{-6})|
}.
\]

Expected behavior:

- \(10^{-6}\) and \(10^{-4}\) should be indistinguishable up to numerical precision;
- \(10^{-2}\) may show a measurable finite-amplitude deviation.

This is sufficient to demonstrate that the principal validation is genuinely in the tangent regime without adding a separate finite-Mach study.

---

## 14. Minimal figures

### Figure 1 — finite-wavenumber transfer relation

Plot

\[
\frac{\eta_K^{LBM}}{A k}
\]

against

\[
k^2.
\]

Overlay

\[
1+B k^2.
\]

Use the three propagation directions for one representative \(\tau\), preferably

\[
\tau=1.
\]

This figure simultaneously shows the leading normalization and angular finite-\(k\) correction.

---

### Figure 2 — asymptotic coefficient and truncation error

Preferred option: two panels.

Panel (a):

\[
R_B(k)
=
\frac{
\eta_K^{LBM}/(Ak)-1
}{
k^2
}
\]

versus \(k^2\), with horizontal lines at the exact analytical \(B\).

Panel (b):

\[
E_{AB}(k)
\]

versus \(k\) on log-log axes, with a reference \(k^4\) slope.

If the paper must remain extremely compact, Figure 1 and panel (a) alone may be sufficient.

---

## 15. Minimal numerical table

A compact table should report fitted asymptotic coefficients.

Suggested columns:

| \(\tau\) | direction | \(A_{\mathrm{exact}}\) | \(A_{\mathrm{fit}}\) | relative error | \(B_{\mathrm{exact}}\) | \(B_{\mathrm{fit}}\) | relative error |
|---:|---|---:|---:|---:|---:|---:|---:|

The fit should use only the smallest \(k\) values where the asymptotic expansion is demonstrably converged.

The table is more useful than listing raw simulation values.

---

## 16. Fitting procedure

For each \((\theta,\tau)\), fit

\[
\frac{\eta_K^{LBM}}{k}
=
A
+
AB\,k^2
+
Ck^4.
\]

A simple least-squares fit over the smallest 4–6 wavenumbers is sufficient.

Then

\[
A_{\mathrm{fit}}
=
\text{intercept},
\]

and

\[
B_{\mathrm{fit}}
=
\frac{
\text{coefficient of }k^2
}{
A_{\mathrm{fit}}
}.
\]

Including the \(k^4\) term reduces bias from the next neglected asymptotic correction.

The fitted values should be compared with the analytical \(A\) and \(B\).

---

## 17. Numerical precision and reproducibility

Use:

- double precision for the LBM evolution;
- complex double precision for Fourier coefficients and eigenvectors;
- direct dense eigensolution of the \(9\times9\) amplification matrix;
- deterministic mode ordering and branch-selection criteria;
- exact velocity and weight tables shared with the theoretical symbolic scripts.

All projector matrices should be generated once from the exact D2Q9 velocity set and weights.

The complete experiment should be reproducible with a small Python/NumPy implementation.

No external LBM library is required.

---

## 18. Important implementation checks

Before producing paper-quality results, verify:

1. mass perturbation of the selected shear mode is negligible;
2. momentum perturbation is transverse:
   \[
   \mathbf k\cdot\delta\mathbf j\approx0;
   \]
3. the measured decay factor agrees with the selected eigenvalue:
   \[
   \widehat f(t+1)/\widehat f(t)\approx\lambda_s;
   \]
4. \(\eta_K^{LBM}\) is constant in time for exact-eigenmode initialization;
5. results are independent of domain size when \(k\) is held fixed;
6. results are independent of perturbation amplitude in the linear regime;
7. the Fourier transform convention is consistent with the sign used in
   \[
   \mathsf S_{ii}=e^{-i\mathbf k\cdot\mathbf c_i}.
   \]

These checks should be automated as assertions whenever possible.

---

## 19. Interpretation criteria

The experiment should be considered successful if the following are observed:

### Leading CE2 relation

\[
\frac{\eta_K^{LBM}}{k}
\rightarrow
A(\theta,\tau).
\]

### Next-order finite-\(k\) correction

\[
R_B(k)
\rightarrow
B(\theta,\tau).
\]

### Correct asymptotic truncation behavior

\[
E_{AB}\sim k^4.
\]

### Linear-amplitude independence

\[
\eta_K^{LBM}
\]

is unchanged as \(\varepsilon\rightarrow0\).

These four observations would constitute direct numerical evidence that the analytical kinetic-observability transfer function is present in the actual D2Q9 BGK dynamics.

---

## 20. What this experiment does not attempt to establish

This experiment deliberately does **not** test:

- turbulence;
- finite-Mach nonlinear populations;
- boundary effects;
- forcing schemes;
- MRT or cumulant collision;
- under-resolution sensing;
- adaptive mesh refinement;
- the HIT \(P/\Omega\) identity.

Its sole purpose is the clean numerical verification of the D2Q9 spectral corollary derived in the theoretical section.

Keeping this experiment narrowly focused is essential: any discrepancy can then be attributed to the spectral theory, implementation, or asymptotic truncation rather than to unrelated flow physics.

---

## 21. Expected role in the paper

The numerical subsection can be compact.

A suitable title is:

> **Numerical verification of the D2Q9 shear-mode transfer function**

The associated text should emphasize that the exact eigenmode initialization isolates the discrete shear branch and that the comparison is performed against the complete nonlinear collision-streaming implementation, not by evaluating the analytical amplification matrix alone.

The main scientific result of the experiment should be stated in terms of asymptotic convergence:

\[
\boxed{
\eta_K^{LBM}
=
A(\theta,\tau)k
\left[
1+B(\theta,\tau)k^2+O(k^4)
\right]
}
\]

with the analytical coefficients recovered numerically from time-evolved lattice populations.
