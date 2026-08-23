# Experiment III — Production Protocol Refinement

## 1. Objective

This study defines the production protocol for the Taylor–Green experiment used to test the CE2 population operator in a weakly nonlinear, non-modal three-dimensional flow.

Three practical choices must be established:

1. the startup interval to discard after equilibrium initialization;
2. the final nondimensional simulation time;
3. the grid resolution required to expose a clean amplitude dependence without unnecessary computational cost.

Experiment III measures the **raw kinetic population**

$$
g_Q^{LBM} = P_{K,Q}(f-f^{eq}),
$$

without subtracting a finite-amplitude CE1 contribution. Consequently, the measured discrepancy from the linear CE2 prediction contains both finite-wavenumber and finite-amplitude effects. Grid refinement alone is therefore not expected to reduce the normalized error monotonically.

---

## 2. Startup transient

The populations are initialized from local equilibrium. Therefore,

$$
g_Q^{LBM}(t=0)=0,
$$

whereas the CE2 curvature response is already nonzero. A short kinetic transient is consequently unavoidable.

A D3Q19 pilot calculation at $N=32$, $U_0=0.02$, and $\tau=0.8$ gives:

| Step | $\mathcal E_{19}$ | $\mathcal R_{19}$ | $\mathcal X_{19}$ |
|---:|---:|---:|---:|
| 0  | 1.0000 | 0.0000 | 0.0368 |
| 1  | 1.1248 | 2.1114 | 0.9929 |
| 2  | 0.4791 | 0.5902 | 0.9478 |
| 3  | 0.2422 | 1.1745 | 0.9880 |
| 5  | 0.1709 | 1.0373 | 0.9866 |
| 8  | 0.1639 | 1.0264 | 0.9873 |
| 10 | 0.1604 | 1.0261 | 0.9878 |
| 15 | 0.1517 | 1.0243 | 0.9891 |
| 20 | 0.1429 | 1.0219 | 0.9902 |

The strongest startup oscillations have disappeared by approximately step 5. Step 10 provides a more conservative cutoff: the norm ratio is already close to unity and the population alignment subsequently evolves smoothly.

The adopted startup discard is therefore

$$
n_{\mathrm{discard}} = 10.
$$

---

## 3. Grid-size study

A D3Q19 grid study was performed at

$$
N = 32,\ 48,\ 64,
$$

with $\tau=0.8$ and two amplitudes, $U_0=0.01$ and $0.04$. Diagnostics were evaluated at step 10.

### Low amplitude: $U_0=0.01$

| $N$ | $\mathcal E_{19}$ | $\mathcal R_{19}$ | $\mathcal X_{19}$ |
|---:|---:|---:|---:|
| 32 | 0.0811 | 1.0161 | 0.9969 |
| 48 | 0.1272 | 1.0138 | 0.9921 |
| 64 | 0.1729 | 1.0180 | 0.9855 |

### Higher amplitude: $U_0=0.04$

| $N$ | $\mathcal E_{19}$ | $\mathcal R_{19}$ | $\mathcal X_{19}$ |
|---:|---:|---:|---:|
| 32 | 0.3199 | 1.0652 | 0.9540 |
| 48 | 0.5081 | 1.1290 | 0.8930 |
| 64 | 0.6917 | 1.2196 | 0.8236 |

At fixed $U_0$, the relative CE2 error **increases** with $N$. This is not evidence of a spatial-resolution failure. It follows from the different scaling of the target CE2 signal and the finite-amplitude kinetic contribution retained in the raw population.

The CE2 curvature response scales as

$$
g_Q^{CE2} = O(U_0 k^2),
$$

whereas a nonlinear CE1-type contribution may scale as

$$
g_Q^{NL} = O(U_0^2 k).
$$

Their relative magnitude therefore scales as

$$
\frac{\lVert g_Q^{NL}\rVert}{\lVert g_Q^{CE2}\rVert} = O\left(\frac{U_0}{k}\right).
$$

For the fundamental Taylor–Green mode,

$$
k \propto \frac{1}{N},
$$

so that, at fixed $U_0$,

$$
\mathcal E_Q^{NL} \sim U_0 N.
$$

Using the characteristic Taylor–Green lattice wavenumber

$$
k_{TGV} = \frac{2\pi\sqrt{3}}{N},
$$

the quantity

$$
\frac{\mathcal E_{19}k_{TGV}}{U_0}
$$

is approximately constant across the three grids for the low-amplitude cases, supporting this interpretation.

Therefore, increasing the resolution to $N=64$ does not improve the intended raw-population comparison. The intermediate grid $N=48$ provides a better balance between numerical resolution, separation of the CE2 signal from finite-amplitude contributions, and computational cost.

---

## 4. Amplitude dependence at $N=48$

The amplitude dependence was examined at $N=48$ and $\tau=0.8$ using

$$
U_0 = 0.005,\ 0.01,\ 0.02,\ 0.04.
$$

Diagnostics were evaluated immediately after the startup interval.

### D3Q19

| $U_0$ | $\mathcal E_{19}$ | $\mathcal R_{19}$ | $\mathcal X_{19}$ |
|---:|---:|---:|---:|
| 0.005 | 0.0638 | 1.0077 | 0.9980 |
| 0.010 | 0.1272 | 1.0138 | 0.9921 |
| 0.020 | 0.2541 | 1.0378 | 0.9696 |
| 0.040 | 0.5081 | 1.1290 | 0.8930 |

A log-log fit gives

$$
\mathcal E_{19} \propto U_0^{0.998}.
$$

### D3Q27

| $U_0$ | $\mathcal E_{27}$ | $\mathcal R_{27}$ | $\mathcal X_{27}$ |
|---:|---:|---:|---:|
| 0.005 | 0.0814 | 1.0089 | 0.9968 |
| 0.010 | 0.1624 | 1.0188 | 0.9872 |
| 0.020 | 0.3246 | 1.0573 | 0.9517 |
| 0.040 | 0.6491 | 1.1991 | 0.8409 |

The corresponding fit gives

$$
\mathcal E_{27} \propto U_0^{0.998}.
$$

Both lattices therefore exhibit an essentially linear decrease of the relative population-vector discrepancy as $U_0$ decreases. This behavior is consistent with

$$
\mathcal E_Q^{NL} \sim \frac{U_0}{k}
$$

at fixed grid resolution.

The population norm and direction show the same tangent-limit trend:

$$
\mathcal R_Q \rightarrow 1,
\qquad
\mathcal X_Q \rightarrow 1
\qquad
\text{as } U_0 \rightarrow 0.
$$

---

## 5. Production amplitudes

The $U_0=0.04$ case already exhibits a significant finite-amplitude departure from the CE2 population direction:

$$
\mathcal X_{19} \simeq 0.893,
\qquad
\mathcal X_{27} \simeq 0.841.
$$

It is therefore useful as a stress test, but it is less appropriate for the primary tangent-limit comparison.

The production calculations use

$$
U_0 = 0.005,\ 0.01,\ 0.02.
$$

These amplitudes span a factor of four and provide a clear amplitude trend while retaining good population-space alignment.

The $U_0=0.04$ case may be retained as an optional higher-amplitude test illustrating departure from the tangent regime.

---

## 6. Time-window study

The nondimensional time is defined as

$$
t^* = \frac{U_0 t}{N}.
$$

For D3Q19 at $N=48$, $U_0=0.01$, and $\tau=0.8$:

| Step | $t^*$ | $\mathcal E_{19}$ | $\mathcal R_{19}$ | $\mathcal X_{19}$ |
|---:|---:|---:|---:|---:|
| 10 | 0.00208 | 0.1272 | 1.0138 | 0.9921 |
| 20 | 0.00417 | 0.1209 | 1.0131 | 0.9929 |
| 40 | 0.00833 | 0.1090 | 1.0113 | 0.9942 |
| 60 | 0.01250 | 0.0984 | 1.0105 | 0.9953 |
| 80 | 0.01667 | 0.0889 | 1.0096 | 0.9961 |
| 96 | 0.02000 | 0.0818 | 1.0091 | 0.9967 |

For D3Q27 under the same conditions:

| Step | $t^*$ | $\mathcal E_{27}$ | $\mathcal R_{27}$ | $\mathcal X_{27}$ |
|---:|---:|---:|---:|---:|
| 10 | 0.00208 | 0.1624 | 1.0188 | 0.9872 |
| 40 | 0.00833 | 0.1392 | 1.0149 | 0.9906 |
| 96 | 0.02000 | 0.1044 | 1.0110 | 0.9947 |

After the startup transient, the agreement with the CE2 prediction does not deteriorate over the investigated interval. Instead, the relative error decreases and the population alignment improves as the Taylor–Green field decays.

There is therefore no need to evolve the calculation to an eddy-turnover time or into a strongly broadband regime. The adopted final time is

$$
t^*_{\max} = 0.02.
$$

This interval is sufficient to establish that the post-transient agreement persists while keeping the experiment within a controlled weakly nonlinear regime.

---

## 7. Output schedule

For each amplitude, diagnostics should be stored approximately at

$$
t^* = 0,\ 0.002,\ 0.005,\ 0.010,\ 0.020.
$$

Raw steps $1$, $2$, $5$, and $10$ may additionally be retained for startup diagnostics.

Only results satisfying

$$
t \ge 10
$$

should be used for the quantitative CE2 comparison.

---

## 8. Final production protocol

The adopted production configuration is:

| Parameter | Value |
|---|---|
| Lattices | D3Q19, D3Q27 |
| Grid | $N=48$ |
| Relaxation time | $\tau=0.8$ |
| Amplitudes | $U_0=0.005,\ 0.01,\ 0.02$ |
| Startup discard | 10 timesteps |
| Final time | $t^*_{\max}=0.02$ |
| Boundary conditions | Periodic |
| Spatial derivatives | Spectral FFT |
| Initialization | Local quadratic equilibrium |

This requires six principal simulations: three amplitudes for each lattice.

An additional D3Q19/D3Q27 pair at $U_0=0.04$ may be used as a finite-amplitude stress test, but it is not part of the primary production set.

---

## 9. Interpretation

Experiment III is **not a conventional grid-convergence test**. It probes whether the full nonlinear kinetic population approaches the independently derived linear CE2 population response in the tangent limit.

The numerical evidence shows that, at fixed resolution,

$$
\mathcal E_Q \rightarrow 0,
\qquad
\mathcal R_Q \rightarrow 1,
\qquad
\mathcal X_Q \rightarrow 1
\qquad
\text{as } U_0 \rightarrow 0.
$$

In particular, the approximately linear behavior

$$
\mathcal E_Q \propto U_0
$$

is consistent with finite-amplitude kinetic contributions becoming negligible relative to the linear CE2 response as the uniform-rest limit is approached.

The appropriate conclusion is therefore:

**The measured kinetic population approaches the linear CE2 prediction as the flow amplitude tends to zero.**

This is the relevant consistency test for an observability operator derived as a low-Mach linear tangent model. It does not establish a finite-amplitude reconstruction formula, nor does it separately quantify finite-wavenumber and nonlinear corrections.
