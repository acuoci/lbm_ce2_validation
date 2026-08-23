# Experiment III — Production Protocol Refinement

## 1. Objective of the protocol study

The purpose of this refinement is to select the cheapest and scientifically cleanest Taylor–Green reserve calculation capable of testing whether the linear-CE2 population operator remains useful in a weakly nonlinear, non-modal three-dimensional flow.

The protocol selection focused on three questions:

1. how many timesteps must be discarded after equilibrium initialization;
2. how far in nondimensional time the reserve calculation needs to be evolved;
3. whether \(N=48\) or \(N=64\) is preferable for exposing a clean Mach-number trend.

The answer is not simply “use the finest grid.” Because Experiment III measures the **raw kinetic population**

\[
g_Q^{LBM}
=
P_{K,Q}(f-f^{eq}),
\]

without subtracting a finite-amplitude CE1 reconstruction, the relative importance of nonlinear kinetic contamination changes with both \(U_0\) and the lattice wavenumber.

---

## 2. Startup transient

The populations are initialized from equilibrium, so

\[
g_Q^{LBM}(t=0)=0,
\]

whereas the CE2 curvature prediction is already nonzero.

A dense D3Q19 pilot at \(N=32\), \(U_0=0.02\), \(\tau=0.8\) gave

| step | \(\mathcal E_{19}\) | \(\mathcal R_{19}\) | \(\mathcal X_{19}\) |
|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0000 | 0.0368 |
| 1 | 1.1248 | 2.1114 | 0.9929 |
| 2 | 0.4791 | 0.5902 | 0.9478 |
| 3 | 0.2422 | 1.1745 | 0.9880 |
| 5 | 0.1709 | 1.0373 | 0.9866 |
| 8 | 0.1639 | 1.0264 | 0.9873 |
| 10 | 0.1604 | 1.0261 | 0.9878 |
| 15 | 0.1517 | 1.0243 | 0.9891 |
| 20 | 0.1429 | 1.0219 | 0.9902 |

The strong kinetic startup oscillation is essentially over by approximately step 5. However, using step 10 as the first accepted diagnostic point is preferable because the norm ratio has already stabilized close to unity and the population alignment evolves smoothly thereafter.

The recommended startup discard is therefore

\[
\boxed{n_{\rm discard}=10\ \text{timesteps}.}
\]

This choice is conservative while adding negligible computational cost.

---

## 3. Grid-size study

A D3Q19 pilot was performed at

\[
N=32,\quad48,\quad64,
\]

at fixed

\[
\tau=0.8
\]

and at two amplitudes,

\[
U_0=0.01,\qquad0.04.
\]

The comparison was made at step 10, after the startup discard.

### \(U_0=0.01\)

| \(N\) | \(\mathcal E_{19}\) | \(\mathcal R_{19}\) | \(\mathcal X_{19}\) |
|---:|---:|---:|---:|
| 32 | 0.0811 | 1.0161 | 0.9969 |
| 48 | 0.1272 | 1.0138 | 0.9921 |
| 64 | 0.1729 | 1.0180 | 0.9855 |

### \(U_0=0.04\)

| \(N\) | \(\mathcal E_{19}\) | \(\mathcal R_{19}\) | \(\mathcal X_{19}\) |
|---:|---:|---:|---:|
| 32 | 0.3199 | 1.0652 | 0.9540 |
| 48 | 0.5081 | 1.1290 | 0.8930 |
| 64 | 0.6917 | 1.2196 | 0.8236 |

The relative CE2 error therefore **increases** with \(N\) at fixed \(U_0\).

This is not a spatial-resolution failure. It follows naturally from the fact that the measured raw kinetic population contains finite-amplitude contributions that are absent from the linear tangent operator.

The target CE2 curvature response scales as

\[
g_Q^{CE2}
=
O(U_0k^2),
\]

whereas a nonlinear CE1-type kinetic contamination can scale as

\[
g_Q^{NL}
=
O(U_0^2k).
\]

Their relative magnitude therefore behaves as

\[
\frac{\|g_Q^{NL}\|}{\|g_Q^{CE2}\|}
=
O\left(\frac{U_0}{k}\right).
\]

For the Taylor–Green fundamental mode,

\[
k\propto\frac1N.
\]

Thus, at fixed Mach number,

\[
\mathcal E_Q^{NL}
\sim U_0N.
\]

The pilot results quantitatively support this interpretation. Using the characteristic Taylor–Green lattice wavenumber

\[
k_{TGV}
=
\frac{2\pi\sqrt3}{N},
\]

the quantity

\[
\frac{\mathcal E_{19}k_{TGV}}{U_0}
\]

is approximately constant over the three grids for the low-amplitude data.

The important conclusion is therefore:

\[
\boxed{
N=64\ \text{is not automatically preferable for the raw-population reserve test.}
}
\]

A moderate lattice makes the linear CE2 curvature signal easier to separate from the deliberately unsubtracted finite-amplitude kinetic contribution.

---

## 4. Mach-number trend at \(N=48\)

At

\[
N=48,
\qquad
\tau=0.8,
\]

four amplitudes were tested at step 10:

\[
U_0=
0.005,\ 0.01,\ 0.02,\ 0.04.
\]

### D3Q19

| \(U_0\) | \(\mathcal E_{19}\) | \(\mathcal R_{19}\) | \(\mathcal X_{19}\) |
|---:|---:|---:|---:|
| 0.005 | 0.0638 | 1.0077 | 0.9980 |
| 0.010 | 0.1272 | 1.0138 | 0.9921 |
| 0.020 | 0.2541 | 1.0378 | 0.9696 |
| 0.040 | 0.5081 | 1.1290 | 0.8930 |

A log-log fit gives

\[
\boxed{
\mathcal E_{19}\propto U_0^{0.998}.
}
\]

### D3Q27

| \(U_0\) | \(\mathcal E_{27}\) | \(\mathcal R_{27}\) | \(\mathcal X_{27}\) |
|---:|---:|---:|---:|
| 0.005 | 0.0814 | 1.0089 | 0.9968 |
| 0.010 | 0.1624 | 1.0188 | 0.9872 |
| 0.020 | 0.3246 | 1.0573 | 0.9517 |
| 0.040 | 0.6491 | 1.1991 | 0.8409 |

The corresponding fit gives

\[
\boxed{
\mathcal E_{27}\propto U_0^{0.998}.
}
\]

This almost perfectly linear Mach-amplitude dependence is a highly useful result. It strongly supports the scaling argument

\[
\mathcal E_Q^{NL}
\sim
\frac{U_0}{k}
\]

and shows that the reserve experiment can expose a clean finite-amplitude trend without requiring a larger grid.

---

## 5. Recommended amplitudes

The original proposal used

\[
U_0=\{0.01,0.02,0.04\}.
\]

The pilot study shows that \(U_0=0.04\) already produces a substantial rotation of the kinetic population vector:

\[
\mathcal X_{19}\simeq0.893,
\qquad
\mathcal X_{27}\simeq0.841
\]

at \(N=48\), step 10.

This is useful as a deliberate stress point, but it is no longer an especially clean linear-CE2 comparison.

The recommended production amplitudes are therefore

\[
\boxed{
U_0=\{0.005,\ 0.01,\ 0.02\}.
}
\]

These span a factor of four while keeping the lowest two cases close to the linear population direction.

Optionally,

\[
U_0=0.04
\]

may be retained as a fourth “breakdown” curve if a reviewer specifically asks to demonstrate the departure from the tangent model.

---

## 6. Time-window study

For D3Q19 at

\[
N=48,
\qquad
U_0=0.01,
\qquad
\tau=0.8,
\]

the diagnostics evolve as follows after startup:

| step | \(t^*=U_0t/N\) | \(\mathcal E_{19}\) | \(\mathcal R_{19}\) | \(\mathcal X_{19}\) |
|---:|---:|---:|---:|---:|
| 10 | 0.00208 | 0.1272 | 1.0138 | 0.9921 |
| 20 | 0.00417 | 0.1209 | 1.0131 | 0.9929 |
| 40 | 0.00833 | 0.1090 | 1.0113 | 0.9942 |
| 60 | 0.01250 | 0.0984 | 1.0105 | 0.9953 |
| 80 | 0.01667 | 0.0889 | 1.0096 | 0.9961 |
| 96 | 0.02000 | 0.0818 | 1.0091 | 0.9967 |

For D3Q27 under the same conditions,

| step | \(t^*\) | \(\mathcal E_{27}\) | \(\mathcal R_{27}\) | \(\mathcal X_{27}\) |
|---:|---:|---:|---:|---:|
| 10 | 0.00208 | 0.1624 | 1.0188 | 0.9872 |
| 40 | 0.00833 | 0.1392 | 1.0149 | 0.9906 |
| 96 | 0.02000 | 0.1044 | 1.0110 | 0.9947 |

The agreement therefore does not deteriorate over this early-time interval; it actually improves as the initially imposed Taylor–Green field decays.

There is consequently no need to evolve to an eddy-turnover time or to a strongly broadband state for the intended reviewer-response purpose.

The recommended final nondimensional time is

\[
\boxed{
t^*_{\max}=0.02.
}
\]

This is long enough to show that the CE2 agreement survives well beyond the startup transient, but short enough to retain the experiment as a controlled weakly nonlinear test.

---

## 7. Recommended output schedule

Because the number of lattice steps corresponding to a fixed \(t^*\) depends on \(U_0\), define

\[
t^*
=
\frac{U_0 t}{N}.
\]

For each amplitude, output at approximately

\[
\boxed{
t^*=
0,\quad
0.002,\quad
0.005,\quad
0.010,\quad
0.020.
}
\]

Additionally retain raw steps

\[
1,\ 2,\ 5,\ 10
\]

for the startup audit if desired.

Only data after

\[
t\ge10
\]

should be used in the CE2 robustness discussion.

---

## 8. Final recommended production protocol

The preferred reserve calculation is:

\[
\boxed{
\begin{aligned}
&\text{lattices:} &&D3Q19,\ D3Q27,\\
&N: &&48,\\
&\tau: &&0.8,\\
&U_0: &&0.005,\ 0.01,\ 0.02,\\
&\text{startup discard:} &&10\ \text{steps},\\
&t^*_{\max}: &&0.02,\\
&\text{boundary conditions:} &&\text{periodic},\\
&\text{derivatives:} &&\text{spectral FFT},\\
&\text{initialization:} &&f=f^{eq}(\rho_0,\mathbf u_{TGV}).
\end{aligned}
}
\]

This produces only six principal simulations.

A seventh/eighth pair at

\[
U_0=0.04
\]

may be added only if one explicitly wants to illustrate breakdown away from the tangent regime.

---

## 9. Why \(N=48\) is preferable to \(N=64\)

For an ordinary grid-convergence study one would normally prefer \(N=64\). That logic does not apply directly here because the numerator of the error contains finite-amplitude kinetic physics that is **not included in the theoretical model being tested**.

At fixed \(U_0\),

\[
\text{linear CE2 signal}\sim k^2,
\]

whereas

\[
\text{finite-amplitude contamination}\sim k.
\]

Refining the grid lowers the modal lattice wavenumber and therefore reduces the target CE2 signal more rapidly than the leading nonlinear contamination.

Thus \(N=48\) is the better compromise:

- it is sufficiently fine for smooth spectral differentiation;
- density fluctuations remain small;
- it gives a clean approximately linear Mach-number trend;
- the population alignment remains close to unity for the lower amplitudes;
- it is substantially cheaper than \(64^3\);
- it avoids artificially weakening the CE2 signal relative to the unsubtracted nonlinear contribution.

---

## 10. Scientific interpretation

The pilot data already reveal a useful physical lesson.

The Taylor–Green calculation does **not** simply test spatial resolution. It probes the distinction between

\[
\text{linear CE2 tangent response}
\]

and

\[
\text{finite-amplitude kinetic population}.
\]

The nearly exact scaling

\[
\mathcal E_Q\propto U_0
\]

at fixed grid is consistent with the expected presence of nonlinear kinetic terms one order lower in spatial differentiation but one order higher in amplitude.

Therefore, if Experiment III is ever presented to a referee, its strongest result would not be “the CE2 error goes to zero under grid refinement.” Instead it would be:

\[
\boxed{
\text{the measured kinetic population approaches the linear CE2 prediction as the flow amplitude tends to zero.}
}
\]

That is precisely the correct robustness statement for a theory explicitly derived as a low-Mach linear tangent model.
