# LBM CE2 population-observability: numerical validation

This repository contains the Python code and reproducibility material associated with the manuscript **“Observability of Velocity Derivatives from Kinetic Populations in the Lattice Boltzmann Method”**, by Alberto Cuoci, submitted for publication to *Physics of Fluids*.

The numerical calculations are deliberately small, transparent BGK collision-streaming implementations written in NumPy. They are intended to verify specific analytical predictions directly from time-evolved lattice populations, not to provide a general-purpose LBM solver. All theoretical operators are archived as exact rational CSV files and are verified in exact arithmetic (SymPy, Python `fractions`).

Release history: see [`CHANGELOG.md`](CHANGELOG.md). Version 0.2.0 accompanies the revised manuscript and adds the D3Q15 lattice, the constructive curvature-reconstruction analysis and Experiment IV.

## Theoretical operators

### Normalized CE2 observability operators

`operators/Lhat_D3Q{15,19,27}.csv` contain the exact normalized second-order Chapman–Enskog (CE2) population operators $\widehat L_Q\in\mathbb Q^{Q\times15}$. The kinetic (non-hydrodynamic) population response to a velocity-curvature field is

$$
\mathbf g_Q^{\mathrm{CE2}}=C_0\widehat L_Q\mathbf q,\qquad C_0=2\rho_0\tau\left(\tau-\tfrac12\right)c_s^4 ,
$$

where $\mathbf q$ holds the 15 independent curvature components $\partial_\beta\partial_\gamma u_\alpha$ of an incompressible field. The D3Q19 and D3Q27 operators come from the symbolic analysis of the manuscript. The D3Q15 operator is regenerated and checked by `scripts/generate_d3q15_operator.py`, which also reproduces the D3Q19 and D3Q27 files exactly. File format, ordering and structure (ranks, kernels, Hermite moments): [`operators/README.md`](operators/README.md).

### Constructive curvature reconstruction

A curvature functional $d=\mathbf d^T\mathbf q$ is observable from kinetic populations on lattice $Q$ if and only if $\mathbf d\in\operatorname{row}(\widehat L_Q)$. For each lattice the observable functionals are

| Lattice | Reconstructed functionals $\mathbf d_Q=D_Q\mathbf q$ | rank $\widehat L_Q$ |
|---|---|---|
| D3Q15 | $S_1, S_2, S_3, T$ | 4 |
| D3Q19 | $A_x, B_x, B_y, A_z, A_y, B_z$ | 6 |
| D3Q27 | $A_x, B_x, B_y, A_z, A_y, B_z, T$ | 7 |

with $A_x=2u_{x,xy}+u_{y,xx}$, $B_x=2u_{x,xz}+u_{z,xx}$ (and cyclic), $T=u_{x,yz}+u_{y,xz}+u_{z,xy}$, and $S_1=A_x+B_z$, $S_2=B_x+A_y$, $S_3=B_y+A_z$. They are reconstructed by explicit third-order Hermite operators $R^H_Q$, which satisfy exactly

$$
R^H_Q\,\widehat L_Q=D_Q,\qquad Q=15,19,27,
$$

and coincide with the weighted minimum-norm reconstructions. The archived, machine-readable files are in `operators/reconstruction/`:

| File | Content |
|---|---|
| `observable_functionals.csv` | coefficients of $A_*$, $B_*$, $T$, $S_1$–$S_3$ on $\mathbf q$ |
| `D_D3Q{15,19,27}.csv` | $D_Q$ (rows = functionals, columns = $\mathbf q$) |
| `RH_D3Q{15,19,27}.csv` | $(R^H_Q)^T$, one row per population, in the population order of `Lhat_D3Q*.csv` |
| `sensitivity_factors.csv` | squared sensitivity factors $\chi^2_{Q,j}=\mathbf d_j^TG_Q^+\mathbf d_j$ |

They are written by `scripts/generate_reconstruction_operators.py` from `src/curvature_reconstruction_exact.py`. The script's `--check` mode rebuilds them in memory and verifies $R^H_Q\widehat L_Q=D_Q$ from the files alone. Derivation and verified results: [`docs/experiment_IV_reconstruction_operators.md`](docs/experiment_IV_reconstruction_operators.md). The NumPy interface that applies $R^H_Q/C_0$ to population arrays is `src/curvature_reconstruction.py` ([`docs/experiment_IV_numerical_reconstruction.md`](docs/experiment_IV_numerical_reconstruction.md)).

## Numerical experiments

### Experiment I — D2Q9 finite-wavenumber shear response

`src/experiment_I_d2q9_shear.py` verifies the transverse-shear expansion

$$
\eta_K^{\mathrm{pop}} =
A(\theta,\tau)k
\left[1+B(\theta,\tau)k^2+O(k^4)\right],
$$

where $k = |\mathbf k|$ is the lattice wavenumber. It uses exact discrete shear-eigenmode initialization followed by the full nonlinear D2Q9 BGK collision-streaming algorithm. The production sweep uses a 256 x 256 periodic lattice, three propagation families ([10], [21], [11]), three relaxation times, and eight harmonics per family. Details: [`docs/experiment_I_D2Q9_exact_shear_mode.md`](docs/experiment_I_D2Q9_exact_shear_mode.md).

```bash
python scripts/reproduce_experiment_I.py          # writes reference_results/experiment_I/
```

### Experiment II — D3Q19/D3Q27 CE2 population operator

`src/experiment_II_3D_shear.py` compares the complete measured kinetic population vector with $g_Q^{\mathrm{CE2}}=C_0\widehat L_Q\mathbf q_F$ for exact three-dimensional transverse shear eigenmodes. The production calculation uses D3Q19 and D3Q27 at $\tau=1$, directions [100], [110], and [111], and N = 24, 32, 40, 48, 64, 80. Symmetry-distinct transverse polarizations are retained. Details: [`docs/experiment_II_3D_shear_population_response.md`](docs/experiment_II_3D_shear_population_response.md).

```bash
python scripts/reproduce_experiment_II.py         # writes reference_results/experiment_II/
```

### Experiment III — weakly nonlinear Taylor–Green cross-check

`src/experiment_III_taylor_green.py` performs the non-modal Taylor–Green tangent-limit consistency check. The production drivers override the exploratory defaults in the solver module and use the manuscript protocol: D3Q19 and D3Q27, N = 48, $\tau$ = 0.8, $U_0$ = 0.005, 0.01, 0.02, equilibrium initialization, spectral differentiation of the instantaneous velocity field, and the first 10 timesteps excluded from quantitative comparison. Details: [`docs/experiment_III_production_protocol_refined.md`](docs/experiment_III_production_protocol_refined.md).

```bash
python scripts/reproduce_experiment_III.py                 # amplitude sweep to t* = 0.02 -> reference_results/experiment_III/
python scripts/reproduce_experiment_III_time_history.py    # manuscript time-history figure -> reference_results/experiment_III/time_history/
```

The manuscript time-history figure, `reference_results/experiment_III/time_history/figure_III_1_global_CE2_error.{pdf,png}`, shows $\mathcal E_Q$ for D3Q19 and D3Q27 at N = 48, $\tau$ = 0.8, $U_0$ = 0.02. The diagnostics are taken at timesteps $t$ = 10, 20, 30, 40, 48, and $t^*=U_0t/N$, so the first point is timestep 10. `reproduce_experiment_III_time_history.py` uses the unmodified solver functions. It also recomputes all 32 tabulated protocol values at their stated timesteps (`experiment_III_table_audit.csv`, all reproduced to 4 decimals). At timestep 10 it gives $\mathcal E_{19}=0.2541$ and $\mathcal E_{27}=0.3246$. The stored time history also contains the numerator and denominator of $\mathcal E_Q$ and the RMS velocity. These show that the decrease of $\mathcal E_Q$ over this window follows viscous amplitude decay, $\exp(-\nu|\mathbf k|^2t)$. Note that `reproduce_experiment_III.py` writes a multi-amplitude diagnostic plot with the same file name one directory higher; the manuscript figure is the one in `time_history/`.

### Experiment IV — curvature reconstruction from BGK populations

Experiment IV tests the constructive reconstruction $\mathbf d^{rec}_Q=R^H_Q\,\mathbf g^{LBM}_{K,Q}/C_0$ on D3Q15, D3Q19 and D3Q27. It compares the result against $D_Q\mathbf q_F$ for transverse shear modes along $\mathbf k=\frac{2\pi}{N}(1,2,3)$ at $\tau=1$, on both shear branches.

| Step | Code / driver | Output | Documentation |
|---|---|---|---|
| Exact linearized BGK eigenmodes, N = 24–128 | `src/experiment_IV_linearized_reconstruction.py`, `scripts/reproduce_experiment_IV_linearized.py` | `reference_results/experiment_IV_linearized/` | [`docs/experiment_IV_linearized_reconstruction.md`](docs/experiment_IV_linearized_reconstruction.md) |
| Velocity-parity analysis of the finite-$k$ error | `src/experiment_IV_parity_analysis.py`, `scripts/reproduce_experiment_IV_parity.py` | `reference_results/experiment_IV_parity/` | [`docs/experiment_IV_parity_analysis.md`](docs/experiment_IV_parity_analysis.md) |
| Fully evolved nonlinear-equilibrium BGK populations, $U_0=10^{-6}$, N = 24–80 | `src/experiment_IV_full_bgk_reconstruction.py`, `scripts/reproduce_experiment_IV_full_bgk.py` | `reference_results/experiment_IV_full_bgk/` | [`docs/experiment_IV_full_bgk_reconstruction.md`](docs/experiment_IV_full_bgk_reconstruction.md) |

The kinetic population error converges as $O(k)$, while the reconstructed observables converge as $O(k^2)$. The reason is that the leading population error has even velocity parity and is annihilated exactly by the odd Hermite operators ($R^H_QP_+=0$). The fully evolved BGK reconstruction reproduces the linearized results.

```bash
python scripts/reproduce_experiment_IV_linearized.py
python scripts/reproduce_experiment_IV_parity.py
python scripts/reproduce_experiment_IV_full_bgk.py           # about 1 min, about 0.6 GB peak memory
python scripts/plot_experiment_IV_paper_figures.py           # publication figures from the stored results
```

The publication figures are drawn from the stored results; no simulation is rerun. The plotting script verifies the data before and after plotting, and writes nothing if any check fails. Outputs:

- `Figures/figure_IV_population_vs_reconstruction.{pdf,png}`
- `Figures/figure_IV_full_bgk_vs_linearized.{pdf,png}`
- `Figures/figure_IV_paper_verification.txt`

## Repository layout

```text
.
├── README.md, CHANGELOG.md, REPRODUCIBILITY.md, CITATION.cff, LICENSE, requirements.txt
├── ARCHITECTURE_NOTES.md               # code-structure notes
├── operators/
│   ├── Lhat_D3Q15.csv, Lhat_D3Q19.csv, Lhat_D3Q27.csv   # normalized CE2 observability operators
│   ├── reconstruction/                                  # D_Q, R^H_Q, functionals, chi^2 (exact)
│   └── README.md
├── src/
│   ├── experiment_I_d2q9_shear.py
│   ├── experiment_II_3D_shear.py
│   ├── experiment_III_taylor_green.py
│   ├── curvature_reconstruction_exact.py     # exact D_Q, R^H_Q, R*_Q, observability (SymPy)
│   ├── curvature_reconstruction.py           # NumPy reconstruction interface
│   ├── experiment_IV_linearized_reconstruction.py
│   ├── experiment_IV_parity_analysis.py
│   └── experiment_IV_full_bgk_reconstruction.py
├── scripts/
│   ├── reproduce_experiment_{I,II,III}.py
│   ├── reproduce_experiment_III_time_history.py
│   ├── reproduce_experiment_IV_{linearized,parity,full_bgk}.py
│   ├── plot_experiment_IV_paper_figures.py
│   ├── generate_d3q15_operator.py
│   ├── generate_reconstruction_operators.py
│   └── check_repository.py
├── tests/                                    # unittest suite
├── docs/                                     # per-experiment protocols and verified results
├── reference_results/
│   ├── experiment_III/time_history/
│   ├── experiment_IV_linearized/
│   ├── experiment_IV_parity/
│   └── experiment_IV_full_bgk/
└── Figures/                                  # Experiment IV publication figures
```

## Installation

Python 3.10 or newer is recommended. The code depends on NumPy, Matplotlib and SymPy (SymPy is used for the exact operator verification and the reconstruction operators).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validation

```bash
python scripts/check_repository.py          # fast: files, metadata, exact identities, stored values (seconds)
python scripts/check_repository.py --full   # complete validation suite (about one minute)
python -m unittest discover -s tests        # unit tests only
```

The fast check verifies:

- the presence of all operators, reference results and figures;
- metadata consistency;
- $R^H_Q\widehat L_Q=D_Q$ for all three lattices, computed from the archived files;
- the Experiment III timestep-10 values against the manuscript tables;
- the stored Experiment IV figure verification.

`--full` additionally runs:

- the unit tests;
- the exact D3Q15 and reconstruction-operator generators in `--check` mode;
- the exact reconstruction report;
- the Experiment IV figure verification;
- a rerun of the Experiment III table audit and time history, compared with the stored CSVs.

## Reproducing the manuscript calculations

Run each production script from the repository root. The scripts encode the parameters used in the manuscript and are the authoritative reproduction entry points. The command-line defaults inside the underlying experiment modules are kept mainly for inexpensive demonstrations and exploratory calculations. In particular, they should not be assumed to represent the final Taylor–Green production protocol.

The D2Q9 experiment is self-contained. Experiments II–IV require the exact operator CSVs; their population and curvature-coordinate ordering is part of the definition of the experiments. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the complete parameter sets.

## License

The software is released under the MIT License; see `LICENSE`. This is a permissive open-source license that allows reuse, modification, and redistribution while retaining the copyright and license notice.

The license choice is independent of the publication license of the associated article and Supplementary Material.
