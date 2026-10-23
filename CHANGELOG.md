# Changelog

## [0.2.0] — 2026-10-23

Reproducibility material for the revised Physics of Fluids manuscript. The scope expands from observability to constructive curvature reconstruction, adds the D3Q15 lattice, and adds a fourth numerical experiment. The numerical methods and results of Experiments I–III are unchanged.

### Observability and reconstruction operators

- **D3Q15 observability operator.** New `operators/Lhat_D3Q15.csv`, the exact normalized CE2 observability operator (rank 4). It is generated and verified in exact arithmetic by `scripts/generate_d3q15_operator.py`, which also reproduces the existing D3Q19 and D3Q27 operators exactly.
- **Constructive reconstruction.** New `src/curvature_reconstruction_exact.py`, which builds in exact arithmetic:
  - the observable functionals $D_Q$;
  - the explicit third-order Hermite reconstructions $R^H_Q$;
  - the weighted minimum-norm reconstructions $R^\star_Q$;
  - the sensitivity factors $\chi^2_{Q,j}$;
  - exact observability tests with nullspace witnesses.

  It covers D3Q15 ($S_1,S_2,S_3,T$), D3Q19 ($A_x,B_x,B_y,A_z,A_y,B_z$) and D3Q27 (the six $A/B$ functionals and $T$). $R^H_Q\widehat L_Q=D_Q$ holds exactly, and $R^H_Q=R^\star_Q$.
- **Archived reconstruction operators.** New `operators/reconstruction/`, containing `observable_functionals.csv`, `D_D3Q*.csv`, `RH_D3Q*.csv` and `sensitivity_factors.csv`. These are written deterministically by `scripts/generate_reconstruction_operators.py`, whose `--check` mode verifies $R^H_Q\widehat L_Q=D_Q$ from the files alone.
- **NumPy interface.** New `src/curvature_reconstruction.py`, which applies $R^H_Q/C_0$ to population arrays of any shape.

### Experiment IV — curvature reconstruction from BGK populations (new)

All three parts use D3Q15, D3Q19 and D3Q27, $\mathbf k=\frac{2\pi}{N}(1,2,3)$, $\tau=1$, and both shear branches.

- **Exact linearized BGK shear eigenmodes, N = 24–128.** The kinetic population error converges as $O(k)$, and the reconstructed observables as $O(k^2)$. Code: `src/experiment_IV_linearized_reconstruction.py`.
- **Velocity-parity analysis.** The leading population error is even in $\mathbf c$ and is annihilated exactly by the odd Hermite reconstruction ($R^H_QP_+=0$). Code: `src/experiment_IV_parity_analysis.py`.
- **Fully evolved nonlinear-equilibrium BGK populations, $U_0=10^{-6}$, N = 24–80.** These reproduce the linearized reconstruction errors. Code: `src/experiment_IV_full_bgk_reconstruction.py`.
- **Drivers, results and figures.**
  - Drivers: `scripts/reproduce_experiment_IV_{linearized,parity,full_bgk}.py`.
  - Reference results: `reference_results/experiment_IV_{linearized,parity,full_bgk}/`.
  - Publication figures, with a data-verification report: `Figures/`, from `scripts/plot_experiment_IV_paper_figures.py`.

### Experiment III — Taylor–Green time history (corrected figure)

- **New driver.** `scripts/reproduce_experiment_III_time_history.py` uses the unmodified solver. It writes the manuscript time-history figure for N = 48, $\tau$ = 0.8, $U_0$ = 0.02 at timesteps 10, 20, 30, 40, 48, with $t^*=U_0t/N$, so the first point is timestep 10.
- **Stored diagnostics.** The time history includes the numerator and denominator of $\mathcal E_Q$ and the RMS velocity. Together they show that the decrease of $\mathcal E_Q$ follows viscous amplitude decay.
- **Table audit.** All 32 values of the Experiment III protocol tables are recomputed at their stated timesteps and reproduced to 4 decimals. At timestep 10, $\mathcal E_{19}=0.2541$ and $\mathcal E_{27}=0.3246$.
- **Output location.** `reference_results/experiment_III/time_history/`.

### Archived reference results

- **Production outputs added.** The Experiment I, II and III production outputs are now archived in `reference_results/experiment_{I,II,III}/`. They were regenerated with the unchanged production scripts.
- **Earlier results now archived.** The Experiment IV results and figures, and the Experiment III time history, are archived as well.
- **Byte-exact storage.** A new `.gitattributes` keeps CSV data byte-exact and stores PNG/PDF files as binary.

### Validation and metadata

- **Repository checks.** `scripts/check_repository.py` now checks:
  - all sources and required operator, result and figure files;
  - metadata consistency;
  - the exact reconstruction identities from the archived files;
  - Experiment III timestep-10 values against the manuscript tables;
  - the Experiment IV verification report.

  `--full` runs the complete validation suite.
- **Tests.** The unittest suite now covers the D3Q15 operator, the exact and numerical reconstruction operators, the archived reconstruction files and Experiment IV.
- **Documentation.** `README.md` and `REPRODUCIBILITY.md` describe all four experiments and the reconstruction analysis.
- **CITATION.cff.**
  - Version set to 0.2.0; the previous file stated 1.0.0, inconsistent with tag `0.1.0`.
  - Abstract and keywords updated.
  - The Zenodo DOI is moved from the article citation to the software `identifiers`, as the DOI of the earlier archive.
- **SymPy.** It is now listed among the documented dependencies (it was already in `requirements.txt`).

## [0.1.0] — 2026-08-23

Initial release for the submitted manuscript. It contains Experiment I (D2Q9 finite-wavenumber shear response), Experiment II (D3Q19/D3Q27 CE2 population operator) and Experiment III (weakly nonlinear Taylor–Green cross-check), with the `Lhat_D3Q19.csv` and `Lhat_D3Q27.csv` operators.
