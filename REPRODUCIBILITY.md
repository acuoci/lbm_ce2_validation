# Reproducibility notes

## Scope

The repository reproduces the numerical validation reported in Section V of the associated manuscript:

1. D2Q9 finite-wavenumber transverse-shear validation.
2. D3Q19/D3Q27 CE2 population-vector validation.
3. Weakly nonlinear Taylor-Green tangent-limit cross-check.

## Required scientific inputs

Experiments II and III require the exact machine-readable normalized CE2 operators:

- `operators/Lhat_D3Q19.csv`
- `operators/Lhat_D3Q27.csv`

They must be the same files used in the symbolic analysis and Supplementary Material. Do not regenerate them independently inside the numerical repository unless the symbolic generation procedure itself is also archived and versioned.

## Production parameters

The production scripts are intentionally explicit rather than relying on CLI defaults.

### Experiment I

- grid: 256 x 256
- tau: 0.9, 1.0, 1.1
- directions: [10], [21], [11]
- harmonics: 1,...,8
- perturbation amplitude: 1e-6
- time steps: 8

### Experiment II

- lattices: D3Q19, D3Q27
- tau: 1.0
- perturbation amplitude: 1e-6
- directions: [100], [110], [111]
- N: 24, 32, 40, 48, 64, 80
- fundamental mode on each grid
- symmetry-distinct transverse polarizations retained

### Experiment III

- lattices: D3Q19, D3Q27
- N: 48
- tau: 0.8
- U0: 0.005, 0.01, 0.02
- equilibrium initialization
- startup discard: first 10 steps
- final nondimensional time: t* = 0.02
- spectral derivatives

These Experiment III values encode the refined production protocol used in the current manuscript rather than the older generic defaults retained in the exploratory solver module.

## Commands

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_repository.py
python scripts/reproduce_experiment_I.py
python scripts/reproduce_experiment_II.py
python scripts/reproduce_experiment_III.py
```

The three production calculations are independent and may be run separately.

## What should be archived with a paper release

For a version corresponding to manuscript submission or acceptance, archive the exact Git commit and include the compact CSV tables and paper-facing figures supporting the reported numerical values. Tag the release (for example `v1.0.0`) and archive that release in a DOI-minting repository such as Zenodo. Record the DOI in `CITATION.cff` and in the manuscript Data Availability Statement/reference list.
