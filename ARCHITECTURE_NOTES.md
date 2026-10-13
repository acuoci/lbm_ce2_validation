# Architecture Notes

Shared reference for extending `lbm_ce2_validation`. Describes the code as of
2026-10-13 (release 1.0.0, see `CITATION.cff`). Update this file whenever the
structure changes.

---

## 1. What the repository is

Not a general LBM framework: three independent, self-contained numerical
experiments that validate analytical predictions from the manuscript
*"Observability of Velocity Derivatives from Kinetic Populations in the Lattice
Boltzmann Method"* (Physics of Fluids, submitted). Each experiment runs a full
nonlinear BGK collision-streaming simulation and compares measured kinetic
populations against theory.

- **Dependencies:** `numpy>=1.24,<3`, `matplotlib>=3.7,<4` (`requirements.txt`).
  Python ≥ 3.10.
- **No packaging:** no `pyproject.toml`, no `__init__.py`, no test suite.
- **Scientific inputs:** `operators/Lhat_D3Q19.csv`, `operators/Lhat_D3Q27.csv`
  (exact symbolic CE2 operators; must not be regenerated independently).

```
.
├── src/                         # one standalone module per experiment
│   ├── experiment_I_d2q9_shear.py        (781 lines)
│   ├── experiment_II_3D_shear.py         (813 lines)
│   └── experiment_III_taylor_green.py    (666 lines)
├── scripts/                     # authoritative production drivers
│   ├── reproduce_experiment_I.py
│   ├── reproduce_experiment_II.py
│   ├── reproduce_experiment_III.py
│   └── check_repository.py      # syntax + input-file presence only
├── operators/                   # Lhat CSVs (input data contract)
├── docs/                        # per-experiment design/protocol notes
└── reference_results/           # created on first production run (outputs)
```

---

## 2. Core module breakdown

All three modules share the same five-layer anatomy. There are **no imports
between modules**; shared logic is duplicated.

| Layer | Experiment I — D2Q9 | Experiment II — D3Q19/27 shear | Experiment III — D3Q19/27 Taylor-Green |
|---|---|---|---|
| **Lattice** | Module-level constants `C`, `W`, `WINV`, `CS2`; `PROJECTORS` dict built at import by `build_projectors()` | `Lattice` dataclass from `load_lattice(csv)` | Same `Lattice` / `load_lattice` (duplicated) |
| **Linear theory** | `collision_matrix`, `amplification_matrix`, `select_shear_mode` → `ShearMode` | Same, `lat` as first argument; `select_shear_mode` adds target polarization + degenerate-eigenspace handling | — (non-modal) |
| **Solver** | `macroscopic`, `equilibrium`, `collide_and_stream` (2-D, axes `(y,x)`) | Same, 3-D, axes `(z,y,x)` | Same (duplicated) |
| **Initialization** | `initialize_exact_mode` — real part of exact complex eigenmode, amplitude ε | `initialize_exact_mode` with `(mx,my,mz)` and polarization | `taylor_green_velocity` + `initialize_taylor_green` (equilibrium at uniform ρ₀) |
| **Diagnostics** | `extract_population_fourier_mode`, `diagnose_mode` → `ModeDiagnostic` (η = ‖P_K f^neq‖_w / ‖P_2 f^neq‖_w) | `extract_fourier_mode`, `q_fourier`, `ce2_prefactor`, `diagnose` → `Diagnostic` (E, R, χ) | Spectral derivatives, `curvature_q_field`, `ce2_population_field`, `measured_kinetic_field`, `global_population_metrics` (E, R, X), flow diagnostics → `TGVRecord` |
| **Theory reference** | `theory_A_B(theta, tau)` (closed-form A, B) | `Lhat @ q_F` with C₀ = 2ρ₀τ(τ−½)c_s⁴ | `Lhat · q(x)`; `gamma_q` (23/35, 26/35) for centroid estimator |
| **Drivers** | `run_case` → `CaseResult`; `run_sweep`; `direction_modes` | `run_case` → `CaseResult`; `run_sweep`; `direction_definition` | `run_case` → `List[TGVRecord]`; `run_sweep`; `suggested_output_steps` |
| **Output** | `write_results_csv`, `write_fit_table`, `make_plots`, `print_case_summary` | `write_csv`, `fit_convergence`, `write_fit_csv`, `make_plots`, `print_case` | `write_records`, `make_case_plots` |
| **CLI** | `main()`: `--mode demo\|sweep` | `main()`: `--mode demo\|sweep` | `main()`: `--mode demo\|sweep` |

### 2.1 The `Lattice` object (II, III)

| Field | Shape | Meaning |
|---|---|---|
| `name` | str | `"D3Q{Q}"`, derived from CSV row count |
| `c` | (Q,3) int | Lattice velocities |
| `w` | (Q,) | Quadrature weights (checked to sum to 1) |
| `lhat` | (Q,15) | Normalized CE2 operator L̂_Q |
| `q_names` | list[15] | Curvature-coordinate names from CSV header |
| `Winv` | (Q,Q) | diag(1/w) — weighted (equilibrium-Hessian) inner product |
| `P_eq` | (Q,Q) | Linearized equilibrium projector around ρ=1, u=0 |
| `P_le1`, `P_le2` | (Q,Q) | Weighted projectors onto Hermite span ≤1, ≤2 |
| `P2` | (Q,Q) | `P_le2 − P_le1` (second-order sector) |
| `PK` | (Q,Q) | `I − P_le2` (kinetic / ghost sector) |
| `N` | (Q,Q) | `I − P_eq` (linear non-equilibrium projector) |

`load_lattice` audits `PK` idempotency and orthogonality to `P_le2` (tol 5e-12).
Experiment I holds the D2Q9 equivalents in the `PROJECTORS` dict.

### 2.2 Data contract: operator CSVs

Header: `cx,cy,cz,w,` + 15 curvature columns in this exact order:

```
ux_xy, ux_xz, ux_yy, ux_zz, ux_yz,
uy_xy, uy_yz, uy_xx, uy_zz, uy_xz,
uz_xz, uz_yz, uz_xx, uz_yy, uz_xy
```

- Row order **defines population ordering** for the whole simulation.
- Entries are exact rationals (`1/18`, `-1/4`), parsed via `fractions.Fraction`.
- The 15-column order is **hard-coded a second time** in `q_fourier` (II) and
  `curvature_q_field` (III). Any change must touch all three places.

### 2.3 Array conventions

- Populations: `f[i, y, x]` (2-D) / `f[i, z, y, x]` (3-D), population index first.
- Velocity: `u[a, ...]`, a ∈ {x,y,z}.
- Fourier convention: δf(x) = f̂ exp(i k·x); streaming phase S_ii = exp(−i k·c_i).
- Fourier extraction normalizes by grid volume, so Re[v e^{ikx}] → v/2. Both
  measured and predicted sides are linear in f̂, so the factor cancels.
- Spectral derivatives (III) use lattice wavenumbers k = 2π·fftfreq(N).
- Convective time (III): t* = U₀·step/N.

---

## 3. Data flow

### 3.1 Production pipeline

```
scripts/reproduce_experiment_X.py
  │  importlib.spec_from_file_location(src/...)  → registers in sys.modules
  │  fixed paper parameters (authoritative)
  ▼
src/experiment_X.py
  │  [II, III] load_lattice(operators/Lhat_D3Q{19,27}.csv) → Lattice
  │  loop over parameter grid → run_case(...)
  ▼
reference_results/experiment_X/
     *.csv  (raw per-case results, fitted coefficients / convergence orders)
     *.png  (figures)
```

### 3.2 Inside a case (I and II: modal experiments)

```
(k, τ, polarization)
   → amplification_matrix → eig → select_shear_mode → ShearMode (eigvec, λ)
   → initialize_exact_mode: f = w ρ₀ + ε Re[v e^{ik·x}]
   → for step in 0..nsteps:
        f̂      = extract Fourier mode at k            (Q,) complex
        f^neq  = N @ f̂                                (linearized)
        I : η  = ‖P_K f^neq‖_w / ‖P_2 f^neq‖_w
        II: g_LBM = P_K f^neq ;  g_CE2 = C₀ L̂ q_F(k, j/ρ₀)
            E = ‖g_LBM − g_CE2‖_w/‖g_CE2‖_w,  R = ‖g_LBM‖/‖g_CE2‖,  χ = alignment
        step ratio f̂_n / f̂_{n−1}  (compared to exact λ)
        f = collide_and_stream(f, τ)                    (full nonlinear BGK)
   → time-average → CaseResult
```

### 3.3 Inside a case (III: Taylor-Green)

```
(N, U₀, τ)
   → initialize_taylor_green: f = f^eq(ρ₀, u_TG)
   → for step in 0..max(output_steps):
        if step in output_steps:
            ρ, u    = macroscopic(f)
            f^neq   = f − f^eq(ρ, u)                    (fully nonlinear, pointwise)
            g_LBM   = PK @ f^neq
            q(x)    = spectral ∂²u (15 components)
            g_CE2   = C₀ L̂ q(x)
            E, R, X = volume-averaged weighted metrics
            + mass, max |ρ−ρ₀|, Mach, div rms, Ω, P, k*² estimators
            → TGVRecord
        f = collide_and_stream(f, τ)
   → List[TGVRecord]
```

Key difference: I/II measure f^neq with the **linear** projector `N` on a single
Fourier coefficient; III uses the **nonlinear** pointwise f − f^eq(ρ,u).

### 3.4 Production parameters (from `scripts/`)

| | Lattices | Grid | τ | Other |
|---|---|---|---|---|
| I | D2Q9 | 256² | 0.9, 1.0, 1.1 | dirs axis/oblique/diag, m=1..8 (k ≤ 0.58), ε=1e-6, 8 steps |
| II | D3Q19, D3Q27 | N=24,32,40,48,64,80 | 1.0 | dirs 100/110/111, m=1, 2 polarizations each, ε=1e-6, 1 step |
| III | D3Q19, D3Q27 | N=48 | 0.8 | U₀=0.005, 0.01, 0.02; t* outputs 0.002–0.02; discard first 10 steps |

Module CLI defaults differ from these (especially III) and are for exploration
only.

---

## 4. Design patterns

| Pattern | Where | Notes |
|---|---|---|
| Procedural / functional core | Everywhere | Pure array-in/array-out functions; side effects confined to writers and plots |
| Dataclasses as value records | `ShearMode`, `ModeDiagnostic`, `Diagnostic`, `CaseResult`, `TGVRecord`, `Lattice` | Derived metrics as `@property` (`eta_over_k`, `RB`, `EAB` in I) |
| Explicit context object | `Lattice` passed as first arg (II, III) | Makes one solver serve D3Q19 and D3Q27; I uses module globals instead |
| Precomputed projector algebra | `weighted_projector`, `PROJECTORS`, `Lattice.P*` | All diagnostics are weighted norms/overlaps of projected population vectors |
| Layered drivers | `run_case` → `run_sweep` → writers/plots → `scripts/` | Production scripts pin parameters; CLI provides demo/sweep |
| Dynamic file-path module loading | `scripts/*.py` | Avoids packaging; registers module in `sys.modules` (required for dataclasses) |
| Fail-fast validation | `load_lattice`, scripts, `select_shear_mode` | Weight sum, Lhat shape, projector audits, missing files, negligible momentum |
| Data-driven configuration | Operator CSVs | Population ordering and L̂ come from data, not code |
| Score-based selection heuristic | `select_shear_mode` (I, II) | Picks eigenmode by hydrodynamic content rather than eigenvalue tracking |

---

## 5. Known issues and extension hotspots

Observed during review; nothing has been changed yet.

1. **Code duplication.** `Lattice`, `load_lattice`, `weighted_projector`,
   `macroscopic`, `equilibrium`, `collide_and_stream`, `ce2_prefactor`,
   `_parse_rational` are near-identical in II and III; I reimplements them for
   D2Q9 with globals. Natural first refactor: a shared core module
   (lattice + solver + projectors), with D2Q9 expressed as a `Lattice` too.
   Any refactor must reproduce existing `reference_results/` bit-for-bit or to
   round-off.
2. **Definition-order dependence (II).** `fit_convergence`, `make_plots`,
   `run_sweep` precede `CaseResult` / `direction_definition`; works only via
   `from __future__ import annotations` and call-time name resolution.
3. **Redundant FFTs (III).** Each `diagnose_state` recomputes first derivatives
   4× (divergence, strain, vorticity, enstrophy) and enstrophy twice;
   second derivatives compute 27 components where 15 are needed.
4. **Silent parameter semantics (III).** `max_tstar` is ignored when
   `output_steps` is given; `startup_discard` only affects plots and console
   tags; the CSV keeps all records.
5. **Hard-coded heuristics.** Mode-selection weights (0.65/0.35, 0.8 in I;
   `exp(-4·long)`, `exp(-2·dens)`, 5e-4 tolerances, `degeneracy_tol=2e-8` in II).
6. **Duplicated contract.** The 15-component curvature ordering exists in the
   CSV header and in two code locations (see §2.2).
7. **Minor.** Unused `_group()` in I; III docstring says matplotlib optional
   but imports it unconditionally; `gamma_q` keyed on name string; equilibrium
   assembly loops over Q in Python (vectorized over grid only).
8. **Testing.** `check_repository.py` only checks syntax and file presence. No
   regression tests against reference outputs; adding small fast-running cases
   (e.g. N=16) before refactoring is recommended.

### Ground rules for extensions

- Treat `operators/*.csv` and their ordering as immutable inputs.
- Keep `scripts/reproduce_*.py` as the authoritative production entry points.
- Keep dependencies to NumPy + Matplotlib unless there is a strong reason.
- Validate any change against existing outputs before trusting new results.
