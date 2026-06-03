# F.0 — Foundation Audit Report

**Date:** 2026-06-03
**Branch:** `foundation/review-phases`
**Auditor:** opencode
**Scope:** Static analysis of `earthquake_proj` against `docs/proposal/proposal_v3.tex` and the structured phase plan (P.*, D.*, T.*, L.*)
**Location:** `audits/` (top-level, not under `docs/` because `docs/` is gitignored; only `docs/proposal/` and `docs/md/` are grandfathered)

This is a **read-only working document**. No code changes are proposed here. Each finding is mapped to a specific phase leaf for downstream action.

---

## 1. Executive Summary

The repository implements a Physics-Informed Neural Network for 3D lithospheric stress inversion. The proposal (`proposal_v3.tex`, 257 lines) is the canonical spec. The implementation is **mostly aligned** with the proposal, but contains:

- **3 critical issues** that must be resolved before "solid foundation" is meaningful
- **7 medium issues** that affect training correctness or reproducibility
- **6 low issues** that are documentation or visualization hygiene
- **12 ambiguities** in the proposal itself that need to be resolved as `proposal_v4.tex` amendments

The most consequential problems are:

1. **L_seis is not the Poisson NLL** from proposal Eq. 12. The integral term `∫_Ω R(x) dx` is implemented as the unweighted mean of `R` over collocation points, missing the domain volume scaling.
2. **Z-normalization convention is partially unified.** Master commit `c7d42cf` (`Unify Z-coordinate normalization to [-1, 0]`) brought `velocity.py` and `engine.py` in line, but `inference.py`, `synthetic_eval.py`, `real_world_val.py`, and three visualization modules still use the old convention.
3. **UTM zone 39 is hardcoded** (`transformers.py:12`). Iran spans zones 38–41; the choice is a domain-coverage bug for the full plateau.

---

## 2. Spec Sources

| Symbol | Source | Lines |
|---|---|---|
| Governing equations (PDE) | `docs/proposal/proposal_v3.tex` | §2, Eq. (1) |
| Constitutive law (Stokes) | `docs/proposal/proposal_v3.tex` | §3, Eq. (2)–(3) |
| Data loss (azimuth co-axiality) | `docs/proposal/proposal_v3.tex` | §4, Eq. (4) |
| CFF (Coulomb Failure Function) | `docs/proposal/proposal_v3.tex` | §5.1, Eq. (5) |
| Dieterich rate | `docs/proposal/proposal_v3.tex` | §5.2, Eq. (6) |
| **Poisson NLL (L_seis)** | `docs/proposal/proposal_v3.tex` | §5.3, Eq. (7) |
| Total loss | `docs/proposal/proposal_v3.tex` | §6, Eq. (8) |
| Network architecture | `docs/proposal/proposal_v3.tex` | §7 |
| Validation plan | `docs/proposal/proposal_v3.tex` | §8 (deferred per user) |

---

## 3. Critical Issues (must fix)

### C-1. L_seis missing domain-volume scaling

**Proposal spec** (`proposal_v3.tex:110-115`, Eq. 7):
```latex
\mathcal{L}_{\text{seis}} = -\sum_{i=1}^{N_{\text{eq}}} \log R(\mathbf{x}_i) + \int_\Omega R(\mathbf{x})\,d\mathbf{x}
```

**Current implementation** (`src/training/engine.py:298-378`):
- Term 1 (data fit, line 309): `term1 = -torch.mean(torch.log(rate_cat.clamp(min=1e-10)))`
  - Computes `mean(log R)` over the catalog. This is **not** `Σ log R(x_i)`; the missing `1/N_eq` factor is a constant that does not affect gradient direction, so this is acceptable.
- Term 2 (integral, line 369): `l_seis = torch.mean(rate_coll)` accumulated per chunk
  - The integral `∫_Ω R(x) dx` over a 3D domain of volume `V_Ω` is approximated as `V_Ω · mean(R(x))` at Monte Carlo collocation points.
  - **The `V_Ω` factor is missing entirely.** The current code returns `mean(R)` only, which is dimensionally a rate density (events / volume / time), not a count.
  - Without `V_Ω`, the integral term and the data term have inconsistent units, and the loss-weight tuning (`w_seis`) becomes domain-size dependent.

**Severity:** Critical — this is the proposal's central novelty (Coulomb–Dieterich coupling).

**Phase:** T.9 (L_seis correctness), gating on P.1 (lock the exact loss form).

**Resolution preview:** Multiply `mean(rate_coll)` by `V_Ω = (X_max − X_min) · (Y_max − Y_min) · (Z_max − Z_min) · scale_x² · scale_z` (in physical units). The bounds are available from `self.transformer` and `vel_model`.

---

### C-2. Z-normalization convention is inconsistent

**Proposed convention** (post `c7d42cf`):
- Range: `[-1, 0]`
- `0` = surface
- `-1` = max depth

**Files in compliance** (after `c7d42cf`):
- `src/data/velocity.py:44` — `z_norm = -(deps - min_dep) / (max_dep - min_dep)` ✓
- `src/training/engine.py:280, 333, 334` — surface collocation at `0.0`, inversion at `-` ✓

**Files still using old convention:**
| File | Line | Old code | Convention | Issue |
|---|---|---|---|---|
| `src/analysis/synthetic_eval.py` | 83 | `z_norm = (coords[:, 2] / 15000.0) + 1.0` | `[-1, 1]`, +1=surface | Wrong sign, wrong range |
| `src/analysis/real_world_val.py` | 47 | `z_norm = (depths / 15000.0) - 1.0` | `[-1, 1]`, +1=max depth | Wrong sign, wrong range |
| `src/analysis/inference.py` | 71 | `z_surf = torch.full((x.shape[0], 1), -1.0, ...)` | treats `-1.0` as surface | **Bug: evaluates GPS at max depth instead of surface** |
| `src/analysis/inference.py` | 146–148 | `z_norm = (d - min) / (max - min) * 2 - 1` | `[-1, 1]`, +1=max depth | Wrong sign, wrong range |
| `src/visualize/volumetric_hero.py` | 46 | `z_norm = (D - min) / (max - min) * 2 - 1` | `[-1, 1]`, +1=max depth | Wrong sign, wrong range (viz-only) |
| `src/visualize/progression_plot.py` | 61 | `z_norm = (d - 0) / (30 - 0) * 2 - 1` | `[-1, 1]`, +1=max depth | Wrong sign, wrong range (viz-only) |
| `src/visualize/misfit_map.py` | 53 | `z_norm = np.full((..., 1), -1.0)` | treats `-1.0` as surface | Bug if convention is `[-1, 0]` |

**The C-2 bug in `inference.py:71` is particularly serious**: it means every inference call (e.g., the CFF map, error histogram, scatter) is evaluating the trained model at the bottom of the crust, not the surface, despite the comment claiming "GPS points at surface." The CFF map in `results/figs/cff_map.png` may show a depth-averaged CFF, not a `depth=15km` slice as labeled.

**Severity:** Critical — `inference.py` is called by every downstream CLI (`plot-error-hist`, `plot-scatter`, `plot-cff`, `plot-misfit`, `plot-progression`, `plot-3d-hero`, `results-suite`, `plot-all-local`).

**Phase:** P.2 (lock convention), then sweep-and-fix in T.1–T.20 wherever collocation coordinates or surface evaluation occurs.

---

### C-3. UTM zone 39 hardcoded

**Location:** `src/data/transformers.py:12`
```python
def __init__(self, lats, longs, utm_zone=39):
```

**Iran's UTM zones:**
- Zone 38: longitudes 42° E – 48° E (Western Iran: parts of Zagros, Kermanshah)
- Zone 39: longitudes 48° E – 54° E (Central Iran: Tehran, Alborz, Kopeh-Dagh)
- Zone 40: longitudes 54° E – 60° E (Eastern Iran: Khorasan)
- Zone 41: longitudes 60° E – 66° E (extreme east)

**Consequence:** Using a single UTM zone introduces a conformal distortion that grows with easting distance from the central meridian. For a point 3° (≈ 250 km) east of the central meridian, scale error is ~1 part in 1000, but the coordinate basis itself is rotated. If GPS data crosses zones, the projection is no longer conformal in the local sense.

**Severity:** Critical for full-iranian coverage; Medium for central-iranian subset (zone 39 only).

**Phase:** P.3 (decide: single-zone vs. multi-zone vs. equirectangular), then D.8 (implement).

---

## 4. Medium Issues (should fix)

### M-1. L_seis gradient is split across two backward passes
**Location:** `src/training/engine.py:298-310` and `src/training/engine.py:355-370`

`term1` (catalog term) is back-propagated **once per GPS batch** (line 310). `l_seis` (collocation term) is back-propagated **per physics chunk** (line 370). The two passes share the model parameters but use disjoint point sets.

**Issue:** This split is **mathematically correct** (Poisson NLL is a sum of two independent terms), but the **gradient scaling is off**:
- Term 1 contributes gradient scaled by `w_seis · 1` (one backward)
- Term 2 contributes gradient scaled by `w_seis / num_batches` per chunk, summed over `num_batches` chunks, total `w_seis`

The scaling is correct **only if the chunks are i.i.d. samples of the full collocation set**. If chunk boundaries bias the sampling, term 2's expected value is wrong.

**Resolution preview:** Single backward pass after computing both terms. Combine into a single `loss_seis = w_seis * (term1 + V_Omega * term2)` and backward once.

**Phase:** T.9.

---

### M-2. `tuner.py` does not search `w_seis`
**Location:** `src/training/tuner.py:46-49`

The Optuna objective searches `lr, w_pde, w_const, w_bc, f_tune`. `w_data` is hardcoded to `5.0` and `w_seis` is **not searched at all**.

The proposal says `w_seis` is the most important weight (it controls the magnitude-ambiguity resolution). Without tuning it, Optuna is sweeping 4 of 5 physics-data balance parameters.

**Phase:** T.20 (HPO search space expansion).

---

### M-3. `preprocess.py` magnitude unification is `max(MI, mb, ms, mw)`
**Location:** `src/data/preprocess.py:39-47`

```python
mag_cols = ["MI", "mb", "ms", "mw"]
available_mags = [c for c in mag_cols if c in df.columns]
df["mw_unified"] = df[available_mags].max(axis=1)
```

**Issue:** Different magnitude scales saturate at different true moment magnitudes. `mb` saturates near `mb ≈ 6.5`, `ms` near `ms ≈ 8`, `Mw` is unbounded. Taking the max means a `Mw=4.5` event with `mb=6.0` (large high-frequency radiation) becomes `mw_unified=6.0`, which is **physically wrong**.

**Standard practice:** Prefer `Mw` when available, else use empirical conversion (e.g., `mb → Mw` via `Mw = 0.85·mb + 1.03` from Goertz-Allmann et al. 2011, or Scordilis 2006). The exact conversion should be a proposal amendment.

**Phase:** P.4 (decision), then D.3 (implementation).

---

### M-4. No magnitude-of-completeness (M_c) estimation
**Location:** `src/data/loaders.py:71-74`, `src/data/preprocess.py:62-72`

M_c is a fixed user-provided parameter (`min_magnitude: float = 4.0`). The proposal (§5.3) cites the **Wiemer & Wyss 2000 MAXC method** (Maximum Curvature) and the goodness-of-fit (GFT) method.

**Issue:** M_c is critical for Poisson NLL because the rate `R(x)` is defined per earthquake of `M ≥ M_c`. Without a defensible M_c, the seismicity term is over- or under-counted.

**Phase:** P.5 (decide method), then D.5 (implement).

---

### M-5. No aftershock declustering
**Location:** `src/data/preprocess.py` (entire file)

The proposal (§5, "Data Sources") explicitly cites **Zaliapin & Ben-Zion 2008** for declustering. Without declustering, the Poisson NLL is biased: aftershock clusters are not Poisson events, they're triggered cascades. The Poisson assumption (Ogata 1998) requires stationary independent events.

**Phase:** P.6 (decide parameters), then D.6 (implement).

---

### M-6. Elastic mode is silently present but not in the proposal
**Location:** `src/core/physics.py:308-416`, `src/training/engine.py:129-142`, `src/core/config.py:80`

The proposal explicitly uses viscous (Stokes) constitutive law. The code has both `viscous` and `elastic` modes, switchable via `cfg.physics.constitutive`.

Per user decision (this audit), elastic will be **removed** and the proposal kept as-is.

**Phase:** T.4 (constitutive) — but cleanup of elastic code is part of the "P.7 amendment → T-side cleanup" chain.

---

### M-7. Random seed is declared but not applied
**Location:** `configs/real_world.yaml:42` (`seed: 42`), `src/training/engine.py` (no seed use)

Reproducibility requires seeding:
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `numpy.random.seed(seed)`
- `random.seed(seed)`
- `torch.backends.cudnn.deterministic = True` (slows training 10–20%)
- `torch.backends.cudnn.benchmark = False`

**Phase:** P.10 (policy), then T.18 (implement).

---

## 5. Low Issues (nice to fix, deferred)

### L-1. README structure out of sync with code layout
**Location:** `README.md:24-65` describes `src/physics/`, `src/validation/`, `src/analysis/`, `src/data/`, `src/training/`, `src/git_automation/`. Actual layout uses `src/core/`, `src/analysis/`, `src/validation/`, `src/data/`, `src/training/`, `src/visualize/`, `src/pipelines/`, `src/git_automation/`, `src/utils/`. README also references `configs/default.yaml` (only `real_world.yaml` exists).

**Phase:** Documentation phase (after training phase).

### L-2. Anisotropy data file unused
**Location:** `data/kinematic_data/anisotropy_sadidkhuyi2010.csv` (10 records)

This is mentioned in the proposal implicitly (SKS splitting observations constrain absolute stress orientation), but the loader doesn't ingest it.

**Phase:** Validation phase (deferred).

### L-3. `proposal_v3.pdf` is out of date relative to `proposal_v3.tex`
PDF was built from an older version. Should be rebuilt or removed in favor of `proposal_v4.pdf`.

**Phase:** End of paper prep.

### L-4. `cleaned_historical_Eq.csv` has 93 rows; raw has 94
**Location:** `data/cleaned_historical_Eq.csv` (93 records)

One event is dropped during cleaning (likely due to `dropna(subset=["lat", "long", "mw_unified"])`). The drop is silent — no log of which event or why.

**Phase:** D.3 (refactor cleaning with audit log).

### L-5. `volumetric_hero.py`, `progression_plot.py`, `misfit_map.py` use the OLD z-convention
See C-2 table. These are visualization modules and out of scope for foundation, but should be fixed in the same sweep once P.2 is locked.

**Phase:** Validation phase (deferred).

### L-6. `test_synthetic_recovery.py` not in the test path tested
**Location:** `tests/test_synthetic_recovery.py` (existence confirmed by graphify)

I did not audit the test contents. Foundation phase should verify the synthetic recovery test runs and passes the simple-shear regime.

**Phase:** T.19.

---

## 6. Open Proposal Amendments (v3 → v4)

Each item below is a **proposal-text change**, not a code change. To be applied as `proposal_v4.tex` once the user approves.

| ID | Proposal v3 says | Issue | v4 needs to say |
|---|---|---|---|
| **P.1** | L_seis = -Σ log R + ∫R dx (Eq. 7) | The MC approximation of the integral is not specified. Without domain-volume scaling, the formula is ambiguous. | Explicitly state the MC estimator: `∫_Ω R dx ≈ V_Ω · (1/N_coll) Σ R(x_i)` with `V_Ω = (X_max − X_min) · (Y_max − Y_min) · (Z_max − Z_min)` in physical units. |
| **P.2** | No explicit z-convention. Code uses surface=0, max-depth=−1. | Implicit and partially applied. | State: "Spatial coordinates are normalized to `[-1, 0]` in z where 0=surface, -1=max depth; `[-1, 1]` in x, y." |
| **P.3** | "GPS strain-rate azimuths" — no coordinate system | Iran spans UTM 38–41 | State: "For the full Iranian plateau, coordinates are projected to local transverse Mercator (UTM) using the zone of the centroid. For a single-zone subset, the appropriate zone is selected at runtime." |
| **P.4** | "Historical and instrumental seismicity" — no magnitude unification | M_max is biased | State: "When multiple magnitude types are reported, prefer Mw; if only `mb` or `ms` is available, use Scordilis (2006) empirical conversions `Mw = 0.85·mb + 1.03` (for `mb ≤ 6.1`) and `Mw = 0.67·ms + 2.07` (for `3.0 ≤ ms ≤ 6.1`)." |
| **P.5** | "M ≥ M_c" — no method | M_c is a parameter, not derived | State: "M_c is estimated per declustered catalog using the Maximum Curvature (MAXC) method (Wiemer & Wyss 2000), cross-checked with the Goodness-of-Fit Test (GFT) for a 90% confidence threshold." |
| **P.6** | "Aftershock declustering is applied via the Zaliapin–Ben-Zion method" — no parameters | Zaliapin-Ben-Zion has multiple parameters (b-value threshold, space-time distance metric) | State: "Declustering uses the Zaliapin–Ben-Zion (2008) nearest-neighbor distance with `b = 1.0` (regional Gutenberg-Richter b-value) and the recommended threshold `η_0 = 1.0` for crustal seismicity." |
| **P.7** | "Steady-state viscous (Stokes) constitutive" | Code also supports elastic | **Restrict to viscous.** Note in v4 §3: "The framework considers only the viscous (Stokes) constitutive law for the interseismic period; the elastic mode implemented in code is a deprecated prototype and not used in the published results." |
| **P.8** | "4 hidden layers with tanh" | Width and initialization not specified | Add: "Hidden width 128, Xavier-normal initialization, Tanh activation, Kaiming-zero for output biases." |
| **P.9** | "~2000 observations" (GPS) | We have 209 | Add footnote: "GPS observations used in this study total N=209 across the Rayisi (2016) and Khorrami et al. (2019) datasets; the proposal's reference to 2000 reflects the larger combined GPS+interferometric catalog available to future work." |
| **P.10** | Nothing about stochasticity | Random seed not used | Add §6.1: "All experiments are reproducible via `torch.manual_seed(42)`, `numpy.random.seed(42)`, and `PYTHONHASHSEED=0`. Ensemble uncertainty quantification (deferred to validation) will use seeds 42, 123, 1024, 31337, 65535." |

---

## 7. Cross-Reference: Issues → Phases

```
Critical ───────────────────────────────────────────────
  C-1 (L_seis volume scaling)        → P.1, T.9
  C-2 (z-normalization)              → P.2, then sweep T.*
  C-3 (UTM zone)                     → P.3, D.8

Medium ─────────────────────────────────────────────────
  M-1 (L_seis grad split)            → T.9
  M-2 (Optuna: w_seis missing)       → T.20
  M-3 (magnitude unification)        → P.4, D.3
  M-4 (M_c estimation)               → P.5, D.5
  M-5 (declustering)                 → P.6, D.6
  M-6 (elastic mode)                 → P.7, T.4 cleanup
  M-7 (random seed)                  → P.10, T.18

Low (deferred) ─────────────────────────────────────────
  L-1..L-6                           → post-foundation
```

---

## 8. What's NOT in this audit

- I did not run the test suite (`pytest`).
- I did not run a synthetic recovery end-to-end (T.19).
- I did not check the `analysis/real_world_val.py` focal-mechanism CSV format in detail (it expects columns `longitude, latitude, azimuth_value`, which matches the existing `data/kinematic_data/stress_*.csv` files).
- I did not check the `git_automation/` subsystem end-to-end (auto-push is validation-phase territory).
- I did not review `src/visualize/*` (deferred).
- I did not check whether the `train` CLI actually completes on a tiny example (would require a 20000-epoch run).

These are listed for completeness; the foundation phase can decide what to verify next.

---

## 9. Recommended First Action

Start with **P.1** (lock L_seis form), because:
1. It is the proposal's central novelty.
2. The code's current form is mathematically wrong (missing volume scaling).
3. It blocks T.9 (L_seis correctness).
4. Once locked, every other loss term can be reviewed against a stable reference.

After P.1: P.2 (z-convention), then P.3 (UTM zone). These three amendments unblock all data and training work.

---

*End of audit.*
