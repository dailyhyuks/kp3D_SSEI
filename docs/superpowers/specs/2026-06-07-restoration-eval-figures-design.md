# Restoration Evaluation Figures — Design Spec

- Date: 2026-06-07
- Topic: Regenerate restoration evaluation figures for the updated Restoration V3 algorithm
- Status: Approved design, pending spec review

## 1. Background & Problem

The Restoration V3 pipeline (Split-Radius FFT notch → Spatial-Adaptive NLM → Contour
Enhancement) has been updated in the source modules
(`src/kp3d/modules/weave_removal/`). The existing paper figures under
`figures/F_restoration_*.png` were rendered from stale assets and must be regenerated
from the current algorithm.

The user wants a dedicated **restoration evaluation** figure set (6 figures) built in a
**separate folder**, leaving all existing figures and assets untouched.

### Confirmed decisions

| Topic | Decision |
|-------|----------|
| Algorithm change state | Code already updated; only assets/figures need regeneration (no module edits) |
| Input cases | Generate assets for **all 29 cases** in `data_original_painting/data_anno/*.png` |
| Output location | New folder `figures/restoration_eval/` (existing figures preserved) |
| Case layout | **Hybrid**: Effects/Ablation (fig 1–4) use single representative case; Baseline/Detail (fig 5–6) use multiple cases as rows |
| Representative case | `1_0022` (consistent with existing figures) |
| Quantitative metrics | **Include** GridE & EdgePres in fig 4 (ablation) and fig 5 (baseline); also emit `metrics.csv` |

## 2. Existing assets reused (read-only, not modified)

- Asset generator pattern: `figures/overview_assets_1_0022/_run_weave_removal_v3.py`
- Module API:
  - `WeaveRemovalPreset.V3.to_config()`
  - `process_image_patchwise(...)` (Stage 1; `include_cross_harmonics`, `split_radius`, `peak_radius`, `cross_peak_radius`)
  - `compute_narrow_region_mask(img_bgr, nlm_cfg)` (Stage 2 mask)
  - `spatial_adaptive_nlm(img_bgr, split_out, nlm_cfg)` (Stage 2)
  - `enhance_contours(img_bgr, nlm_out, boost, block_size, thresh_c)` (Stage 3)
- Metrics API (numpy, BGR-friendly): `experiments/neural_restoration/metrics.py`
  - `compute_grid_energy_reduction(original, restored, k=20)` → GridE
  - `compute_edge_preservation(original, restored)` → EdgePres
  - `compute_all_metrics(original, restored)` → dict
- Baseline filters (computed inline inside figure scripts, as in existing
  `_make_F_restoration_baselines_grid.py`): Butterworth FFT notch, Median, Bilateral,
  NL-Means, Guided.

## 3. Output directory structure

```
figures/restoration_eval/
├── _run_assets.py                  # unified asset generator (29-case loop)
├── assets/
│   └── {case}/                     # e.g. 1_0022/, 1_0036/, ...
│       ├── 00_original.png
│       ├── fft_00_original.png
│       ├── 01a_after_split_cross_off.png   # NEW: include_cross_harmonics=False
│       ├── 01_after_split_radius.png        # cross_harmonics ON (full Stage 1)
│       ├── fft_01_after_split.png
│       ├── diff_01_orig_vs_split.png
│       ├── 02_narrow_region_mask.png
│       ├── 02_narrow_region_mask_color.png
│       ├── 03_after_nlm_adaptive.png
│       ├── fft_03_after_nlm.png
│       ├── diff_03_split_vs_nlm.png
│       ├── 04_after_contour_enhance.png     # final
│       ├── fft_04_final.png
│       ├── diff_04_nlm_vs_final.png
│       ├── diff_total_orig_vs_final.png
│       └── zoom_{00,01a,01,03,04}.png       # 96x96 @ 4x NN
├── metrics.csv                     # NEW: per-case, per-stage GridE & EdgePres
├── _make_fig1_cross_harmonic.py    → F_eval_1_cross_harmonic.png
├── _make_fig2_adaptive_nlm.py      → F_eval_2_adaptive_nlm.png
├── _make_fig3_contour.py           → F_eval_3_contour.png
├── _make_fig4_ablation.py          → F_eval_4_ablation.png
├── _make_fig5_baselines.py         → F_eval_5_baselines.png
└── _make_fig6_input_nlm_ours.py    → F_eval_6_input_nlm_ours.png
```

## 4. Component design

### 4.1 `_run_assets.py` (asset generator)

Single-responsibility: take an input painting, run the V3 pipeline stage-by-stage, and
persist every intermediate the figures need.

- Inputs: loop over all 29 `data_anno/*.png`. Optional CLI arg to restrict to a case
  subset (default: all).
- For each case, write per-stage PNGs (table in §3) into `assets/{case}/`.
- **New vs old generator:**
  1. Add a `cross_harmonics OFF` Stage-1 variant (`01a_after_split_cross_off.png`,
     `zoom_01a...`) by calling `process_image_patchwise(..., include_cross_harmonics=False)`.
     The normal `01_after_split_radius.png` keeps `include_cross_harmonics=True`.
  2. Compute metrics per stage via `compute_all_metrics(original, stage_output)` and append
     rows to `metrics.csv` with columns:
     `case, stage, gride, edgepres` where `stage ∈ {split_cross_off, split, nlm, contour}`.
- FFT visualization reuses the existing `fft_log_uint8` (LAB-L, Hann window, log magnitude,
  INFERNO colormap).
- Zoom crop reuses the existing 96×96 @ (cy=40, cx=40), 4× nearest upscale.
- Idempotent: re-running overwrites `assets/{case}/` deterministically.

### 4.2 Figure labeling convention (publication-facing text)

**Hard rule:** No internal case IDs (e.g. `1_0022`) may appear in any rendered figure text —
titles, subtitles, panel captions, row/column labels, axis labels, or annotations.

- Refer to samples generically: `Example`, `Example (a)`, `Example (b)`, ...
- Use sub-figure letters for panels/steps: `(a)`, `(b)`, `(c)`, ... optionally with a short
  descriptor (e.g. `(a) Input`, `(b) + Split-Radius`).
- Multi-case rows (fig 5/6) are labeled `Example (a)`, `Example (b)`, ... top-to-bottom — never
  the case ID.
- Case IDs remain allowed only in non-rendered places: source file paths, `assets/{case}/`
  directory names, and the `case` column of `metrics.csv` (data file, not a figure).
- Output figure filenames stay generic (`F_eval_1_cross_harmonic.png`, etc.) — already ID-free.

### 4.3 Figure scripts (consume assets only; no pipeline recompute except baselines)

All figure scripts load PNGs from `assets/{case}/`. Baseline filters (fig 5/6) are computed
inline from `00_original.png`, matching the existing pattern. Metrics are read from
`metrics.csv`. All rendered text follows the §4.2 labeling convention.

**fig1 — Cross-harmonic Effects** (`1_0022`)
- Compare `01a_after_split_cross_off` vs `01_after_split_radius`.
- Layout: 2 columns (cross OFF / cross ON). Row 1 full image with ROI box; Row 2 zoomed ROI
  (nearest). Optional 3rd FFT row showing cross-harmonic peak suppression.

**fig2 — Adaptive-NLM Effects** (`1_0022`)
- Compare `01_after_split_radius` (input to NLM) vs `03_after_nlm_adaptive`, plus the
  `02_narrow_region_mask_color` to show where NLM is gated.
- Layout: 3 panels (Split result / Narrow-region mask / NLM result) + zoom row of the
  narrow-region detail.

**fig3 — Contour Enhancement Effects** (`1_0022`)
- Compare `03_after_nlm_adaptive` vs `04_after_contour_enhance`.
- Layout: 2 columns (NLM / +Contour), full + zoom row emphasizing recovered ink-line detail.

**fig4 — Ablation (Stages 1→3)** (`1_0022`)
- Progressive panels: Original → +Split → +NLM → +Contour(final), each with a zoom inset.
- Quantitative: GridE & EdgePres bar chart (one group per stage) read from `metrics.csv`.

**fig5 — Baseline Comparison** (multiple cases as rows)
- Columns: Input / Butterworth / Median / Bilateral / NL-Means / Guided / **Ours**.
- Rows: a curated set of cases (zoom patches), "Ours" = `04_after_contour_enhance`.
- Quantitative: per-method GridE/EdgePres summary (mean over cases) read/computed and
  annotated under the grid or as a side bar chart.

**fig6 — Input vs NL-Means vs Ours (detail)** (multiple cases as rows)
- Columns: Input / NL-Means(h=15) / Ours. Rows: case zoom patches.
- Purpose: show NL-Means over-smoothing vs Ours preserving line detail.

## 5. Data flow

```
data_anno/{case}.png
   │  _run_assets.py (V3 stages + cross-off variant + metrics)
   ▼
assets/{case}/*.png  +  metrics.csv
   │  _make_fig{1..6}_*.py (load PNG, inline baselines, read metrics)
   ▼
figures/restoration_eval/F_eval_{1..6}_*.png
```

## 6. Error handling & edge cases

- Missing/failed image read → skip case with a logged warning; continue the loop.
- Variable image sizes across the 29 cases → ROI/zoom boxes must be clamped to image bounds
  (existing generator already clamps `ph/pw`). Figure ROI coords are per-figure constants
  validated against the representative case size.
- `metrics.csv` regenerated fresh each full run (no stale append): open in write mode, write
  header once.
- Baseline filters operate on `00_original.png` only, so they are unaffected by asset variant
  naming.

## 7. Testing / verification

- Smoke: run `_run_assets.py` restricted to `1_0022`, assert all expected PNGs exist and
  `metrics.csv` has 4 stage rows for the case.
- Visual: open each `F_eval_*.png` and confirm panels are populated (no missing-asset blanks).
- Sanity on metrics: GridE for `contour` stage should be lower than `original`/baseline
  (grid reduced); EdgePres should remain within an expected band (per WORK_REPORT, ~0.60–0.75).
- Full run: generate all 29 cases, then render all 6 figures.

## 8. Out of scope (YAGNI)

- No changes to `src/kp3d/modules/weave_removal/` (algorithm already updated).
- No changes to existing `figures/F_restoration_*.png` or `overview_assets_1_0022/`.
- No new metric definitions; reuse `experiments/neural_restoration/metrics.py`.
- The legacy `tools/generate_{split_radius,contour_enhancement}_figure.py` and their
  intermediate inputs are superseded by fig1/fig3 here and will not be revived.
