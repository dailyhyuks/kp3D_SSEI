# Restoration Evaluation Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the 6 restoration-evaluation figures from the updated Restoration V3 algorithm into a new isolated folder `figures/restoration_eval/`, with all rendered text using generic labels (`Example`, `(a)`, `(b)`...) instead of internal case IDs.

**Architecture:** A single asset generator runs the V3 pipeline (Split-Radius FFT → Adaptive NLM → Contour) plus a `cross_harmonics=OFF` variant over all 29 painting cases, saving per-stage PNGs and a `metrics.csv` (GridE, EdgePres). Six figure scripts consume those PNGs (and inline baseline filters) to render the publication figures. A shared `_common.py` holds I/O, FFT/zoom helpers, and the case→`Example` label mapping so no case ID leaks into rendered text.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, Matplotlib. Reuses `kp3d.modules.weave_removal` and `experiments/neural_restoration/metrics.py`.

**Note on git/TDD:** The project is NOT a git repo, so there are no commit steps. Because outputs are visual artifacts, "tests" are smoke checks asserting expected files exist and have correct shape/contents; run them with plain `python`.

**Note on the spec:** See `docs/superpowers/specs/2026-06-07-restoration-eval-figures-design.md`. §4.2 labeling convention is mandatory for every rendered string.

---

## File Structure

- Create: `figures/restoration_eval/_common.py` — shared helpers (I/O, FFT, diff, zoom, metrics import shim, label mapping)
- Create: `figures/restoration_eval/_run_assets.py` — asset generator (29-case loop + cross-off variant + metrics.csv)
- Create: `figures/restoration_eval/_make_fig1_cross_harmonic.py`
- Create: `figures/restoration_eval/_make_fig2_adaptive_nlm.py`
- Create: `figures/restoration_eval/_make_fig3_contour.py`
- Create: `figures/restoration_eval/_make_fig4_ablation.py`
- Create: `figures/restoration_eval/_make_fig5_baselines.py`
- Create: `figures/restoration_eval/_make_fig6_input_nlm_ours.py`
- Generated (not authored): `figures/restoration_eval/assets/{case}/*.png`, `figures/restoration_eval/metrics.csv`, `figures/restoration_eval/F_eval_{1..6}_*.png`

Reference templates (read-only, do NOT modify):
- `figures/overview_assets_1_0022/_run_weave_removal_v3.py` (asset-gen pattern)
- `figures/_make_F_restoration_baselines_grid.py` (baseline filters + grid layout)
- `figures/_make_F_restoration_nlmeans_vs_ours.py` (Input/NLM/Ours layout)

**Python invocation:** All scripts run from project root with the venv python. Use:
`C:/Users/admin/korean-painting-3d/venv_3d/Scripts/python.exe <script>` (fallback: `python` if that path is absent — verify in Task 1 Step 1).

---

## Task 1: Shared helpers (`_common.py`)

**Files:**
- Create: `figures/restoration_eval/_common.py`
- Verify: choose the working python interpreter

- [ ] **Step 1: Pick the python interpreter**

Run (from project root):
```bash
ls figures/ ; ls venv_3d/Scripts/python.exe 2>/dev/null || ls venv_3d/bin/python 2>/dev/null || echo "use plain python"
```
Record the interpreter path (call it `PY`). Used for every later `Run:` step.

- [ ] **Step 2: Write `_common.py`**

Create `figures/restoration_eval/_common.py` with full contents:

```python
"""Shared helpers for restoration_eval figures.

Centralizes image I/O, FFT/diff/zoom visualization, the metrics import shim,
and the case-ID -> generic "Example" label mapping (spec §4.2: no internal
case IDs may appear in any rendered figure text).
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:/Users/admin/korean-painting-3d")
SRC = ROOT / "src"
EVAL_DIR = ROOT / "figures" / "restoration_eval"
ASSETS_DIR = EVAL_DIR / "assets"
DATA_ANNO = ROOT / "data_original_painting" / "data_anno"
METRICS_CSV = EVAL_DIR / "metrics.csv"

# Make kp3d and the metrics module importable.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
_METRICS_DIR = ROOT / "experiments" / "neural_restoration"
if str(_METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(_METRICS_DIR))


def imread_bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fft_log_uint8(img_bgr: np.ndarray) -> np.ndarray:
    """LAB L-channel log-magnitude FFT (Hann-windowed), INFERNO-colormapped."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    L = L - L.mean()
    h, w = L.shape
    Lw = L * (np.hanning(h)[:, None] * np.hanning(w)[None, :])
    F = np.fft.fftshift(np.fft.fft2(Lw))
    mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
    return cv2.applyColorMap((mag * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def diff_map_uint8(a: np.ndarray, b: np.ndarray, gain: float = 4.0) -> np.ndarray:
    d = cv2.absdiff(a, b).astype(np.float32)
    return np.clip(d * gain, 0, 255).astype(np.uint8)


def zoom4(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (arr.shape[1] * 4, arr.shape[0] * 4),
                      interpolation=cv2.INTER_NEAREST)


def crop_patch(bgr: np.ndarray, box: tuple[int, int, int, int], up: int = 4) -> np.ndarray:
    """box = (y, x, h, w); returns up-scaled (nearest) crop, clamped to bounds."""
    y, x, h, w = box
    H, W = bgr.shape[:2]
    y = max(0, min(y, H - 1)); x = max(0, min(x, W - 1))
    h = min(h, H - y); w = min(w, W - x)
    p = bgr[y:y + h, x:x + w]
    return cv2.resize(p, (p.shape[1] * up, p.shape[0] * up), interpolation=cv2.INTER_NEAREST)


def list_cases() -> list[str]:
    """All case stems in data_anno (e.g. '1_0022'), sorted."""
    return sorted(p.stem for p in DATA_ANNO.glob("*.png"))


def example_label(index: int) -> str:
    """0 -> 'Example (a)', 1 -> 'Example (b)', ... (spec §4.2). No case IDs."""
    return f"Example ({chr(ord('a') + index)})"


def panel_label(index: int, descriptor: str = "") -> str:
    """0 -> '(a)', 1 -> '(b)'; with descriptor -> '(a) Input'."""
    base = f"({chr(ord('a') + index)})"
    return f"{base} {descriptor}".strip()


def load_metrics() -> list[dict]:
    """Read metrics.csv into a list of dict rows (case, stage, gride, edgepres)."""
    import csv
    if not METRICS_CSV.exists():
        return []
    with open(METRICS_CSV, newline="") as f:
        return list(csv.DictReader(f))
```

- [ ] **Step 3: Smoke-check the helpers**

Run:
```bash
<PY> -c "import sys; sys.path.insert(0,'figures/restoration_eval'); import _common as c; print(len(c.list_cases()), c.example_label(0), c.panel_label(1,'Input'))"
```
Expected: prints `29 Example (a) (b) Input` (case count may differ if data changes; must be ≥1).

---

## Task 2: Asset generator core (single-case smoke)

**Files:**
- Create: `figures/restoration_eval/_run_assets.py`

- [ ] **Step 1: Write `_run_assets.py`**

Create `figures/restoration_eval/_run_assets.py` with full contents:

```python
"""Generate restoration-eval assets for all cases (or a CLI-supplied subset).

For each case, runs the V3 pipeline stage-by-stage plus a cross-harmonics=OFF
Stage-1 variant, saves per-stage PNGs into assets/{case}/, and appends GridE &
EdgePres rows to metrics.csv. See spec 2026-06-07-restoration-eval-figures-design.md.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from kp3d.modules.weave_removal import (  # noqa: E402
    WeaveRemovalPreset,
    process_image_patchwise,
    spatial_adaptive_nlm,
    compute_narrow_region_mask,
    enhance_contours,
    SpatialAdaptiveNLMConfig,
)
from metrics import compute_grid_energy_reduction, compute_edge_preservation  # noqa: E402

# 96x96 zoom crop origin (matches existing generator)
ZCY, ZCX, ZPH, ZPW = 40, 40, 96, 96


def _stage1(img_bgr, cfg, include_cross: bool):
    out, conf = process_image_patchwise(
        img_bgr,
        patch_size=cfg.patch_size, overlap_ratio=cfg.overlap_ratio,
        alpha=cfg.alpha, method=cfg.method, min_prominence=cfg.min_prominence,
        channel_mode=cfg.channel_mode, edge_aware=cfg.edge_aware,
        edge_alpha_min=cfg.edge_alpha_min, peak_radius=cfg.peak_radius,
        cross_peak_radius=cfg.cross_peak_radius, adaptive_radius=cfg.adaptive_radius,
        split_radius=cfg.split_radius, include_cross_harmonics=include_cross,
        cross_harmonic_threshold=cfg.cross_harmonic_threshold,
    )
    return out


def _nlm_cfg(cfg) -> SpatialAdaptiveNLMConfig:
    return SpatialAdaptiveNLMConfig(
        h_base=cfg.nlm_h_base, h_max=cfg.nlm_h_max,
        h_color_base=cfg.nlm_h_color_base, h_color_max=cfg.nlm_h_color_max,
        narrow_threshold=cfg.nlm_narrow_threshold, edge_threshold=cfg.nlm_edge_threshold,
        template_window=cfg.nlm_template_window, search_window=cfg.nlm_search_window,
        n_clusters=cfg.nlm_n_clusters, min_cluster_area=cfg.nlm_min_cluster_area,
        blur_sigma=cfg.nlm_blur_sigma,
    )


def process_case(case: str) -> list[dict]:
    """Run pipeline for one case, save assets, return metric rows."""
    img_path = c.DATA_ANNO / f"{case}.png"
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  WARN: cannot read {img_path}; skipping")
        return []

    out_dir = c.ASSETS_DIR / case
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        cv2.imwrite(str(out_dir / name), arr)

    cfg = WeaveRemovalPreset.V3.to_config()
    ncfg = _nlm_cfg(cfg)

    # Stage 0
    save("00_original.png", img_bgr)
    save("fft_00_original.png", c.fft_log_uint8(img_bgr))

    # Stage 1 variants
    split_off = _stage1(img_bgr, cfg, include_cross=False)
    split_on = _stage1(img_bgr, cfg, include_cross=True)
    save("01a_after_split_cross_off.png", split_off)
    save("01_after_split_radius.png", split_on)
    save("fft_01_after_split.png", c.fft_log_uint8(split_on))
    save("diff_01_orig_vs_split.png", c.diff_map_uint8(img_bgr, split_on))

    # Stage 2 mask + NLM
    narrow_w = compute_narrow_region_mask(img_bgr, ncfg)
    narrow_u8 = (narrow_w * 255).astype(np.uint8)
    save("02_narrow_region_mask.png", narrow_u8)
    save("02_narrow_region_mask_color.png",
         cv2.applyColorMap(narrow_u8, cv2.COLORMAP_VIRIDIS))
    nlm_out = spatial_adaptive_nlm(img_bgr, split_on, ncfg)
    save("03_after_nlm_adaptive.png", nlm_out)
    save("fft_03_after_nlm.png", c.fft_log_uint8(nlm_out))
    save("diff_03_split_vs_nlm.png", c.diff_map_uint8(split_on, nlm_out))

    # Stage 3 contour
    final_out = enhance_contours(
        img_bgr, nlm_out,
        boost=cfg.contour_boost, block_size=cfg.contour_block_size,
        thresh_c=cfg.contour_thresh_c,
    )
    save("04_after_contour_enhance.png", final_out)
    save("fft_04_final.png", c.fft_log_uint8(final_out))
    save("diff_04_nlm_vs_final.png", c.diff_map_uint8(nlm_out, final_out))
    save("diff_total_orig_vs_final.png", c.diff_map_uint8(img_bgr, final_out))

    # Zoom crops (clamped)
    H, W = img_bgr.shape[:2]
    ph, pw = min(ZPH, H - ZCY), min(ZPW, W - ZCX)
    for name, arr in [
        ("zoom_00_original.png", img_bgr),
        ("zoom_01a_after_split_cross_off.png", split_off),
        ("zoom_01_after_split.png", split_on),
        ("zoom_03_after_nlm.png", nlm_out),
        ("zoom_04_after_contour.png", final_out),
    ]:
        crop = arr[ZCY:ZCY + ph, ZCX:ZCX + pw]
        save(name, c.zoom4(crop))

    # Metrics (GridE: higher=more grid removed; EdgePres in [0,1])
    rows = []
    for stage, restored in [
        ("split_cross_off", split_off),
        ("split", split_on),
        ("nlm", nlm_out),
        ("contour", final_out),
    ]:
        rows.append({
            "case": case, "stage": stage,
            "gride": round(compute_grid_energy_reduction(img_bgr, restored), 6),
            "edgepres": round(compute_edge_preservation(img_bgr, restored), 6),
        })
    return rows


def main():
    cases = sys.argv[1:] or c.list_cases()
    c.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case}")
        all_rows.extend(process_case(case))

    # Write metrics.csv fresh (header once)
    with open(c.METRICS_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["case", "stage", "gride", "edgepres"])
        wtr.writeheader()
        wtr.writerows(all_rows)
    print(f"Wrote {len(all_rows)} metric rows -> {c.METRICS_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run single-case smoke**

Run:
```bash
<PY> figures/restoration_eval/_run_assets.py 1_0022
```
Expected: prints `[1/1] 1_0022` then `Wrote 4 metric rows`. No traceback.

- [ ] **Step 3: Verify single-case assets exist**

Run:
```bash
<PY> -c "import sys;sys.path.insert(0,'figures/restoration_eval');import _common as c; d=c.ASSETS_DIR/'1_0022'; req=['00_original.png','01a_after_split_cross_off.png','01_after_split_radius.png','02_narrow_region_mask_color.png','03_after_nlm_adaptive.png','04_after_contour_enhance.png','fft_00_original.png','zoom_01_after_split.png']; miss=[x for x in req if not (d/x).exists()]; print('MISSING',miss) if miss else print('ALL ASSETS OK'); r=c.load_metrics(); print('rows',len(r),'stages',sorted({x[\"stage\"] for x in r}))"
```
Expected: `ALL ASSETS OK` and `rows 4 stages ['contour', 'nlm', 'split', 'split_cross_off']`.

---

## Task 3: Full 29-case asset generation

**Files:** none (runs Task 2 script over all cases)

- [ ] **Step 1: Run all cases**

Run (may take several minutes; run in background if needed):
```bash
<PY> figures/restoration_eval/_run_assets.py
```
Expected: prints `[1/29] ...` through `[29/29] ...` then `Wrote 116 metric rows` (29×4). Skipped cases (if any unreadable) log a WARN and reduce the count.

- [ ] **Step 2: Verify coverage**

Run:
```bash
<PY> -c "import sys;sys.path.insert(0,'figures/restoration_eval');import _common as c; cases=c.list_cases(); ok=[x for x in cases if (c.ASSETS_DIR/x/'04_after_contour_enhance.png').exists()]; print(f'{len(ok)}/{len(cases)} cases have final output'); r=c.load_metrics(); print('metric rows',len(r))"
```
Expected: `29/29 cases have final output` (or N/N for available cases) and `metric rows 116`.

- [ ] **Step 3: Sanity-check metrics direction**

Run:
```bash
<PY> -c "import sys;sys.path.insert(0,'figures/restoration_eval');import _common as c; r=[x for x in c.load_metrics() if x['case']=='1_0022']; d={x['stage']:(float(x['gride']),float(x['edgepres'])) for x in r}; print(d); assert d['nlm'][0]>=d['split_cross_off'][0]-0.05, 'grid removal at NLM should not be worse than cross-off baseline'; assert d['contour'][1]>0.0, 'contour should preserve some edges'; print('SANITY OK')"
```
Expected: prints the 4-stage dict and `SANITY OK`. If the assertion fails, STOP and report the metric values (do not fudge).

**Note on contour-stage GridE:** `compute_grid_energy_reduction` (metrics.py:210-213) clips a negative
`reduction_ratio` to 0.0. The contour stage is an ink-line *enhancement* step that adds high-frequency
energy, so energy at the original grid-peak frequencies increases → ratio goes negative → reported as
0.0. This is expected, not a bug: grid removal is performed (and measured) at the split/nlm stages
(GridE ≈ 0.079/0.094 for 1_0022), while the contour stage is evaluated via EdgePres. The sanity-check
therefore asserts grid reduction at the **nlm** stage and edge preservation (> 0) at the **contour** stage.

---

## Task 4: Figure 1 — Cross-harmonic Effects

**Files:**
- Create: `figures/restoration_eval/_make_fig1_cross_harmonic.py`

- [ ] **Step 1: Write the script**

Create `figures/restoration_eval/_make_fig1_cross_harmonic.py`:

```python
"""F_eval_1_cross_harmonic.png — cross-harmonic OFF vs ON in Stage 1.

Representative example only. No case IDs in rendered text (spec §4.2).
Layout: 2 columns (cross OFF / cross ON), 2 rows (full + ROI zoom) + FFT row.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

CASE = "1_0022"          # representative example (not rendered)
ROI = (10, 175, 120, 120)  # (y,x,h,w) zoom region with visible weave
OUT = c.EVAL_DIR / "F_eval_1_cross_harmonic.png"

d = c.ASSETS_DIR / CASE
off = c.imread_bgr(d / "01a_after_split_cross_off.png")
on = c.imread_bgr(d / "01_after_split_radius.png")
fft_off = c.fft_log_uint8(off)
fft_on = c.fft_log_uint8(on)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 120})
fig = plt.figure(figsize=(9, 12))
fig.suptitle("Cross-Harmonic Peak Removal in Spectral Interpolation",
             fontsize=15, fontweight="bold", y=0.98)
gs = GridSpec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.18, wspace=0.06,
              top=0.93, bottom=0.02, left=0.04, right=0.98)

cols = [("(a) cross-harmonic OFF", off, fft_off),
        ("(b) cross-harmonic ON (adopted)", on, fft_on)]
for ci, (title, img, fft) in enumerate(cols):
    # Row 0: full image with ROI box
    ax = fig.add_subplot(gs[0, ci]); ax.imshow(c.to_rgb(img))
    ax.add_patch(mpatches.Rectangle((ROI[1], ROI[0]), ROI[3], ROI[2],
                 ec="#E53935", fc="none", lw=1.8))
    ax.set_title(title, pad=6); ax.set_xticks([]); ax.set_yticks([])
    # Row 1: ROI zoom
    axz = fig.add_subplot(gs[1, ci]); axz.imshow(c.to_rgb(c.crop_patch(img, ROI)))
    axz.set_title("zoomed ROI (nearest)", fontsize=10); axz.set_xticks([]); axz.set_yticks([])
    # Row 2: FFT magnitude
    axf = fig.add_subplot(gs[2, ci]); axf.imshow(c.to_rgb(fft))
    axf.set_title("FFT magnitude", fontsize=10); axf.set_xticks([]); axf.set_yticks([])

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
plt.close(fig)
```

- [ ] **Step 2: Run and verify**

Run:
```bash
<PY> figures/restoration_eval/_make_fig1_cross_harmonic.py
```
Expected: `Saved: ...F_eval_1_cross_harmonic.png`. Then verify file exists and is non-trivial:
```bash
<PY> -c "import os;p='figures/restoration_eval/F_eval_1_cross_harmonic.png';print('OK',os.path.getsize(p)) if os.path.getsize(p)>50000 else print('TOO SMALL')"
```
Expected: `OK <size>`.

- [ ] **Step 3: Label audit**

Confirm no case ID in the script's rendered strings (titles/labels). The string `"1_0022"` may appear ONLY as the `CASE` constant, never inside `set_title`/`suptitle`/`set_ylabel`. Visually confirm by reading the file.

---

## Task 5: Figure 2 — Adaptive-NLM Effects

**Files:**
- Create: `figures/restoration_eval/_make_fig2_adaptive_nlm.py`

- [ ] **Step 1: Write the script**

Create `figures/restoration_eval/_make_fig2_adaptive_nlm.py`:

```python
"""F_eval_2_adaptive_nlm.png — Stage 2 spatial-adaptive NLM effect.

3 columns: Split result | Narrow-region mask | NLM result, + ROI zoom row.
Representative example only; no case IDs in rendered text (spec §4.2).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

CASE = "1_0022"
ROI = (75, 30, 120, 120)
OUT = c.EVAL_DIR / "F_eval_2_adaptive_nlm.png"

d = c.ASSETS_DIR / CASE
split = c.imread_bgr(d / "01_after_split_radius.png")
mask = c.imread_bgr(d / "02_narrow_region_mask_color.png")
nlm = c.imread_bgr(d / "03_after_nlm_adaptive.png")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 120})
fig = plt.figure(figsize=(13, 9))
fig.suptitle("Spatial-Adaptive NLM (Narrow-Region Gated)",
             fontsize=15, fontweight="bold", y=0.98)
gs = GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.16, wspace=0.06,
              top=0.92, bottom=0.02, left=0.03, right=0.99)

cols = [("(a) after Split-Radius", split, True),
        ("(b) narrow-region mask", mask, False),
        ("(c) after Adaptive NLM", nlm, True)]
for ci, (title, img, boxed) in enumerate(cols):
    ax = fig.add_subplot(gs[0, ci]); ax.imshow(c.to_rgb(img))
    if boxed:
        ax.add_patch(mpatches.Rectangle((ROI[1], ROI[0]), ROI[3], ROI[2],
                     ec="#E53935", fc="none", lw=1.8))
    ax.set_title(title, pad=6); ax.set_xticks([]); ax.set_yticks([])
    axz = fig.add_subplot(gs[1, ci]); axz.imshow(c.to_rgb(c.crop_patch(img, ROI)))
    axz.set_title("zoomed ROI (nearest)", fontsize=10); axz.set_xticks([]); axz.set_yticks([])

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
plt.close(fig)
```

- [ ] **Step 2: Run and verify**

Run:
```bash
<PY> figures/restoration_eval/_make_fig2_adaptive_nlm.py
```
Expected: `Saved: ...F_eval_2_adaptive_nlm.png`. Verify size:
```bash
<PY> -c "import os;p='figures/restoration_eval/F_eval_2_adaptive_nlm.png';print('OK',os.path.getsize(p)) if os.path.getsize(p)>50000 else print('TOO SMALL')"
```
Expected: `OK <size>`.

---

## Task 6: Figure 3 — Contour Enhancement Effects

**Files:**
- Create: `figures/restoration_eval/_make_fig3_contour.py`

- [ ] **Step 1: Write the script**

Create `figures/restoration_eval/_make_fig3_contour.py`:

```python
"""F_eval_3_contour.png — Stage 3 contour enhancement effect.

2 columns: after NLM | + Contour, with ROI zoom row emphasizing ink-line detail.
Representative example only; no case IDs in rendered text (spec §4.2).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

CASE = "1_0022"
ROI = (170, 5, 120, 120)
OUT = c.EVAL_DIR / "F_eval_3_contour.png"

d = c.ASSETS_DIR / CASE
nlm = c.imread_bgr(d / "03_after_nlm_adaptive.png")
final = c.imread_bgr(d / "04_after_contour_enhance.png")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 120})
fig = plt.figure(figsize=(9, 9))
fig.suptitle("Contour (Ink-Line) Enhancement",
             fontsize=15, fontweight="bold", y=0.98)
gs = GridSpec(2, 2, hspace=0.16, wspace=0.06,
              top=0.92, bottom=0.02, left=0.04, right=0.98)

cols = [("(a) after Adaptive NLM", nlm),
        ("(b) + Contour Enhancement (final)", final)]
for ci, (title, img) in enumerate(cols):
    ax = fig.add_subplot(gs[0, ci]); ax.imshow(c.to_rgb(img))
    ax.add_patch(mpatches.Rectangle((ROI[1], ROI[0]), ROI[3], ROI[2],
                 ec="#E53935", fc="none", lw=1.8))
    ax.set_title(title, pad=6); ax.set_xticks([]); ax.set_yticks([])
    axz = fig.add_subplot(gs[1, ci]); axz.imshow(c.to_rgb(c.crop_patch(img, ROI)))
    axz.set_title("zoomed ROI (nearest)", fontsize=10); axz.set_xticks([]); axz.set_yticks([])

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
plt.close(fig)
```

- [ ] **Step 2: Run and verify**

Run:
```bash
<PY> figures/restoration_eval/_make_fig3_contour.py
```
Expected: `Saved: ...F_eval_3_contour.png`. Verify size > 50000 bytes as in Task 4 Step 2.

---

## Task 7: Figure 4 — Ablation (Stages 1→3) with metrics

**Files:**
- Create: `figures/restoration_eval/_make_fig4_ablation.py`

- [ ] **Step 1: Write the script**

Create `figures/restoration_eval/_make_fig4_ablation.py`:

```python
"""F_eval_4_ablation.png — progressive ablation with quantitative metrics.

Row 1: Original -> +Split -> +NLM -> +Contour (full images).
Row 2: matching ROI zooms.
Row 3: GridE & EdgePres bar charts per stage (from metrics.csv).
Representative example only; no case IDs in rendered text (spec §4.2).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

CASE = "1_0022"
ROI = (40, 40, 96, 96)
OUT = c.EVAL_DIR / "F_eval_4_ablation.png"

d = c.ASSETS_DIR / CASE
panels = [
    ("(a) Original", c.imread_bgr(d / "00_original.png"), "original"),
    ("(b) + Split-Radius", c.imread_bgr(d / "01_after_split_radius.png"), "split"),
    ("(c) + Adaptive NLM", c.imread_bgr(d / "03_after_nlm_adaptive.png"), "nlm"),
    ("(d) + Contour (final)", c.imread_bgr(d / "04_after_contour_enhance.png"), "contour"),
]

# Metrics for this case keyed by stage. 'original' has no row (it's the reference);
# treat its GridE as 0.0 and EdgePres as 1.0 for the chart baseline.
rows = {r["stage"]: r for r in c.load_metrics() if r["case"] == CASE}
def gride(stage): return 0.0 if stage == "original" else float(rows[stage]["gride"])
def edgep(stage): return 1.0 if stage == "original" else float(rows[stage]["edgepres"])

stages = [p[2] for p in panels]
xlabels = ["Orig", "+Split", "+NLM", "+Contour"]
gride_vals = [gride(s) for s in stages]
edgep_vals = [edgep(s) for s in stages]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.dpi": 120})
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Pipeline Ablation: Split-Radius -> Adaptive NLM -> Contour",
             fontsize=15, fontweight="bold", y=0.98)
gs = GridSpec(3, 4, height_ratios=[1.2, 1.0, 1.1], hspace=0.22, wspace=0.06,
              top=0.93, bottom=0.06, left=0.04, right=0.98)

for ci, (title, img, _) in enumerate(panels):
    ax = fig.add_subplot(gs[0, ci]); ax.imshow(c.to_rgb(img))
    ax.set_title(title, pad=6); ax.set_xticks([]); ax.set_yticks([])
    axz = fig.add_subplot(gs[1, ci]); axz.imshow(c.to_rgb(c.crop_patch(img, ROI)))
    axz.set_xticks([]); axz.set_yticks([])

# GridE bar (higher = more grid removed)
axg = fig.add_subplot(gs[2, 0:2])
axg.bar(xlabels, gride_vals, color="#1f77b4")
axg.set_title("Grid Energy Reduction (higher = better)", fontsize=11)
axg.set_ylim(0, max(gride_vals + [0.01]) * 1.2)
for i, v in enumerate(gride_vals):
    axg.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# EdgePres bar (higher = better)
axe = fig.add_subplot(gs[2, 2:4])
axe.bar(xlabels, edgep_vals, color="#d62728")
axe.set_title("Edge Preservation (higher = better)", fontsize=11)
axe.set_ylim(0, 1.05)
for i, v in enumerate(edgep_vals):
    axe.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
plt.close(fig)
```

- [ ] **Step 2: Run and verify**

Run:
```bash
<PY> figures/restoration_eval/_make_fig4_ablation.py
```
Expected: `Saved: ...F_eval_4_ablation.png` with no KeyError (requires metrics.csv from Task 3). Verify size > 50000 bytes.

---

## Task 8: Figure 5 — Baseline Comparison (multi-case + metrics)

**Files:**
- Create: `figures/restoration_eval/_make_fig5_baselines.py`

- [ ] **Step 1: Write the script**

Create `figures/restoration_eval/_make_fig5_baselines.py`:

```python
"""F_eval_5_baselines.png — Ours vs classical baselines across multiple examples.

Columns: Input | Butterworth | Median | Bilateral | NL-Means | Guided | Ours.
Rows: 4 examples (labeled 'Example (a)'..'(d)'; no case IDs, spec §4.2),
chosen by largest GridE at the contour stage. A mean-GridE bar per method is
appended below the grid.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from metrics import compute_grid_energy_reduction  # noqa: E402

OUT = c.EVAL_DIR / "F_eval_5_baselines.png"
ZBOX = (40, 40, 110, 110)  # (y,x,h,w) zoom per example

# ----- baseline filters (verbatim from existing baselines script) -----
def b_butterworth(img_bgr, order=4, cut_lo=0.18, cut_hi=0.35):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]; h, w = L.shape; cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w)); dy = (yy - cy) / cy; dx = (xx - cx) / cx
    r = np.sqrt(dx * dx + dy * dy); eps = 1e-6
    H = 1.0 / (1.0 + ((r + eps) / cut_lo) ** (2 * order))
    H += 1.0 / (1.0 + (cut_hi / (r + eps)) ** (2 * order))
    H = np.clip(H, 0.0, 1.0)
    F = np.fft.fftshift(np.fft.fft2(L - L.mean()))
    L2 = np.real(np.fft.ifft2(np.fft.ifftshift(F * H))) + L.mean()
    lab[:, :, 0] = np.clip(L2, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def b_median(img): return cv2.medianBlur(img, 5)
def b_bilateral(img): return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
def b_nlmeans(img): return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                                           templateWindowSize=7, searchWindowSize=21)
def b_guided(img):
    try:
        from cv2.ximgproc import guidedFilter
        return guidedFilter(guide=img, src=img, radius=8, eps=int(0.04 * 255 * 255))
    except Exception:
        return cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)

METHODS = [("Input", None), ("Butterworth", b_butterworth), ("Median", b_median),
           ("Bilateral", b_bilateral), ("NL-Means", b_nlmeans), ("Guided", b_guided),
           ("Ours", None)]

# ----- pick 4 examples with strongest grid (largest contour-stage GridE) -----
metrics = [r for r in c.load_metrics() if r["stage"] == "contour"]
metrics.sort(key=lambda r: float(r["gride"]), reverse=True)
chosen = [r["case"] for r in metrics[:4]] or c.list_cases()[:4]

# ----- compute method outputs + per-method mean GridE -----
def ours(case): return c.imread_bgr(c.ASSETS_DIR / case / "04_after_contour_enhance.png")
def orig(case): return c.imread_bgr(c.ASSETS_DIR / case / "00_original.png")

grid_acc = {m: [] for m, _ in METHODS}
cell_imgs = {}  # (case, method) -> bgr
for case in chosen:
    o = orig(case)
    for label, fn in METHODS:
        img = o if label == "Input" else (ours(case) if label == "Ours" else fn(o))
        cell_imgs[(case, label)] = img
        if label != "Input":
            grid_acc[label].append(compute_grid_energy_reduction(o, img))
mean_grid = {m: (np.mean(v) if v else 0.0) for m, v in grid_acc.items()}

# ----- figure -----
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.dpi": 110})
n_rows, n_cols = len(chosen), len(METHODS)
fig = plt.figure(figsize=(2.3 * n_cols + 0.6, 2.3 * n_rows + 1.8))
fig.suptitle("Weave Removal: Ours vs Classical Baselines", fontsize=15,
             fontweight="bold", y=0.99)
gs = GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.9],
              wspace=0.04, hspace=0.06, top=0.95, bottom=0.04, left=0.05, right=0.99)

for ri, case in enumerate(chosen):
    for ci, (label, _) in enumerate(METHODS):
        ax = fig.add_subplot(gs[ri, ci])
        ax.imshow(c.to_rgb(c.crop_patch(cell_imgs[(case, label)], ZBOX, up=3)))
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(label, pad=8, fontsize=12)
        if ci == 0:
            ax.set_ylabel(c.example_label(ri), fontsize=12, fontweight="bold", labelpad=10)

# mean GridE bar spanning full width
axb = fig.add_subplot(gs[n_rows, :])
labels = [m for m, _ in METHODS if m != "Input"]
vals = [mean_grid[m] for m in labels]
axb.bar(labels, vals, color=["#777"] * (len(labels) - 1) + ["#b3001b"])
axb.set_title("Mean Grid Energy Reduction across examples (higher = better)", fontsize=11)
for i, v in enumerate(vals):
    axb.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}  (examples: {len(chosen)})")
plt.close(fig)
```

- [ ] **Step 2: Run and verify**

Run:
```bash
<PY> figures/restoration_eval/_make_fig5_baselines.py
```
Expected: `Saved: ...F_eval_5_baselines.png  (examples: 4)`. Verify size > 80000 bytes.

- [ ] **Step 3: Label audit**

Read the file and confirm rendered labels use `c.example_label(ri)` (→ `Example (a)`...) and method names only — no case IDs in any `set_title`/`set_ylabel`/`suptitle`.

---

## Task 9: Figure 6 — Input vs NL-Means vs Ours (detail)

**Files:**
- Create: `figures/restoration_eval/_make_fig6_input_nlm_ours.py`

- [ ] **Step 1: Write the script**

Create `figures/restoration_eval/_make_fig6_input_nlm_ours.py`:

```python
"""F_eval_6_input_nlm_ours.png — detail comparison Input | NL-Means | Ours.

Rows: 4 examples (zoom patches), labeled 'Example (a)'..'(d)' (no case IDs, §4.2).
NL-Means uses strong h=15 to expose over-smoothing vs Ours' line preservation.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

OUT = c.EVAL_DIR / "F_eval_6_input_nlm_ours.png"
ZBOX = (40, 40, 120, 120)

def nlmeans_strong(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=15, hColor=15,
                                           templateWindowSize=7, searchWindowSize=21)

# Choose 4 examples with strongest grid (contour-stage GridE), like fig5.
metrics = [r for r in c.load_metrics() if r["stage"] == "contour"]
metrics.sort(key=lambda r: float(r["gride"]), reverse=True)
chosen = [r["case"] for r in metrics[:4]] or c.list_cases()[:4]

COLS = ["Input", "NL-Means", "Ours"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.titlesize": 16, "axes.titleweight": "bold",
                     "figure.dpi": 110})
n_rows = len(chosen)
fig = plt.figure(figsize=(4.2 * 3 + 0.7, 4.2 * n_rows + 0.6))
gs = GridSpec(n_rows, 3, wspace=0.025, hspace=0.025,
              top=0.96, bottom=0.005, left=0.05, right=0.998)

for ri, case in enumerate(chosen):
    o = c.imread_bgr(c.ASSETS_DIR / case / "00_original.png")
    ours = c.imread_bgr(c.ASSETS_DIR / case / "04_after_contour_enhance.png")
    nlm = nlmeans_strong(o)
    for ci, (label, img) in enumerate(zip(COLS, [o, nlm, ours])):
        ax = fig.add_subplot(gs[ri, ci])
        ax.imshow(c.to_rgb(c.crop_patch(img, ZBOX, up=4)))
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(label, pad=10, fontsize=18)
        if ci == 0:
            ax.set_ylabel(c.example_label(ri), fontsize=14, fontweight="bold", labelpad=12)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}  (examples: {len(chosen)})")
plt.close(fig)
```

- [ ] **Step 2: Run and verify**

Run:
```bash
<PY> figures/restoration_eval/_make_fig6_input_nlm_ours.py
```
Expected: `Saved: ...F_eval_6_input_nlm_ours.png  (examples: 4)`. Verify size > 80000 bytes.

---

## Task 10: Final verification pass

**Files:** none

- [ ] **Step 1: Confirm all 6 figures exist and are non-trivial**

Run:
```bash
<PY> -c "import os; fs=['F_eval_1_cross_harmonic','F_eval_2_adaptive_nlm','F_eval_3_contour','F_eval_4_ablation','F_eval_5_baselines','F_eval_6_input_nlm_ours']; base='figures/restoration_eval/'; [print(f, os.path.getsize(base+f+'.png')) for f in fs]; bad=[f for f in fs if os.path.getsize(base+f+'.png')<50000]; print('BAD',bad) if bad else print('ALL 6 FIGURES OK')"
```
Expected: prints each filename + size, then `ALL 6 FIGURES OK`.

- [ ] **Step 2: Global case-ID leak audit**

Run:
```bash
grep -nE "set_title|suptitle|set_ylabel|set_xlabel|ax.text|axb.text|axg.text|axe.text" figures/restoration_eval/_make_fig*.py | grep -E "1_0[0-9]{3}|ceramic_painting" && echo "LEAK FOUND" || echo "NO CASE-ID LEAK IN RENDERED TEXT"
```
Expected: `NO CASE-ID LEAK IN RENDERED TEXT`.

- [ ] **Step 3: Visual spot-check (human/agent review)**

Open `figures/restoration_eval/F_eval_4_ablation.png` and `F_eval_5_baselines.png`. Confirm:
- Panels are populated (no blank/missing-asset axes).
- Metric bars render with numeric labels.
- All visible text uses `(a)/(b)...` or `Example (a)...` — zero case IDs.

Report any panel that looks wrong; do not claim completion until all 6 figures pass.

---

## Self-Review Notes (author)

- **Spec coverage:** fig1↔Cross-harmonic(§4.2 fig1), fig2↔NLM, fig3↔Contour, fig4↔Ablation+metrics, fig5↔Baseline+metrics, fig6↔Input/NLM/Ours. Asset gen + metrics.csv ↔ §4.1. Label rule ↔ §4.2 (enforced via `_common.example_label/panel_label` + Task 10 audit). 29-case generation ↔ §3/§4.1.
- **Placeholders:** none — every script is complete.
- **Type consistency:** metric stage keys (`split_cross_off/split/nlm/contour`) are produced in Task 2 and consumed in Task 7/8/9; `crop_patch(box, up)` signature consistent across figs; `example_label`/`panel_label` defined in Task 1.
- **ROI constants** are per-figure and clamped by `crop_patch`; safe across variable image sizes.
