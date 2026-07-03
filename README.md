# Design of a 3D Reconstruction Pipeline of Occluded Objects in Korean Royal Court Paintings

A research implementation of the paper's 4-stage pipeline for 3D reconstruction of occluded objects in traditional Korean royal court paintings (Korean: **"한국 궁중 회화 속 가려진 기물의 3D 복원 파이프라인 설계"**).

## Pipeline Overview

This pipeline processes digitized high-resolution scans of Korean traditional paintings to separate occluded objects, restore hidden regions, and optionally reconstruct 3D meshes.

### Stage 1: Restoration / De-Weaving

Removes periodic silk weave patterns (fabric grid artifacts) from digitized paintings using FFT spectral notch filtering, spatial-adaptive NLM blending, and contour (ink line) enhancement.

**Module:** `kp3d.modules.weave_removal`

- **FFT Spectral Interpolation:** Detects and suppresses harmonic peaks caused by the silk weave pattern
- **Spatial-Adaptive NLM:** Applies Non-Local Means denoising selectively to narrow regions where grid artifacts persist
- **Contour Enhancement:** Restores ink line sharpness after spectral processing

> **Note:** This pipeline intentionally excludes Real-ESRGAN upscaling. The paper's methodology uses weave removal as the sole preprocessing step.

### Stage 2: Object Segmentation

Segments individual objects using manual LabelMe polygon annotations, with optional SAM (Segment Anything Model) refinement for tighter mask boundaries.

**Module:** `kp3d.modules.occlusion.segmentation`, `kp3d.modules.occlusion.sam_mask_refiner`

- **LabelMe Polygons:** Manual annotations define object boundaries and layer ordering
- **SAM Refinement:** Optional neural mask refinement for cleaner edges
- **Layer Order:** Annotation's `layer_order` field determines occlusion relationships

### Stage 3: SSEI Inpainting (Style-consistent Self-Exemplar Inpainting)

Detects occluded regions where objects overlap and inpaints the hidden areas using style-consistent boundary-guided methods.

**Module:** `kp3d.modules.occlusion.occlusion_detection`, `kp3d.modules.occlusion.inpainting`

- **Occlusion Detection:** Identifies hidden regions based on mask overlap and layer ordering
- **Boundary-Guided Inpainting:** Samples colors from occlusion boundary neighborhood for adaptive color matching
- **RGBA Extraction:** Outputs each object as a separate RGBA image with transparent background

### Stage 4: 3D Reconstruction (Optional)

Reconstructs 3D meshes from the separated RGBA object images using multi-view generation methods.

**Module:** `kp3d.modules.reconstruction`

- **Wonder3D** (default): Multi-view diffusion for consistent normal and color map generation
- **InstantMesh:** Fast mesh reconstruction from single images
- **LGM:** Large Gaussian Model for 3D generation

## Repository Structure

```
korean-painting-3d/
├── src/kp3d/
│   ├── modules/
│   │   ├── weave_removal/     # Stage 1: FFT + NLM + Contour
│   │   ├── occlusion/         # Stages 2-3: Segmentation + Inpainting
│   │   │   ├── segmentation.py
│   │   │   ├── occlusion_detection.py
│   │   │   ├── inpainting.py
│   │   │   ├── sam_mask_refiner.py
│   │   │   └── pipeline.py
│   │   └── reconstruction/    # Stage 4: 3D Reconstruction
│   │       ├── wonder3d.py
│   │       ├── instantmesh.py
│   │       └── lgm.py
│   ├── pipelines/
│   │   ├── paper_pipeline.py  # Paper-aligned 4-stage orchestrator
│   │   └── integrated.py      # Full-featured E2E pipeline (with upscaling)
│   └── core/
│       ├── base.py
│       ├── config.py
│       └── registry.py
├── configs/                   # Pipeline configuration files
├── data/                      # Input images and annotations
├── outputs/                   # Pipeline outputs
├── docs/                      # Documentation
└── requirements.txt
```

## Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/korean-painting-3d.git
cd korean-painting-3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running the Paper Pipeline

```python
from kp3d.pipelines.paper_pipeline import PaperPipeline, PaperPipelineConfig

# Configure pipeline
config = PaperPipelineConfig(
    weave_removal_preset="v3",       # FFT + NLM Adaptive + Contour
    inpaint_method="boundary_guided", # V21 boundary-guided inpainting
    use_reconstruction=False,         # Disable 3D reconstruction
    output_dir="outputs/my_painting"
)

# Create pipeline
pipeline = PaperPipeline(config)

# Process painting with LabelMe annotation
result = pipeline.process(
    image_path="data/paintings/court_scene.png",
    annotation_path="data/annotations/court_scene.json"
)

# Access results
print(f"Extracted {len(result.extracted)} objects")
for label, rgba in result.extracted.items():
    print(f"  {label}: {rgba.shape}")
```

### Using the Convenience Function

```python
from kp3d.pipelines.paper_pipeline import run

result = run(
    image_path="data/paintings/court_scene.png",
    annotation_path="data/annotations/court_scene.json",
    output_dir="outputs/court_scene",
    weave_removal_preset="v3",
    use_reconstruction=True  # Enable 3D reconstruction
)
```

### LabelMe Annotation Format

Annotations should include `layer_order` to define occlusion relationships:

```json
{
  "shapes": [
    {
      "label": "vase",
      "points": [[100, 100], [200, 100], [200, 300], [100, 300]],
      "shape_type": "polygon",
      "layer_order": 1
    },
    {
      "label": "table",
      "points": [[50, 250], [250, 250], [250, 400], [50, 400]],
      "shape_type": "polygon",
      "layer_order": 2
    }
  ]
}
```

Lower `layer_order` = foreground (occluder), higher = background (occludee).

## Output Structure

```
outputs/court_scene/
├── 00_original.png          # Input image
├── 01_deweaved.png          # After Stage 1 weave removal
├── 02_vase_rgba.png         # Extracted vase with alpha
├── 02_vase_vis.png          # Vase on white background
├── 02_table_rgba.png        # Extracted table with inpainted region
├── 02_table_vis.png         # Table on white background
├── 03_vase.obj              # 3D mesh (if reconstruction enabled)
├── 03_table.obj
└── metadata.json            # Processing metadata and timing
```

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:

- **PyTorch >= 2.0.0** - Deep learning framework
- **OpenCV >= 4.8.0** - Image processing
- **NumPy, SciPy** - Numerical operations
- **loguru** - Logging
- **Pydantic >= 2.5.0** - Configuration validation

Optional dependencies for full functionality:
- **SAM2** - Segment Anything Model for mask refinement
- **Wonder3D / InstantMesh** - 3D reconstruction backends

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{korean_painting_3d_2026,
  title={Design of a 3D Reconstruction Pipeline of Occluded Objects in Korean Royal Court Paintings},
  author={...},
  journal={...},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author:** dailyhyuks
- **Email:** dailyhyuks@naver.com
- **Repository:** https://github.com/dailyhyuks/kp3D_SSEI
