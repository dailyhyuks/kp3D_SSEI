# Shade Generation Module

MiDaS-based depth estimation and physically-based shading generation optimized for traditional Korean painting aesthetics.

## Overview

The shade generation module provides:

1. **Depth Estimation**: MiDaS monocular depth estimation
2. **Normal Mapping**: Surface normal computation from depth gradients
3. **Lighting Simulation**: Lambert diffuse shading with multiple light sources
4. **Shadow Generation**: Soft shadow computation based on depth occlusion
5. **Artistic Preservation**: Blending with original image to preserve traditional painting aesthetics

## Architecture

```
ShadeGeneratorModule
├── MiDaSDepthEstimator      # Depth estimation using MiDaS
│   └── torch.hub MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_small)
└── LightingSimulator         # Lighting and shadow computation
    ├── compute_normals()     # Sobel-based normal map
    ├── apply_lighting()      # Lambert diffuse shading
    └── generate_shadows()    # Soft shadow generation
```

## Usage

### Basic Usage

```python
from kp3d.modules.shade import ShadeGeneratorModule, ShadeConfig, LightSource

# Create configuration
config = ShadeConfig(
    depth_model="DPT_Large",
    shade_intensity=0.6,
    preserve_original_tones=True,
    normal_smoothing=0.5,
    shadow_softness=0.3,
)

# Initialize module
shade_gen = ShadeGeneratorModule(config=config)

# Process image
import torch
image = torch.randn(1, 3, 512, 512)  # RGB image in [0, 1]
output = shade_gen(image)

# Access results
shaded_image = output.result
depth_map = output.intermediate["depth_map"]
normal_map = output.intermediate["normal_map"]
shadow_map = output.intermediate["shadow_map"]
```

### Custom Lighting

```python
from kp3d.modules.shade import LightSource, ShadeConfig

# Define multiple light sources
lights = [
    LightSource(
        direction=(0.0, -1.0, 1.0),  # Top-right light
        intensity=1.2,
        color=(1.0, 0.95, 0.9),      # Warm white
        ambient=0.2,
    ),
    LightSource(
        direction=(1.0, 0.0, 0.5),   # Side fill light
        intensity=0.6,
        color=(0.9, 0.95, 1.0),      # Cool white
        ambient=0.1,
    ),
]

config = ShadeConfig(light_sources=lights)
shade_gen = ShadeGeneratorModule(config=config)
```

### Edge-Aware Depth Refinement

```python
# Provide edge map to refine depth estimation
edge_map = edge_detector(image)  # From edge module
output = shade_gen(image, edge_map=edge_map)
```

## Configuration

### ShadeConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth_model` | str | `"DPT_Large"` | MiDaS model type (`DPT_Large`, `DPT_Hybrid`, `MiDaS_small`) |
| `light_sources` | List[LightSource] | `[LightSource()]` | List of light sources |
| `normal_smoothing` | float | `0.5` | Normal map smoothing factor (0-1) |
| `shadow_softness` | float | `0.3` | Shadow edge softness (0-1) |
| `preserve_original_tones` | bool | `True` | Blend with original to preserve artistic style |
| `shade_intensity` | float | `0.5` | Overall shading intensity (0-1) |

### LightSource

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | Tuple[float, float, float] | `(0.0, 0.0, 1.0)` | Light direction unit vector |
| `intensity` | float | `1.0` | Light intensity multiplier |
| `color` | Tuple[float, float, float] | `(1.0, 1.0, 1.0)` | RGB light color |
| `ambient` | float | `0.2` | Ambient light contribution (0-1) |

## Components

### MiDaSDepthEstimator

Estimates depth from a single RGB image using Intel's MiDaS model.

**Models:**
- `DPT_Large`: Highest quality, slower (recommended)
- `DPT_Hybrid`: Balanced quality/speed
- `MiDaS_small`: Fastest, lower quality

**Features:**
- Automatic model loading from torch.hub
- Edge-aware depth refinement
- Normalized depth output [0, 1]

### LightingSimulator

Generates realistic shading from depth maps.

**Methods:**

#### `compute_normals(depth_map, smoothing=0.5)`
Computes surface normals using Sobel gradients.

**Args:**
- `depth_map`: (B, 1, H, W) depth tensor
- `smoothing`: Normal smoothing factor (0-1)

**Returns:**
- `normal_map`: (B, 3, H, W) normalized normal vectors

#### `apply_lighting(image, normal_map, light_sources, intensity=1.0)`
Applies Lambert diffuse shading.

**Args:**
- `image`: (B, C, H, W) original image
- `normal_map`: (B, 3, H, W) surface normals
- `light_sources`: List of LightSource objects
- `intensity`: Shading intensity multiplier

**Returns:**
- `shaded_image`: (B, C, H, W) shaded result

#### `generate_shadows(depth_map, light_direction, softness=0.3)`
Generates soft shadows based on depth occlusion.

**Args:**
- `depth_map`: (B, 1, H, W) depth tensor
- `light_direction`: (x, y, z) light direction
- `softness`: Shadow edge softness (0-1)

**Returns:**
- `shadow_map`: (B, 1, H, W) shadow mask [0, 1]

## Output

### ModuleOutput

```python
output = shade_gen(image)

# Primary result
output.result  # (B, C, H, W) shaded image

# Intermediate results
output.intermediate = {
    "depth_map": Tensor,    # (B, 1, H, W) depth map
    "normal_map": Tensor,   # (B, 3, H, W) surface normals
    "shading": Tensor,      # (B, C, H, W) pure shading
    "shadow_map": Tensor,   # (B, 1, H, W) shadow mask
}

# Metadata
output.metadata = {
    "depth_model": str,          # MiDaS model used
    "num_lights": int,           # Number of light sources
    "shade_intensity": float,    # Applied intensity
    "preserve_original": bool,   # Blending enabled
}
```

## Implementation Details

### Depth Estimation

1. Image preprocessing via MiDaS transforms
2. Inference through selected MiDaS model
3. Resize to original resolution (bicubic)
4. Invert and normalize to [0, 1] (near=1, far=0)
5. Optional edge-aware smoothing

### Normal Computation

1. Optional Gaussian smoothing of depth
2. Sobel gradient computation (∂z/∂x, ∂z/∂y)
3. Cross product to get surface normal: n = (-∂z/∂x, -∂z/∂y, 1)
4. Normalization to unit vectors

### Lighting Application

1. For each light source:
   - Compute N·L (normal dot light direction)
   - Clamp to [0, 1] for diffuse term
   - Apply ambient + diffuse shading
   - Multiply by light intensity and color
2. Average all light contributions
3. Multiply original image by shading

### Shadow Generation

1. Compute depth gradient in light direction
2. Negative gradient → potential occlusion
3. Apply Gaussian blur for soft edges
4. Convert to shadow mask (0=shadow, 1=lit)

### Artistic Preservation

For traditional Korean paintings, original artistic tones are preserved by blending:

```python
result = 0.6 * shaded_image + 0.4 * original_image
```

This maintains the hand-painted aesthetic while adding depth-based shading.

## Performance

### Model Comparison

| Model | Quality | Speed | Memory | Use Case |
|-------|---------|-------|--------|----------|
| DPT_Large | Excellent | Slow | High | Production, high quality |
| DPT_Hybrid | Good | Medium | Medium | Balanced workflow |
| MiDaS_small | Fair | Fast | Low | Prototyping, real-time |

### Optimization Tips

1. **Batch Processing**: Process multiple images in a batch
2. **Half Precision**: Use `.half()` for faster GPU inference
3. **Model Selection**: Choose appropriate model for use case
4. **Edge Hints**: Provide edge maps to improve depth accuracy

## Examples

### Example 1: Subtle Shading for Traditional Art

```python
config = ShadeConfig(
    depth_model="DPT_Large",
    shade_intensity=0.3,        # Subtle effect
    preserve_original_tones=True,
    normal_smoothing=0.7,       # Smooth normals
    shadow_softness=0.5,        # Soft shadows
)

shade_gen = ShadeGeneratorModule(config=config)
output = shade_gen(traditional_painting)
```

### Example 2: Dramatic Lighting

```python
lights = [
    LightSource(
        direction=(-1.0, -1.0, 1.0),
        intensity=1.5,
        ambient=0.1,
    )
]

config = ShadeConfig(
    light_sources=lights,
    shade_intensity=0.8,
    preserve_original_tones=False,
)

shade_gen = ShadeGeneratorModule(config=config)
output = shade_gen(image)
```

### Example 3: Edge-Enhanced Depth

```python
from kp3d.modules.edge import EdgeDetectionModule

# Detect edges first
edge_detector = EdgeDetectionModule()
edge_output = edge_detector(image)
edge_map = edge_output.result

# Use edges to refine depth
shade_gen = ShadeGeneratorModule()
output = shade_gen(image, edge_map=edge_map)
```

## Dependencies

- **PyTorch**: ≥1.10
- **torch.hub**: For MiDaS model loading
- **timm**: MiDaS dependency
- **pydantic**: Configuration validation

MiDaS models are automatically downloaded from torch.hub on first use.

## References

- [MiDaS: Towards Robust Monocular Depth Estimation](https://github.com/isl-org/MiDaS)
- [DPT: Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

## License

MiDaS is licensed under MIT. See torch.hub for details.
