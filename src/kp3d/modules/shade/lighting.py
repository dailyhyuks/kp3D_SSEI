"""Lighting simulation for shade generation."""

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from kp3d.core.base import ModuleOutput
from .base import BaseShadeGeneration, LightSource, ShadeConfig
from .midas import MiDaSDepthEstimator


class LightingSimulator:
    """Lighting-based shading generator.

    Computes surface normals from depth maps and applies
    physically-based lighting (Lambert diffuse) to generate
    realistic shading for 2D images.
    """

    def compute_normals(self, depth_map: Tensor, smoothing: float = 0.5) -> Tensor:
        """Compute surface normal map from depth using Sobel gradients.

        Args:
            depth_map: Depth map tensor (B, 1, H, W).
            smoothing: Smoothing factor for normals (0-1).

        Returns:
            Normal map tensor (B, 3, H, W) with normalized vectors.
        """
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth_map.dtype,
            device=depth_map.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth_map.dtype,
            device=depth_map.device
        ).view(1, 1, 3, 3)

        # Optional smoothing before gradient
        if smoothing > 0:
            kernel_size = int(smoothing * 4) * 2 + 1  # Odd kernel size
            depth_smooth = F.avg_pool2d(
                F.pad(depth_map, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate'),
                kernel_size=kernel_size,
                stride=1
            )
        else:
            depth_smooth = depth_map

        # Compute gradients
        grad_x = F.conv2d(
            F.pad(depth_smooth, (1, 1, 1, 1), mode='replicate'),
            sobel_x
        )
        grad_y = F.conv2d(
            F.pad(depth_smooth, (1, 1, 1, 1), mode='replicate'),
            sobel_y
        )

        # Build normal vectors: cross product of tangent vectors
        # Tangent in x: (1, 0, dz/dx)
        # Tangent in y: (0, 1, dz/dy)
        # Normal = cross(tangent_x, tangent_y) = (-dz/dx, -dz/dy, 1)

        normals = torch.cat([
            -grad_x,  # nx
            -grad_y,  # ny
            torch.ones_like(grad_x)  # nz
        ], dim=1)

        # Normalize to unit vectors
        norm = torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True)) + 1e-6
        normals = normals / norm

        return normals

    def apply_lighting(
        self,
        image: Tensor,
        normal_map: Tensor,
        light_sources: List[LightSource],
        intensity: float = 1.0
    ) -> Tensor:
        """Apply Lambert diffuse shading from multiple light sources.

        Args:
            image: Original image tensor (B, C, H, W).
            normal_map: Surface normal map (B, 3, H, W).
            light_sources: List of light source configurations.
            intensity: Overall shading intensity multiplier.

        Returns:
            Shaded image tensor (B, C, H, W).
        """
        batch_size, channels, height, width = image.shape
        device = image.device

        # Accumulate lighting from all sources
        total_lighting = torch.zeros(
            batch_size, 1, height, width,
            dtype=image.dtype,
            device=device
        )

        for light in light_sources:
            # Light direction as unit vector
            light_dir = torch.tensor(
                light.direction,
                dtype=image.dtype,
                device=device
            ).view(1, 3, 1, 1)

            # Normalize
            light_dir = light_dir / (torch.norm(light_dir) + 1e-6)

            # Lambert diffuse: I = max(N · L, 0)
            dot_product = torch.sum(normal_map * light_dir, dim=1, keepdim=True)
            diffuse = torch.clamp(dot_product, min=0.0)

            # Apply light intensity and ambient
            lighting = light.ambient + (1.0 - light.ambient) * diffuse * light.intensity

            # Apply light color if needed
            if light.color != (1.0, 1.0, 1.0):
                light_color = torch.tensor(
                    light.color,
                    dtype=image.dtype,
                    device=device
                ).view(1, 3, 1, 1)
                lighting = lighting * light_color.mean()

            total_lighting += lighting

        # Normalize by number of lights
        if len(light_sources) > 0:
            total_lighting = total_lighting / len(light_sources)

        # Apply to image with intensity control
        shaded = image * (1.0 - intensity + intensity * total_lighting)

        return torch.clamp(shaded, 0.0, 1.0)

    def generate_shadows(
        self,
        depth_map: Tensor,
        light_direction: Tuple[float, float, float],
        softness: float = 0.3
    ) -> Tensor:
        """Generate soft shadows using depth-based ray marching approximation.

        Args:
            depth_map: Depth map tensor (B, 1, H, W).
            light_direction: Light direction as (x, y, z) tuple.
            softness: Shadow edge softness (0-1).

        Returns:
            Shadow map tensor (B, 1, H, W) where 0=shadow, 1=lit.
        """
        # Simplified shadow generation using depth gradient
        # More sophisticated methods would use ray marching

        batch_size, _, height, width = depth_map.shape
        device = depth_map.device

        # Normalize light direction
        light_dir = torch.tensor(light_direction, dtype=depth_map.dtype, device=device)
        light_dir = light_dir / (torch.norm(light_dir) + 1e-6)

        # Compute depth gradient in light direction
        # Positive gradient = facing away from light = potential shadow
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth_map.dtype,
            device=device
        ).view(1, 1, 3, 3) * light_dir[0].item()

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth_map.dtype,
            device=device
        ).view(1, 1, 3, 3) * light_dir[1].item()

        grad_light = F.conv2d(
            F.pad(depth_map, (1, 1, 1, 1), mode='replicate'),
            sobel_x + sobel_y
        )

        # Negative gradient suggests occlusion
        shadow_strength = torch.clamp(-grad_light, min=0.0, max=1.0)

        # Apply softness via Gaussian blur
        if softness > 0:
            kernel_size = int(softness * 10) * 2 + 1
            shadow_strength = F.avg_pool2d(
                F.pad(shadow_strength, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate'),
                kernel_size=kernel_size,
                stride=1
            )

        # Convert to shadow map (0=full shadow, 1=no shadow)
        shadow_map = 1.0 - shadow_strength * 0.7  # Max 70% shadow

        return shadow_map


class ShadeGeneratorModule(BaseShadeGeneration):
    """Complete shade generation module integrating MiDaS depth + lighting.

    Generates realistic shading for 2D images by:
    1. Estimating depth using MiDaS
    2. Computing surface normals from depth
    3. Applying physically-based lighting
    4. Generating soft shadows
    5. Blending with original to preserve artistic style
    """

    def __init__(self, config: Optional[ShadeConfig] = None, **kwargs):
        """Initialize shade generator.

        Args:
            config: Shade generation configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config=config, **kwargs)

        # Initialize depth estimator
        self.depth_estimator = MiDaSDepthEstimator(config=self.config, device=self.device)

        # Initialize lighting simulator
        self.lighting = LightingSimulator()

        self._initialized = True

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights.

        Args:
            checkpoint_path: Path to checkpoint (passed to depth estimator).
        """
        self.depth_estimator.load_weights(checkpoint_path)
        self._initialized = True

    def forward(self, image: Tensor, edge_map: Optional[Tensor] = None, **kwargs: Any) -> ModuleOutput:
        """Generate shading for input image.

        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1].
            edge_map: Optional edge map for depth refinement (B, C, H, W).
            **kwargs: Additional arguments.

        Returns:
            ModuleOutput with shaded image and intermediate results.
        """
        # 1. Estimate depth
        depth_output = self.depth_estimator(image, edge_map=edge_map)
        depth_map = depth_output.result

        # 2. Compute surface normals
        normal_map = self.lighting.compute_normals(
            depth_map,
            smoothing=self.config.normal_smoothing
        )

        # 3. Apply lighting from all sources
        shaded_image = self.lighting.apply_lighting(
            image,
            normal_map,
            self.config.light_sources,
            intensity=self.config.shade_intensity
        )

        # 4. Generate shadows (using primary light source)
        shadow_map = None
        if len(self.config.light_sources) > 0:
            primary_light = self.config.light_sources[0]
            shadow_map = self.lighting.generate_shadows(
                depth_map,
                primary_light.direction,
                softness=self.config.shadow_softness
            )

            # Apply shadows to shaded image
            shaded_image = shaded_image * shadow_map

        # 5. Blend with original to preserve artistic tones
        if self.config.preserve_original_tones:
            blend_factor = 0.6  # 60% shaded, 40% original
            result = blend_factor * shaded_image + (1 - blend_factor) * image
        else:
            result = shaded_image

        result = torch.clamp(result, 0.0, 1.0)

        # Build output
        intermediate = {
            "depth_map": depth_map,
            "normal_map": normal_map,
            "shading": shaded_image,
        }

        if shadow_map is not None:
            intermediate["shadow_map"] = shadow_map

        return ModuleOutput(
            result=result,
            intermediate=intermediate,
            metadata={
                "depth_model": self.config.depth_model,
                "num_lights": len(self.config.light_sources),
                "shade_intensity": self.config.shade_intensity,
                "preserve_original": self.config.preserve_original_tones,
            }
        )

    @property
    def name(self) -> str:
        """Module name."""
        return "shade_generator"
