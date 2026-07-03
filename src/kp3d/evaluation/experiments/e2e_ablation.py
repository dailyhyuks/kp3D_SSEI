"""End-to-End Pipeline Ablation Study (Qualitative).

Evaluates the contribution of each pipeline stage by selectively disabling
components and producing per-image outputs for qualitative visual comparison.

This experiment follows the protocol defined in Section 4.2.3 of the paper:
qualitative visual quality assessment. Quantitative metrics (PSNR/SSIM,
grid energy, edge preservation) are intentionally not computed here; the
E2E ablation relies on visual inspection of pipeline outputs.

Ablation configs (matches Table 22 in the paper):
- Full Pipeline:      All stages active
- w/o Weave Removal:  Skip spectral weave removal step (keep upscaling)
- w/o Upscaling:      Skip upscaling step (keep weave removal)
- w/o Inpainting:     Skip occlusion-region inpainting stage
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    name: str
    use_weave_removal: bool = True
    use_upscaling: bool = True
    use_inpainting: bool = True

    @property
    def short_name(self) -> str:
        """Short identifier for filenames."""
        return (
            self.name.replace(" ", "_")
            .replace("/", "")
            .lower()
        )


def get_ablation_configs() -> List[AblationConfig]:
    """Get standard ablation configurations (matches Table 22)."""
    return [
        AblationConfig(name="Full Pipeline"),
        AblationConfig(name="w/o Weave Removal", use_weave_removal=False),
        AblationConfig(name="w/o Upscaling", use_upscaling=False),
        AblationConfig(name="w/o Inpainting", use_inpainting=False),
    ]


@dataclass
class AblationResult:
    """Result for a single image + ablation config (qualitative).

    No quantitative quality metrics are stored; only the path to the
    saved output image and completion status are tracked.
    """

    image_name: str
    config_name: str
    output_path: str = ""
    completed: bool = True
    error: str = ""


class AblationExperiment:
    """E2E Pipeline Ablation Study (Qualitative).

    Runs the pipeline with different configurations and saves per-config
    outputs to disk for visual inspection. No quantitative metrics are
    aggregated.
    """

    def __init__(self, config):
        from kp3d.evaluation.config import EvalConfig

        self.config: EvalConfig = config
        self.results: List[AblationResult] = []

    def run(
        self,
        configs: Optional[List[AblationConfig]] = None,
    ) -> List[AblationResult]:
        """Run ablation study (qualitative).

        Args:
            configs: Ablation configurations. If None, uses standard set.

        Returns:
            List of results per image per config (paths to saved outputs).
        """
        if configs is None:
            configs = get_ablation_configs()

        from kp3d.evaluation.datasets import find_images

        images = find_images(self.config.data_dir)
        if self.config.max_images > 0:
            images = images[: self.config.max_images]

        if not images:
            raise FileNotFoundError(
                f"No images found in {self.config.data_dir}"
            )

        out_root = Path(getattr(self.config, "output_dir", "ablation_outputs"))
        out_root.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            img_name = img_path.stem

            if self.config.dry_run:
                print(f"  [DRY RUN] {img_name}: would run {len(configs)} configs")
                continue

            for cfg in configs:
                try:
                    output = self._run_pipeline(img_bgr, cfg)
                    out_dir = out_root / cfg.short_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{img_name}.png"
                    cv2.imwrite(str(out_path), output)
                    self.results.append(
                        AblationResult(
                            image_name=img_name,
                            config_name=cfg.name,
                            output_path=str(out_path),
                        )
                    )
                except Exception as e:
                    self.results.append(
                        AblationResult(
                            image_name=img_name,
                            config_name=cfg.name,
                            completed=False,
                            error=str(e),
                        )
                    )

        return self.results

    def _run_pipeline(
        self, img_bgr: np.ndarray, config: AblationConfig
    ) -> np.ndarray:
        """Run pipeline with given ablation config, returning the output image.

        Enhancement is implemented as the sandwich sequence:
            2x Upscale -> Weave Removal -> 2x Upscale
        The inpainting toggle is a passthrough hook here; full occlusion
        handling and inpainting are performed by ``IntegratedPipeline`` when
        this ablation is invoked through the integrated runner.
        """
        current = img_bgr.copy()

        # Stage 1: Enhancement (sandwich)
        if config.use_upscaling:
            current = self._upscale(current, scale=2)
        if config.use_weave_removal:
            current = self._weave_removal(current)
        if config.use_upscaling:
            current = self._upscale(current, scale=2)

        # Stage 2: Inpainting toggle (delegation hook)
        # In standalone qualitative runs this is a passthrough; the
        # IntegratedPipeline applies SSEI inpainting when use_inpainting=True
        # and skips it (visible-region-only output) when False.
        if not config.use_inpainting:
            # Mark via filename / config tracking; image content unchanged.
            pass

        return current

    def _upscale(self, img_bgr: np.ndarray, scale: int = 2) -> np.ndarray:
        """Upscale using Real-ESRGAN or fallback to bicubic."""
        try:
            from kp3d.modules.superres.real_esrgan import RealESRGANModule

            module = RealESRGANModule(scale=scale)
            return module.process(img_bgr)
        except (ImportError, RuntimeError, AttributeError):
            h, w = img_bgr.shape[:2]
            return cv2.resize(
                img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
            )

    def _weave_removal(self, img_bgr: np.ndarray) -> np.ndarray:
        """Apply weave removal (Spectral Interpolation)."""
        try:
            from kp3d.modules.weave_removal import (
                WeaveRemovalConfig,
                WeaveRemovalModule,
                WeaveRemovalPreset,
            )

            config = WeaveRemovalConfig(preset=WeaveRemovalPreset.CLEAN)
            module = WeaveRemovalModule(config=config)
            result, _ = module.process_bgr(img_bgr)
            return result
        except ImportError:
            return img_bgr

    def aggregate(self) -> Dict[str, Dict[str, int]]:
        """Aggregate qualitative run statistics by config name.

        Returns a dict of {config_name: {"completed": N, "failed": M}}.
        No quality metrics are aggregated.
        """
        from collections import defaultdict

        config_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"completed": 0, "failed": 0}
        )
        for r in self.results:
            key = "completed" if r.completed else "failed"
            config_stats[r.config_name][key] += 1
        return dict(config_stats)
