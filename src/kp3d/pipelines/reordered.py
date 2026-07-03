"""Reordered Pipeline v6 - PoC for alternative stage ordering.

Pipeline order (differs from v4/v5):
  Stage A: Upscale 2x (original 320 → 640)
  Stage B: Weave Removal (spectral Butterworth notch filter at 640)
  Stage C: Inpainting (occlusion detection + per-object inpainting)
  Stage D: Upscale 2x (scene + per-object RGBA → 1280)

Unlike v4/v5 which run: spectral_grid_removal(orig) → upscale_1 → upscale_2 → inpainting,
this pipeline upscales first, then removes weave, then inpaints, then final upscale.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class ReorderedPipeline:
    """V6 Reordered Pipeline: Upscale → Weave → Inpaint → Upscale.

    This PoC variant explores different stage ordering for comparison:
    - Upscaling before weave removal may help spectral analysis
    - Inpainting at 640 resolution (before final upscale) may reduce artifacts
    - Final upscale applies to both scene and per-object RGBA
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        use_sam_refinement: bool = False,
        use_refined_mask_for_inpaint: bool = False,
        inpaint_mask_dilate_px: int = 0,
        skip_upscale: bool = False,
    ):
        """Initialize the reordered pipeline.

        Args:
            device: Computation device. Auto-detects CUDA if None.
            output_dir: Base output directory for all results.
            use_sam_refinement: If True, enable SAM mask refinement (for extraction).
            use_refined_mask_for_inpaint: If True, use SAM-refined polygon for inpaint mask.
            inpaint_mask_dilate_px: Pixels to dilate refined mask before inpainting.
            skip_upscale: If True, skip both upscale stages (A and D).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/e2e_ablation_v6_reordered")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SAM refinement must be enabled if using refined mask for inpaint
        if use_refined_mask_for_inpaint and not use_sam_refinement:
            use_sam_refinement = True

        self.use_sam_refinement = use_sam_refinement
        self.use_refined_mask_for_inpaint = use_refined_mask_for_inpaint
        self.inpaint_mask_dilate_px = inpaint_mask_dilate_px
        self.skip_upscale = skip_upscale

        # Lazy-loaded components
        self._enhancement_pipeline = None
        self._occlusion_pipeline = None

        print(f"[v6] ReorderedPipeline initialized on {self.device}")
        if use_sam_refinement:
            print(f"[v6]   SAM refinement: ENABLED")
        else:
            print(f"[v6]   SAM refinement: DISABLED")
        if use_refined_mask_for_inpaint:
            print(f"[v6]   Inpaint mask: SAM-refined (tight){' + ' + str(inpaint_mask_dilate_px) + 'px dilate' if inpaint_mask_dilate_px > 0 else ''}")
        else:
            print(f"[v6]   Inpaint mask: Original (wide)")
        if skip_upscale:
            print(f"[v6]   Upscale stages: SKIPPED (Stage A and D disabled)")

    @property
    def enhancement_pipeline(self):
        """Lazy load EnhancementPipeline for upscale and spectral grid methods."""
        if self._enhancement_pipeline is None:
            from kp3d.modules.enhancement import EnhancementPipeline
            from kp3d.modules.enhancement.config import EnhancementConfig

            config = EnhancementConfig(
                use_spectral_grid=True,
                enable_grid_removal=True,
                enable_first_upscale=True,
                enable_second_upscale=True,
            )
            self._enhancement_pipeline = EnhancementPipeline(
                config=config, device=self.device
            )
            print("[v6] EnhancementPipeline loaded")
        return self._enhancement_pipeline

    @property
    def occlusion_pipeline(self):
        """Lazy load OcclusionPipeline for inpainting."""
        if self._occlusion_pipeline is None:
            from kp3d.modules.occlusion.pipeline import OcclusionPipeline
            from kp3d.modules.occlusion.base import OcclusionConfig

            config = OcclusionConfig(
                inpaint_method="boundary_guided",
                use_sam_refinement=self.use_sam_refinement,
                use_refined_mask_for_inpaint=self.use_refined_mask_for_inpaint,
                inpaint_mask_dilate_px=self.inpaint_mask_dilate_px,
            )
            self._occlusion_pipeline = OcclusionPipeline(
                config=config,
                output_dir=str(self.output_dir / "occlusion_tmp"),
            )
            print("[v6] OcclusionPipeline loaded")
        return self._occlusion_pipeline

    def _numpy_to_tensor(self, img_rgb: np.ndarray) -> Tensor:
        """Convert RGB uint8 numpy (H, W, 3) to tensor (1, 3, H, W) float [0,1]."""
        tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=self.device, dtype=torch.float32)

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor (1, 3, H, W) float [0,1] to RGB uint8 numpy (H, W, 3)."""
        img = tensor[0].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def _scale_annotation(
        self,
        annotation_path: str,
        scale: float,
        output_path: str,
    ) -> str:
        """Scale annotation coordinates by a factor and save to new path.

        Args:
            annotation_path: Path to original labelme JSON
            scale: Scale factor for coordinates
            output_path: Path to save scaled JSON

        Returns:
            Path to the scaled annotation file
        """
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            scaled_points = [[p[0] * scale, p[1] * scale] for p in shape["points"]]
            shape["points"] = scaled_points

        # Update image dimensions if present
        if "imageHeight" in data:
            data["imageHeight"] = int(data["imageHeight"] * scale)
        if "imageWidth" in data:
            data["imageWidth"] = int(data["imageWidth"] * scale)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path

    def _upscale_rgba_2x(self, rgba: np.ndarray) -> np.ndarray:
        """Upscale RGBA image 2x: RGB via RealESRGAN, alpha via cv2.resize.

        Args:
            rgba: RGBA image (H, W, 4) uint8

        Returns:
            Upscaled RGBA (2H, 2W, 4) uint8
        """
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        # Upscale RGB via RealESRGAN
        try:
            rgb_tensor = self._numpy_to_tensor(rgb)
            rgb_up_tensor, _ = self.enhancement_pipeline._upscale_2x(rgb_tensor, "rgba_upscale")
            rgb_up = self._tensor_to_numpy(rgb_up_tensor)
        except Exception as e:
            print(f"[v6] Warning: RealESRGAN failed for RGBA RGB, using bicubic: {e}")
            rgb_up = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Upscale alpha via cv2.resize (linear interpolation for smooth alpha)
        alpha_up = cv2.resize(alpha, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Merge back
        h, w = rgb_up.shape[:2]
        rgba_up = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_up[:, :, :3] = rgb_up
        rgba_up[:, :, 3] = alpha_up

        return rgba_up

    def process(
        self,
        image_path: str,
        annotation_path: str,
        stem: str,
        config_name: str = "full_pipeline",
    ) -> Dict[str, Any]:
        """Run the reordered v6 pipeline on a single image.

        Pipeline: Upscale 2x → Weave Removal → Inpainting → Upscale 2x

        Args:
            image_path: Path to input RGB image (expected 320x320)
            annotation_path: Path to labelme JSON annotation
            stem: Image stem name for output organization
            config_name: Configuration variant name (default: full_pipeline)

        Returns:
            Dict with metadata including timing per stage and file paths
        """
        start_total = time.time()
        metadata = {
            "stem": stem,
            "config": config_name,
            "stages": {},
            "sizes": {},
            "objects": [],
        }

        out_dir = self.output_dir / config_name / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load original image
        print(f"\n[v6] Processing {stem}")
        image_320 = cv2.imread(str(image_path))
        if image_320 is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_320_rgb = cv2.cvtColor(image_320, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_320_rgb.shape[:2]
        metadata["sizes"]["00_original"] = (h_orig, w_orig)
        print(f"[v6]   Original: {w_orig}x{h_orig}")

        # Save original
        cv2.imwrite(str(out_dir / "00_original.png"), image_320)

        # ============ Stage A: Upscale 2x (320 → 640) ============
        if self.skip_upscale:
            print("[v6]   Stage A: SKIPPED (no upscale)")
            image_640_rgb = image_320_rgb  # Keep variable name for downstream compatibility
            h_640, w_640 = h_orig, w_orig
            tensor_320 = self._numpy_to_tensor(image_320_rgb)
            tensor_640 = tensor_320  # Keep variable name for downstream compatibility
            metadata["stages"]["A_upscale"] = 0.0
            metadata["sizes"]["01_upscale1"] = (h_orig, w_orig)
        else:
            t_a = time.time()
            print("[v6]   Stage A: Upscale 2x...")
            tensor_320 = self._numpy_to_tensor(image_320_rgb)
            tensor_640, meta_a = self.enhancement_pipeline._upscale_2x(tensor_320, "upscale_1")
            image_640_rgb = self._tensor_to_numpy(tensor_640)
            h_640, w_640 = image_640_rgb.shape[:2]
            metadata["stages"]["A_upscale"] = time.time() - t_a
            metadata["sizes"]["01_upscale1"] = (h_640, w_640)
            print(f"[v6]     → {w_640}x{h_640} ({metadata['stages']['A_upscale']:.2f}s)")

            # Save Stage A output
            cv2.imwrite(
                str(out_dir / "01_upscale1.png"),
                cv2.cvtColor(image_640_rgb, cv2.COLOR_RGB2BGR)
            )

        # ============ Stage B: Weave Removal (at 640) ============
        t_b = time.time()
        print("[v6]   Stage B: Weave Removal...")
        tensor_640_weave, meta_b = self.enhancement_pipeline._spectral_grid_remove(tensor_640)
        image_640_weave_rgb = self._tensor_to_numpy(tensor_640_weave)
        metadata["stages"]["B_weave_removal"] = time.time() - t_b
        metadata["sizes"]["02_weave_removed"] = image_640_weave_rgb.shape[:2]
        print(f"[v6]     → {image_640_weave_rgb.shape[1]}x{image_640_weave_rgb.shape[0]} ({metadata['stages']['B_weave_removal']:.2f}s)")

        # Save Stage B output
        cv2.imwrite(
            str(out_dir / "02_weave_removed.png"),
            cv2.cvtColor(image_640_weave_rgb, cv2.COLOR_RGB2BGR)
        )

        # ============ Stage C: Inpainting (at 640) ============
        t_c = time.time()
        print("[v6]   Stage C: Inpainting...")

        # Scale annotation from 320 to 640 (or keep at 320 if skip_upscale)
        scaled_anno_path = str(out_dir / "_scaled_anno.json")
        scale_factor = 1.0 if self.skip_upscale else 2.0
        self._scale_annotation(annotation_path, scale=scale_factor, output_path=scaled_anno_path)

        # Run OcclusionPipeline.process_full_pipeline
        # upscale=False (we already upscaled), restore=False, save_outputs=False
        occ_result = self.occlusion_pipeline.process_full_pipeline(
            image=image_640_weave_rgb,
            annotation_path=scaled_anno_path,
            upscale=False,
            restore=False,
            save_outputs=False,
        )

        # Extract results
        processed_image_640 = occ_result["image"]  # Original scene (unchanged by inpainting)
        extracted_objects = occ_result["extracted"]  # Dict[label, RGBA HxWx4 uint8]
        inpainted_objects = occ_result["inpainted"]  # Dict[label, full-scene RGB with inpainted pixels]

        # Composite per-object inpainted content back into scene
        # This makes the inpaint mask source toggle produce visible scene differences
        inpainted_scene_640 = processed_image_640.copy()
        total_inpainted_px = 0

        for label, inp_full in inpainted_objects.items():
            # inp_full is full-scene-sized image with inpainted pixels in occlusion area
            # Find where it differs from original scene → those are the inpainted pixels
            diff = np.abs(inp_full.astype(np.int32) - processed_image_640.astype(np.int32)).sum(axis=2)
            diff_mask = diff > 5  # Threshold to ignore noise
            inpainted_scene_640[diff_mask] = inp_full[diff_mask]
            total_inpainted_px += diff_mask.sum()

        print(f"[v6]     → Composited {total_inpainted_px} inpainted pixels into scene")

        metadata["stages"]["C_inpainting"] = time.time() - t_c
        metadata["sizes"]["03_inpainted_scene"] = inpainted_scene_640.shape[:2]
        metadata["inpainted_pixels"] = int(total_inpainted_px)
        print(f"[v6]     → Scene: {inpainted_scene_640.shape[1]}x{inpainted_scene_640.shape[0]} ({metadata['stages']['C_inpainting']:.2f}s)")
        print(f"[v6]     → Objects: {list(extracted_objects.keys())}")

        # Save Stage C scene (now with visible inpainting)
        cv2.imwrite(
            str(out_dir / "03_inpainted_scene.png"),
            cv2.cvtColor(inpainted_scene_640, cv2.COLOR_RGB2BGR)
        )

        # ============ Stage D: Upscale 2x on scene + per-object RGBA (640 → 1280) ============
        if self.skip_upscale:
            print("[v6]   Stage D: SKIPPED (no upscale)")
            # Save inpainted scene at native resolution as "04_scene_upscaled.png"
            cv2.imwrite(
                str(out_dir / "04_scene_upscaled.png"),
                cv2.cvtColor(inpainted_scene_640, cv2.COLOR_RGB2BGR)
            )
            metadata["sizes"]["04_scene_upscaled"] = inpainted_scene_640.shape[:2]

            # Save objects at native resolution
            for label, rgba_orig in extracted_objects.items():
                out_path = out_dir / f"05_object_{label}_rgba.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(rgba_orig, cv2.COLOR_RGBA2BGRA))
                metadata["objects"].append({
                    "label": label,
                    "size_640": (rgba_orig.shape[0], rgba_orig.shape[1]),
                    "size_1280": None,
                })
            metadata["stages"]["D_upscale_final"] = 0.0
        else:
            t_d = time.time()
            print("[v6]   Stage D: Upscale 2x (scene + objects)...")

            # Upscale scene (use composited scene with visible inpainting)
            scene_tensor = self._numpy_to_tensor(inpainted_scene_640)
            scene_1280_tensor, _ = self.enhancement_pipeline._upscale_2x(scene_tensor, "upscale_scene")
            scene_1280_rgb = self._tensor_to_numpy(scene_1280_tensor)
            metadata["sizes"]["04_scene_upscaled"] = scene_1280_rgb.shape[:2]
            print(f"[v6]     → Scene: {scene_1280_rgb.shape[1]}x{scene_1280_rgb.shape[0]}")

            # Save scene
            cv2.imwrite(
                str(out_dir / "04_scene_upscaled.png"),
                cv2.cvtColor(scene_1280_rgb, cv2.COLOR_RGB2BGR)
            )

            # Upscale each object RGBA
            for label, rgba_640 in extracted_objects.items():
                print(f"[v6]     → Upscaling object: {label}")
                try:
                    rgba_1280 = self._upscale_rgba_2x(rgba_640)
                    out_path = out_dir / f"05_object_{label}_rgba.png"
                    cv2.imwrite(str(out_path), cv2.cvtColor(rgba_1280, cv2.COLOR_RGBA2BGRA))
                    metadata["objects"].append({
                        "label": label,
                        "size_640": (rgba_640.shape[0], rgba_640.shape[1]),
                        "size_1280": (rgba_1280.shape[0], rgba_1280.shape[1]),
                    })
                except Exception as e:
                    print(f"[v6]     Warning: Failed to upscale {label}: {e}")
                    # Fallback: save 640 version
                    out_path = out_dir / f"05_object_{label}_rgba.png"
                    cv2.imwrite(str(out_path), cv2.cvtColor(rgba_640, cv2.COLOR_RGBA2BGRA))
                    metadata["objects"].append({
                        "label": label,
                        "size_640": (rgba_640.shape[0], rgba_640.shape[1]),
                        "size_1280": None,
                        "error": str(e),
                    })

            metadata["stages"]["D_upscale_final"] = time.time() - t_d
            print(f"[v6]     ({metadata['stages']['D_upscale_final']:.2f}s)")

        # Total time
        metadata["total_time"] = time.time() - start_total
        print(f"[v6]   Total: {metadata['total_time']:.2f}s")

        # Save metadata
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata
