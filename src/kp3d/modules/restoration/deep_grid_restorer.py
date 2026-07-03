"""Deep Grid Restoration wrapper for v13 multiplicative deconvolution.

BaseRestoration wrapper around MultiplicativeGridRemover and NeuralResidualRefiner.
Handles Tensor <-> numpy conversion and ModuleOutput generation.
"""

import time

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Any, Optional

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig
from kp3d.modules.restoration.multiplicative_grid import MultiplicativeGridRemover
from kp3d.modules.restoration.neural_residual import NeuralResidualRefiner


class DeepGridRestorer(BaseRestoration):
    """Deep grid restoration for Korean traditional paintings (v13).

    Combines multiplicative deconvolution with neural residual refinement.
    Supports three modes via config.dg_method:
    - "deconv_only": Only multiplicative deconvolution
    - "deconv_neural": Deconvolution + SCUNet refinement (default)
    - "neural_only": Only SCUNet/bilateral refinement
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = True

        # Core processors
        self._grid_remover = MultiplicativeGridRemover()

        # Neural refiner (lazy init - only created if needed)
        self._refiner = None
        if self.config.dg_method in ("deconv_neural", "neural_only"):
            if self.config.dg_neural_model != "none":
                self._refiner = NeuralResidualRefiner(
                    model_name=self.config.dg_neural_model,
                    device=self.device,
                )

    @property
    def name(self) -> str:
        return "deep_grid"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy_bgr(self, tensor: Tensor) -> np.ndarray:
        """Convert [C,H,W] or [1,C,H,W] float tensor (0-1, RGB) to BGR uint8 numpy."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        arr_bgr = arr[:, :, ::-1].copy()
        return (np.clip(arr_bgr, 0, 1) * 255).astype(np.uint8)

    def _numpy_bgr_to_tensor(self, bgr: np.ndarray) -> Tensor:
        """Convert BGR uint8 numpy to [C,H,W] float tensor (0-1, RGB)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def _mask_to_tensor(self, mask: np.ndarray) -> Tensor:
        """Convert single-channel uint8 mask to 3-channel tensor for visualization."""
        vis = np.stack([mask, mask, mask], axis=-1)
        arr = vis.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Run deep grid restoration.

        Args:
            image: Input image tensor [1,C,H,W] or [C,H,W], float32, 0-1, RGB.

        Returns:
            ModuleOutput with result, intermediates, and metadata.
        """
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        img_bgr = self._tensor_to_numpy_bgr(image[0])

        method = self.config.dg_method
        intermediates_raw = {}

        if method in ("deconv_only", "deconv_neural"):
            # Stage 1-3: Multiplicative deconvolution
            result_bgr, grid_intermediates = self._grid_remover.process(
                img_bgr,
                period_detection=self.config.dg_period_detection,
                manual_period_x=self.config.dg_manual_period_x,
                manual_period_y=self.config.dg_manual_period_y,
                template_method=self.config.dg_template_method,
                deconv_strength=self.config.dg_deconv_strength,
                clamp_min=self.config.dg_deconv_clamp_min,
                clamp_max=self.config.dg_deconv_clamp_max,
                edge_protection=self.config.dg_edge_protection,
                ink_l_threshold=self.config.dg_ink_l_threshold,
                notch_width=self.config.dg_notch_width,
                notch_harmonics=self.config.dg_notch_harmonics,
                notch_attenuation=self.config.dg_notch_attenuation,
                edge_enhance=self.config.dg_edge_enhance,
                edge_detail_strength=self.config.dg_edge_detail_strength,
                edge_detail_sigma=self.config.dg_edge_detail_sigma,
                final_sharpen=self.config.dg_final_sharpen,
                final_sharpen_strength=self.config.dg_final_sharpen_strength,
                final_sharpen_sigma=self.config.dg_final_sharpen_sigma,
                final_sharpen_edge_threshold=self.config.dg_final_sharpen_edge_threshold,
                object_edge_enhance=self.config.dg_oee_enabled,
                oee_edge_sigma_scale=self.config.dg_oee_edge_sigma_scale,
                oee_detail_source=self.config.dg_oee_detail_source,
                oee_detail_sigma=self.config.dg_oee_detail_sigma,
                oee_enhance_strength=self.config.dg_oee_enhance_strength,
                oee_edge_low=self.config.dg_oee_edge_low,
                oee_edge_high=self.config.dg_oee_edge_high,
                oee_periodicity_rejection=self.config.dg_oee_periodicity_rejection,
            )
            intermediates_raw.update(grid_intermediates)
        else:
            result_bgr = img_bgr

        if method in ("deconv_neural", "neural_only"):
            # Stage 4: Neural refinement
            if self._refiner is not None:
                intermediates_raw["before_neural"] = result_bgr.copy()
                result_bgr = self._refiner.refine(
                    result_bgr, strength=self.config.dg_neural_strength
                )

        elapsed = time.time() - start

        # Build intermediates for ModuleOutput
        intermediates = {}
        if self.config.store_intermediates:
            intermediates["original"] = self._numpy_bgr_to_tensor(img_bgr)
            if "deconvolved" in intermediates_raw:
                intermediates["deconvolved"] = self._numpy_bgr_to_tensor(
                    intermediates_raw["deconvolved"]
                )
            if "before_neural" in intermediates_raw:
                intermediates["before_neural"] = self._numpy_bgr_to_tensor(
                    intermediates_raw["before_neural"]
                )
            if "oee_mask" in intermediates_raw:
                intermediates["oee_mask"] = self._mask_to_tensor(
                    intermediates_raw["oee_mask"]
                )
            if "template" in intermediates_raw:
                # Visualize template (normalize to 0-255)
                tpl = intermediates_raw["template"]
                tpl_vis = np.clip(tpl * 128, 0, 255).astype(np.uint8)
                if len(tpl_vis.shape) == 3:
                    intermediates["template"] = self._numpy_bgr_to_tensor(tpl_vis)

        result_tensor = self._numpy_bgr_to_tensor(result_bgr).unsqueeze(0)

        metadata = {
            "method": "deep_grid",
            "dg_method": method,
            "processing_time": elapsed,
            "period_x": intermediates_raw.get("period_x", -1),
            "period_y": intermediates_raw.get("period_y", -1),
            "dg_template_method": self.config.dg_template_method,
            "dg_deconv_strength": self.config.dg_deconv_strength,
            "dg_neural_model": self.config.dg_neural_model,
            "dg_neural_strength": self.config.dg_neural_strength,
            "dg_edge_protection": self.config.dg_edge_protection,
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata=metadata,
        )
