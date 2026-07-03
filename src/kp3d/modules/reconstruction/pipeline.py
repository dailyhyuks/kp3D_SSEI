"""High-level pipeline for 3D reconstruction of painting objects."""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

from .base import (
    BaseReconstructor,
    ReconstructionModel,
    ReconstructionConfig,
    ReconstructionResult,
)


class ReconstructionPipeline:
    """High-level pipeline for object 3D reconstruction.

    This pipeline integrates multiple processing stages:
        1. Segmentation → Identify individual objects
        2. Background removal → Clean object isolation
        3. Inpainting → Complete occluded regions (optional)
        4. 3D Reconstruction → Generate mesh for each object
        5. Export → Save meshes to files

    The pipeline can process single objects or multiple objects from
    the same painting in batch mode.

    Example:
        >>> pipeline = ReconstructionPipeline(
        ...     model=ReconstructionModel.WONDER3D
        ... )
        >>>
        >>> # Single object reconstruction
        >>> result = pipeline.reconstruct_single(
        ...     image=painting_crop,
        ...     mask=object_mask,
        ...     output_path="output/pine_tree.obj"
        ... )
        >>>
        >>> # Multiple objects from segmentation
        >>> objects = [
        ...     {"mask": mask1, "label": "pine_tree"},
        ...     {"mask": mask2, "label": "crane"},
        ... ]
        >>> results = pipeline.reconstruct_multi(
        ...     image=full_painting,
        ...     objects=objects,
        ...     output_dir="output/objects/"
        ... )

    Attributes:
        model: Which reconstruction model to use
        config: Reconstruction configuration
        _reconstructor: Lazy-loaded reconstructor instance
    """

    def __init__(
        self,
        model: ReconstructionModel = ReconstructionModel.WONDER3D,
        config: Optional[ReconstructionConfig] = None
    ):
        """Initialize reconstruction pipeline.

        Args:
            model: Which reconstruction model to use
            config: Optional reconstruction configuration
        """
        self.model = model
        self.config = config or ReconstructionConfig(model=model)
        self._reconstructor: Optional[BaseReconstructor] = None

    @property
    def reconstructor(self) -> BaseReconstructor:
        """Lazy load appropriate reconstructor based on model selection.

        Returns:
            Reconstructor instance for the selected model
        """
        if self._reconstructor is None:
            if self.model == ReconstructionModel.WONDER3D:
                from .wonder3d import Wonder3DReconstructor
                self._reconstructor = Wonder3DReconstructor(self.config)
            elif self.model == ReconstructionModel.INSTANTMESH:
                from .instantmesh import InstantMeshReconstructor
                self._reconstructor = InstantMeshReconstructor(self.config)
            elif self.model == ReconstructionModel.LGM:
                from .lgm import LGMReconstructor
                self._reconstructor = LGMReconstructor(self.config)
            else:
                raise ValueError(f"Unknown reconstruction model: {self.model}")

        return self._reconstructor

    def reconstruct_single(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: Optional[str] = None,
        inpaint: bool = False
    ) -> ReconstructionResult:
        """Reconstruct a single object from image and mask.

        Args:
            image: Input RGB image (H, W, 3)
            mask: Binary mask (H, W) for the object
            output_path: Optional path to save the mesh
            inpaint: Whether to inpaint occluded regions before reconstruction

        Returns:
            ReconstructionResult containing mesh and metadata
        """
        # Optional: Inpaint occluded regions
        if inpaint:
            image = self._inpaint_occluded(image, mask)

        # Run 3D reconstruction
        result = self.reconstructor.reconstruct(image, mask)

        # Save mesh if output path provided
        if output_path is not None and result.vertices is not None:
            self._save_result(result, output_path)
            result.mesh_path = output_path

        return result

    def reconstruct_multi(
        self,
        image: np.ndarray,
        objects: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        inpaint: bool = False
    ) -> List[ReconstructionResult]:
        """Reconstruct multiple separated objects from the same painting.

        Args:
            image: Full painting image (H, W, 3)
            objects: List of objects, each containing:
                - mask: Binary mask (H, W)
                - label: Object label/name (optional)
                - bbox: Bounding box [x1, y1, x2, y2] (optional)
            output_dir: Directory to save meshes (uses object labels for filenames)
            inpaint: Whether to inpaint occluded regions

        Returns:
            List of ReconstructionResults, one per object
        """
        results = []

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, obj in enumerate(objects):
            mask = obj["mask"]
            label = obj.get("label", f"object_{i:03d}")

            # Crop to bounding box if provided (for efficiency)
            if "bbox" in obj:
                x1, y1, x2, y2 = obj["bbox"]
                crop_image = image[y1:y2, x1:x2]
                crop_mask = mask[y1:y2, x1:x2]
            else:
                crop_image = image
                crop_mask = mask

            # Determine output path
            output_path = None
            if output_dir is not None:
                output_path = str(
                    Path(output_dir) / f"{label}.{self.config.output_format}"
                )

            # Reconstruct this object
            result = self.reconstruct_single(
                image=crop_image,
                mask=crop_mask,
                output_path=output_path,
                inpaint=inpaint
            )

            # Add object metadata
            result.metadata["label"] = label
            result.metadata["object_index"] = i

            results.append(result)

        return results

    def _inpaint_occluded(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Inpaint occluded regions to complete the object.

        This is useful when objects are partially occluded in the painting.
        The inpainting can help produce more complete 3D reconstructions.

        Args:
            image: Input RGB image (H, W, 3)
            mask: Binary mask (H, W) for the object

        Returns:
            Inpainted image with occluded regions filled
        """
        # TODO: Integrate inpainting module when available
        # from ..inpainting import InpaintingPipeline
        # inpainter = InpaintingPipeline()
        # return inpainter.inpaint(image, mask)

        # For now, return original image
        return image

    def _save_result(
        self,
        result: ReconstructionResult,
        output_path: str
    ) -> None:
        """Save reconstruction result to file.

        Args:
            result: ReconstructionResult to save
            output_path: Where to save the mesh
        """
        if result.vertices is None or result.faces is None:
            raise ValueError("Cannot save result: no mesh data available")

        # Use reconstructor's save method
        self.reconstructor.save_mesh(
            output_path=output_path,
            vertices=result.vertices,
            faces=result.faces,
            textures=result.textures
        )
