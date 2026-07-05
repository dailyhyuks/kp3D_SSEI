"""Base classes and types for 3D reconstruction."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class ReconstructionModel(Enum):
    """Supported 3D reconstruction models."""

    WONDER3D = "wonder3d"
    INSTANTMESH = "instantmesh"
    LGM = "lgm"


@dataclass
class ReconstructionConfig:
    """Configuration for 3D reconstruction.

    Attributes:
        model: Which reconstruction model to use
        output_format: Output mesh format ("obj", "glb", "ply")
        resolution: Resolution for texture and normal maps
        num_views: Number of views to generate for multi-view methods
        device: Compute device ("cuda", "cpu", "mps")
    """

    model: ReconstructionModel = ReconstructionModel.INSTANTMESH
    output_format: str = "obj"  # "obj", "glb", "ply"
    resolution: int = 256
    num_views: int = 6
    device: str = "cuda"


@dataclass
class ReconstructionResult:
    """Result from 3D reconstruction.

    Attributes:
        mesh_path: Path to saved mesh file
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
        textures: Texture coordinates or vertex colors
        normal_maps: Generated normal maps for multi-view methods
        multi_view_images: Generated multi-view color images
        metadata: Additional metadata (processing time, model info, etc.)
    """

    mesh_path: Optional[str] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    textures: Optional[np.ndarray] = None
    normal_maps: Optional[List[np.ndarray]] = None
    multi_view_images: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReconstructor:
    """Base class for 3D reconstruction modules.

    All reconstruction implementations should inherit from this class
    and implement the reconstruct method.
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """Initialize reconstructor with configuration.

        Args:
            config: Reconstruction configuration. Uses defaults if None.
        """
        self.config = config or ReconstructionConfig()

    def reconstruct(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> ReconstructionResult:
        """Reconstruct 3D mesh from single image.

        Args:
            image: Input image (H, W, 3) in RGB format
            mask: Optional binary mask (H, W) for foreground object

        Returns:
            ReconstructionResult containing mesh and metadata

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement reconstruct()")

    def preprocess_image(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Remove background and prepare RGBA image for reconstruction.

        Args:
            image: Input RGB image (H, W, 3)
            mask: Optional binary mask (H, W) where 1 = foreground, 0 = background

        Returns:
            RGBA image (H, W, 4) with background removed
        """
        if mask is None:
            # No mask provided, return image with full alpha channel
            alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            return np.concatenate([image, alpha], axis=-1)

        # Apply mask to create RGBA image
        mask_3d = np.expand_dims(mask, axis=-1)
        rgba = np.concatenate([image, mask_3d * 255], axis=-1)
        return rgba.astype(np.uint8)

    def save_mesh(
        self,
        output_path: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        textures: Optional[np.ndarray] = None
    ) -> str:
        """Save mesh to file.

        Args:
            output_path: Where to save the mesh
            vertices: Vertex positions (N, 3)
            faces: Face indices (M, 3)
            textures: Optional texture coordinates or vertex colors

        Returns:
            Path to saved mesh file

        Raises:
            NotImplementedError: Subclasses should implement format-specific saving
        """
        raise NotImplementedError("Subclasses should implement save_mesh()")
