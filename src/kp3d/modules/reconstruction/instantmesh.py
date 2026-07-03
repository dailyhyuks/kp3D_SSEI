"""InstantMesh-based 3D reconstruction implementation."""

from typing import Optional
import numpy as np
import logging

from .base import BaseReconstructor, ReconstructionConfig, ReconstructionResult

logger = logging.getLogger(__name__)


class InstantMeshReconstructor(BaseReconstructor):
    """InstantMesh-based 3D reconstruction.

    InstantMesh is a feed-forward 3D reconstruction method that directly
    predicts mesh from a single image without per-shape optimization.
    This makes it very fast (~10 seconds per object) but may produce
    less detailed results compared to optimization-based methods.

    Pipeline:
        1. Remove background (if mask provided)
        2. Encode image to latent representation
        3. Decode latent to triplane representation
        4. Extract mesh from triplane using marching cubes
        5. Optionally refine with texture

    Features:
        - Very fast inference (~10 seconds)
        - No per-shape optimization needed
        - Good for simple, symmetric shapes
        - Lower quality than Wonder3D for complex objects

    Reference:
        https://github.com/TencentARC/InstantMesh

    Attributes:
        config: Reconstruction configuration
        _model: Lazy-loaded InstantMesh/OpenLRM model instance
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """Initialize InstantMesh reconstructor.

        Args:
            config: Reconstruction configuration. Uses defaults if None.
        """
        super().__init__(config)
        self._model = None
        self._encoder = None
        self._decoder = None

    @property
    def model(self):
        """Lazy load InstantMesh model using OpenLRM as implementation.

        Returns:
            Model instance or None if loading fails
        """
        if self._model is None:
            try:
                logger.info("Loading OpenLRM model for InstantMesh reconstruction...")
                import torch
                from transformers import AutoModel, AutoConfig

                # Try loading OpenLRM (open-source large reconstruction model)
                # This is a feed-forward model similar to InstantMesh
                model_name = "zxhezexin/openlrm-mix-base-1.1"

                dtype = torch.float16 if self.config.device == "cuda" and torch.cuda.is_available() else torch.float32

                try:
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    self._model = AutoModel.from_pretrained(
                        model_name,
                        config=config,
                        torch_dtype=dtype,
                        trust_remote_code=True
                    )

                    if self.config.device == "cuda" and torch.cuda.is_available():
                        self._model = self._model.to("cuda")
                        logger.info("Model loaded on CUDA")
                    else:
                        logger.warning("CUDA not available, using CPU (this will be slow)")
                        self._model = self._model.to("cpu")

                    self._model.eval()
                    logger.info("OpenLRM model loaded successfully")

                except Exception as e:
                    logger.warning(f"Could not load OpenLRM: {e}")
                    logger.info("Will use fallback triplane-based reconstruction")
                    self._model = None

            except ImportError as e:
                logger.error(f"Could not import required libraries: {e}")
                logger.warning("Please install: pip install transformers torch")
                self._model = None

        return self._model

    def reconstruct(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> ReconstructionResult:
        """Reconstruct 3D mesh from single image using InstantMesh.

        Args:
            image: Input RGB image (H, W, 3), range [0, 255]
            mask: Optional binary mask (H, W) for foreground object

        Returns:
            ReconstructionResult with mesh and metadata
        """
        logger.info("Starting InstantMesh reconstruction...")

        # Step 1: Preprocess image - remove background
        rgba_image = self.preprocess_image(image, mask)
        logger.info("Image preprocessed")

        # Try to use full model if available
        if self.model is not None:
            try:
                return self._reconstruct_with_model(rgba_image)
            except Exception as e:
                logger.error(f"Model-based reconstruction failed: {e}")
                logger.warning("Falling back to simple reconstruction")

        # Fallback to simple triplane-based reconstruction
        return self._reconstruct_simple(rgba_image)

    def _reconstruct_with_model(self, rgba_image: np.ndarray) -> ReconstructionResult:
        """Reconstruct using OpenLRM model.

        Args:
            rgba_image: RGBA input image (H, W, 4)

        Returns:
            ReconstructionResult
        """
        import torch
        from PIL import Image

        # Convert to PIL and resize
        pil_image = Image.fromarray(rgba_image)
        pil_image = pil_image.resize((self.config.resolution, self.config.resolution))

        # Prepare input tensor
        img_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        if self.config.device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")

        # Run inference
        with torch.no_grad():
            output = self._model(img_tensor)

        # Extract mesh from model output
        if hasattr(output, 'vertices') and hasattr(output, 'faces'):
            vertices = output.vertices.cpu().numpy()
            faces = output.faces.cpu().numpy()
        else:
            # Model might output triplane representation
            triplane = output.cpu().numpy() if torch.is_tensor(output) else output
            vertices, faces = self._extract_mesh_from_triplane(triplane)

        # Apply texture
        textures = self._apply_texture(vertices, faces, rgba_image)

        logger.info(f"Reconstruction complete: {len(vertices)} vertices, {len(faces)} faces")

        return ReconstructionResult(
            vertices=vertices,
            faces=faces,
            textures=textures,
            metadata={
                "model": "instantmesh",
                "resolution": self.config.resolution,
                "num_vertices": len(vertices),
                "num_faces": len(faces),
            }
        )

    def _reconstruct_simple(self, rgba_image: np.ndarray) -> ReconstructionResult:
        """Simple reconstruction using shape-from-silhouette.

        Args:
            rgba_image: RGBA input image (H, W, 4)

        Returns:
            ReconstructionResult
        """
        logger.info("Using simple shape-from-silhouette reconstruction")

        # Extract silhouette
        if rgba_image.shape[-1] == 4:
            silhouette = rgba_image[..., 3] / 255.0
        else:
            silhouette = np.mean(rgba_image, axis=-1) / 255.0

        # Create voxel grid from silhouette
        vertices, faces = self._silhouette_to_mesh(silhouette)

        # Apply colors from input image
        textures = self._apply_texture(vertices, faces, rgba_image)

        logger.info(f"Simple reconstruction complete: {len(vertices)} vertices, {len(faces)} faces")

        return ReconstructionResult(
            vertices=vertices,
            faces=faces,
            textures=textures,
            metadata={
                "model": "instantmesh_simple",
                "resolution": self.config.resolution,
                "num_vertices": len(vertices),
                "num_faces": len(faces),
            }
        )

    def _silhouette_to_mesh(self, silhouette: np.ndarray, grid_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Convert silhouette to 3D mesh using rotation and carving.

        Args:
            silhouette: 2D silhouette image (H, W)
            grid_size: Voxel grid resolution

        Returns:
            Tuple of (vertices, faces)
        """
        try:
            from skimage import measure
            from scipy.ndimage import gaussian_filter
            import cv2

            # Validate and preprocess silhouette
            if silhouette is None or silhouette.size == 0:
                logger.warning("Empty silhouette, using fallback sphere")
                return self._create_simple_sphere()

            # Ensure silhouette is 2D
            if len(silhouette.shape) != 2:
                silhouette = silhouette.mean(axis=-1) if len(silhouette.shape) == 3 else silhouette.squeeze()

            # Ensure values are in [0, 1]
            if silhouette.max() > 1.0:
                silhouette = silhouette / 255.0

            # Check if silhouette has enough foreground pixels
            foreground_ratio = (silhouette > 0.5).sum() / silhouette.size
            if foreground_ratio < 0.01:
                logger.warning(f"Silhouette has too few foreground pixels ({foreground_ratio:.2%}), using fallback sphere")
                return self._create_simple_sphere()

            # Resize silhouette to grid_size using cv2 for robustness
            silhouette_resized = cv2.resize(
                silhouette.astype(np.float32),
                (grid_size, grid_size),
                interpolation=cv2.INTER_LINEAR
            )

            # Create voxel grid - start with cylinder shape based on silhouette
            voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

            # Create initial shape by extruding silhouette along Z axis
            for z in range(grid_size):
                voxel_grid[:, :, z] = silhouette_resized

            # Carve from multiple views (simulate rotation)
            num_views = 4  # Reduced for stability
            for i in range(num_views):
                angle = i * (360.0 / num_views)
                self._carve_voxels(voxel_grid, silhouette_resized, angle, grid_size)

            # Smooth voxels
            voxel_grid = gaussian_filter(voxel_grid, sigma=1.0)

            # Check if voxel grid has valid range for marching cubes
            vmin, vmax = voxel_grid.min(), voxel_grid.max()
            if vmax - vmin < 0.01:
                logger.warning(f"Voxel grid has no variation (min={vmin:.3f}, max={vmax:.3f}), using fallback sphere")
                return self._create_simple_sphere()

            # Choose appropriate level for marching cubes
            level = (vmin + vmax) / 2
            if level <= vmin or level >= vmax:
                level = vmin + (vmax - vmin) * 0.3

            logger.debug(f"Marching cubes: vmin={vmin:.3f}, vmax={vmax:.3f}, level={level:.3f}")

            # Extract mesh using marching cubes
            vertices, faces, normals, _ = measure.marching_cubes(voxel_grid, level=level)

            # Normalize vertices to [-1, 1]
            vertices = (vertices / grid_size) * 2 - 1

            if len(vertices) < 4 or len(faces) < 1:
                logger.warning("Marching cubes produced insufficient geometry, using fallback sphere")
                return self._create_simple_sphere()

            return vertices.astype(np.float32), faces.astype(np.int32)

        except Exception as e:
            logger.warning(f"Silhouette to mesh failed: {e}, using simple sphere")
            return self._create_simple_sphere()

    def _carve_voxels(self, voxel_grid: np.ndarray, silhouette: np.ndarray, angle: float, grid_size: int = None):
        """Carve voxel grid based on silhouette from given angle.

        Args:
            voxel_grid: 3D voxel grid to carve
            silhouette: 2D silhouette image (should already be resized to grid_size)
            angle: Viewing angle in degrees
            grid_size: Grid size (optional, inferred from voxel_grid if None)
        """
        import math

        if grid_size is None:
            grid_size = voxel_grid.shape[0]

        # Ensure silhouette matches grid size
        sil_h, sil_w = silhouette.shape[:2]
        if sil_h != grid_size or sil_w != grid_size:
            import cv2
            silhouette = cv2.resize(
                silhouette.astype(np.float32),
                (grid_size, grid_size),
                interpolation=cv2.INTER_LINEAR
            )

        # Carve along viewing direction
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        half_grid = grid_size / 2.0

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Rotate point around Y axis
                    x = i - half_grid
                    z = k - half_grid
                    x_rot = cos_a * x - sin_a * z

                    # Project to image coordinates (clamped)
                    img_x = int(x_rot + half_grid)
                    img_y = j

                    # Clamp indices to valid range
                    img_x = max(0, min(grid_size - 1, img_x))
                    img_y = max(0, min(grid_size - 1, img_y))

                    # Carve where silhouette is background
                    if silhouette[img_y, img_x] < 0.5:
                        voxel_grid[i, j, k] *= 0.5  # Soft carving instead of hard

    def _extract_mesh_from_triplane(self, triplane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract mesh from triplane representation.

        Args:
            triplane: Triplane representation (3, H, W, C)

        Returns:
            Tuple of (vertices, faces)
        """
        # Convert triplane to voxel grid
        # This is a simplified approach - proper triplane decoding would query
        # the triplane at 3D coordinates and combine features

        try:
            from skimage import measure

            grid_size = 64

            # Create query points
            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            z = np.linspace(-1, 1, grid_size)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

            # Query triplane (simplified)
            # In practice, would interpolate from all 3 planes and combine
            voxel_grid = np.zeros((grid_size, grid_size, grid_size))

            # Simple averaging of triplane features
            if len(triplane.shape) == 4 and triplane.shape[0] >= 3:
                xy_plane = triplane[0].mean(axis=-1)
                xz_plane = triplane[1].mean(axis=-1)
                yz_plane = triplane[2].mean(axis=-1)

                for i in range(grid_size):
                    for j in range(grid_size):
                        for k in range(grid_size):
                            # Sample from each plane
                            val_xy = xy_plane[min(i, xy_plane.shape[0]-1), min(j, xy_plane.shape[1]-1)]
                            val_xz = xz_plane[min(i, xz_plane.shape[0]-1), min(k, xz_plane.shape[1]-1)]
                            val_yz = yz_plane[min(j, yz_plane.shape[0]-1), min(k, yz_plane.shape[1]-1)]
                            voxel_grid[i, j, k] = (val_xy + val_xz + val_yz) / 3

            # Extract mesh
            vertices, faces, _, _ = measure.marching_cubes(voxel_grid, level=voxel_grid.mean())

            # Normalize
            vertices = (vertices / grid_size) * 2 - 1

            return vertices, faces

        except Exception as e:
            logger.error(f"Triplane extraction failed: {e}")
            return self._create_simple_sphere()

    def _apply_texture(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        image: np.ndarray
    ) -> np.ndarray:
        """Apply texture to mesh from input image.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            image: Input RGBA image

        Returns:
            Vertex colors (N, 3)
        """
        if image.shape[-1] == 4:
            img_rgb = image[..., :3]
        else:
            img_rgb = image

        # Map vertices to image coordinates
        v_norm = (vertices + 1) / 2  # Normalize to [0, 1]

        img_h, img_w = img_rgb.shape[:2]
        img_x = np.clip(v_norm[:, 0] * img_w, 0, img_w - 1).astype(int)
        img_y = np.clip(v_norm[:, 1] * img_h, 0, img_h - 1).astype(int)

        # Sample colors
        vertex_colors = img_rgb[img_y, img_x] / 255.0

        return vertex_colors

    def _create_simple_sphere(self, radius: float = 1.0, subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Create a simple sphere mesh as fallback.

        Args:
            radius: Sphere radius
            subdivisions: Number of subdivisions

        Returns:
            Tuple of (vertices, faces)
        """
        try:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
            return np.array(sphere.vertices), np.array(sphere.faces)
        except ImportError:
            # Manual icosphere
            phi = (1 + np.sqrt(5)) / 2
            vertices = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ]) * radius / np.sqrt(1 + phi**2)

            faces = np.array([
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ])

            return vertices, faces

    def save_mesh(
        self,
        output_path: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        textures: Optional[np.ndarray] = None
    ) -> str:
        """Save mesh to file in OBJ format.

        Args:
            output_path: Where to save the mesh
            vertices: Vertex positions (N, 3)
            faces: Face indices (M, 3)
            textures: Optional vertex colors (N, 3)

        Returns:
            Path to saved mesh file
        """
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import trimesh

            # Create mesh with vertex colors if provided
            if textures is not None:
                vertex_colors = (textures * 255).astype(np.uint8)
                if vertex_colors.shape[1] == 3:
                    # Add alpha channel
                    alpha = np.ones((len(vertex_colors), 1), dtype=np.uint8) * 255
                    vertex_colors = np.hstack([vertex_colors, alpha])
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors
                )
            else:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            mesh.export(str(output_path))
            logger.info(f"Mesh saved to {output_path}")

        except ImportError:
            # Manual OBJ export
            with open(output_path, 'w') as f:
                f.write("# OBJ file generated by InstantMesh reconstructor\n")

                # Write vertices with colors if available
                for i, v in enumerate(vertices):
                    if textures is not None and i < len(textures):
                        c = textures[i]
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
                    else:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Write faces (1-indexed)
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            logger.info(f"Mesh saved to {output_path} (manual OBJ export)")

        return str(output_path)
