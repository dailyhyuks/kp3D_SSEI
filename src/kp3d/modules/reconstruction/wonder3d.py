"""Wonder3D-based 3D reconstruction implementation.

Now uses TripoSR (Stability AI) as the backend for reliable single-image-to-3D.
"""

from typing import Optional
import numpy as np
import logging

from .base import BaseReconstructor, ReconstructionConfig, ReconstructionResult

logger = logging.getLogger(__name__)


class Wonder3DReconstructor(BaseReconstructor):
    """Wonder3D-based 3D reconstruction using TripoSR backend.

    TripoSR is Stability AI's open-source single-image-to-3D model that
    provides fast and reliable mesh generation from a single image.

    Pipeline:
        1. Remove background (if mask provided)
        2. Run TripoSR to generate 3D mesh
        3. Extract mesh vertices and faces
        4. Apply texture from input image

    Features:
        - Fast inference (~5-10 seconds on GPU)
        - High-quality mesh output
        - Supports textured output
        - Works well for isolated objects

    Attributes:
        config: Reconstruction configuration
        _model: Lazy-loaded TripoSR model instance
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """Initialize Wonder3D reconstructor with TripoSR backend.

        Args:
            config: Reconstruction configuration. Uses defaults if None.
        """
        super().__init__(config)
        self._model = None
        self._use_triposr = True

    @property
    def model(self):
        """Lazy load TripoSR model.

        Returns:
            TSR model instance or None if loading fails
        """
        if self._model is None:
            try:
                logger.info("Loading TripoSR model...")
                import torch

                # Try TripoSR
                try:
                    from tsr.system import TSR

                    device = "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"

                    self._model = TSR.from_pretrained(
                        "stabilityai/TripoSR",
                        config_name="config.yaml",
                        weight_name="model.ckpt"
                    )
                    self._model.to(device)
                    self._model.renderer.set_chunk_size(8192)  # Lower for memory efficiency

                    logger.info(f"TripoSR model loaded on {device}")
                    self._use_triposr = True

                except ImportError:
                    logger.warning("TripoSR not available, trying alternative...")
                    self._model = None
                    self._use_triposr = False

            except Exception as e:
                logger.error(f"Could not load 3D model: {e}")
                self._model = None

        return self._model

    def reconstruct(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> ReconstructionResult:
        """Reconstruct 3D mesh from single image.

        Args:
            image: Input RGB image (H, W, 3), range [0, 255]
            mask: Optional binary mask (H, W) for foreground object

        Returns:
            ReconstructionResult with mesh, textures, and metadata
        """
        logger.info("Starting 3D reconstruction...")

        # Step 1: Preprocess image - remove background
        rgba_image = self.preprocess_image(image, mask)
        logger.info("Image preprocessed")

        # Try TripoSR first
        if self.model is not None and self._use_triposr:
            try:
                return self._reconstruct_triposr(rgba_image)
            except Exception as e:
                logger.error(f"TripoSR reconstruction failed: {e}")
                logger.warning("Falling back to simple reconstruction")

        # Fallback to simple shape-from-silhouette
        return self._fallback_reconstruction(rgba_image)

    def _reconstruct_triposr(self, rgba_image: np.ndarray) -> ReconstructionResult:
        """Reconstruct using TripoSR model.

        Args:
            rgba_image: RGBA input image (H, W, 4)

        Returns:
            ReconstructionResult
        """
        import torch
        from PIL import Image

        # Convert to PIL Image
        pil_image = Image.fromarray(rgba_image)

        # Run TripoSR
        logger.info("Running TripoSR inference...")
        with torch.no_grad():
            scene_codes = self._model([pil_image], device=self._model.device)

        # Extract mesh
        logger.info("Extracting mesh...")
        meshes = self._model.extract_mesh(
            scene_codes,
            resolution=self.config.resolution,
            threshold=25.0,
        )

        mesh = meshes[0]

        # Get vertices and faces
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # Get vertex colors if available
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            textures = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
        else:
            textures = self._apply_texture(vertices, faces, rgba_image)

        # Generate multi-view renders
        multi_view_images = self._render_multiview(vertices, faces, textures)

        logger.info(f"TripoSR reconstruction complete: {len(vertices)} vertices, {len(faces)} faces")

        return ReconstructionResult(
            vertices=vertices,
            faces=faces,
            textures=textures,
            multi_view_images=multi_view_images,
            metadata={
                "model": "triposr",
                "num_views": len(multi_view_images),
                "resolution": self.config.resolution,
                "num_vertices": len(vertices),
                "num_faces": len(faces),
            }
        )

    def _render_multiview(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        textures: np.ndarray,
        num_views: int = 6
    ) -> list[np.ndarray]:
        """Render mesh from multiple viewpoints using matplotlib.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            textures: Vertex colors (N, 3)
            num_views: Number of views to render

        Returns:
            List of rendered images
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import io
            from PIL import Image

            views = []

            # Normalize vertices to [-1, 1]
            v_center = vertices.mean(axis=0)
            vertices_centered = vertices - v_center
            v_scale = np.abs(vertices_centered).max()
            if v_scale > 0:
                vertices_norm = vertices_centered / v_scale
            else:
                vertices_norm = vertices_centered

            for i in range(num_views):
                angle = i * (360.0 / num_views)

                fig = plt.figure(figsize=(3, 3), dpi=85)
                ax = fig.add_subplot(111, projection='3d')

                # Create face colors from vertex colors
                if textures is not None and len(textures) > 0:
                    face_colors = []
                    for face in faces:
                        # Average color of face vertices
                        fc = textures[face].mean(axis=0)
                        face_colors.append(fc)
                    face_colors = np.array(face_colors)
                else:
                    face_colors = np.ones((len(faces), 3)) * 0.7

                # Create polygon collection
                mesh_faces = vertices_norm[faces]
                poly = Poly3DCollection(mesh_faces, alpha=1.0)
                poly.set_facecolor(face_colors)
                poly.set_edgecolor('none')
                ax.add_collection3d(poly)

                # Set view angle
                ax.view_init(elev=20, azim=angle)

                # Set axis limits
                ax.set_xlim([-1.2, 1.2])
                ax.set_ylim([-1.2, 1.2])
                ax.set_zlim([-1.2, 1.2])

                # Clean up axes
                ax.set_axis_off()
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')

                # Render to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight',
                           pad_inches=0, facecolor='white', edgecolor='none')
                buf.seek(0)
                img = Image.open(buf).convert('RGB')

                # Resize to 256x256
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                view = np.array(img)
                views.append(view)

                plt.close(fig)
                buf.close()

            return views

        except Exception as e:
            logger.warning(f"Matplotlib rendering failed: {e}, using simple projection")
            return self._render_simple_projection(vertices, faces, textures, num_views)

    def _render_simple_projection(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        textures: np.ndarray,
        num_views: int = 6
    ) -> list[np.ndarray]:
        """Simple orthographic projection rendering using OpenCV.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            textures: Vertex colors (N, 3)
            num_views: Number of views

        Returns:
            List of rendered images
        """
        import cv2
        import math

        views = []
        img_size = 256

        # Normalize vertices
        v_center = vertices.mean(axis=0)
        vertices_centered = vertices - v_center
        v_scale = np.abs(vertices_centered).max()
        if v_scale > 0:
            vertices_norm = vertices_centered / v_scale
        else:
            vertices_norm = vertices_centered

        for i in range(num_views):
            angle = i * (360.0 / num_views)
            rad = math.radians(angle)

            # Create rotation matrix around Y axis
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            rot_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])

            # Rotate vertices
            rotated = vertices_norm @ rot_matrix.T

            # Project to 2D (orthographic)
            proj_x = (rotated[:, 0] * 0.4 + 0.5) * img_size
            proj_y = (0.5 - rotated[:, 1] * 0.4) * img_size

            # Create image
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

            # Sort faces by depth (back to front)
            face_depths = []
            for face in faces:
                z_avg = rotated[face, 2].mean()
                face_depths.append(z_avg)
            sorted_indices = np.argsort(face_depths)

            # Draw faces
            for fi in sorted_indices:
                face = faces[fi]
                pts = np.array([
                    [proj_x[face[0]], proj_y[face[0]]],
                    [proj_x[face[1]], proj_y[face[1]]],
                    [proj_x[face[2]], proj_y[face[2]]]
                ], dtype=np.int32)

                # Get face color
                if textures is not None and len(textures) > 0:
                    color = (textures[face].mean(axis=0) * 255).astype(int)
                    color = tuple(color.tolist())
                else:
                    color = (180, 180, 180)

                # Add simple shading based on face normal
                v0, v1, v2 = rotated[face[0]], rotated[face[1]], rotated[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                normal_len = np.linalg.norm(normal)
                if normal_len > 0:
                    normal = normal / normal_len
                    light_dir = np.array([0.3, 0.5, 0.8])
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    shade = max(0.3, min(1.0, np.dot(normal, light_dir) * 0.5 + 0.5))
                    color = tuple(int(c * shade) for c in color)

                cv2.fillPoly(img, [pts], color)

            views.append(img)

        return views

    def _create_placeholder_view(self, view_idx: int, total_views: int) -> np.ndarray:
        """Create a placeholder view image.

        Args:
            view_idx: Index of this view
            total_views: Total number of views

        Returns:
            Placeholder image
        """
        import cv2

        img = np.ones((256, 256, 3), dtype=np.uint8) * 220

        # Draw text
        text = f"View {view_idx + 1}/{total_views}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        x = (256 - text_size[0]) // 2
        y = (256 + text_size[1]) // 2
        cv2.putText(img, text, (x, y), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        return img

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

        # Normalize vertices to [0, 1]
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_range = v_max - v_min
        v_range[v_range == 0] = 1.0
        v_norm = (vertices - v_min) / v_range

        img_h, img_w = img_rgb.shape[:2]
        img_x = np.clip(v_norm[:, 0] * img_w, 0, img_w - 1).astype(int)
        img_y = np.clip((1 - v_norm[:, 1]) * img_h, 0, img_h - 1).astype(int)  # Flip Y

        # Sample colors
        vertex_colors = img_rgb[img_y, img_x] / 255.0

        return vertex_colors

    def _create_simple_sphere(self, radius: float = 1.0, subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Create a simple sphere mesh as fallback.

        Args:
            radius: Sphere radius
            subdivisions: Number of subdivisions (higher = smoother)

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

    def _fallback_reconstruction(self, rgba_image: np.ndarray) -> ReconstructionResult:
        """Fallback reconstruction using shape-from-silhouette.

        Args:
            rgba_image: RGBA input image

        Returns:
            ReconstructionResult with simple mesh
        """
        logger.warning("Using fallback reconstruction with shape-from-silhouette")

        # Extract silhouette
        if rgba_image.shape[-1] == 4:
            silhouette = rgba_image[..., 3] / 255.0
        else:
            silhouette = np.mean(rgba_image, axis=-1) / 255.0

        # Create mesh from silhouette
        vertices, faces = self._silhouette_to_mesh(silhouette)

        # Apply texture
        textures = self._apply_texture(vertices, faces, rgba_image)

        # Generate actual multi-view renders of the mesh
        multi_view_images = self._render_multiview(vertices, faces, textures, num_views=6)

        return ReconstructionResult(
            vertices=vertices,
            faces=faces,
            textures=textures,
            multi_view_images=multi_view_images,
            metadata={
                "model": "fallback_silhouette",
                "num_vertices": len(vertices),
                "num_faces": len(faces),
                "warning": "Using fallback reconstruction"
            }
        )

    def _silhouette_to_mesh(self, silhouette: np.ndarray, grid_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Convert silhouette to 3D mesh using voxel carving.

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

            # Validate silhouette
            if silhouette is None or silhouette.size == 0:
                return self._create_simple_sphere()

            if len(silhouette.shape) != 2:
                silhouette = silhouette.mean(axis=-1) if len(silhouette.shape) == 3 else silhouette.squeeze()

            if silhouette.max() > 1.0:
                silhouette = silhouette / 255.0

            # Check foreground
            foreground_ratio = (silhouette > 0.5).sum() / silhouette.size
            if foreground_ratio < 0.01:
                return self._create_simple_sphere()

            # Resize
            silhouette_resized = cv2.resize(
                silhouette.astype(np.float32),
                (grid_size, grid_size),
                interpolation=cv2.INTER_LINEAR
            )

            # Create voxel grid by extruding silhouette
            voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            for z in range(grid_size):
                voxel_grid[:, :, z] = silhouette_resized

            # Carve from multiple angles
            for i in range(4):
                angle = i * 90.0
                self._carve_voxels(voxel_grid, silhouette_resized, angle, grid_size)

            # Smooth
            voxel_grid = gaussian_filter(voxel_grid, sigma=1.0)

            # Extract mesh
            vmin, vmax = voxel_grid.min(), voxel_grid.max()
            if vmax - vmin < 0.01:
                return self._create_simple_sphere()

            level = (vmin + vmax) / 2
            vertices, faces, _, _ = measure.marching_cubes(voxel_grid, level=level)
            vertices = (vertices / grid_size) * 2 - 1

            if len(vertices) < 4:
                return self._create_simple_sphere()

            return vertices.astype(np.float32), faces.astype(np.int32)

        except Exception as e:
            logger.warning(f"Silhouette to mesh failed: {e}")
            return self._create_simple_sphere()

    def _carve_voxels(self, voxel_grid: np.ndarray, silhouette: np.ndarray, angle: float, grid_size: int):
        """Carve voxels based on silhouette from viewing angle."""
        import math
        import cv2

        sil_h, sil_w = silhouette.shape[:2]
        if sil_h != grid_size or sil_w != grid_size:
            silhouette = cv2.resize(silhouette.astype(np.float32), (grid_size, grid_size))

        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        half_grid = grid_size / 2.0

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x = i - half_grid
                    z = k - half_grid
                    x_rot = cos_a * x - sin_a * z
                    img_x = int(x_rot + half_grid)
                    img_y = j
                    img_x = max(0, min(grid_size - 1, img_x))
                    img_y = max(0, min(grid_size - 1, img_y))
                    if silhouette[img_y, img_x] < 0.5:
                        voxel_grid[i, j, k] *= 0.5

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
                f.write("# OBJ file generated by Wonder3D reconstructor\n")

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
