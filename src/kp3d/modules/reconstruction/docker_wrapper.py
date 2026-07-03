"""Docker wrapper for 3D reconstruction models.

This module provides a unified interface to run 3D reconstruction
models (Wonder3D, TripoSR) inside Docker containers.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DockerReconstructionWrapper:
    """Wrapper for Docker-based 3D reconstruction.

    Supports Wonder3D and TripoSR models running in isolated Docker containers.

    Usage:
        wrapper = DockerReconstructionWrapper(model="wonder3d")
        result = wrapper.reconstruct("input.png", "output.obj")
    """

    SUPPORTED_MODELS = ["wonder3d", "triposr"]

    def __init__(
        self,
        model: str = "wonder3d",
        docker_image: Optional[str] = None,
        gpu_id: int = 0
    ):
        """Initialize Docker wrapper.

        Args:
            model: Model to use ("wonder3d" or "triposr")
            docker_image: Custom Docker image name (default: kp3d-{model}:latest)
            gpu_id: GPU device ID to use
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of {self.SUPPORTED_MODELS}")

        self.model = model
        self.docker_image = docker_image or f"kp3d-{model}:latest"
        self.gpu_id = gpu_id

        # Check if Docker is available
        self._check_docker()

    def _check_docker(self):
        """Check if Docker is available and image exists."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Docker not found")

            logger.info(f"Docker available: {result.stdout.strip()}")

        except FileNotFoundError:
            raise RuntimeError("Docker not installed. Please install Docker first.")

    def _check_image(self) -> bool:
        """Check if Docker image exists."""
        result = subprocess.run(
            ["docker", "images", "-q", self.docker_image],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())

    def build_image(self, dockerfile_dir: Optional[str] = None):
        """Build Docker image if not exists.

        Args:
            dockerfile_dir: Directory containing Dockerfile
        """
        if self._check_image():
            logger.info(f"Image {self.docker_image} already exists")
            return

        if dockerfile_dir is None:
            # Default location
            project_root = Path(__file__).parent.parent.parent.parent.parent
            dockerfile_dir = project_root / "docker" / "triposr"

        dockerfile = "Dockerfile" if self.model == "wonder3d" else "Dockerfile.triposr"

        logger.info(f"Building Docker image {self.docker_image}...")

        result = subprocess.run(
            [
                "docker", "build",
                "-t", self.docker_image,
                "-f", str(dockerfile_dir / dockerfile),
                str(dockerfile_dir)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")

        logger.info(f"Successfully built {self.docker_image}")

    def reconstruct(
        self,
        input_path: str,
        output_path: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Run 3D reconstruction on single image.

        Args:
            input_path: Path to input RGBA image
            output_path: Path to output .obj file
            timeout: Timeout in seconds

        Returns:
            Dict with reconstruction results
        """
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Docker command
        cmd = [
            "docker", "run",
            "--rm",
            "--gpus", f"device={self.gpu_id}",
            "-v", f"{input_path.parent}:/input:ro",
            "-v", f"{output_path.parent}:/output",
            self.docker_image,
            "--input", f"/input/{input_path.name}",
            "--output", f"/output/{output_path.name}"
        ]

        logger.info(f"Running Docker: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                logger.error(f"Docker run failed:\n{result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info(f"Reconstruction complete: {output_path}")

            return {
                "success": True,
                "output_path": str(output_path),
                "stdout": result.stdout
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Docker run timed out after {timeout}s")
            return {"success": False, "error": "Timeout"}

    def reconstruct_batch(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.png",
        timeout: int = 1800
    ) -> Dict[str, Any]:
        """Run batch reconstruction.

        Args:
            input_dir: Directory with input images
            output_dir: Directory for output meshes
            pattern: File pattern to match
            timeout: Timeout in seconds

        Returns:
            Dict with batch results
        """
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker", "run",
            "--rm",
            "--gpus", f"device={self.gpu_id}",
            "-v", f"{input_dir}:/input:ro",
            "-v", f"{output_dir}:/output",
            self.docker_image,
            "--input-dir", "/input",
            "--output-dir", "/output"
        ]

        logger.info(f"Running batch reconstruction...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "output_dir": str(output_dir),
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout"}


class DockerReconstructor:
    """Integration with existing reconstruction interface.

    Drop-in replacement for Wonder3DReconstructor that uses Docker backend.
    """

    def __init__(self, config=None, model: str = "wonder3d"):
        """Initialize Docker-based reconstructor.

        Args:
            config: ReconstructionConfig (for compatibility)
            model: Model to use ("wonder3d" or "triposr")
        """
        self.config = config
        self.wrapper = DockerReconstructionWrapper(model=model)
        self._built = False

    def _ensure_image(self):
        """Ensure Docker image is built."""
        if not self._built:
            if not self.wrapper._check_image():
                logger.info("Docker image not found, building...")
                self.wrapper.build_image()
            self._built = True

    def reconstruct_single(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        output_path: str = "output.obj",
        inpaint: bool = False
    ):
        """Reconstruct 3D mesh from image.

        Args:
            image: RGB image (H, W, 3)
            mask: Alpha mask (H, W)
            output_path: Output mesh path
            inpaint: Whether to inpaint (ignored, should be done before)

        Returns:
            ReconstructionResult-like object
        """
        import tempfile
        import cv2
        from .base import ReconstructionResult

        self._ensure_image()

        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_input = f.name

        try:
            # Save RGBA image
            if mask is not None:
                rgba = np.dstack([image, mask])
            else:
                rgba = np.dstack([image, np.ones(image.shape[:2], dtype=np.uint8) * 255])

            cv2.imwrite(temp_input, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

            # Run reconstruction
            result = self.wrapper.reconstruct(temp_input, output_path)

            if not result["success"]:
                raise RuntimeError(f"Reconstruction failed: {result.get('error', 'Unknown')}")

            # Load mesh
            import trimesh
            mesh = trimesh.load(output_path)

            return ReconstructionResult(
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.faces),
                textures=self._get_vertex_colors(mesh),
                multi_view_images=[],
                metadata={
                    "model": self.wrapper.model,
                    "docker": True,
                    "output_path": output_path
                }
            )

        finally:
            # Cleanup temp file
            Path(temp_input).unlink(missing_ok=True)

    def _get_vertex_colors(self, mesh) -> np.ndarray:
        """Extract vertex colors from mesh."""
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            return np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
        return np.ones((len(mesh.vertices), 3)) * 0.7


# Convenience function
def reconstruct_with_docker(
    input_path: str,
    output_path: str,
    model: str = "wonder3d"
) -> str:
    """Convenience function for Docker-based reconstruction.

    Args:
        input_path: Input RGBA image path
        output_path: Output .obj path
        model: "wonder3d" or "triposr"

    Returns:
        Output mesh path
    """
    wrapper = DockerReconstructionWrapper(model=model)
    result = wrapper.reconstruct(input_path, output_path)

    if not result["success"]:
        raise RuntimeError(f"Reconstruction failed: {result.get('error')}")

    return result["output_path"]
