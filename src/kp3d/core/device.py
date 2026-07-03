"""Device management utilities for GPU/CPU fallback and memory management."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from loguru import logger


@dataclass
class DeviceInfo:
    """Information about an available compute device."""

    device: torch.device
    name: str
    is_cuda: bool
    total_memory: int = 0  # bytes
    free_memory: int = 0  # bytes
    compute_capability: Optional[Tuple[int, int]] = None

    @property
    def memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory / (1024**3)

    @property
    def free_memory_gb(self) -> float:
        """Free memory in GB."""
        return self.free_memory / (1024**3)


class DeviceManager:
    """Manages device selection, memory monitoring, and GPU/CPU fallback.

    Provides utilities for intelligent device selection based on available
    resources and memory requirements.
    """

    _instance: Optional["DeviceManager"] = None

    def __new__(cls) -> "DeviceManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize device manager."""
        if self._initialized:
            return

        self._current_device: Optional[torch.device] = None
        self._devices: List[DeviceInfo] = []
        self._scan_devices()
        self._initialized = True

    def _scan_devices(self) -> None:
        """Scan and catalog available compute devices."""
        self._devices = []

        # Add CPU
        self._devices.append(DeviceInfo(
            device=torch.device("cpu"),
            name="CPU",
            is_cuda=False,
        ))

        # Add CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                props = torch.cuda.get_device_properties(i)

                # Get memory info
                total_mem = props.total_memory
                free_mem = total_mem - torch.cuda.memory_allocated(i)

                self._devices.append(DeviceInfo(
                    device=device,
                    name=props.name,
                    is_cuda=True,
                    total_memory=total_mem,
                    free_memory=free_mem,
                    compute_capability=(props.major, props.minor),
                ))

        logger.info(f"Found {len(self._devices)} compute devices")
        for dev in self._devices:
            if dev.is_cuda:
                logger.info(f"  - {dev.name}: {dev.memory_gb:.1f} GB VRAM")
            else:
                logger.info(f"  - {dev.name}")

    @property
    def devices(self) -> List[DeviceInfo]:
        """Get list of all available devices."""
        return self._devices.copy()

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    @property
    def current_device(self) -> torch.device:
        """Get the current default device."""
        if self._current_device is None:
            self._current_device = self.get_optimal_device()
        return self._current_device

    def set_device(self, device: torch.device) -> None:
        """Set the current default device.

        Args:
            device: Device to set as default.
        """
        self._current_device = device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        logger.info(f"Set current device to: {device}")

    def get_optimal_device(
        self,
        min_memory_gb: float = 0.0,
        prefer_cuda: bool = True,
    ) -> torch.device:
        """Get the optimal device based on requirements.

        Args:
            min_memory_gb: Minimum required GPU memory in GB.
            prefer_cuda: Whether to prefer CUDA over CPU.

        Returns:
            The optimal device meeting requirements.
        """
        if not prefer_cuda or not self.cuda_available:
            return torch.device("cpu")

        # Find best CUDA device
        cuda_devices = [d for d in self._devices if d.is_cuda]

        if min_memory_gb > 0:
            # Filter by memory requirement
            suitable = [d for d in cuda_devices if d.memory_gb >= min_memory_gb]
            if suitable:
                # Return device with most free memory
                best = max(suitable, key=lambda d: d.free_memory)
                return best.device

        # Return first CUDA device or fall back to CPU
        if cuda_devices:
            return cuda_devices[0].device

        logger.warning("No suitable CUDA device found, falling back to CPU")
        return torch.device("cpu")

    def get_memory_info(self, device: Optional[torch.device] = None) -> Tuple[int, int]:
        """Get memory information for a device.

        Args:
            device: Device to query. Uses current device if None.

        Returns:
            Tuple of (allocated_bytes, total_bytes).
        """
        device = device or self.current_device

        if device.type != "cuda":
            # CPU memory info not easily available
            return (0, 0)

        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return (allocated, total)

    def get_free_memory_gb(self, device: Optional[torch.device] = None) -> float:
        """Get free memory in GB for a device.

        Args:
            device: Device to query. Uses current device if None.

        Returns:
            Free memory in GB.
        """
        allocated, total = self.get_memory_info(device)
        if total == 0:
            return float("inf")  # CPU has "unlimited" memory
        return (total - allocated) / (1024**3)

    def clear_cache(self, device: Optional[torch.device] = None) -> None:
        """Clear CUDA cache to free memory.

        Args:
            device: Device to clear. Clears all CUDA devices if None.
        """
        if torch.cuda.is_available():
            if device is not None and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    def memory_guard(
        self,
        required_gb: float,
        device: Optional[torch.device] = None,
    ) -> torch.device:
        """Ensure sufficient memory is available, falling back to CPU if needed.

        Args:
            required_gb: Required memory in GB.
            device: Preferred device. Uses current device if None.

        Returns:
            Device with sufficient memory (may be CPU as fallback).
        """
        device = device or self.current_device

        if device.type != "cuda":
            return device

        free_gb = self.get_free_memory_gb(device)

        if free_gb >= required_gb:
            return device

        # Try clearing cache
        self.clear_cache(device)
        free_gb = self.get_free_memory_gb(device)

        if free_gb >= required_gb:
            return device

        # Fall back to CPU
        logger.warning(
            f"Insufficient GPU memory ({free_gb:.1f}GB free, {required_gb:.1f}GB needed). "
            "Falling back to CPU."
        )
        return torch.device("cpu")


def get_optimal_device(
    min_memory_gb: float = 0.0,
    prefer_cuda: bool = True,
) -> torch.device:
    """Convenience function to get optimal device.

    Args:
        min_memory_gb: Minimum required GPU memory in GB.
        prefer_cuda: Whether to prefer CUDA over CPU.

    Returns:
        The optimal device meeting requirements.
    """
    return DeviceManager().get_optimal_device(min_memory_gb, prefer_cuda)


def clear_gpu_cache() -> None:
    """Convenience function to clear GPU cache."""
    DeviceManager().clear_cache()
