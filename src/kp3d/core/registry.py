"""Module registry for dynamic module registration and retrieval."""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from loguru import logger

from kp3d.core.base import BasePreprocessModule


T = TypeVar("T", bound=BasePreprocessModule)


class ModuleRegistry:
    """Central registry for preprocessing modules.

    Provides a singleton pattern for registering, retrieving, and managing
    preprocessing modules throughout the application lifecycle.
    """

    _instance: Optional["ModuleRegistry"] = None
    _modules: Dict[str, Type[BasePreprocessModule]] = {}
    _instances: Dict[str, BasePreprocessModule] = {}

    def __new__(cls) -> "ModuleRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a module class.

        Args:
            name: Optional custom name. Uses class name if not provided.

        Returns:
            Decorator function.

        Example:
            @ModuleRegistry.register("real_esrgan")
            class RealESRGANModule(BasePreprocessModule):
                ...
        """
        def decorator(module_cls: Type[T]) -> Type[T]:
            module_name = name or module_cls.__name__
            if module_name in cls._modules:
                logger.warning(f"Module '{module_name}' already registered, overwriting")
            cls._modules[module_name] = module_cls
            logger.debug(f"Registered module: {module_name}")
            return module_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BasePreprocessModule:
        """Get or create a module instance by name.

        Args:
            name: Registered module name.
            **kwargs: Arguments passed to module constructor.

        Returns:
            Module instance.

        Raises:
            KeyError: If module name is not registered.
        """
        if name not in cls._modules:
            available = ", ".join(cls._modules.keys())
            raise KeyError(
                f"Module '{name}' not found. Available modules: {available or 'none'}"
            )

        # Create new instance with provided kwargs
        return cls._modules[name](**kwargs)

    @classmethod
    def get_singleton(cls, name: str, **kwargs: Any) -> BasePreprocessModule:
        """Get a singleton instance of a module.

        Creates the instance on first call, returns cached instance thereafter.

        Args:
            name: Registered module name.
            **kwargs: Arguments passed to module constructor (only on first call).

        Returns:
            Cached module instance.
        """
        if name not in cls._instances:
            cls._instances[name] = cls.get(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def list_modules(cls) -> List[str]:
        """List all registered module names.

        Returns:
            List of registered module names.
        """
        return list(cls._modules.keys())

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a module is registered.

        Args:
            name: Module name to check.

        Returns:
            True if module is registered.
        """
        return name in cls._modules

    @classmethod
    def clear(cls) -> None:
        """Clear all registered modules and cached instances.

        Primarily for testing purposes.
        """
        cls._modules.clear()
        cls._instances.clear()
        logger.debug("Module registry cleared")

    @classmethod
    def clear_instances(cls) -> None:
        """Clear cached singleton instances but keep registrations.

        Useful for resetting state between runs.
        """
        cls._instances.clear()
        logger.debug("Module instances cleared")


# Convenience functions for module-level access
def register_module(name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
    """Register a preprocessing module.

    Convenience wrapper around ModuleRegistry.register().

    Args:
        name: Optional custom name for the module.

    Returns:
        Decorator function.

    Example:
        @register_module("my_module")
        class MyModule(BasePreprocessModule):
            ...
    """
    return ModuleRegistry.register(name)


def get_module(name: str, **kwargs: Any) -> BasePreprocessModule:
    """Get a module instance by name.

    Convenience wrapper around ModuleRegistry.get().

    Args:
        name: Registered module name.
        **kwargs: Arguments passed to module constructor.

    Returns:
        Module instance.
    """
    return ModuleRegistry.get(name, **kwargs)


def list_available_modules() -> List[str]:
    """List all available registered modules.

    Returns:
        List of module names.
    """
    return ModuleRegistry.list_modules()
