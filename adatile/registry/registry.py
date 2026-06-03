"""Registry / Factory pattern for extensible module management."""

from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for decorator-based module registration.

    Usage:
        BACKBONE = Registry("backbone")

        @BACKBONE.register()
        class ViTBackbone(nn.Module):
            ...
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Type[T]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class in this registry."""

        def _register(cls: Type[T]) -> Type[T]:
            key = name or cls.__name__
            if key in self._registry:
                raise KeyError(
                    f"Module '{key}' already registered in registry '{self._name}'."
                )
            self._registry[key] = cls
            return cls

        return _register

    def get(self, name: str) -> Type[T]:
        """Retrieve a registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Module '{name}' not found in registry '{self._name}'. "
                f"Available: {available}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered class by name with given kwargs."""
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list:
        """List all registered module names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry('{self._name}', modules={self.list()})"


# ── Core Module Registries ──────────────────────────────────────────

BACKBONE = Registry("backbone")
"""Registry for backbone networks (ViT, ResNet, FastSAM encoder, etc.)."""

SPARSE = Registry("sparse")
"""Registry for sparse importance predictors (Ada-SPM, etc.)."""

TOKENIZER = Registry("tokenizer")
"""Registry for dynamic tile tokenizers."""

ROUTER = Registry("router")
"""Registry for dynamic token routers (DTR-v2, etc.)."""

DECODER = Registry("decoder")
"""Registry for segmentation decoders."""

PROTOTYPE = Registry("prototype")
"""Registry for prototype memory modules."""

SEGMENTATION = Registry("segmentation")
"""Registry for full segmentation pipelines."""

DATASET = Registry("dataset")
"""Registry for dataset classes."""

TRANSFORM = Registry("transform")
"""Registry for data transforms / augmentations."""

LOSS = Registry("loss")
"""Registry for loss functions."""
