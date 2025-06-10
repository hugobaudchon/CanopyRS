from typing import Dict, Type


class Registry:
    """Registry for automatic model registration."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, *names):
        """Decorator to register a class with one or more names."""

        def decorator(cls):
            for name in names:
                self[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """Get a registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"'{name}' not found in {self.name} registry. Available: {available}")
        return self._registry[name]

    def list_available(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    @property
    def registry(self) -> Dict[str, Type]:
        """Access the internal registry dict."""
        return self._registry

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Get number of registered items."""
        return len(self._registry)

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"Registry('{self.name}', {len(self._registry)} items)"

    def __getitem__(self, name: str) -> Type:
        """Enable subscript access: registry[name] -> registry.get(name)"""
        return self.get(name)

    def __setitem__(self, name: str, cls: Type) -> None:
        """Enable subscript assignment: registry[name] = cls"""
        self._registry[name] = cls


DETECTOR_REGISTRY = Registry("detector")
SEGMENTER_REGISTRY = Registry("segmenter")
CLASSIFIER_REGISTRY = Registry("classifier")
EMBEDDER_REGISTRY = Registry("embedder")
