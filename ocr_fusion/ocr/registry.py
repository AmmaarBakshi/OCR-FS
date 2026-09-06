"""Provider registry.

Concrete engines register a factory here; the pipeline and the UI look them up
by id. This is the seam that lets a new engine (PaddleOCR, EasyOCR, a cloud
API) be added by writing one module and registering it - no UI change and no
pipeline change (spec s11).

Factories take :class:`~ocr_fusion.config.schema.AppSettings` and return a
provider, so construction stays lazy: importing the registry must never import
torch or open a network connection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ocr_fusion.config.schema import AppSettings
from ocr_fusion.ocr.interface import OCRProvider

ProviderFactory = Callable[[AppSettings], OCRProvider]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Registry entry describing one engine."""

    provider_id: str
    display_name: str
    factory: ProviderFactory
    description: str = ""
    enabled_check: Callable[[AppSettings], bool] | None = None
    """Reads the engine's own ``enabled`` flag out of settings."""

    tags: tuple[str, ...] = field(default_factory=tuple)

    def is_enabled(self, settings: AppSettings) -> bool:
        return True if self.enabled_check is None else bool(self.enabled_check(settings))


class ProviderRegistry:
    """An ordered, name-addressable collection of provider specs."""

    def __init__(self) -> None:
        self._specs: dict[str, ProviderSpec] = {}

    def register(self, spec: ProviderSpec, *, replace: bool = False) -> ProviderSpec:
        """Add a provider spec.

        Registering a duplicate id is an error unless ``replace`` is set, which
        catches the common mistake of two engines claiming the same id.
        """
        if spec.provider_id in self._specs and not replace:
            raise ValueError(f"Provider {spec.provider_id!r} is already registered")
        self._specs[spec.provider_id] = spec
        return spec

    def unregister(self, provider_id: str) -> None:
        self._specs.pop(provider_id, None)

    def get(self, provider_id: str) -> ProviderSpec:
        try:
            return self._specs[provider_id]
        except KeyError:
            known = ", ".join(sorted(self._specs)) or "none"
            raise KeyError(
                f"Unknown provider {provider_id!r}. Registered providers: {known}"
            ) from None

    def has(self, provider_id: str) -> bool:
        return provider_id in self._specs

    def specs(self) -> list[ProviderSpec]:
        return list(self._specs.values())

    def ids(self) -> list[str]:
        return list(self._specs)

    def create(self, provider_id: str, settings: AppSettings) -> OCRProvider:
        """Instantiate one provider."""
        return self.get(provider_id).factory(settings)

    def create_many(
        self, provider_ids: Iterable[str], settings: AppSettings
    ) -> list[OCRProvider]:
        return [self.create(pid, settings) for pid in provider_ids]

    def enabled_specs(self, settings: AppSettings) -> list[ProviderSpec]:
        """Specs whose engine is switched on in settings, in registration order."""
        return [spec for spec in self._specs.values() if spec.is_enabled(settings)]

    def describe(self) -> list[dict[str, Any]]:
        """Registry contents for the Settings UI and documentation."""
        return [
            {
                "provider_id": spec.provider_id,
                "display_name": spec.display_name,
                "description": spec.description,
                "tags": list(spec.tags),
            }
            for spec in self._specs.values()
        ]

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, provider_id: object) -> bool:
        return provider_id in self._specs


#: The registry the application uses. Populated by :mod:`ocr_fusion.ocr.providers`.
default_registry = ProviderRegistry()


def register_provider(spec: ProviderSpec, *, replace: bool = False) -> ProviderSpec:
    """Register a spec on the default registry."""
    return default_registry.register(spec, replace=replace)


__all__ = [
    "ProviderFactory",
    "ProviderRegistry",
    "ProviderSpec",
    "default_registry",
    "register_provider",
]
