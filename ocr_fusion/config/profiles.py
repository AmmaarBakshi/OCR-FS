"""Named points on the speed/accuracy curve.

There is no single right configuration. Reading a thousand statements to file
them is a different job from reading one contract someone will sign, and the
setting that serves both badly is the one chosen without knowing which job it
is for.

So rather than one tuned default, three points on a curve that was actually
measured. The numbers below come from the DPI sweep in
:class:`~ocr_fusion.config.schema.DocumentSettings`: a real tax-form page
transcribed by qwen2.5vl:3b and scored against the text layer that page was
authored with, on a 4-core i5-10310U with no usable GPU.

    profile     DPI   seconds/page   word error   recall
    fast         96            326        0.080    0.927
    balanced    120            398        0.028    0.990
    accurate    150            551        0.021    0.997

Those are per *scanned* page. A page with a usable text layer costs
milliseconds under every profile, which is why the profile matters far less
than the routing does: on the reference corpus 84.9% of pages never reach a
model at all.

``balanced`` is the default. ``fast`` is a real trade - at 96 DPI the word
error rate nearly triples, because blurred text does not just get misread, it
makes the model ramble - so it suits bulk triage where a human reviews the
flagged output, not a document being relied on unread.
"""

from __future__ import annotations

from enum import Enum

from ocr_fusion.config.schema import AppSettings, EngineMode


class PerformanceProfile(str, Enum):
    """A named speed/accuracy trade-off."""

    FAST = "fast"
    """Bulk triage. Cheapest render, single pass, no second opinion."""

    BALANCED = "balanced"
    """The default. Near-ceiling accuracy at 72% of the cost of ``accurate``."""

    ACCURATE = "accurate"
    """Highest render resolution, and a second engine on anything doubtful."""


#: What each profile changes. Everything not named here is left as configured,
#: so a profile is a starting point rather than a reset.
PROFILE_SETTINGS: dict[PerformanceProfile, dict[str, object]] = {
    PerformanceProfile.FAST: {
        "pdf_render_dpi": 96,
        "max_image_dimension": 1600,
        "qwen_max_tokens": 2048,
        "fallback_enabled": False,
        "engine_mode": EngineMode.CASCADE,
    },
    PerformanceProfile.BALANCED: {
        "pdf_render_dpi": 120,
        "max_image_dimension": 2048,
        "qwen_max_tokens": 3072,
        "fallback_enabled": True,
        "engine_mode": EngineMode.CASCADE,
    },
    PerformanceProfile.ACCURATE: {
        "pdf_render_dpi": 150,
        "max_image_dimension": 2048,
        "qwen_max_tokens": 4096,
        "fallback_enabled": True,
        "engine_mode": EngineMode.CASCADE,
    },
}

#: Measured cost and accuracy per scanned page, for the Settings UI. Pages
#: answered from a text layer cost milliseconds under every profile.
PROFILE_MEASUREMENTS: dict[PerformanceProfile, dict[str, float]] = {
    PerformanceProfile.FAST: {"seconds_per_scanned_page": 326, "word_error_rate": 0.080, "recall": 0.927},
    PerformanceProfile.BALANCED: {"seconds_per_scanned_page": 398, "word_error_rate": 0.028, "recall": 0.990},
    PerformanceProfile.ACCURATE: {"seconds_per_scanned_page": 551, "word_error_rate": 0.021, "recall": 0.997},
}


def apply_profile(settings: AppSettings, profile: PerformanceProfile) -> AppSettings:
    """Set the performance knobs to ``profile``, in place.

    Touches only the settings that move the speed/accuracy trade-off. Engine
    choice, prompts, hosts, privacy and output settings are the operator's and
    are left exactly as they were.
    """
    values = PROFILE_SETTINGS[profile]
    settings.documents.pdf_render_dpi = int(values["pdf_render_dpi"])
    settings.documents.max_image_dimension = int(values["max_image_dimension"])
    settings.qwen.max_tokens = int(values["qwen_max_tokens"])
    settings.pipeline.fallback_enabled = bool(values["fallback_enabled"])
    settings.pipeline.engine_mode = values["engine_mode"]  # type: ignore[assignment]
    return settings


def describe_profile(profile: PerformanceProfile) -> str:
    """One line for the Settings UI, carrying the measured numbers."""
    measured = PROFILE_MEASUREMENTS[profile]
    dpi = PROFILE_SETTINGS[profile]["pdf_render_dpi"]
    return (
        f"{dpi} DPI - about {measured['seconds_per_scanned_page']:.0f}s a scanned "
        f"page on this class of machine, {measured['recall']:.1%} of words "
        "recovered."
    )


def current_profile(settings: AppSettings) -> PerformanceProfile | None:
    """Which profile these settings match, or ``None`` for a custom mix."""
    for profile, values in PROFILE_SETTINGS.items():
        if (
            settings.documents.pdf_render_dpi == values["pdf_render_dpi"]
            and settings.qwen.max_tokens == values["qwen_max_tokens"]
            and settings.pipeline.fallback_enabled == values["fallback_enabled"]
            and settings.pipeline.engine_mode == values["engine_mode"]
        ):
            return profile
    return None


__all__ = [
    "PROFILE_MEASUREMENTS",
    "PROFILE_SETTINGS",
    "PerformanceProfile",
    "apply_profile",
    "current_profile",
    "describe_profile",
]
