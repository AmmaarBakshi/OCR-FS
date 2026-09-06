"""Configuration loading, layering and persistence."""

from __future__ import annotations

import json

import pytest

from ocr_fusion.config import AppSettings, load_settings, reset_settings, save_settings
from ocr_fusion.config.prompts import DEFAULT_PROMPTS, render
from ocr_fusion.config.schema import FusionStrategy, UnlimitedBackend
from ocr_fusion.config.store import env_overrides


class TestDefaults:
    def test_ships_with_a_runnable_configuration(self):
        settings = AppSettings()
        assert settings.qwen.model == "qwen2.5vl:3b"
        assert settings.qwen.enabled
        assert settings.unlimited_ocr.enabled
        assert settings.pipeline.run_comparison
        assert settings.pipeline.run_fusion

    def test_transcription_defaults_to_deterministic_sampling(self):
        # Any creativity in an OCR pass is invention, so temperature starts at 0.
        assert AppSettings().qwen.temperature == 0.0

    def test_demo_mode_is_the_default(self):
        assert AppSettings().general.developer_mode is False

    def test_documents_are_not_persisted_by_default(self):
        privacy = AppSettings().privacy
        assert privacy.persist_documents is False
        assert privacy.persist_results is False
        assert privacy.redact_text_in_logs is True

    @pytest.mark.parametrize(
        "field,value",
        [
            ("temperature", -0.1),
            ("temperature", 2.5),
            ("max_tokens", 0),
            ("timeout_seconds", 0),
            ("retry_count", 9),
        ],
    )
    def test_rejects_out_of_range_values(self, field, value):
        with pytest.raises(Exception):
            AppSettings(qwen={field: value})

    def test_host_trailing_slash_is_normalised(self):
        # Otherwise every request URL ends up with a double slash.
        assert AppSettings(ollama={"host": "http://x:11434/"}).ollama.host == "http://x:11434"


class TestPersistence:
    def test_round_trips_through_a_file(self, tmp_path):
        settings = AppSettings()
        settings.qwen.model = "custom-model:7b"
        settings.pipeline.fusion_strategy = FusionStrategy.LLM
        path = save_settings(settings, tmp_path / "settings.json")

        loaded = load_settings(path)
        assert loaded.qwen.model == "custom-model:7b"
        assert loaded.pipeline.fusion_strategy is FusionStrategy.LLM

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "settings.json"
        save_settings(AppSettings(), path)
        # No temp files are left behind on success.
        assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        # A broken settings file must not stop the app from starting.
        path = tmp_path / "settings.json"
        path.write_text("{ this is not json")
        assert load_settings(path).qwen.model == "qwen2.5vl:3b"

    def test_invalid_values_fall_back_to_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"qwen": {"temperature": 99}}))
        assert load_settings(path).qwen.temperature == 0.0

    def test_missing_file_gives_defaults(self, tmp_path):
        assert load_settings(tmp_path / "absent.json").qwen.model == "qwen2.5vl:3b"

    def test_reset_removes_the_file(self, tmp_path):
        path = tmp_path / "settings.json"
        save_settings(AppSettings(), path)
        reset_settings(path)
        assert not path.exists()

    def test_unknown_keys_are_ignored(self, tmp_path):
        # An older or newer file must not break loading.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"qwen": {"model": "x:1b"}, "future_section": {"a": 1}}))
        assert load_settings(path).qwen.model == "x:1b"


class TestEnvironmentOverrides:
    def test_env_beats_the_settings_file(self, tmp_path, monkeypatch):
        settings = AppSettings()
        settings.ollama.host = "http://from-file:11434"
        path = save_settings(settings, tmp_path / "settings.json")

        monkeypatch.setenv("OCRFS_OLLAMA_HOST", "http://from-env:11434")
        assert load_settings(path).ollama.host == "http://from-env:11434"

    @pytest.mark.parametrize(
        "raw,expected", [("true", True), ("1", True), ("false", False), ("off", False)]
    )
    def test_booleans_are_coerced(self, monkeypatch, raw, expected):
        monkeypatch.setenv("OCRFS_DEVELOPER_MODE", raw)
        assert env_overrides()["general"]["developer_mode"] is expected

    def test_numbers_are_coerced(self, monkeypatch):
        monkeypatch.setenv("OCRFS_MAX_PAGES", "12")
        monkeypatch.setenv("OCRFS_QWEN_TEMPERATURE", "0.25")
        overrides = env_overrides()
        assert overrides["pipeline"]["max_pages"] == 12
        assert overrides["qwen"]["temperature"] == 0.25

    def test_empty_variable_is_ignored(self, monkeypatch):
        monkeypatch.setenv("OCRFS_OLLAMA_HOST", "")
        assert "ollama" not in env_overrides()

    def test_backend_can_be_selected_by_environment(self, monkeypatch, tmp_path):
        # Point the default path at a temp dir so the test never reads or
        # depends on the developer's own settings file.
        monkeypatch.setenv("OCRFS_SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.setenv("OCRFS_UNLIMITED_BACKEND", "http")
        assert load_settings().unlimited_ocr.backend is UnlimitedBackend.HTTP


class TestPrompts:
    def test_every_prompt_has_a_default(self):
        prompts = AppSettings().prompts
        for key, default in DEFAULT_PROMPTS.items():
            assert getattr(prompts, key) == default

    def test_a_prompt_can_be_reset_individually(self):
        prompts = AppSettings().prompts
        prompts.qwen_system = "changed"
        prompts.reset_field("qwen_system")
        assert prompts.qwen_system == DEFAULT_PROMPTS["qwen_system"]

    def test_resetting_an_unknown_prompt_raises(self):
        with pytest.raises(KeyError):
            AppSettings().prompts.reset_field("nope")

    def test_render_fills_placeholders(self):
        assert render("A={a} B={b}", a=1, b=2) == "A=1 B=2"

    def test_render_survives_user_typos(self):
        # A user editing a prompt in Settings must not be able to crash a run.
        assert render("hello {unknown}") == "hello {unknown}"
        assert render("unbalanced {brace") == "unbalanced {brace"

    def test_fusion_prompt_carries_both_engine_texts(self):
        filled = render(
            AppSettings().prompts.fusion_user,
            engine_a="A",
            text_a="alpha",
            engine_b="B",
            text_b="beta",
        )
        assert "alpha" in filled and "beta" in filled


class TestRedaction:
    def test_api_key_is_removed_from_snapshots(self):
        # The snapshot is embedded in exports, so a key must never reach it.
        settings = AppSettings()
        settings.unlimited_ocr.api_key = "sk-secret-value"
        assert settings.redacted()["unlimited_ocr"]["api_key"] == "***"
        assert "sk-secret-value" not in json.dumps(settings.redacted())

    def test_absent_key_stays_empty(self):
        assert AppSettings().redacted()["unlimited_ocr"]["api_key"] == ""
