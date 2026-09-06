"""Pipeline execution, stage tracking and failure isolation."""

from __future__ import annotations

from ocr_fusion.ocr.interface import OCRStatus
from ocr_fusion.pipeline import OCRPipeline, build_pipeline
from ocr_fusion.pipeline.events import EventLog, LogLevel, StageStatus
from ocr_fusion.pipeline.pipeline import STAGE_COMPARISON, STAGE_FUSION, STAGE_UPLOAD
from tests.conftest import FakeProvider


class TestHappyPath:
    def test_runs_every_engine_in_order(self, settings, single_page_document, qwen_like, unlimited_like):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        assert [r.provider_id for r in result.engine_results] == ["qwen_vl", "unlimited_ocr"]
        assert qwen_like.process_calls == 1
        assert unlimited_like.process_calls == 1

    def test_all_stages_complete(self, settings, single_page_document, qwen_like, unlimited_like):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        statuses = {stage.key: stage.status for stage in result.stages}
        assert statuses[STAGE_UPLOAD] is StageStatus.COMPLETED
        assert statuses["qwen_vl"] is StageStatus.COMPLETED
        assert statuses[STAGE_COMPARISON] is StageStatus.COMPLETED
        assert statuses[STAGE_FUSION] is StageStatus.COMPLETED

    def test_every_completed_stage_is_timed(self, settings, single_page_document, qwen_like, unlimited_like):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        for stage in result.stages:
            if stage.status is StageStatus.COMPLETED:
                assert stage.duration_seconds is not None
        assert result.total_duration_seconds > 0

    def test_produces_a_final_result(self, settings, single_page_document, qwen_like, unlimited_like):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        assert result.succeeded
        assert "INV-1024" in result.final_text
        assert result.fusion is not None

    def test_comparison_is_produced(self, settings, single_page_document, qwen_like, unlimited_like):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        assert result.comparison is not None
        assert 0 < result.comparison.similarity <= 1.0

    def test_multi_page_documents_keep_page_numbers(self, settings, multi_page_document, qwen_like):
        result = OCRPipeline(settings, [qwen_like]).execute(multi_page_document)
        pages = result.engine_results[0].pages
        assert [p.page_number for p in pages] == [1, 2, 3]


class TestFailureIsolation:
    def test_a_failed_engine_does_not_stop_the_run(self, settings, single_page_document, qwen_like):
        broken = FakeProvider("unlimited_ocr", "Unlimited-OCR", "", fail=True)
        result = OCRPipeline(settings, [qwen_like, broken]).execute(single_page_document)

        assert result.succeeded
        assert "INV-1024" in result.final_text
        assert result.engine_results[1].status is OCRStatus.FAILED

    def test_failed_engine_becomes_a_failed_stage_with_a_remedy(
        self, settings, single_page_document, qwen_like
    ):
        broken = FakeProvider("unlimited_ocr", "Unlimited-OCR", "", fail=True)
        result = OCRPipeline(settings, [qwen_like, broken]).execute(single_page_document)
        stage = next(s for s in result.stages if s.key == "unlimited_ocr")
        assert stage.status is StageStatus.FAILED
        assert result.engine_results[1].remedy == "Start the engine"

    def test_an_engine_that_raises_is_contained(self, settings, single_page_document, qwen_like):
        exploding = FakeProvider("unlimited_ocr", "Unlimited-OCR", "", raises=True)
        result = OCRPipeline(settings, [qwen_like, exploding]).execute(single_page_document)
        assert result.succeeded
        assert "failed unexpectedly" in result.engine_results[1].error

    def test_comparison_is_skipped_when_only_one_engine_succeeds(
        self, settings, single_page_document, qwen_like
    ):
        broken = FakeProvider("unlimited_ocr", "Unlimited-OCR", "", fail=True)
        result = OCRPipeline(settings, [qwen_like, broken]).execute(single_page_document)
        stage = next(s for s in result.stages if s.key == STAGE_COMPARISON)
        assert stage.status is StageStatus.SKIPPED
        assert result.comparison is None

    def test_fusion_still_runs_with_a_single_engine(self, settings, single_page_document, qwen_like):
        broken = FakeProvider("unlimited_ocr", "Unlimited-OCR", "", fail=True)
        result = OCRPipeline(settings, [qwen_like, broken]).execute(single_page_document)
        assert result.fusion is not None
        assert result.final_text

    def test_all_engines_failing_is_reported_not_crashed(self, settings, single_page_document):
        a = FakeProvider("a", "A", "", fail=True)
        b = FakeProvider("b", "B", "", fail=True)
        result = OCRPipeline(settings, [a, b]).execute(single_page_document)

        assert result.succeeded is False
        assert result.final_text == ""
        assert len(result.failed_engines) == 2
        assert result.comparison is None
        assert result.fusion is None

    def test_partial_page_failure_is_partial_not_failed(self, settings, multi_page_document):
        flaky = FakeProvider("a", "A", "text", failing_pages=(2,))
        result = OCRPipeline(settings, [flaky]).execute(multi_page_document)
        assert result.engine_results[0].status is OCRStatus.PARTIAL
        assert result.succeeded

    def test_stop_on_error_when_configured(self, settings, single_page_document):
        settings.pipeline.continue_on_provider_error = False
        broken = FakeProvider("a", "A", "", fail=True)
        second = FakeProvider("b", "B", "text")
        OCRPipeline(settings, [broken, second]).execute(single_page_document)
        assert second.process_calls == 0

    def test_no_providers_yields_a_clean_empty_result(self, settings, single_page_document):
        result = OCRPipeline(settings, []).execute(single_page_document)
        assert result.succeeded is False
        assert result.engine_results == []


class TestStageToggles:
    def test_comparison_can_be_switched_off(self, settings, single_page_document, qwen_like, unlimited_like):
        settings.pipeline.run_comparison = False
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        assert result.comparison is None
        assert not any(s.key == STAGE_COMPARISON for s in result.stages)

    def test_fusion_can_be_switched_off(self, settings, single_page_document, qwen_like, unlimited_like):
        settings.pipeline.run_fusion = False
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        assert result.fusion is None
        # The raw engine output must still be reachable.
        assert result.final_text

    def test_page_limit_is_applied(self, settings, multi_page_document, qwen_like):
        settings.pipeline.max_pages = 2
        result = OCRPipeline(settings, [qwen_like]).execute(multi_page_document)
        assert result.document.page_count == 2
        assert result.document.warnings


class TestEventLog:
    def test_records_the_run(self, settings, single_page_document, qwen_like, unlimited_like):
        pipeline = OCRPipeline(settings, [qwen_like, unlimited_like])
        pipeline.execute(single_page_document)
        text = pipeline.log.as_text()
        assert "Document loaded" in text
        assert "Qwen2.5-VL" in text
        assert "Final result generated" in text

    def test_events_are_timestamped_in_the_documented_format(
        self, settings, single_page_document, qwen_like
    ):
        pipeline = OCRPipeline(settings, [qwen_like])
        pipeline.execute(single_page_document)
        first = pipeline.log.events[0]
        assert first.format().startswith("[")
        assert len(first.clock) == 8  # HH:MM:SS

    def test_document_text_is_kept_out_of_the_log(self, settings, single_page_document, qwen_like):
        # Documents may be confidential (spec s14).
        pipeline = OCRPipeline(settings, [qwen_like])
        pipeline.execute(single_page_document)
        text = pipeline.log.as_text()
        assert "INV-1024" not in text
        assert "characters" in text

    def test_preview_is_allowed_when_redaction_is_off(self, settings, single_page_document, qwen_like):
        settings.privacy.redact_text_in_logs = False
        settings.privacy.log_preview_chars = 40
        log = EventLog(redact_text=False, preview_chars=40)
        OCRPipeline(settings, [qwen_like], log=log).execute(single_page_document)
        assert "ACME" in log.as_text()

    def test_subscribers_receive_events(self, settings, single_page_document, qwen_like):
        seen = []
        pipeline = OCRPipeline(settings, [qwen_like])
        pipeline.subscribe(seen.append)
        pipeline.execute(single_page_document)
        assert len(seen) > 3

    def test_a_broken_subscriber_does_not_abort_the_run(
        self, settings, single_page_document, qwen_like
    ):
        # A UI callback failing must not lose minutes of completed OCR.
        def explode(event):
            raise RuntimeError("ui blew up")

        pipeline = OCRPipeline(settings, [qwen_like])
        pipeline.subscribe(explode)
        assert pipeline.execute(single_page_document).succeeded

    def test_failures_are_logged_at_error_level(self, settings, single_page_document):
        pipeline = OCRPipeline(settings, [FakeProvider("a", "A", "", fail=True)])
        pipeline.execute(single_page_document)
        assert any(e.level is LogLevel.ERROR for e in pipeline.log.events)

    def test_log_can_be_cleared(self, settings, single_page_document, qwen_like):
        pipeline = OCRPipeline(settings, [qwen_like])
        pipeline.execute(single_page_document)
        pipeline.log.clear()
        assert pipeline.log.as_text() == ""


class TestComposition:
    def test_build_pipeline_uses_enabled_engines(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.unlimited_ocr.enabled = False
        pipeline = build_pipeline(settings)
        assert [p.provider_id for p in pipeline.providers] == ["qwen_vl"]

    def test_build_pipeline_respects_explicit_ids(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        pipeline = build_pipeline(settings, ["unlimited_ocr"])
        assert [p.provider_id for p in pipeline.providers] == ["unlimited_ocr"]

    def test_providers_can_be_added_after_construction(self, settings, single_page_document):
        # Adding an engine requires no pipeline change (spec s11).
        pipeline = OCRPipeline(settings, [])
        pipeline.add_provider(FakeProvider("new_engine", "New Engine", "text"))
        result = pipeline.execute(single_page_document)
        assert result.engine_results[0].provider_id == "new_engine"

    def test_a_third_engine_needs_no_pipeline_change(self, settings, single_page_document):
        engines = [
            FakeProvider("a", "A", "line one\nline two"),
            FakeProvider("b", "B", "line one\nline 2"),
            FakeProvider("c", "C", "line one\nline two"),
        ]
        result = OCRPipeline(settings, engines).execute(single_page_document)
        assert len(result.engine_results) == 3
        assert any(s.label == "C" for s in result.stages)


class TestHealthReport:
    def test_reports_each_engine(self, settings):
        pipeline = OCRPipeline(
            settings,
            [FakeProvider("a", "A", "t"), FakeProvider("b", "B", "t", healthy=False)],
        )
        report = pipeline.health_report()
        assert report["a"]["available"] is True
        assert report["b"]["available"] is False
        assert report["b"]["remedy"] == "Start the engine"
