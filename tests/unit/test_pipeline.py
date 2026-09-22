"""Pipeline execution, stage tracking and failure isolation.

The pipeline has two engine modes and both are covered here. The classes
below that declare ``_all_engines`` describe the all-engines contract - every
enabled engine reads every page - because that is what the behaviour they
assert is about. :class:`TestCascade` covers the default, where each engine
handles only what the engines before it could not.
"""

from __future__ import annotations

import pytest

from ocr_fusion.config.schema import EngineMode
from ocr_fusion.ocr.interface import OCRStatus
from ocr_fusion.pipeline import OCRPipeline, build_pipeline
from ocr_fusion.pipeline.events import EventLog, LogLevel, StageStatus
from ocr_fusion.pipeline.pipeline import (
    STAGE_COMPARISON,
    STAGE_FUSION,
    STAGE_ROUTING,
    STAGE_UPLOAD,
    STAGE_VERIFICATION,
)
from tests.conftest import FakeProvider


@pytest.fixture
def all_engines(settings):
    """Settings that make every enabled engine read every page."""
    settings.pipeline.engine_mode = EngineMode.ALL_ENGINES
    return settings


@pytest.fixture
def scanned_document():
    """Three pages of a scan: no text layer, and no two alike.

    The pages must differ byte for byte or routing will - correctly - call them
    duplicates and read only the first.
    """
    from ocr_fusion.documents.models import Document, DocumentKind, DocumentPage

    return Document(
        filename="scan.pdf",
        kind=DocumentKind.PDF,
        pages=[
            DocumentPage(number=n, image_bytes=f"page-{n}".encode(), width=1240, height=1755)
            for n in (1, 2, 3)
        ],
    )


class TestHappyPath:
    @pytest.fixture(autouse=True)
    def _all_engines(self, settings):
        settings.pipeline.engine_mode = EngineMode.ALL_ENGINES

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
    @pytest.fixture(autouse=True)
    def _all_engines(self, settings):
        settings.pipeline.engine_mode = EngineMode.ALL_ENGINES

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
    @pytest.fixture(autouse=True)
    def _all_engines(self, settings):
        settings.pipeline.engine_mode = EngineMode.ALL_ENGINES

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
    @pytest.fixture(autouse=True)
    def _all_engines(self, settings):
        settings.pipeline.engine_mode = EngineMode.ALL_ENGINES

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
    @pytest.fixture(autouse=True)
    def _all_engines(self, settings):
        settings.pipeline.engine_mode = EngineMode.ALL_ENGINES

    def test_build_pipeline_uses_enabled_engines(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.unlimited_ocr.enabled = False
        pipeline = build_pipeline(settings)
        # Text-layer extraction leads: it answers most pages for nothing, so
        # the models only ever see what it could not.
        assert [p.provider_id for p in pipeline.providers] == ["text_layer", "qwen_vl"]

    def test_text_layer_extraction_can_be_switched_off(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.text_layer.enabled = False
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


class TestCascade:
    """The default mode: each engine handles only what the earlier ones could not.

    This is where the run time went. Every assertion below is an assertion
    about a model call that does not happen.
    """

    def test_a_clean_primary_result_costs_no_second_engine(
        self, settings, single_page_document, qwen_like, unlimited_like
    ):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(
            single_page_document
        )
        assert qwen_like.process_calls == 1
        assert unlimited_like.process_calls == 0
        assert result.succeeded

    def test_the_skipped_engine_says_why(
        self, settings, single_page_document, qwen_like, unlimited_like
    ):
        result = OCRPipeline(settings, [qwen_like, unlimited_like]).execute(
            single_page_document
        )
        stage = next(s for s in result.stages if s.key == "unlimited_ocr")
        assert stage.status is StageStatus.SKIPPED
        assert "looks complete" in stage.detail

    def test_a_flagged_page_does_reach_the_second_engine(
        self, settings, scanned_document, unlimited_like
    ):
        # An engine that returns nothing for page 2 is the clearest possible
        # signal that page 2 needs another look.
        primary = FakeProvider("qwen_vl", "Qwen2.5-VL", "Invoice INV-1024", failing_pages=(2,))
        OCRPipeline(settings, [primary, unlimited_like]).execute(scanned_document)
        assert unlimited_like.process_calls == 1
        # Only the flagged page, not the whole document.
        assert unlimited_like.pages_seen == [2]

    def test_verification_reports_what_it_flagged(
        self, settings, scanned_document, unlimited_like
    ):
        primary = FakeProvider("qwen_vl", "Qwen2.5-VL", "Invoice INV-1024", failing_pages=(2,))
        result = OCRPipeline(settings, [primary, unlimited_like]).execute(scanned_document)
        stage = next(s for s in result.stages if s.key == STAGE_VERIFICATION)
        assert stage.status is StageStatus.COMPLETED
        assert [c.page_number for c in result.confidence if c.needs_second_opinion] == [2]

    def test_fallback_can_be_switched_off(
        self, settings, scanned_document, unlimited_like
    ):
        settings.pipeline.fallback_enabled = False
        primary = FakeProvider("qwen_vl", "Qwen2.5-VL", "Invoice INV-1024", failing_pages=(2,))
        OCRPipeline(settings, [primary, unlimited_like]).execute(scanned_document)
        assert unlimited_like.process_calls == 0

    def test_a_document_the_primary_engine_failed_throughout_is_capped(
        self, settings, scanned_document, unlimited_like
    ):
        # Paying twice for every page would hide a configuration problem.
        primary = FakeProvider("qwen_vl", "Qwen2.5-VL", "", failing_pages=(1, 2, 3))
        result = OCRPipeline(settings, [primary, unlimited_like]).execute(scanned_document)
        assert unlimited_like.pages_seen == [1]
        assert any("more than the fallback is allowed" in e.message for e in result.log.events)

    def test_routing_stage_is_recorded(self, settings, single_page_document, qwen_like):
        result = OCRPipeline(settings, [qwen_like]).execute(single_page_document)
        stage = next(s for s in result.stages if s.key == STAGE_ROUTING)
        assert stage.status is StageStatus.COMPLETED
        assert result.routing is not None

    def test_routing_can_be_switched_off_entirely(
        self, settings, single_page_document, qwen_like, unlimited_like
    ):
        settings.routing.enabled = False
        OCRPipeline(settings, [qwen_like, unlimited_like]).execute(single_page_document)
        assert qwen_like.process_calls == 1
        assert unlimited_like.process_calls == 1

    def test_pages_no_engine_can_read_are_reported_not_dropped(
        self, settings, scanned_document
    ):
        # A result quietly missing pages is worse than a slow one.
        result = OCRPipeline(settings, []).execute(scanned_document)
        assert any(
            "need an OCR engine and none is enabled" in event.message
            for event in result.log.events
        )

    def test_comparison_finds_the_two_engines_that_share_pages(
        self, settings, scanned_document, unlimited_like
    ):
        # text_layer covers nothing here, the primary covers 1-3 and the
        # fallback only page 2. Comparing the first two engines in order would
        # find no overlap and skip - losing the one real disagreement.
        primary = FakeProvider("qwen_vl", "Qwen2.5-VL", "Invoice INV-1024", failing_pages=(2,))
        empty = FakeProvider("text_layer", "PDF text layer", "", fail=True)
        result = OCRPipeline(settings, [empty, primary, unlimited_like]).execute(
            scanned_document
        )
        stage = next(s for s in result.stages if s.key == STAGE_COMPARISON)
        assert stage.status is StageStatus.SKIPPED
        # Page 2 is the only page two engines both produced text for, and the
        # primary produced nothing for it - so there is still nothing to
        # compare, and the run says so rather than inventing an agreement.
        assert "nothing to compare" in stage.detail or "fewer than two" in stage.detail

    def test_comparison_uses_the_overlapping_pages_when_a_fallback_ran(
        self, settings, scanned_document
    ):
        primary = FakeProvider("qwen_vl", "Qwen2.5-VL", "Total 38,085.10")
        # A fallback that read every page, so pages genuinely overlap.
        second = FakeProvider("unlimited_ocr", "Unlimited-OCR", "Total 38,085.60")
        settings.pipeline.engine_mode = EngineMode.ALL_ENGINES
        result = OCRPipeline(settings, [primary, second]).execute(scanned_document)
        assert result.comparison is not None
        assert result.comparison.numeric_conflicts


class TestRunProgress:
    """A nine-page run takes tens of minutes; a spinner is not enough."""

    def test_no_estimate_before_a_page_has_finished(self):
        from ocr_fusion.pipeline.pipeline import RunProgress

        progress = RunProgress(pages_total=9)
        assert progress.estimated_remaining_seconds is None
        assert "before estimating" in progress.describe()

    def test_the_estimate_comes_from_this_document(self):
        # Page cost varies tenfold with how much text is on the page, so a
        # constant would be worse than useless.
        from ocr_fusion.pipeline.pipeline import RunProgress

        progress = RunProgress(pages_total=9, pages_done=1, durations=[400.0])
        assert progress.estimated_remaining_seconds == pytest.approx(3200.0)
        assert "53 min" in progress.describe()

    def test_nothing_is_shown_when_no_page_needs_an_engine(self):
        from ocr_fusion.pipeline.pipeline import RunProgress

        assert RunProgress().describe() == ""

    def test_the_run_reports_progress_as_it_goes(self, settings, scanned_document, qwen_like):
        pipeline = OCRPipeline(settings, [qwen_like])
        pipeline.execute(scanned_document)
        assert pipeline.progress.pages_total == 3
        # Two of the three pages were timed - the last has no successor to
        # report it, which is honest rather than rounded up.
        assert len(pipeline.progress.durations) == 2


class TestEngineOrder:
    """Which engine leads is a configuration choice, not a code change."""

    def test_the_registry_order_is_the_default(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        pipeline = build_pipeline(settings)
        assert [p.provider_id for p in pipeline.providers] == [
            "text_layer",
            "qwen_vl",
            "unlimited_ocr",
        ]

    def test_an_explicit_order_chooses_the_primary_engine(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.pipeline.engine_order = ["text_layer", "unlimited_ocr", "qwen_vl"]
        pipeline = build_pipeline(settings)
        assert [p.provider_id for p in pipeline.providers] == [
            "text_layer",
            "unlimited_ocr",
            "qwen_vl",
        ]

    def test_an_engine_left_out_of_the_order_still_runs(self, settings):
        # A partial order must not silently drop an enabled engine.
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.pipeline.engine_order = ["unlimited_ocr"]
        pipeline = build_pipeline(settings)
        ids = [p.provider_id for p in pipeline.providers]
        assert ids[0] == "unlimited_ocr"
        assert set(ids) == {"text_layer", "qwen_vl", "unlimited_ocr"}

    def test_a_disabled_engine_named_in_the_order_is_ignored(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.unlimited_ocr.enabled = False
        settings.pipeline.engine_order = ["unlimited_ocr", "qwen_vl"]
        pipeline = build_pipeline(settings)
        assert "unlimited_ocr" not in [p.provider_id for p in pipeline.providers]
