"""Batch state, checkpointing and resuming.

The property under test throughout is the one that matters over a batch
measured in days: work that has been done is never done again, and one bad
document does not cost the other nine thousand.
"""

from __future__ import annotations

import json

import pytest

from ocr_fusion.batch import (
    DEFAULT_MEMORY_FLOOR_MB,
    BatchRunner,
    BatchState,
    DocumentJob,
    JobState,
    collect_documents,
    elapsed_text,
    memory_floor_for,
)
from ocr_fusion.config.schema import AppSettings, OutputFormat, UnlimitedBackend


@pytest.fixture
def checkpoint(tmp_path):
    return tmp_path / "batch.json"


def _pdf(tmp_path, name: str, pages: int = 1, text: str = "Invoice Number: INV-1024") -> object:
    import fitz

    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 100), f"{text} page {index + 1}", fontsize=11)
        # Enough words that the text layer passes the usability gate.
        for line in range(12):
            page.insert_text((72, 130 + line * 14), f"Line {line} total 1,250.00 due 05/18/21", fontsize=9)
    path = tmp_path / name
    path.write_bytes(document.tobytes())
    document.close()
    return path


class TestJobState:
    def test_finished_states_are_left_alone_by_a_resume(self):
        assert JobState.COMPLETED.is_finished
        assert JobState.FAILED.is_finished
        assert JobState.REVIEW_REQUIRED.is_finished
        assert JobState.CANCELLED.is_finished

    def test_unfinished_states_are_picked_up_again(self):
        assert not JobState.QUEUED.is_finished
        assert not JobState.PROCESSING.is_finished
        assert not JobState.RETRYING.is_finished

    def test_a_document_interrupted_mid_flight_is_retried(self):
        # The crash case: the process died while this was PROCESSING, so a
        # resume has to treat it as unfinished rather than assume it worked.
        assert not JobState.PROCESSING.is_finished


class TestCheckpoint:
    def test_state_survives_a_round_trip(self, checkpoint, tmp_path):
        state = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        state.mark(state.jobs[0], JobState.COMPLETED, pages=9, seconds=12.5)

        reloaded = BatchState.load(checkpoint)
        assert reloaded is not None
        assert reloaded.jobs[0].state is JobState.COMPLETED
        assert reloaded.jobs[0].pages == 9

    def test_a_resume_keeps_finished_documents_and_queues_new_ones(
        self, checkpoint, tmp_path
    ):
        first = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        first.mark(first.jobs[0], JobState.COMPLETED, pages=3)

        second = BatchState.for_documents(
            [tmp_path / "a.pdf", tmp_path / "b.pdf"], checkpoint
        )
        assert second.jobs[0].state is JobState.COMPLETED
        assert second.jobs[1].state is JobState.QUEUED
        assert [j.source for j in second.pending] == [str(tmp_path / "b.pdf")]

    def test_a_document_no_longer_present_is_dropped(self, checkpoint, tmp_path):
        first = BatchState.for_documents(
            [tmp_path / "a.pdf", tmp_path / "b.pdf"], checkpoint
        )
        first.save()
        second = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        assert len(second.jobs) == 1

    def test_missing_checkpoint_is_not_an_error(self, checkpoint):
        assert BatchState.load(checkpoint) is None

    def test_a_corrupt_checkpoint_does_not_crash_the_batch(self, checkpoint):
        checkpoint.write_text("{not json", encoding="utf-8")
        assert BatchState.load(checkpoint) is None

    def test_the_checkpoint_holds_no_document_text(self, checkpoint, tmp_path):
        state = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        state.mark(state.jobs[0], JobState.COMPLETED, pages=2)
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert "text" not in json.dumps(payload).lower().replace("context", "")

    def test_the_write_is_atomic(self, checkpoint, tmp_path):
        state = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        state.save()
        # No stray temporary left behind to be mistaken for the checkpoint.
        assert list(checkpoint.parent.glob("*.tmp")) == []


class TestProgress:
    def test_rate_is_none_before_anything_finishes(self, checkpoint, tmp_path):
        state = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        assert state.pages_per_minute is None
        assert state.progress()["pages_per_minute"] is None

    def test_eta_is_none_rather_than_invented(self, checkpoint, tmp_path):
        state = BatchState.for_documents([tmp_path / "a.pdf"], checkpoint)
        assert state.estimated_remaining_seconds() is None

    def test_rate_comes_from_what_was_observed(self, checkpoint, tmp_path):
        state = BatchState.for_documents(
            [tmp_path / "a.pdf", tmp_path / "b.pdf"], checkpoint
        )
        state.mark(state.jobs[0], JobState.COMPLETED, pages=6, seconds=60.0)
        assert state.pages_per_minute == pytest.approx(6.0)
        assert state.estimated_remaining_seconds() == pytest.approx(60.0)

    def test_elapsed_text_says_n_a_for_nothing(self):
        assert elapsed_text(None) == "n/a"
        assert elapsed_text(90) == "1m 30s"
        assert elapsed_text(3700) == "1h 01m"


class TestMemoryFloor:
    def test_a_model_run_reserves_room(self):
        settings = AppSettings()
        assert memory_floor_for(settings) == DEFAULT_MEMORY_FLOOR_MB

    def test_a_text_only_run_reserves_nothing(self):
        # Refusing to read a text layer because a model would not have fitted
        # is a guard protecting against nothing.
        settings = AppSettings()
        settings.qwen.enabled = False
        settings.unlimited_ocr.enabled = False
        assert memory_floor_for(settings) == 0.0

    def test_a_remote_engine_reserves_nothing_locally(self):
        settings = AppSettings()
        settings.qwen.enabled = False
        settings.unlimited_ocr.backend = UnlimitedBackend.HTTP
        assert memory_floor_for(settings) == 0.0


class TestRunner:
    @pytest.fixture
    def settings(self) -> AppSettings:
        settings = AppSettings()
        # Text-layer extraction only: the suite must not need Ollama.
        settings.qwen.enabled = False
        settings.unlimited_ocr.enabled = False
        return settings

    def test_every_document_gets_its_own_result_file(self, settings, tmp_path, checkpoint):
        sources = [_pdf(tmp_path, f"doc{n}.pdf", pages=2) for n in range(3)]
        state = BatchState.for_documents(sources, checkpoint)

        BatchRunner(settings, tmp_path / "out", output_format=OutputFormat.TXT).run(state)

        assert state.counts()["completed"] == 3
        assert len(list((tmp_path / "out").glob("*.txt"))) == 3

    def test_results_are_written_as_each_document_finishes(
        self, settings, tmp_path, checkpoint
    ):
        # A run stopped after nine hours must have nine hours of output, not
        # none - so the file has to exist before the batch ends.
        sources = [_pdf(tmp_path, f"doc{n}.pdf") for n in range(3)]
        state = BatchState.for_documents(sources, checkpoint)
        seen: list[int] = []

        def watch(kind, payload):
            if kind == "document_finished":
                seen.append(len(list((tmp_path / "out").glob("*.txt"))))

        BatchRunner(
            settings, tmp_path / "out", output_format=OutputFormat.TXT, on_event=watch
        ).run(state)
        assert seen == [1, 2, 3]

    def test_a_corrupt_document_does_not_stop_the_batch(
        self, settings, tmp_path, checkpoint
    ):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 truncated nonsense")
        sources = [broken, _pdf(tmp_path, "good.pdf")]
        state = BatchState.for_documents(sources, checkpoint)

        BatchRunner(settings, tmp_path / "out", output_format=OutputFormat.TXT).run(state)

        assert state.jobs[0].state is JobState.FAILED
        assert state.jobs[0].error
        assert state.jobs[1].state is JobState.COMPLETED

    def test_a_corrupt_document_is_not_retried_forever(
        self, settings, tmp_path, checkpoint
    ):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 truncated nonsense")
        state = BatchState.for_documents([broken], checkpoint)

        BatchRunner(
            settings, tmp_path / "out", output_format=OutputFormat.TXT, max_attempts=3
        ).run(state)
        # Unreadable now means unreadable on the third attempt too.
        assert state.jobs[0].attempts == 1

    def test_a_finished_document_is_not_processed_again(
        self, settings, tmp_path, checkpoint
    ):
        sources = [_pdf(tmp_path, "a.pdf"), _pdf(tmp_path, "b.pdf")]
        state = BatchState.for_documents(sources, checkpoint)
        runner = BatchRunner(settings, tmp_path / "out", output_format=OutputFormat.TXT)
        runner.run(state)

        resumed = BatchState.for_documents(sources, checkpoint)
        started: list[str] = []
        BatchRunner(
            settings,
            tmp_path / "out",
            output_format=OutputFormat.TXT,
            on_event=lambda kind, p: started.append(p["source"])
            if kind == "document_started"
            else None,
        ).run(resumed)
        assert started == []

    def test_cancelling_stops_cleanly_and_loses_nothing(
        self, settings, tmp_path, checkpoint
    ):
        sources = [_pdf(tmp_path, f"doc{n}.pdf") for n in range(3)]
        state = BatchState.for_documents(sources, checkpoint)
        calls = {"n": 0}

        def cancel_after_one() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        BatchRunner(
            settings,
            tmp_path / "out",
            output_format=OutputFormat.TXT,
            should_cancel=cancel_after_one,
        ).run(state)

        assert state.jobs[0].state is JobState.COMPLETED
        # The rest stay resumable rather than being marked done or lost.
        assert BatchState.load(checkpoint) is not None

    def test_a_broken_listener_does_not_end_the_batch(
        self, settings, tmp_path, checkpoint
    ):
        state = BatchState.for_documents([_pdf(tmp_path, "a.pdf")], checkpoint)

        def explode(kind, payload):
            raise RuntimeError("listener is broken")

        BatchRunner(
            settings, tmp_path / "out", output_format=OutputFormat.TXT, on_event=explode
        ).run(state)
        assert state.jobs[0].state is JobState.COMPLETED

    def test_documents_with_the_same_name_both_survive(
        self, settings, tmp_path, checkpoint
    ):
        # Two client folders routinely hold a "report.pdf" each, and keeping
        # both results matters more than a tidy filename.
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        sources = [
            _pdf(tmp_path / "one", "report.pdf"),
            _pdf(tmp_path / "two", "report.pdf"),
        ]
        state = BatchState.for_documents(sources, checkpoint)

        BatchRunner(settings, tmp_path / "out", output_format=OutputFormat.TXT).run(state)
        assert len(list((tmp_path / "out").glob("*.txt"))) == 2


class TestCollect:
    def test_finds_supported_documents_only(self, tmp_path):
        _pdf(tmp_path, "a.pdf")
        (tmp_path / "notes.docx").write_bytes(b"not supported")
        (tmp_path / "sub").mkdir()
        _pdf(tmp_path / "sub", "b.pdf")

        found = collect_documents(tmp_path)
        assert [p.name for p in found] == ["a.pdf", "b.pdf"]

    def test_a_single_file_target_works(self, tmp_path):
        path = _pdf(tmp_path, "a.pdf")
        assert collect_documents(path) == [path]
