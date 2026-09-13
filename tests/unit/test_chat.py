"""Asking questions about a transcribed document."""

from __future__ import annotations

import pytest

from ocr_fusion.chat import ChatAnswer, ChatTurn, DocumentChat, format_history, trim_to_budget
from ocr_fusion.ocr.ollama_client import (
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from tests.unit.test_providers import StubOllama

INVOICE = """ACME LOGISTICS LTD
Invoice No: INV-2024-0042
Date: 11 March 2024
Subtotal 10,900.00
IGST 18% 1,962.00
Total 12,862.00"""


@pytest.fixture
def chat(settings):
    settings.chat.model = "qwen2.5:1.5b"
    return DocumentChat(settings, StubOllama(models=["qwen2.5:1.5b"]))


class TestTrimming:
    def test_short_text_is_untouched(self):
        text, trimmed = trim_to_budget(INVOICE, 10_000)
        assert text == INVOICE
        assert trimmed is False

    def test_both_ends_survive_a_trim(self):
        """The head identifies the document and the tail totals it."""
        body = "filler line\n" * 5000
        text, trimmed = trim_to_budget(f"ACME LOGISTICS LTD\n{body}\nTotal 12,862.00", 2000)
        assert trimmed is True
        assert "ACME LOGISTICS LTD" in text
        assert "Total 12,862.00" in text

    def test_the_trim_is_announced_inside_the_text(self):
        """The model must know it is reading an excerpt, not the whole thing."""
        text, _ = trim_to_budget("x" * 9000, 1000)
        assert "omitted" in text

    def test_kept_document_text_respects_the_budget(self):
        budget = 1000
        text, _ = trim_to_budget("x" * 9000, budget)
        assert text.count("x") == budget


class TestHistory:
    def test_empty_history_contributes_nothing(self):
        assert format_history([], 6) == ""

    def test_only_the_most_recent_turns_are_sent(self):
        turns = [ChatTurn(f"question {n}", f"answer {n}") for n in range(10)]
        rendered = format_history(turns, 2)
        assert "question 9" in rendered
        assert "question 7" not in rendered

    def test_a_zero_limit_sends_no_history(self):
        assert format_history([ChatTurn("q", "a")], 0) == ""


class TestAsking:
    def test_answers_from_the_transcription(self, chat):
        answer = chat.ask("What is the total?", document_text=INVOICE)
        assert answer.succeeded
        assert answer.text
        assert chat.client.calls[0]["prompt"].count("ACME LOGISTICS LTD") == 1

    def test_the_question_and_document_both_reach_the_model(self, chat):
        chat.ask("What is the invoice number?", document_text=INVOICE, filename="inv.pdf")
        prompt = chat.client.calls[0]["prompt"]
        assert "What is the invoice number?" in prompt
        assert "INV-2024-0042" in prompt
        assert "inv.pdf" in prompt

    def test_sampling_is_deterministic(self, chat):
        """A question about what a document says has one right answer."""
        chat.ask("What is the total?", document_text=INVOICE)
        assert chat.client.calls[0]["temperature"] == 0.0

    def test_history_reaches_the_model(self, chat):
        chat.ask(
            "And the date?",
            document_text=INVOICE,
            history=[ChatTurn("What is the total?", "12,862.00")],
        )
        assert "12,862.00" in chat.client.calls[0]["prompt"]

    def test_reports_token_usage(self, chat):
        answer = chat.ask("What is the total?", document_text=INVOICE)
        assert answer.input_tokens == 120
        assert answer.output_tokens == 30

    def test_a_trim_is_reported_to_the_caller(self, settings):
        settings.chat.max_context_characters = 100
        chat = DocumentChat(settings, StubOllama(models=[settings.chat.model]))
        answer = chat.ask("What is the total?", document_text="x" * 5000)
        assert answer.context_was_trimmed is True


class TestExpectedFailuresAreReturned:
    def test_an_empty_question_is_refused_without_a_round_trip(self, chat):
        answer = chat.ask("   ", document_text=INVOICE)
        assert not answer.succeeded
        assert chat.client.calls == []

    def test_asking_before_a_run_is_explained(self, chat):
        answer = chat.ask("What is the total?", document_text="")
        assert not answer.succeeded
        assert "transcription" in answer.error
        assert answer.remedy
        assert chat.client.calls == []

    def test_a_disabled_chat_never_calls_the_model(self, settings):
        settings.chat.enabled = False
        client = StubOllama(models=[settings.chat.model])
        answer = DocumentChat(settings, client).ask("What?", document_text=INVOICE)
        assert not answer.succeeded
        assert client.calls == []

    @pytest.mark.parametrize(
        "error",
        [
            OllamaUnavailableError("Could not reach Ollama.", remedy="Start Ollama."),
            OllamaModelMissingError("Model missing.", remedy="ollama pull x"),
            OllamaTimeoutError("Too slow.", remedy="Raise the timeout."),
        ],
    )
    def test_a_runtime_failure_is_returned_not_raised(self, settings, error):
        """An unreachable runtime is an expected condition, not a bug."""
        chat = DocumentChat(settings, StubOllama(models=[settings.chat.model], error=error))
        answer = chat.ask("What is the total?", document_text=INVOICE)
        assert not answer.succeeded
        assert answer.error == error.message
        assert answer.remedy == error.remedy

    def test_a_failed_answer_carries_no_fabricated_metrics(self, settings):
        """Unknown is None, never 0."""
        chat = DocumentChat(
            settings,
            StubOllama(models=[settings.chat.model], error=OllamaTimeoutError("slow")),
        )
        answer = chat.ask("What is the total?", document_text=INVOICE)
        assert answer.input_tokens is None
        assert answer.output_tokens is None


class TestHealth:
    def test_ready_when_the_model_is_pulled(self, settings):
        chat = DocumentChat(settings, StubOllama(models=[settings.chat.model]))
        assert chat.health_check().available

    def test_a_missing_model_names_the_pull_command(self, settings):
        chat = DocumentChat(settings, StubOllama(models=["something-else"]))
        status = chat.health_check()
        assert not status.available
        assert f"ollama pull {settings.chat.model}" in status.remedy

    def test_an_unreachable_runtime_is_reported_with_its_remedy(self, settings):
        error = OllamaUnavailableError("Could not reach Ollama.", remedy="Start Ollama.")
        chat = DocumentChat(settings, StubOllama(error=error))
        status = chat.health_check()
        assert not status.available
        assert status.remedy == "Start Ollama."

    def test_switched_off_is_reported_as_such(self, settings):
        settings.chat.enabled = False
        status = DocumentChat(settings, StubOllama()).health_check()
        assert not status.available
        assert "switched off" in status.message


class TestChatAnswer:
    def test_a_failure_is_not_a_success(self):
        assert not ChatAnswer.failure("broken").succeeded

    def test_an_answer_with_text_is_a_success(self):
        assert ChatAnswer(text="12,862.00").succeeded

    def test_an_empty_answer_is_still_a_success(self):
        """A model that found nothing to say has not failed."""
        assert ChatAnswer(text="").succeeded
