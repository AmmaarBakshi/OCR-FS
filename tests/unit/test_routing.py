"""Page routing, the text-layer gate and confidence scoring.

These are the tests that keep the optimisation honest. Routing exists to stop
pages reaching a model, and every rule here is a rule about spending eight
minutes of CPU or not spending it - so the cases that matter most are the ones
where the cheap answer must be refused.
"""

from __future__ import annotations

import io

import pytest

from ocr_fusion.config.schema import AppSettings, ConfidenceSettings, RoutingSettings
from ocr_fusion.documents.models import Document, DocumentKind, DocumentPage
from ocr_fusion.documents.textlayer import assess_text_layer, clean_text_layer
from ocr_fusion.ocr.interface import OCRStatus, PageResult
from ocr_fusion.pipeline.confidence import ConfidenceFlag, score_page
from ocr_fusion.pipeline.routing import (
    PageClass,
    measure_ink,
    page_fingerprint,
    route_document,
    subset,
)

REAL_TEXT = (
    "Bank of America, N.A. Customer service 1.800.432.1000 "
    "Your combined statement for May 01, 2021 to May 28, 2021 "
    "Deposits and other additions Date Description Amount "
    "05/18/21 Online scheduled advance from LOC 2000 2,200.00"
)


def _page(number: int, *, text: str | None = None, image: bytes | None = None) -> DocumentPage:
    return DocumentPage(
        number=number,
        image_bytes=image if image is not None else b"not-an-image",
        width=1275,
        height=1650,
        embedded_text=text,
    )


def _image(fill: str = "white", marks: int = 0) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 520), fill)
    draw = ImageDraw.Draw(image)
    for row in range(marks):
        draw.rectangle([10, 10 + row * 12, 390, 18 + row * 12], fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _document(pages: list[DocumentPage]) -> Document:
    return Document(filename="statement.pdf", kind=DocumentKind.PDF, pages=pages)


class TestTextLayerGate:
    def test_a_real_layer_is_usable(self):
        assert assess_text_layer(REAL_TEXT).usable

    def test_a_page_number_stamped_on_a_scan_is_not_a_text_layer(self):
        result = assess_text_layer("Page 4 of 10")
        assert not result.usable
        assert "too few" in result.reason

    def test_a_broken_font_encoding_is_refused(self):
        # A PDF whose glyph map is missing extracts as replacement characters,
        # which is worse than no layer at all: it looks like content.
        result = assess_text_layer("�" * 400 + " " + "word " * 40)
        assert not result.usable
        assert "encoding" in result.reason

    def test_a_scrambled_cid_map_is_refused(self):
        result = assess_text_layer("bcdfg hjklm nprst wxzz " * 12)
        assert not result.usable
        assert result.quality < 0.7

    def test_figures_count_as_language(self):
        # A financial page is mostly numbers; refusing it would send every
        # statement to a model.
        assert assess_text_layer("1,250.00 15,000.00 05/18/21 " * 12).usable

    def test_cleaning_does_not_change_the_words(self):
        cleaned = clean_text_layer("Invoice    No:  INV-1024\r\n\n\n\nTotal   38,085.10")
        assert "INV-1024" in cleaned and "38,085.10" in cleaned
        assert "\r" not in cleaned
        assert "\n\n\n" not in cleaned


class TestRouting:
    @pytest.fixture
    def routing(self) -> RoutingSettings:
        return RoutingSettings()

    def test_a_page_with_a_good_layer_never_reaches_a_model(self, routing):
        plan = route_document(_document([_page(1, text=REAL_TEXT)]), routing)
        assert plan.ocr_pages == []
        assert plan.routes[0].page_class is PageClass.TEXT_LAYER

    def test_a_text_layer_page_is_not_even_rendered(self, routing):
        # The saving is not just the model: rendering costs ~149ms a page and
        # a page answered from its text needs no pixels at all.
        rendered: list[int] = []

        page = DocumentPage(
            number=1,
            width=1275,
            height=1650,
            embedded_text=REAL_TEXT,
            render=lambda: (rendered.append(1), _image())[1],
        )
        route_document(_document([page]), routing)
        assert rendered == []
        assert not page.is_rendered

    def test_a_scan_goes_to_ocr(self, routing):
        plan = route_document(_document([_page(1, image=_image(marks=30))]), routing)
        assert plan.ocr_pages == [1]

    def test_a_blank_page_is_skipped(self, routing):
        plan = route_document(_document([_page(1, image=_image())]), routing)
        assert plan.routes[0].page_class is PageClass.BLANK
        assert plan.ocr_pages == []

    def test_a_page_carrying_only_a_signature_is_still_read(self, routing):
        # Sparse is not blank. The threshold has to sit below what a few
        # strokes of ink produce or signed pages vanish silently.
        plan = route_document(_document([_page(1, image=_image(marks=2))]), routing)
        assert plan.routes[0].page_class is PageClass.NEEDS_OCR

    def test_an_identical_page_is_read_once(self, routing):
        scan = _image(marks=30)
        plan = route_document(
            _document([_page(1, image=scan), _page(2, image=scan)]), routing
        )
        assert plan.ocr_pages == [1]
        assert plan.routes[1].page_class is PageClass.DUPLICATE
        assert plan.routes[1].duplicate_of == 1

    def test_pages_that_differ_by_one_pixel_are_both_read(self, routing):
        # Two statement pages can look alike and differ only in the figures.
        # Nothing here may be allowed to merge them.
        plan = route_document(
            _document([_page(1, image=_image(marks=30)), _page(2, image=_image(marks=31))]),
            routing,
        )
        assert plan.ocr_pages == [1, 2]

    def test_an_undecodable_page_is_read_rather_than_assumed_blank(self, routing):
        plan = route_document(_document([_page(1, image=b"not-an-image")]), routing)
        assert plan.routes[0].page_class is PageClass.NEEDS_OCR
        assert plan.routes[0].ink_ratio is None

    def test_routing_can_be_switched_off_per_rule(self):
        routing = RoutingSettings(use_text_layer=False)
        plan = route_document(_document([_page(1, text=REAL_TEXT)]), routing)
        assert plan.ocr_pages == [1]

    def test_measure_ink_reports_none_for_rubbish(self):
        assert measure_ink(_page(1, image=b"not-an-image")) is None

    def test_fingerprint_is_content_addressed(self):
        scan = _image(marks=8)
        assert page_fingerprint(_page(1, image=scan)) == page_fingerprint(_page(9, image=scan))


class TestSubset:
    def test_keeps_page_numbers_and_shares_pages(self):
        pages = [_page(n) for n in (1, 2, 3)]
        view = subset(_document(pages), [1, 3])
        assert [p.number for p in view.pages] == [1, 3]
        # Shared, not copied: the image bytes must not be duplicated in memory.
        assert view.pages[0] is pages[0]


class TestConfidence:
    @pytest.fixture
    def conf(self) -> ConfidenceSettings:
        return ConfidenceSettings()

    def test_a_clean_page_is_not_flagged(self, conf):
        score = score_page(PageResult(page_number=1, text=REAL_TEXT), conf)
        assert not score.needs_second_opinion
        assert score.score == 1.0

    def test_an_empty_page_is_flagged(self, conf):
        score = score_page(
            PageResult(page_number=1, text="", status=OCRStatus.FAILED), conf
        )
        assert ConfidenceFlag.EMPTY in score.flags

    def test_a_truncated_page_is_flagged(self, conf):
        score = score_page(
            PageResult(
                page_number=1, text=REAL_TEXT, raw_response={"done_reason": "length"}
            ),
            conf,
        )
        assert ConfidenceFlag.TRUNCATED in score.flags

    def test_a_looping_engine_is_flagged(self, conf):
        score = score_page(
            PageResult(page_number=1, text="\n".join(["Total due 38,085.10"] * 40)), conf
        )
        assert ConfidenceFlag.REPETITION in score.flags

    def test_a_repeated_column_of_amounts_is_not_a_loop(self, conf):
        # A real table legitimately repeats a value a few times.
        score = score_page(
            PageResult(page_number=1, text="\n".join(["1,250.00"] * 4 + [REAL_TEXT])), conf
        )
        assert ConfidenceFlag.REPETITION not in score.flags

    def test_a_dense_page_with_two_lines_out_is_flagged(self, conf):
        from ocr_fusion.pipeline.routing import PageRoute

        route = PageRoute(
            page_number=1,
            page_class=PageClass.NEEDS_OCR,
            reason="scan",
            fingerprint="x",
            ink_ratio=0.35,
        )
        score = score_page(PageResult(page_number=1, text="Invoice"), conf, route)
        assert ConfidenceFlag.SPARSE_FOR_INK in score.flags

    def test_symbol_soup_is_flagged(self, conf):
        score = score_page(PageResult(page_number=1, text="@@@ ### $$$ %%% ^^^ &&&"), conf)
        assert ConfidenceFlag.NOT_LANGUAGE in score.flags

    def test_every_flag_explains_itself(self, conf):
        score = score_page(PageResult(page_number=1, text=""), conf)
        assert score.reason and not score.reason.endswith(".")


class TestSettingsCompatibility:
    def test_routing_defaults_are_on(self):
        settings = AppSettings()
        assert settings.routing.enabled
        assert settings.routing.use_text_layer
        assert settings.pipeline.fallback_enabled
