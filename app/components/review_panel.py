"""What routing decided, and which pages a person should actually look at.

The point of the whole optimisation is not that the machine is faster in the
abstract - it is that a person stops waiting, and then stops checking. Both
halves need saying out loud, so this panel answers two questions:

* **Where did the work go?** Which pages were read by a model, and which were
  answered from the document's own text, skipped as blank, or reused because
  an identical page had already been read. That is the run's cost, explained.

* **What still needs a human?** The pages the confidence check flagged, each
  with the reason and the page image beside it. Ten flagged pages out of a
  hundred is a far better outcome than a hundred pages nobody can trust - but
  only if the ten are easy to find.

Nothing here is a verdict on correctness. A page that is not flagged has not
been proved right; it has failed to look wrong in any of the ways this system
can check for. The panel says that plainly rather than implying a guarantee.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from app.theme import badge, empty_state, notice, section_title
from ocr_fusion.config.schema import AppSettings
from ocr_fusion.pipeline.result import PipelineResult
from ocr_fusion.pipeline.routing import PageClass

#: How each routing decision is described and badged.
_CLASS_LABEL: dict[PageClass, tuple[str, str]] = {
    PageClass.TEXT_LAYER: ("From the document's own text", "ok"),
    PageClass.BLANK: ("Blank", "neutral"),
    PageClass.DUPLICATE: ("Identical to an earlier page", "neutral"),
    PageClass.NEEDS_OCR: ("Read by an engine", "info"),
}


def render(result: PipelineResult, settings: AppSettings) -> None:
    """Draw the routing summary and the review queue."""
    _render_routing(result)
    st.write("")
    _render_review_queue(result, settings)


# -- where the work went ---------------------------------------------------


def _render_routing(result: PipelineResult) -> None:
    plan = result.routing
    if plan is None:
        notice(
            "Routing was switched off for this run, so every page was sent to "
            "every engine.",
            "info",
        )
        return

    total = len(plan.routes)
    via_model = len(plan.ocr_pages)

    section_title("Where the work went")
    share = f"{(total - via_model) / total:.0%}" if total else "n/a"
    st.markdown(
        _tiles(
            [
                ("Pages", str(total)),
                ("Read by an engine", str(via_model)),
                ("Answered for free", str(total - via_model)),
                ("Share avoided", share),
            ]
        ),
        unsafe_allow_html=True,
    )

    if via_model < total:
        st.caption(
            "A page answered from the document's own text is not a shortcut - "
            "that text is what the file actually says, rather than a "
            "transcription of a picture of it."
        )

    with st.expander(f"Every page, and why ({total})", expanded=False):
        for route in plan.routes:
            label, kind = _CLASS_LABEL.get(route.page_class, ("Unknown", "neutral"))
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'align-items:baseline;gap:12px;padding:5px 0;'
                'border-bottom:1px solid var(--rule);">'
                f'<span style="font-size:12.5px;"><b>Page {route.page_number}</b>'
                f'<span style="color:var(--ink-faint);"> - {escape(route.reason)}'
                "</span></span>"
                f"{badge(label, kind)}</div>",
                unsafe_allow_html=True,
            )


def _tiles(entries: list[tuple[str, str]]) -> str:
    """Reuse the metric grid rather than inventing a second tile style."""
    cells = [
        f'<div class="ofs-metric">'
        f'<div class="ofs-metric-label">{escape(label)}</div>'
        f'<div class="ofs-metric-value">{escape(value)}</div></div>'
        for label, value in entries
    ]
    return f'<div class="ofs-metrics">{"".join(cells)}</div>'


# -- what still needs a person ---------------------------------------------


def _render_review_queue(result: PipelineResult, settings: AppSettings) -> None:
    section_title("Pages to check")

    flagged = [score for score in result.confidence if score.needs_second_opinion]
    checked = len(result.confidence)

    if not checked:
        empty_state(
            "○",
            "Nothing was checked",
            "No page went through an engine on this run, so there was nothing "
            "to verify.",
        )
        return

    if not flagged:
        notice(
            f"None of the {checked} transcribed page(s) looked wrong. That is "
            "not a guarantee they are right - it means none of them was empty, "
            "cut off, repetitive, or too thin for the amount of ink on the page.",
            "ok",
        )
        return

    notice(
        f"{len(flagged)} of {checked} transcribed page(s) are worth a look. "
        "The rest can be taken as they are.",
        "warn",
    )

    for score in flagged:
        _render_flagged_page(score, result, settings)


def _render_flagged_page(score, result: PipelineResult, settings: AppSettings) -> None:
    """One flagged page: the reason, the transcription, and the page itself."""
    with st.expander(f"Page {score.page_number} - {score.reason}", expanded=False):
        left, right = st.columns([3, 2])

        with left:
            st.caption("What the engine read")
            text = _page_text(result, score.page_number)
            if text.strip():
                st.text_area(
                    "Transcription",
                    value=text,
                    height=260,
                    label_visibility="collapsed",
                    key=f"ofs_review_text_{score.page_number}",
                )
            else:
                st.caption("Nothing came back for this page.")

        with right:
            st.caption("The page itself")
            _render_page_image(result, score.page_number)

        if settings.general.developer_mode:
            st.caption(
                "Signals: "
                + ", ".join(flag.value for flag in score.flags)
                + f" (score {score.score:.2f})"
            )


def _page_text(result: PipelineResult, page_number: int) -> str:
    """The best available transcription of one page, whichever engine read it."""
    for engine in result.engine_results:
        page = engine.page(page_number)
        if page is not None and page.text.strip():
            return page.text
    return ""


def _render_page_image(result: PipelineResult, page_number: int) -> None:
    """Show the page so a person can check the transcription against it.

    Rendering is on demand, so a page the run never needed as an image is
    drawn here for the first time - which is the right moment, because this is
    the first time anybody wants to look at it.
    """
    try:
        page = result.document.page(page_number)
    except KeyError:
        st.caption("The page image is no longer available.")
        return
    try:
        st.image(page.image_bytes, use_container_width=True)
    except Exception:  # noqa: BLE001 - a preview must not break the review
        st.caption("This page could not be displayed.")


__all__ = ["render"]
