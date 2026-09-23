"""Export a :class:`~ocr_fusion.pipeline.result.PipelineResult`.

Four formats (spec s10), each honouring the Output settings so the user
controls what appears in the file, not just on screen:

* **TXT** - the transcription, optionally with a metadata header
* **Markdown** - a readable report with per-engine sections
* **JSON** - the full stable schema for other projects to consume
* **CSV** - one row per page per engine, for spreadsheets

Exporters return ``str`` rather than writing files, so the same function serves
a Streamlit download button, a CLI redirect and a unit test.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable
from datetime import datetime
from html import escape
from typing import Any
from xml.etree import ElementTree as ET

from ocr_fusion.config.schema import AppSettings, OutputFormat
from ocr_fusion.metrics import NOT_AVAILABLE
from ocr_fusion.pipeline.result import PipelineResult


def _metric(value: Any, suffix: str = "") -> str:
    """Render a metric, showing N/A for anything the engine did not report.

    The rule from spec s8: never fabricate a metric, never print 0 for unknown.
    """
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def export_txt(result: PipelineResult, settings: AppSettings) -> str:
    """Plain text: the final transcription, with an optional header."""
    output = settings.output
    lines: list[str] = []

    header: list[str] = []
    if output.show_engine_name:
        engines = ", ".join(r.provider_name for r in result.successful_engines)
        header.append(f"Engines: {engines or 'none'}")
    if output.show_model_name:
        models = ", ".join(
            r.model_name for r in result.successful_engines if r.model_name
        )
        if models:
            header.append(f"Models: {models}")
    if output.show_processing_time:
        header.append(f"Processing time: {result.total_duration_seconds:.2f}s")
    if output.show_character_count:
        header.append(f"Characters: {len(result.final_text)}")
    if output.show_word_count:
        header.append(f"Words: {len(result.final_text.split())}")

    if header:
        lines.append(f"{result.document.filename}")
        lines.extend(header)
        lines.append("-" * 60)
        lines.append("")

    if output.show_page_numbers and result.document.page_count > 1:
        lines.append(_paginated_text(result))
    else:
        lines.append(result.final_text)

    return "\n".join(lines).strip() + "\n"


def _paginated_text(result: PipelineResult) -> str:
    """Final text split by page, when per-page output is available.

    Fusion merges the whole document, so page boundaries survive only in the
    engine results; the best available source is used rather than inventing a
    split of the fused text.
    """
    engine = next(
        (r for r in result.successful_engines if len(r.pages) > 1),
        None,
    )
    if engine is None:
        return result.final_text

    blocks: list[str] = []
    for page in sorted(engine.pages, key=lambda p: p.page_number):
        blocks.append(f"--- Page {page.page_number} ---")
        blocks.append(page.text or "(no text extracted)")
        blocks.append("")
    return "\n".join(blocks)


def export_markdown(result: PipelineResult, settings: AppSettings) -> str:
    """A readable report: final result, then per-engine detail and metrics."""
    output = settings.output
    document = result.document
    parts: list[str] = [f"# OCR Result - {document.filename}", ""]

    meta: list[str] = [
        f"- **Pages:** {document.page_count}",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if output.show_processing_time:
        meta.append(f"- **Total time:** {result.total_duration_seconds:.2f}s")
    if output.show_engine_name:
        names = ", ".join(r.provider_name for r in result.engine_results) or "none"
        meta.append(f"- **Engines:** {names}")
    if result.fusion:
        meta.append(f"- **Fusion strategy:** {result.fusion.strategy}")
    parts.extend(meta)
    parts.append("")

    if output.show_extracted_text:
        parts.extend(["## Final result", "", result.final_text or "_No text extracted._", ""])

    if output.show_comparison and result.comparison is not None:
        comparison = result.comparison
        parts.extend(
            [
                "## Comparison",
                "",
                f"- **Agreement:** {comparison.agreement_percent}%",
                f"- **Lines:** {comparison.total_lines} "
                f"({comparison.equal_lines} identical, {comparison.differing_lines} differing)",
                "",
            ]
        )
        if comparison.numeric_conflicts:
            parts.extend(
                [
                    f"### Numeric disagreements ({len(comparison.numeric_conflicts)})",
                    "",
                    f"| {comparison.engine_a} | {comparison.engine_b} |",
                    "| --- | --- |",
                ]
            )
            for conflict in comparison.numeric_conflicts[:25]:
                parts.append(
                    f"| {_cell(conflict['text_a'])} | {_cell(conflict['text_b'])} |"
                )
            parts.append("")

    if output.show_raw_results:
        parts.extend(["## Engine outputs", ""])
        for engine in result.engine_results:
            parts.append(f"### {engine.provider_name}")
            parts.append("")
            parts.extend(_engine_metric_lines(engine, settings))
            parts.append("")
            if engine.succeeded and engine.text:
                parts.extend([engine.text, ""])
            else:
                parts.extend([f"_Failed: {engine.error or 'unknown error'}_", ""])

    if output.show_processing_logs and result.log is not None:
        parts.extend(["## Processing log", "", "```", result.log.as_text(), "```", ""])

    return "\n".join(parts)


def _cell(text: str) -> str:
    """Escape a value for a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _engine_metric_lines(engine, settings: AppSettings) -> list[str]:
    output = settings.output
    lines: list[str] = [f"- **Status:** {engine.status.value}"]
    if output.show_model_name and engine.model_name:
        lines.append(f"- **Model:** {engine.model_name}")
    if engine.backend:
        lines.append(f"- **Backend:** {engine.backend}")
    if output.show_processing_time:
        lines.append(f"- **Time:** {engine.duration_seconds:.2f}s")
    if output.show_token_usage:
        usage = engine.tokens
        lines.append(
            f"- **Tokens:** in {_metric(usage.input_tokens)}, "
            f"out {_metric(usage.output_tokens)}, "
            f"total {_metric(usage.total_tokens)}"
        )
        lines.append(f"- **Tokens/second:** {_metric(engine.tokens_per_second)}")
    if output.show_character_count:
        lines.append(f"- **Characters:** {engine.character_count}")
    if output.show_word_count:
        lines.append(f"- **Words:** {engine.word_count}")
    return lines


def _structured_payload(result: PipelineResult, settings: AppSettings) -> dict[str, Any]:
    """The stable export schema, trimmed by the Output settings.

    Shared by the JSON and XML exporters so the two can never describe the same
    run differently - XML is a rendering of this dict, not a second schema.
    """
    payload = result.as_dict(include_logs=settings.output.show_processing_logs)
    if not settings.output.show_raw_results:
        for engine in payload["engines"]:
            engine.pop("text", None)
            engine.pop("pages", None)
    if not settings.output.show_comparison:
        payload["comparison"] = None
    return payload


def export_json(result: PipelineResult, settings: AppSettings) -> str:
    """The full stable schema, trimmed by the Output settings."""
    return json.dumps(
        _structured_payload(result, settings), indent=2, ensure_ascii=False
    )


def export_csv(result: PipelineResult, settings: AppSettings) -> str:
    """One row per page per engine, for spreadsheet analysis."""
    output = settings.output
    buffer = io.StringIO()

    columns = ["engine"]
    if output.show_model_name:
        columns.append("model")
    if output.show_page_numbers:
        columns.append("page")
    columns.append("status")
    if output.show_processing_time:
        columns.append("duration_seconds")
    if output.show_character_count:
        columns.append("characters")
    if output.show_word_count:
        columns.append("words")
    if output.show_token_usage:
        columns.extend(["input_tokens", "output_tokens"])
    if output.show_confidence:
        columns.append("confidence")
    if output.show_extracted_text:
        columns.append("text")
    columns.append("error")

    # QUOTE_ALL keeps multi-line OCR text inside one cell in every spreadsheet
    # application, and \r\n is what RFC 4180 and Excel expect.
    writer = csv.DictWriter(
        buffer, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
    )
    writer.writeheader()

    for engine in result.engine_results:
        pages = engine.pages or [None]
        for page in pages:
            row: dict[str, Any] = {"engine": engine.provider_name}
            if output.show_model_name:
                row["model"] = engine.model_name or ""
            if output.show_page_numbers:
                row["page"] = page.page_number if page else ""
            row["status"] = page.status.value if page else engine.status.value
            if output.show_processing_time:
                row["duration_seconds"] = (
                    f"{page.duration_seconds:.3f}"
                    if page and page.duration_seconds is not None
                    else ""
                )
            # A row with no page is an engine that failed before it read
            # anything. It did not measure zero characters - it measured
            # nothing, and the counts have to say so (spec s8).
            if output.show_character_count:
                row["characters"] = page.character_count if page else NOT_AVAILABLE
            if output.show_word_count:
                row["words"] = page.word_count if page else NOT_AVAILABLE
            if output.show_token_usage:
                row["input_tokens"] = (
                    page.tokens.input_tokens if page and page.tokens.input_tokens is not None else "N/A"
                )
                row["output_tokens"] = (
                    page.tokens.output_tokens if page and page.tokens.output_tokens is not None else "N/A"
                )
            if output.show_confidence:
                row["confidence"] = (
                    page.confidence if page and page.confidence is not None else "N/A"
                )
            if output.show_extracted_text:
                row["text"] = page.text if page else ""
            row["error"] = (page.error if page else engine.error) or ""
            writer.writerow(row)

    # The fused result is a document-level row, so it has no page number.
    if result.fusion and output.show_extracted_text:
        row = {column: "" for column in columns}
        row["engine"] = f"Final ({result.fusion.strategy})"
        row["status"] = "success"
        if output.show_character_count:
            row["characters"] = result.fusion.character_count
        if output.show_word_count:
            row["words"] = result.fusion.word_count
        row["text"] = result.fusion.text
        writer.writerow(row)

    return buffer.getvalue()


#: Element name used for a list entry whose parent key has no obvious singular.
_LIST_ITEM = "item"


def _is_xml_char(code: int) -> bool:
    """Whether a code point is representable in XML 1.0."""
    return (
        code in (0x09, 0x0A, 0x0D)
        or 0x20 <= code <= 0xD7FF
        or 0xE000 <= code <= 0xFFFD
        or 0x10000 <= code <= 0x10FFFF
    )


def _xml_safe(text: str) -> str:
    """Drop characters XML 1.0 cannot represent.

    OCR of a noisy scan can emit stray control bytes, and a single one of them
    would leave the consumer with a file that no parser will open.
    """
    return "".join(char for char in text if _is_xml_char(ord(char)))


def _xml_tag(key: str) -> str:
    """Turn a payload key into a legal element name."""
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return tag if tag[:1].isalpha() or tag[:1] == "_" else f"_{tag}"


def _singular(key: str) -> str:
    """The child element name for entries of a list called ``key``."""
    if key.endswith("ies") and len(key) > 3:
        return f"{key[:-3]}y"
    if key.endswith("s") and not key.endswith("ss") and len(key) > 1:
        return key[:-1]
    return _LIST_ITEM


def _xml_value(parent: ET.Element, key: str, value: Any) -> None:
    """Append ``value`` to ``parent`` as one or more child elements.

    ``None`` becomes an empty element marked ``nil``, never a zero or an empty
    string: an unreported metric must stay distinguishable from a measured
    zero, which is the same rule the JSON export follows.
    """
    if isinstance(value, dict):
        child = ET.SubElement(parent, _xml_tag(key))
        for sub_key, sub_value in value.items():
            _xml_value(child, sub_key, sub_value)
        return

    if isinstance(value, list):
        child = ET.SubElement(parent, _xml_tag(key))
        item_tag = _singular(key)
        for entry in value:
            _xml_value(child, item_tag, entry)
        return

    child = ET.SubElement(parent, _xml_tag(key))
    if value is None:
        child.set("nil", "true")
    elif isinstance(value, bool):
        child.text = "true" if value else "false"
    else:
        child.text = _xml_safe(str(value))


def export_xml(result: PipelineResult, settings: AppSettings) -> str:
    """The same schema as the JSON export, rendered as XML.

    Element names match the JSON keys one for one, so a consumer that already
    reads one format can map the other without a lookup table.
    """
    root = ET.Element("ocr-result")
    for key, value in _structured_payload(result, settings).items():
        _xml_value(root, key, value)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)



#: Styling for the HTML export. Inlined rather than linked so the file stays
#: readable after it has been emailed, moved or archived on its own.
_HTML_STYLE = """
  :root { color-scheme: light dark; }
  body { font: 15px/1.65 -apple-system, "Segoe UI", Roboto, sans-serif;
         max-width: 880px; margin: 0 auto; padding: 40px 24px; color: #12141c;
         background: #ffffff; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h2 { font-size: 17px; margin: 34px 0 10px; padding-top: 18px;
       border-top: 1px solid #e4e6ef; }
  h3 { font-size: 14.5px; margin: 22px 0 8px; }
  .meta { color: #5b6072; font-size: 13px; margin-bottom: 8px; }
  .meta span + span::before { content: " \00B7 "; }
  .text { white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace,
          "Cascadia Mono", Consolas, monospace; font-size: 13px; background: #f6f7fb;
          border: 1px solid #e4e6ef; border-radius: 8px; padding: 16px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #e4e6ef;
           vertical-align: top; }
  th { color: #5b6072; font-weight: 600; }
  .kv { width: auto; margin-bottom: 14px; }
  .kv th { width: 150px; font-weight: 500; }
  .failed { color: #b4232c; }
  @media (prefers-color-scheme: dark) {
    body { background: #12141c; color: #e8eaf2; }
    h2 { border-color: #2a2e3f; }
    .text { background: #1a1d29; border-color: #2a2e3f; }
    th, td { border-color: #2a2e3f; }
    .meta, th { color: #9aa0b5; }
  }
"""


def _html_text_block(text: str) -> str:
    return f'<div class="text">{escape(text)}</div>' if text else "<p><em>No text.</em></p>"


def _html_definitions(pairs: list[tuple[str, str]]) -> str:
    """Term/value pairs as a two-column table.

    A table rather than a definition list because the PDF renderer lays out
    tables and ignores ``dl``, and one markup shape for both outputs is worth
    more than the slightly better semantics.
    """
    if not pairs:
        return ""
    rows = "".join(
        f"<tr><th>{escape(term)}</th><td>{escape(value)}</td></tr>"
        for term, value in pairs
    )
    return f'<table class="kv">{rows}</table>'


def _html_engine_metrics(engine, settings: AppSettings) -> list[tuple[str, str]]:
    """The same metric set the Markdown report lists, as term/value pairs."""
    output = settings.output
    pairs: list[tuple[str, str]] = [("Status", engine.status.value)]
    if output.show_model_name and engine.model_name:
        pairs.append(("Model", engine.model_name))
    if engine.backend:
        pairs.append(("Backend", engine.backend))
    if output.show_processing_time:
        pairs.append(("Time", f"{engine.duration_seconds:.2f}s"))
    if output.show_token_usage:
        usage = engine.tokens
        pairs.append(("Input tokens", _metric(usage.input_tokens)))
        pairs.append(("Output tokens", _metric(usage.output_tokens)))
        pairs.append(("Tokens/second", _metric(engine.tokens_per_second)))
    if output.show_character_count:
        pairs.append(("Characters", f"{engine.character_count:,}"))
    if output.show_word_count:
        pairs.append(("Words", f"{engine.word_count:,}"))
    return pairs


def _report_body(result: PipelineResult, settings: AppSettings) -> list[str]:
    """The report as HTML fragments, independent of how it is paged.

    Shared by the HTML and PDF exports so the two can never describe the same
    run differently; only the chrome and the stylesheet around them differ.
    """
    output = settings.output
    document = result.document
    parts: list[str] = []

    meta: list[str] = [f"{document.page_count} page(s)"]
    if output.show_processing_time:
        meta.append(f"{result.total_duration_seconds:.2f}s total")
    if output.show_engine_name:
        names = ", ".join(r.provider_name for r in result.engine_results) or "none"
        meta.append(f"Engines: {names}")
    if result.fusion:
        meta.append(f"Fusion: {result.fusion.strategy}")
    meta.append(f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}")

    parts.append(f"<h1>{escape(document.filename)}</h1>")
    parts.append(f'<div class="meta">{escape(" - ".join(meta))}</div>')

    if output.show_extracted_text:
        parts.append("<h2>Final result</h2>")
        parts.append(_html_text_block(result.final_text))
        counts: list[str] = []
        if output.show_character_count:
            counts.append(f"{len(result.final_text):,} characters")
        if output.show_word_count:
            counts.append(f"{len(result.final_text.split()):,} words")
        if counts:
            parts.append(f'<div class="meta">{escape(" - ".join(counts))}</div>')

    if output.show_comparison and result.comparison is not None:
        parts.extend(_html_comparison(result.comparison))

    if output.show_raw_results:
        parts.append("<h2>Engine outputs</h2>")
        for engine in result.engine_results:
            parts.append(f"<h3>{escape(engine.provider_name)}</h3>")
            parts.append(_html_definitions(_html_engine_metrics(engine, settings)))
            if engine.succeeded and engine.text:
                parts.append(_html_text_block(engine.text))
            else:
                reason = engine.error or "unknown error"
                parts.append(f'<p class="failed">Failed: {escape(reason)}</p>')

    if output.show_processing_logs and result.log is not None:
        parts.append("<h2>Processing log</h2>")
        parts.append(_html_text_block(result.log.as_text()))

    return parts


def export_html(result: PipelineResult, settings: AppSettings) -> str:
    """A self-contained report page.

    Everything is inlined - no stylesheet, no script, no external request - so
    the file renders identically offline and carries no tracking surface for a
    document that may be confidential.
    """
    body = "\n".join(_report_body(result, settings))
    title = escape(result.document.filename)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>OCR result - {title}</title>\n"
        f"<style>{_HTML_STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _html_comparison(comparison) -> list[str]:
    """The comparison section: agreement figures, then numeric disagreements."""
    parts = [
        "<h2>Comparison</h2>",
        _html_definitions(
            [
                ("Agreement", f"{comparison.agreement_percent}%"),
                ("Lines", str(comparison.total_lines)),
                ("Identical", str(comparison.equal_lines)),
                ("Differing", str(comparison.differing_lines)),
            ]
        ),
    ]
    conflicts = comparison.numeric_conflicts
    if not conflicts:
        return parts

    rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            escape(conflict["text_a"]), escape(conflict["text_b"])
        )
        for conflict in conflicts[:25]
    )
    parts.append(f"<h3>Numeric disagreements ({len(conflicts)})</h3>")
    parts.append(
        f"<table><thead><tr><th>{escape(comparison.engine_a)}</th>"
        f"<th>{escape(comparison.engine_b)}</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return parts


#: Print stylesheet. Deliberately a smaller subset than the HTML one: the
#: paginating renderer supports plain block and table layout, not grid or
#: media queries, and silently drops what it cannot apply.
_PDF_STYLE = """
  body { font-family: sans-serif; font-size: 10px; color: #12141c; }
  h1 { font-size: 17px; margin: 0 0 2px 0; }
  h2 { font-size: 12px; margin: 16px 0 5px 0; color: #2f3345; }
  h3 { font-size: 10.5px; margin: 11px 0 4px 0; }
  .meta { font-size: 8.5px; color: #5b6072; margin-bottom: 6px; }
  .text { font-family: monospace; font-size: 8.5px; white-space: pre-wrap;
          background: #f6f7fb; padding: 7px; }
  table { font-size: 8.5px; }
  th { text-align: left; color: #5b6072; font-weight: normal; padding-right: 10px; }
  td { text-align: left; padding-right: 10px; }
  .failed { color: #b4232c; }
"""

#: Page geometry, in points. A4 with a 50pt (~18mm) margin.
_PDF_MARGIN = 50


def export_pdf(result: PipelineResult, settings: AppSettings) -> bytes:
    """The same report as the HTML export, paginated as a PDF.

    Rendered from the shared report body through PyMuPDF, which is already a
    dependency for reading PDFs - so the format that most clients expect to be
    handed costs no new package.
    """
    import fitz  # imported lazily: only this exporter needs it

    body = "\n".join(_report_body(result, settings))
    html = f"<html><body>{body}</body></html>"

    buffer = io.BytesIO()
    story = fitz.Story(html=html, user_css=_PDF_STYLE)
    writer = fitz.DocumentWriter(buffer)
    mediabox = fitz.paper_rect("a4")
    frame = mediabox + (_PDF_MARGIN, _PDF_MARGIN, -_PDF_MARGIN, -_PDF_MARGIN)

    more = True
    # Guarded rather than while-True: a pathological transcription must not be
    # able to spin the renderer forever inside a Streamlit request.
    for _ in range(_PDF_MAX_PAGES):
        if not more:
            break
        device = writer.begin_page(mediabox)
        more, _filled = story.place(frame)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


#: Upper bound on generated pages. Reached only by a transcription far larger
#: than any real scan; the alternative is an unbounded loop.
_PDF_MAX_PAGES = 2000


#: Exporter registry. Add a format by registering a callable here.
EXPORTERS: dict[OutputFormat, Callable[[PipelineResult, AppSettings], str]] = {
    OutputFormat.TXT: export_txt,
    OutputFormat.MARKDOWN: export_markdown,
    OutputFormat.JSON: export_json,
    OutputFormat.CSV: export_csv,
    OutputFormat.XML: export_xml,
    OutputFormat.HTML: export_html,
}

MIME_TYPES: dict[OutputFormat, str] = {
    OutputFormat.TXT: "text/plain",
    OutputFormat.MARKDOWN: "text/markdown",
    OutputFormat.JSON: "application/json",
    OutputFormat.CSV: "text/csv",
    OutputFormat.XML: "application/xml",
    OutputFormat.HTML: "text/html",
    OutputFormat.PDF: "application/pdf",
}


#: Exporters whose payload is binary. A binary format has no meaningful ``str``
#: form, so it is registered here instead of in :data:`EXPORTERS` and is reached
#: through :func:`export_bytes`.
BINARY_EXPORTERS: dict[OutputFormat, Callable[[PipelineResult, AppSettings], bytes]] = {
    OutputFormat.PDF: export_pdf,
}


def is_binary(output_format: OutputFormat) -> bool:
    """Whether ``output_format`` produces bytes rather than text."""
    return output_format in BINARY_EXPORTERS


def export(
    result: PipelineResult, settings: AppSettings, output_format: OutputFormat
) -> str:
    """Render ``result`` in ``output_format`` as text.

    Raises for a binary format rather than returning mojibake; callers that
    accept either kind should use :func:`export_bytes`.
    """
    if is_binary(output_format):
        raise ValueError(
            f"{output_format.value.upper()} is a binary format - use export_bytes()."
        )
    exporter = EXPORTERS.get(output_format)
    if exporter is None:
        raise ValueError(f"Unsupported export format: {output_format}")
    return exporter(result, settings)


def export_bytes(
    result: PipelineResult, settings: AppSettings, output_format: OutputFormat
) -> bytes:
    """Render ``result`` as the bytes that belong in a file of that format.

    The single entry point for anything that writes or serves a download, so a
    caller never has to know whether a format is text or binary.
    """
    binary = BINARY_EXPORTERS.get(output_format)
    if binary is not None:
        return binary(result, settings)
    return export(result, settings, output_format).encode("utf-8")


def export_filename(result: PipelineResult, output_format: OutputFormat) -> str:
    """A download filename derived from the document, without its extension."""
    stem = result.document.filename.rsplit(".", 1)[0] or "ocr-result"
    safe = "".join(char if char.isalnum() or char in "-_ " else "_" for char in stem)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{safe.strip() or 'ocr-result'}-{stamp}.{output_format.value}"


__all__ = [
    "BINARY_EXPORTERS",
    "EXPORTERS",
    "MIME_TYPES",
    "export",
    "export_bytes",
    "export_csv",
    "export_html",
    "export_filename",
    "export_json",
    "export_markdown",
    "export_pdf",
    "export_txt",
    "export_xml",
    "is_binary",
]
