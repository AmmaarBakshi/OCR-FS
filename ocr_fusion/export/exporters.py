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
from typing import Any
from xml.etree import ElementTree as ET

from ocr_fusion.config.schema import AppSettings, OutputFormat
from ocr_fusion.pipeline.result import PipelineResult


def _metric(value: Any, suffix: str = "") -> str:
    """Render a metric, showing N/A for anything the engine did not report.

    The rule from spec s8: never fabricate a metric, never print 0 for unknown.
    """
    if value is None:
        return "N/A"
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
            if output.show_character_count:
                row["characters"] = page.character_count if page else 0
            if output.show_word_count:
                row["words"] = page.word_count if page else 0
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



#: Exporter registry. Add a format by registering a callable here.
EXPORTERS: dict[OutputFormat, Callable[[PipelineResult, AppSettings], str]] = {
    OutputFormat.TXT: export_txt,
    OutputFormat.MARKDOWN: export_markdown,
    OutputFormat.JSON: export_json,
    OutputFormat.CSV: export_csv,
    OutputFormat.XML: export_xml,
}

MIME_TYPES: dict[OutputFormat, str] = {
    OutputFormat.TXT: "text/plain",
    OutputFormat.MARKDOWN: "text/markdown",
    OutputFormat.JSON: "application/json",
    OutputFormat.CSV: "text/csv",
    OutputFormat.XML: "application/xml",
}


def export(
    result: PipelineResult, settings: AppSettings, output_format: OutputFormat
) -> str:
    """Render ``result`` in ``output_format``."""
    exporter = EXPORTERS.get(output_format)
    if exporter is None:
        raise ValueError(f"Unsupported export format: {output_format}")
    return exporter(result, settings)


def export_filename(result: PipelineResult, output_format: OutputFormat) -> str:
    """A download filename derived from the document, without its extension."""
    stem = result.document.filename.rsplit(".", 1)[0] or "ocr-result"
    safe = "".join(char if char.isalnum() or char in "-_ " else "_" for char in stem)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{safe.strip() or 'ocr-result'}-{stamp}.{output_format.value}"


__all__ = [
    "EXPORTERS",
    "MIME_TYPES",
    "export",
    "export_csv",
    "export_filename",
    "export_json",
    "export_markdown",
    "export_txt",
    "export_xml",
]
