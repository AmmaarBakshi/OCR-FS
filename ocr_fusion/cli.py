"""Command-line interface.

Exists for two reasons: batch use without a browser, and as a standing
demonstration that the framework is genuinely UI-independent. If this module
ever needs Streamlit, the separation has been broken.

    ocr-fusion run invoice.pdf --format json --output result.json
    ocr-fusion check
    ocr-fusion providers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr_fusion.config import AppSettings, load_settings
from ocr_fusion.config.schema import OutputFormat, UnlimitedBackend
from ocr_fusion.documents import load_document
from ocr_fusion.documents.errors import DocumentError
from ocr_fusion.export import export
from ocr_fusion.metrics import format_duration
from ocr_fusion.pipeline import build_pipeline
from ocr_fusion.version import __version__

import ocr_fusion.ocr.providers  # noqa: F401  (registers the built-in engines)


def _stdout_supports_unicode() -> bool:
    """Whether the console can render the status glyphs.

    A Windows console defaults to cp1252, which cannot encode the tick and
    cross used in the UI; printing them raises UnicodeEncodeError.
    """
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


_SYMBOLS_UNICODE = {"completed": "✓", "failed": "✕", "skipped": "–", "running": "◐", "waiting": "○"}
_SYMBOLS_ASCII = {"completed": "OK", "failed": "!!", "skipped": "--", "running": "..", "waiting": "  "}


def _symbol(status: str) -> str:
    table = _SYMBOLS_UNICODE if _stdout_supports_unicode() else _SYMBOLS_ASCII
    return table.get(status, "?")


def _apply_overrides(settings: AppSettings, args: argparse.Namespace) -> AppSettings:
    """Apply command-line overrides on top of the loaded settings."""
    if args.model:
        settings.qwen.model = args.model
    if args.host:
        settings.ollama.host = args.host
    if args.backend:
        settings.unlimited_ocr.backend = UnlimitedBackend(args.backend)
    if args.dpi:
        settings.documents.pdf_render_dpi = args.dpi
    if args.max_pages is not None:
        settings.pipeline.max_pages = args.max_pages
    if args.engine:
        # Only the named engines run.
        settings.qwen.enabled = "qwen_vl" in args.engine
        settings.unlimited_ocr.enabled = "unlimited_ocr" in args.engine
        settings.tesseract.enabled = "tesseract" in args.engine
    return settings


def command_check(args: argparse.Namespace) -> int:
    """Report which engines are ready, and how to fix the ones that are not."""
    settings = _apply_overrides(load_settings(), args)
    report = build_pipeline(settings).health_report()

    if not report:
        print("No engines are enabled. Enable one in settings.")
        return 1

    failures = 0
    for info in report.values():
        state = "ready" if info["available"] else "unavailable"
        print(f"{_symbol('completed' if info['available'] else 'failed')} {info['name']}: {state}")
        print(f"    {info['message']}")
        if not info["available"]:
            failures += 1
            if info["remedy"]:
                print(f"    fix: {info['remedy']}")
    return 1 if failures == len(report) else 0


def command_providers(args: argparse.Namespace) -> int:
    """List registered engines."""
    from ocr_fusion.ocr.registry import default_registry

    settings = load_settings()
    enabled = {spec.provider_id for spec in default_registry.enabled_specs(settings)}
    for spec in default_registry.specs():
        mark = "on " if spec.provider_id in enabled else "off"
        print(f"[{mark}] {spec.provider_id:<16} {spec.display_name}")
        if spec.description:
            print(f"       {spec.description}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    """Transcribe a document and write the result."""
    settings = _apply_overrides(load_settings(), args)

    try:
        document = load_document(args.document, settings.documents)
    except DocumentError as exc:
        print(f"error: {exc.user_message}", file=sys.stderr)
        return 2

    pipeline = build_pipeline(settings)
    if not pipeline.providers:
        print("error: no OCR engines are enabled.", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"{document.filename}: {document.page_count} page(s)", file=sys.stderr)
        pipeline.subscribe(lambda event: print(f"  {event.format()}", file=sys.stderr))

    result = pipeline.execute(document)

    if not args.quiet:
        print(file=sys.stderr)
        for stage in result.stages:
            timing = format_duration(stage.duration_seconds)
            print(f"  {_symbol(stage.status.value)} {stage.label:<28} {timing:>9}", file=sys.stderr)
        print(f"  total: {format_duration(result.total_duration_seconds)}", file=sys.stderr)

    payload = export(result, settings, OutputFormat(args.format))
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        if not args.quiet:
            print(f"written to {args.output}", file=sys.stderr)
    else:
        # Results go to stdout so the command composes with a shell pipeline;
        # progress goes to stderr so it does not corrupt them.
        sys.stdout.write(payload)

    return 0 if result.succeeded else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-fusion",
        description="Multi-engine OCR with result comparison and fusion.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="Override the Qwen model name.")
    common.add_argument("--host", help="Override the Ollama address.")
    common.add_argument(
        "--backend",
        choices=[b.value for b in UnlimitedBackend],
        help="Override the Unlimited-OCR backend.",
    )
    common.add_argument("--dpi", type=int, help="PDF render DPI.")
    common.add_argument("--max-pages", type=int, help="Process at most N pages (0 = all).")
    common.add_argument(
        "--engine",
        action="append",
        choices=["qwen_vl", "unlimited_ocr", "tesseract"],
        help="Run only this engine. Repeat for several.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="Transcribe a document.")
    run.add_argument("document", help="Path to a PDF or image.")
    run.add_argument(
        "-f",
        "--format",
        choices=[f.value for f in OutputFormat],
        default="txt",
        help="Output format (default: txt).",
    )
    run.add_argument("-o", "--output", help="Write to a file instead of stdout.")
    run.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    run.set_defaults(func=command_run)

    check = sub.add_parser("check", parents=[common], help="Check engine readiness.")
    check.set_defaults(func=command_check)

    providers = sub.add_parser("providers", parents=[common], help="List registered engines.")
    providers.set_defaults(func=command_providers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
