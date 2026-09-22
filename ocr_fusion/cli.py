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
from ocr_fusion.export import export_bytes
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

    payload = export_bytes(result, settings, OutputFormat(args.format))
    if args.output:
        Path(args.output).write_bytes(payload)
        if not args.quiet:
            print(f"written to {args.output}", file=sys.stderr)
    else:
        # Results go to stdout so the command composes with a shell pipeline;
        # progress goes to stderr so it does not corrupt them. Binary formats
        # are written through the buffer so the console encoding cannot mangle
        # them.
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()

    return 0 if result.succeeded else 1


def _collect_documents(target: str, pattern: str = "*.pdf") -> list[Path]:
    """Every document under ``target``, or just ``target`` if it is a file."""
    path = Path(target)
    if path.is_file():
        return [path]
    return sorted(path.rglob(pattern))


def command_bench(args: argparse.Namespace) -> int:
    """Time a configuration over real documents, and say where the time went.

    A performance claim that cannot be re-run is not a claim, so this is the
    command that has to back every one of them.
    """
    from ocr_fusion.bench import benchmark_documents

    settings = _apply_overrides(load_settings(), args)
    _apply_bench_overrides(settings, args)

    documents = _collect_documents(args.target)
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        print(f"error: no documents found at {args.target}", file=sys.stderr)
        return 2

    print(
        f"Benchmarking {len(documents)} document(s) - "
        f"{settings.pipeline.engine_mode.value}, "
        f"{settings.documents.pdf_render_dpi} DPI, "
        f"routing {'on' if settings.routing.enabled else 'off'}",
        file=sys.stderr,
    )

    def announce(path: Path) -> None:
        print(f"  {path.name}", file=sys.stderr)

    report = benchmark_documents(
        documents, settings, label=args.label, on_document=announce
    )
    _print_bench_report(report)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nwritten to {args.output}", file=sys.stderr)
    return 0


def command_accuracy(args: argparse.Namespace) -> int:
    """Score transcription against ground truth taken from PDF text layers."""
    from ocr_fusion.bench import benchmark_accuracy, build_gold_set

    settings = _apply_overrides(load_settings(), args)
    _apply_bench_overrides(settings, args)

    gold = build_gold_set(
        _collect_documents(args.target),
        max_pages_per_document=args.per_document,
        limit=args.limit or 0,
    )
    if not gold.pages:
        print(
            "error: no page carried a text layer usable as ground truth. "
            "Accuracy can only be scored against born-digital pages.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Scoring {len(gold)} page(s) from {gold.summary()['documents']} document(s) "
        f"at {settings.documents.pdf_render_dpi} DPI",
        file=sys.stderr,
    )

    def announce(entry: object) -> None:
        print(f"  {Path(entry.source).name} p{entry.page_number}", file=sys.stderr)

    report = benchmark_accuracy(gold, settings, label=args.label, on_page=announce)

    print(f"\n  pages            {report['pages']}")
    for key, label in (
        ("mean_seconds", "seconds / page"),
        ("mean_input_tokens", "image tokens"),
        ("mean_wer", "word error rate"),
        ("mean_cer", "char error rate"),
        ("mean_recall", "word recall"),
        ("mean_figure_recall", "figure recall"),
    ):
        value = report.get(key)
        print(f"  {label:<16} {'n/a' if value is None else f'{value:.4f}'}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.output}", file=sys.stderr)
    return 0


def _apply_bench_overrides(settings: AppSettings, args: argparse.Namespace) -> AppSettings:
    """Apply the benchmark-only switches, so two configurations can be compared."""
    from ocr_fusion.config.schema import EngineMode

    if getattr(args, "mode", None):
        settings.pipeline.engine_mode = EngineMode(args.mode)
    if getattr(args, "no_routing", False):
        settings.routing.enabled = False
    if getattr(args, "no_fallback", False):
        settings.pipeline.fallback_enabled = False
    return settings


def _print_bench_report(report: object) -> None:
    """Print the report as a table, with n/a for anything not measured."""
    data = report.as_dict()  # type: ignore[attr-defined]
    totals = data["totals"]
    spread = data["page_seconds"]

    print()
    print(f"  {'documents':<18} {totals['documents']}")
    print(f"  {'pages':<18} {totals['pages']}")
    print(f"  {'pages via a model':<18} {totals['pages_needing_ocr']}")
    print(f"  {'pages avoided':<18} {totals['pages_avoided']}")
    print(f"  {'total time':<18} {format_duration(totals['seconds'])}")
    print(f"  {'pages / minute':<18} {totals['pages_per_minute']}")
    peak = totals["peak_rss_mb"]
    print(f"  {'peak RSS':<18} {'n/a' if peak is None else f'{peak} MB'}")

    print("\n  per page (seconds)")
    for key in ("mean", "median", "p90", "p95", "max"):
        value = spread.get(key)
        print(f"    {key:<8} {'n/a' if value is None else value}")

    print("\n  per document")
    for entry in data["documents"]:
        if entry["error"]:
            print(f"    {entry['filename'][:44]:<46} {entry['error']}")
            continue
        print(
            f"    {entry['filename'][:44]:<46} "
            f"{entry['pages']:>4}p  "
            f"{entry['pages_needing_ocr']:>4} via model  "
            f"{format_duration(entry['seconds']):>10}  "
            f"{entry['pages_per_minute']:>7} p/min"
        )


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

    bench_common = argparse.ArgumentParser(add_help=False)
    bench_common.add_argument(
        "target", help="A document, or a folder to search for PDFs."
    )
    bench_common.add_argument(
        "--mode",
        choices=["cascade", "all_engines"],
        help="Engine mode to benchmark. Default: whatever settings say.",
    )
    bench_common.add_argument(
        "--no-routing",
        action="store_true",
        help="Send every page to every engine, as the pipeline did before routing.",
    )
    bench_common.add_argument(
        "--no-fallback", action="store_true", help="Never spend a second engine."
    )
    bench_common.add_argument("--limit", type=int, help="Use at most N documents/pages.")
    bench_common.add_argument("--label", default="run", help="Name for this run in the report.")
    bench_common.add_argument("-o", "--output", help="Write the full report as JSON.")

    bench = sub.add_parser(
        "bench",
        parents=[common, bench_common],
        help="Time a configuration over real documents.",
    )
    bench.set_defaults(func=command_bench)

    accuracy = sub.add_parser(
        "accuracy",
        parents=[common, bench_common],
        help="Score transcription against ground truth from PDF text layers.",
    )
    accuracy.add_argument(
        "--per-document",
        type=int,
        default=2,
        help="Gold pages to take from each document (default: 2).",
    )
    accuracy.set_defaults(func=command_accuracy)

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
