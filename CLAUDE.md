# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
streamlit run app/main.py                       # http://localhost:8501

# Headless / batch
python -m ocr_fusion.cli check                  # engine readiness + remedies
python -m ocr_fusion.cli providers              # what is registered
python -m ocr_fusion.cli run invoice.pdf -f json -o out.json

# Tests
pytest                                          # default suite: no Ollama, no GPU, no models
pytest -m integration                           # add live-engine tests (they self-skip)
pytest -m "integration and not slow"            # live tests without real inference
pytest tests/unit/test_fusion.py -k line_vote   # one file, one pattern
pytest --cov=ocr_fusion

# Lint / types (optional extras)
ruff check . && mypy ocr_fusion
```

Prerequisites for a real run: `ollama serve`, plus `ollama pull qwen2.5vl:3b`
and `ollama pull deepseek-ocr:3b`.

## Architecture

`app/` (Streamlit) depends on `ocr_fusion/` (framework); the reverse never
happens. `ocr_fusion` has no Streamlit dependency — `cli.py` exists partly to
keep that honest. `app/` contains no OCR logic.

Pipeline: `Document → engines → Comparison → Fusion → PipelineResult`.

Four extension seams, all working the same way — the caller depends on an
interface or a registry entry, never on an implementation:

| Seam | Add a… | Where |
|---|---|---|
| `OCRProvider` | OCR engine | `ocr/providers/`, register in `ocr/providers/__init__.py` |
| `UnlimitedBackendBase` | deployment shape for Unlimited-OCR | `ocr/providers/unlimited/` |
| `FUSION_STRATEGIES` | fusion policy | `pipeline/fusion.py` |
| `LOADERS` / `EXPORTERS` | document type / export format | `documents/loaders.py`, `export/exporters.py` |

The pipeline and UI resolve engines through `default_registry` and never import
a provider class, which is why a new engine needs no UI change. `docs/PROVIDERS.md`
has a full worked example; `docs/ARCHITECTURE.md` covers data flow.

## Invariants

These are enforced by tests. Breaking one is a regression, not a style choice.

- **Unknown metrics are `None`, never `0`.** Tesseract has no tokens, so its
  token fields stay unset and render as `N/A`. A zero is a fabricated
  measurement.
- **Expected failures are returned, not raised.** A provider that cannot reach
  its runtime returns a failed `OCRResult` with a plain-language message and a
  `remedy`. Raising is for bugs; the pipeline catches those too so one broken
  engine cannot end a run.
- **Fusion never invents.** Three strategies select between existing lines. The
  `llm` strategy is verified by word containment afterwards and discarded in
  favour of the deterministic merge if it drifts.
- **Documents stay local and in memory** unless persistence is enabled. Logs
  record counts, not content. Backends needing a file use a temp dir removed
  immediately.
- **The UI reports what actually ran** — `model_name`, `backend`, and
  `is_substitute` where applicable.

## Stage 2 is substituted by default

Upstream Unlimited-OCR needs CUDA. The default `ollama` backend runs
`deepseek-ocr:3b` instead so the two-engine comparison is demonstrable without a
GPU. It is a real model, labelled as a substitute everywhere it surfaces — never
present it as upstream Unlimited-OCR. The `http` backend against a GPU host is
the path to the genuine model.

## Gotchas

- **Streamlit reruns the whole script** on every interaction. Session state
  lives in `app/state.py`; health checks are cached and manual because probing
  costs a round trip per engine.
- **`st.markdown` cannot wrap widgets.** Each call renders into its own element,
  so an unclosed `<div>` is closed immediately. Use `theme.card()`, which is a
  container plus a marker element styled via `:has()`.
- **Never style Streamlit's emotion classes** (`st-emotion-cache-*`) — they
  change between releases. Use our `ofs-*` classes and stable `data-testid`s.
- **Avoid `SequenceMatcher` over whole documents.** It is O(n·m) on characters
  and took ~59 s on a realistic transcription pair. Comparison aggregates from
  aligned lines; fusion verification uses word containment.
- **CPU inference is slow.** A cold 3B VLM load can exceed five minutes, which
  is why the Qwen timeout defaults to 600 s. If both models will not fit in RAM,
  set `keep_alive` to `0`.
- **Write files as LF.** `.gitattributes` pins this; scripted edits on Windows
  otherwise produce whole-file diffs.

## Configuration

Defaults → `runtime/settings.json` → `OCRFS_*` environment variables, each
layer overriding the previous. Nothing is hardcoded: model names, prompts,
hosts, timeouts and output options all live in `config/schema.py` and are
reachable from the Settings page. Prompts are data in `config/prompts.py`, kept
separate from provider logic.
