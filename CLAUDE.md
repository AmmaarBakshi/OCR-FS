# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
streamlit run app/main.py                       # http://localhost:8501

# Headless / batch
python -m ocr_fusion.cli check                  # engine readiness + remedies
python -m ocr_fusion.cli providers              # what is registered
python -m ocr_fusion.cli profiles               # speed/accuracy, with measurements
python -m ocr_fusion.cli run invoice.pdf -f json -o out.json
python -m ocr_fusion.cli batch ./inbox -o ./out # resumable; re-run to continue

# Measure before claiming
python -m ocr_fusion.cli bench ./corpus -o report.json
python -m ocr_fusion.cli accuracy ./corpus --profile fast

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

Pipeline: `Document → Routing → cascade of engines → Comparison → Fusion →
PipelineResult`.

Routing is the load-bearing stage. It decides per page - from the text layer,
an exact content hash and the ink ratio - whether the page needs a model at
all. On the reference corpus 84.9% of pages do not.

`ocr_fusion/chat/` sits after the pipeline, not inside it: it answers questions
about a finished transcription and never re-reads the page.

Five extension seams, all working the same way — the caller depends on an
interface or a registry entry, never on an implementation:

| Seam | Add a… | Where |
|---|---|---|
| `OCRProvider` | OCR engine | `ocr/providers/`, register in `ocr/providers/__init__.py` |
| `UnlimitedBackendBase` | deployment shape for Unlimited-OCR | `ocr/providers/unlimited/` |
| `FUSION_STRATEGIES` | fusion policy | `pipeline/fusion.py` |
| `LOADERS` / `EXPORTERS` | document type / export format | `documents/loaders.py`, `export/exporters.py` |
| `BINARY_EXPORTERS` | export format whose bytes are not text | `export/exporters.py`, reached via `export_bytes()` |

A sixth seam worth knowing: `ocr/cache.py` wraps any provider to add page
caching, and `pipeline/routing.py:subset()` hands a provider a document with
fewer pages. Between them, routing and caching keep pages away from engines
without any engine knowing either exists.

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
- **Nothing reaches a model that did not have to.** Routing runs before any
  engine, and the free test (text layer) runs before any test that needs
  pixels. Adding a check that forces a render before the text-layer check
  silently reinstates the cost this pipeline exists to avoid.
- **A cache hit reports zero seconds for this run.** The time was spent on a
  previous run. Replaying the original duration would claim work this run did
  not do — the same fabrication as a zero token count.
- **A page is never described as verified.** Confidence says a page did not
  look wrong in the ways that are checkable — empty, truncated, looping, too
  thin for its ink, not language. That is not the same as correct, and the UI
  must not round it up.

## Stage 2 is substituted by default

Upstream Unlimited-OCR needs CUDA. The default `ollama` backend runs
`deepseek-ocr:3b` instead so the two-engine comparison is demonstrable without a
GPU. It is a real model, labelled as a substitute everywhere it surfaces — never
present it as upstream Unlimited-OCR. The `http` backend against a GPU host is
the path to the genuine model.

## Where the time actually goes

Full measurements, with method and caveats, are in `docs/PERFORMANCE.md`.
The short version:

Measured on the development machine (i5-10310U, 4 cores, 16 GB, no usable
GPU), one 150 DPI page through `qwen2.5vl:3b`:

| stage | time | share |
|---|---|---|
| PDF render | 0.18 s | 0.04% |
| PNG encode | 0.14 s | 0.03% |
| model load | 15.5 s | 3% |
| **prompt eval — 2,979 image tokens** | **415.9 s** | **84.6%** |
| generation — 283 tokens at 4.85 tok/s | 58.4 s | 12% |

**The image is the cost, not the answer.** Optimising decode, resize or
encoding is optimising 0.07% of the run. The levers that matter, in order:
don't call the model (routing, cache), then make the image smaller (DPI), then
bound the output (`max_tokens`).

DPI swept against ground truth from a page's own text layer:

| DPI | image tokens | seconds | word error | recall |
|---|---|---|---|---|
| 72 | 1,338 | 282 | 0.101 | 0.920 |
| 96 | 1,367 | 326 | 0.080 | 0.927 |
| **120** | 1,957 | 398 | 0.028 | 0.990 |
| 150 | 2,979 | 551 | 0.021 | 0.997 |

120 is the default. Lower is a trap, and the token counts say why: the text
prompt is a fixed 265 tokens whatever the image, and the model pads small
images up to a minimum pixel budget (782/796/775 px per image token at
96/120/150 DPI, but 452 at 72 - the 72 DPI render is being padded back up).
So 72 DPI costs nearly the same tokens as 96 for a blurrier page, and blurred
text makes the model ramble, so output tokens rise too.

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
- **`deepseek-ocr:3b` is pulled as F16, 6.7 GB unquantised** — against
  `qwen2.5vl:3b` at Q4_K_M, 3.2 GB. With both resident on a 16 GB machine the
  box swaps, and a swapping run is slower than no run. That is why the second
  engine is a fallback rather than a second full pass, and why the batch
  runner refuses to start a document with no room for a model.
- **Ollama drops the integrated GPU by default** (`dropping integrated GPU; to
  enable, set OLLAMA_IGPU_ENABLE=1`). Untested here; do not assume it helps
  before measuring, and do not report it as acceleration that exists.
- **Benchmark on documents, not on a page.** A page is a measurement of the
  model; a document is a measurement of the pipeline, and the pipeline's whole
  job is arranging for most pages not to be pages the model sees.
- **Write files as LF.** `.gitattributes` pins this; scripted edits on Windows
  otherwise produce whole-file diffs.

## Configuration

Defaults → `runtime/settings.json` → `OCRFS_*` environment variables, each
layer overriding the previous. Nothing is hardcoded: model names, prompts,
hosts, timeouts and output options all live in `config/schema.py` and are
reachable from the Settings page. Prompts are data in `config/prompts.py`, kept
separate from provider logic.
