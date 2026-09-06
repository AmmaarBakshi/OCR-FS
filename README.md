# OCR Fusion Studio

A multi-engine OCR pipeline that transcribes a document with two independent
models, shows you where they disagree, and produces a single reconciled result.

```
Document  ->  Qwen2.5-VL  ->  Unlimited-OCR  ->  Comparison  ->  Fusion  ->  Final result
```

Both raw transcriptions are always kept. The comparison view puts them side by
side and calls out the disagreements — especially the numeric ones, because a
mistranscribed figure on an invoice is the error that actually costs something.

Underneath the app is `ocr_fusion`, a UI-agnostic framework you can drop into
another project. Adding a third engine means writing one class and registering
it; neither the pipeline nor the interface changes.

---

## What you get

- **Two real OCR engines**, run locally. No document leaves the machine unless
  you point an engine at a remote server, and the interface says so when you do.
- **A comparison view** that aligns the two transcriptions line by line and
  separates formatting differences from genuine disagreements.
- **A final result** produced by a configurable fusion strategy that can recover
  text one engine missed — and that cannot invent text neither engine produced.
- **Honest metrics.** A measurement an engine does not report shows as `N/A`,
  never as a zero.
- **Demo Mode and Developer Mode.** One toggle between a clean client-facing
  view and full metrics, logs, raw output and configuration.
- **Exports** to TXT, Markdown, JSON and CSV, with a documented, stable JSON
  schema for downstream consumers.

---

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| [Ollama](https://ollama.com) | for the Qwen2.5-VL engine |
| Disk | ~4 GB for `qwen2.5vl:3b`, ~7 GB more for the local Stage 2 model |
| RAM | 16 GB is enough; see [Performance](#performance) |
| GPU | optional — everything runs on CPU, just slower |

---

## Install

```bash
git clone https://github.com/AmmaarBakshi/OCR-FS.git
cd OCR-FS
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### Set up Ollama and the models

Install Ollama from [ollama.com/download](https://ollama.com/download), then:

```bash
ollama serve                  # if it is not already running
ollama pull qwen2.5vl:3b      # Stage 1  (~3.2 GB)
ollama pull deepseek-ocr:3b   # Stage 2 on a machine without a GPU (~6.7 GB)
```

Confirm both are present:

```bash
ollama list
```

### Run

```bash
streamlit run app/main.py
```

The app opens at <http://localhost:8501>. Use **Check engines** in the sidebar
to confirm both stages are ready before your first run.

---

## Using it

1. Drag a PDF or image onto the upload area.
2. Press **Run OCR**. The pipeline display shows each stage as it runs.
3. Read the **Final result**, or switch tabs to see each engine's raw output.
4. Open **Comparison** to see where the engines disagreed.
5. Export from the **Export** tab.

Turn on **Developer Mode** in the sidebar for metrics, the processing log, raw
engine responses and the exact configuration a run used.

---

## Stage 2: Unlimited-OCR

Upstream [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) is a ~6.7B MoE
vision model that calls `.cuda()` and expects CUDA 12.9. Rather than tie the app
to one deployment, Stage 2 runs through a backend you choose in
**Settings › Unlimited OCR**:

| Backend | What it runs | Needs |
|---|---|---|
| `http` | vLLM, SGLang or `unlimited-ocr-server` over an OpenAI-compatible API | a GPU host (can be another machine) |
| `transformers` | the model in this process | a local CUDA GPU, ~14 GB download |
| `cli` | an external script or binary, e.g. `infer.py` or `franken_ocr` | that tool installed |
| `ollama` | a locally pulled OCR model — **the default** | Ollama only |

### About the default

Without a GPU, none of the first three backends can run, which would leave
Stage 2 permanently dead and the whole point of the app — comparing two engines
— undemonstrable.

So the default `ollama` backend runs a **different, real OCR model**
(`deepseek-ocr:3b`; Unlimited-OCR is itself DeepSeek-OCR-derived). It is a
substitute, and the app says so everywhere it appears — the stage label, the
results tab, the metrics and the JSON export all report the model that actually
ran. Nothing is simulated: either a real model transcribes the page, or the
stage reports failure.

To run genuine Unlimited-OCR, start a server on a GPU machine:

```bash
python -m sglang.launch_server --model baidu/Unlimited-OCR --context-length 32768
```

then set **Backend** to `http` and **Server address** to `http://<host>:30000/v1`.

---

## Configuration

Settings resolve in this order, each layer overriding the one before:

1. Packaged defaults
2. `runtime/settings.json` — written by the Settings page
3. `OCRFS_*` environment variables

Nothing is hardcoded. Model names, prompts, timeouts, hosts, paths and output
options are all reachable from the Settings page.

Useful variables:

```bash
OCRFS_OLLAMA_HOST=http://localhost:11434
OCRFS_QWEN_MODEL=qwen2.5vl:3b
OCRFS_QWEN_TIMEOUT=600
OCRFS_UNLIMITED_BACKEND=http
OCRFS_UNLIMITED_ENDPOINT=http://gpu-host:30000/v1
OCRFS_FUSION_STRATEGY=line_vote
OCRFS_PDF_DPI=150
OCRFS_MAX_PAGES=10
OCRFS_SETTINGS_PATH=/custom/settings.json
```

Copy `.env.example` if you prefer a file.

### Fusion strategies

| Strategy | Behaviour |
|---|---|
| `line_vote` *(default)* | Merge line by line. Agreed lines kept, disagreements resolved toward the more complete reading, lines found by only one engine preserved. |
| `prefer_primary` | Use the first successful engine verbatim. |
| `prefer_longest` | Use whichever engine extracted the most text. |
| `llm` | Ask a language model to reconcile the two. Its output is checked against both inputs and discarded if it drifts. |

Only `llm` involves a model; the other three are deterministic and cannot
introduce text.

---

## Using the framework directly

`ocr_fusion` has no Streamlit dependency:

```python
from ocr_fusion.config import load_settings
from ocr_fusion.documents import load_document
from ocr_fusion.pipeline import build_pipeline
import ocr_fusion.ocr.providers  # registers the built-in engines

settings = load_settings()
document = load_document("invoice.pdf", settings.documents)
result = build_pipeline(settings).execute(document)

print(result.final_text)
print(result.metrics())

for engine in result.engine_results:
    print(engine.provider_name, engine.status.value, engine.character_count)

if result.comparison:
    print(f"{result.comparison.agreement_percent}% agreement")
    for conflict in result.comparison.numeric_conflicts:
        print("check:", conflict["text_a"], "vs", conflict["text_b"])
```

---

## Export schema

`result.as_dict()` and the JSON export share one stable shape:

```json
{
  "schema_version": "1.0",
  "document":     { "filename": "...", "page_count": 2, "has_text_layer": true },
  "pipeline":     { "stages": [ ... ], "total_duration_seconds": 134.7 },
  "engines":      [ { "provider_id": "...", "text": "...", "pages": [ ... ] } ],
  "comparison":   { "agreement_percent": 86.2, "numeric_conflicts": [ ... ] },
  "final_result": { "text": "...", "strategy": "line_vote" },
  "metrics":      { "total_tokens": 4153, "pages_processed": 1 },
  "configuration": { },
  "logs":         [ ]
}
```

A metric an engine did not report is `null`, never `0`.

---

## Adding an OCR engine

Implement `OCRProvider`, register it, and it appears in the pipeline, the
results tabs, the comparison and the exports with no further changes. See
[`docs/PROVIDERS.md`](docs/PROVIDERS.md) for a worked example.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # the full default suite - no Ollama, no GPU, no models
pytest -m integration       # add tests that use live engines
pytest --cov=ocr_fusion     # with coverage
pytest tests/unit/test_fusion.py -k line_vote   # one file, one pattern
```

The default suite uses mocked providers throughout, so it runs anywhere in a
few seconds. Integration tests skip themselves when their runtime is missing.

---

## Performance

Measured on an Intel i5-10310U (4 cores, no GPU), 16 GB RAM, one A4 page at
150 DPI:

| Stage | Cold | Warm |
|---|---|---|
| Qwen2.5-VL | ~7 min (includes model load) | ~70 s |
| Stage 2 (`deepseek-ocr:3b`) | ~2 min | ~100 s |
| Comparison + fusion | — | < 0.1 s |

A GPU changes this by an order of magnitude. To speed up a CPU demo: lower
**PDF quality (DPI)** in Settings, set a **page limit**, or disable one engine.

If both models will not fit in RAM at once, set **Keep model loaded for** to
`0` in Settings › OCR Models so the first model unloads before the second
loads.

---

## Privacy

- Documents are held in memory and processed locally. Nothing is written to disk
  unless you enable persistence in Settings › Privacy.
- The processing log records counts, not content, by default.
- Each engine reports whether it processes **locally** or in the **cloud**, and
  a remote HTTP endpoint is classified as cloud automatically.
- API keys are redacted from settings snapshots before they reach an export.

---

## Troubleshooting

**"Could not reach Ollama"** — start it with `ollama serve`, or check the
address in Settings › OCR Models.

**"The model … is not installed"** — the message names the exact
`ollama pull` command to run.

**A run times out** — a cold model load on CPU can exceed five minutes. Raise
**Timeout** in Settings › OCR Models, or lower the PDF DPI.

**Stage 2 shows "Unavailable"** — expected without a GPU unless the `ollama`
backend is selected. The message states which backend was tried and what it
needs.

**Out of memory** — reduce **PDF quality (DPI)**, set **Keep model loaded for**
to `0`, or disable one engine.

**Transcription is cut off** — raise **Maximum output length** in Settings. The
app flags truncation when it happens rather than returning a short result
silently.

**A page comes back empty** — reported as a page failure, not silent success.
Try a higher DPI; very low-contrast scans may need preprocessing.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components and data flow
- [`docs/PROVIDERS.md`](docs/PROVIDERS.md) — writing a new OCR provider

## Licence

MIT
