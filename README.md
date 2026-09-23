# OCR Fusion Studio

A multi-engine OCR pipeline that transcribes a document with two independent
models, shows you where they disagree, and produces a single reconciled result.

```
Document  ->  Qwen2.5-VL  ->  Unlimited-OCR  ->  Comparison  ->  Fusion  ->  Final result
```

The Streamlit app runs a document through both engines, shows the two
transcriptions side by side and calls out the disagreements — especially the
numeric ones, because a mistranscribed figure on an invoice is the error that
actually costs something.

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
- **Ask about the document.** Once a run finishes, a question box answers from
  the transcription — "what is in this document?", "list the dates and
  amounts" — and says so when the document does not contain the answer.
- **Exports** to TXT, Markdown, JSON, CSV, XML, HTML and PDF, with a
  documented, stable JSON schema for downstream consumers.

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
4. Open **To check** to see which pages a model read, which were answered from
   the document's own text, and which few pages are worth a human look.
5. Open **Comparison** to see where two engines disagreed.
6. Ask a question about the document in the box below the results.
7. Export from the **Export** tab.

### Many documents at once

```bash
python -m ocr_fusion.cli batch ./inbox -o ./results
```

Each document's result is written the moment it finishes, and progress is
checkpointed on every change. Run the same command again to continue: finished
documents are skipped and new ones picked up, so a batch stopped after nine
hours costs nothing to resume and a crash costs one document.

```bash
python -m ocr_fusion.cli batch ./inbox -o ./results --status   # progress only
python -m ocr_fusion.cli batch ./inbox -o ./results --cache    # remember pages
```

Documents whose pages were flagged come back as **review required** rather
than silently passing — transcribed and usable, with the caveat attached.

### How you get the result

**Settings › Output** chooses between two ways of handing a result over:

| Mode | What the results panel does |
|---|---|
| **On site** (default) | Shows the transcription in the page. |
| **Off site** | Prepares the result as a file in a format you pick — PDF, Markdown, HTML, XML, JSON, CSV or plain text — and offers it for download. |

Every format stays available in the **Export** tab either way; the mode only
decides which one the page leads with.

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

### Where the time goes

One A4 page at 150 DPI through `qwen2.5vl:3b`, on an Intel i5-10310U
(4 cores, 16 GB, no usable GPU):

| Stage | Time | Share |
|---|---|---|
| PDF render | 0.18 s | 0.04% |
| PNG encode | 0.14 s | 0.03% |
| Model load | 15.5 s | 3% |
| **Encoding the image — 2,979 image tokens** | **415.9 s** | **84.6%** |
| Writing the answer — 283 tokens at 4.85 tok/s | 58.4 s | 12% |

The image is the cost, not the answer. Everything below follows from that.

### Most pages never reach a model

Across a 1,666-page reference corpus of real client documents — tax returns,
bank statements, insurance schedules, scanned deeds:

| | |
|---|---|
| Pages that already carry their own text | 1,415 (**84.9%**) |
| Pages that genuinely need OCR | 251 (15.1%) |

A born-digital PDF states its text exactly. Recognising it again with a vision
model costs about 250,000 times as much CPU **and is less accurate**, because
it replaces a fact with a transcription. So the pipeline reads the text layer
first and only sends a model what is left.

Measured, on this machine:

| Document | Old pipeline | Now |
|---|---|---|
| 65-page tax return (born-digital) | ~17 h | **1.0 s** |
| 309 pages across six tax returns | ~84 h | **~3 s** |
| 20-page mixed statement | ~5 h | 3 pages of OCR, the other 17 free |

### Choosing a resolution

DPI is the main lever on a page that *does* need a model, because image tokens
scale with pixel area. Swept on a real form page, scored against the text that
page was authored with:

| DPI | Image tokens | Seconds | Word error | Recall |
|---|---|---|---|---|
| 72 | 1,338 | 282 | 0.101 | 92.0% |
| 96 | 1,367 | 326 | 0.080 | 92.7% |
| **120** (default) | 1,957 | 398 | 0.028 | 99.0% |
| 150 | 2,979 | 551 | 0.021 | 99.7% |

Lower is a trap. Below 120 the error rate nearly triples — blurred text does
not merely get misread, it makes the model ramble, so output tokens go up too.

Pick a point with `--profile fast|balanced|accurate`, or in Settings › Speed.
`python -m ocr_fusion.cli profiles` prints the table above with the current
setting marked.

Full measurements, including the before/after on a document that cannot be
routed around and the head-to-head between the two models, are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

### Measuring it yourself

Every number here came from the bundled harness, on real documents:

```bash
python -m ocr_fusion.cli bench ./your-documents -o report.json
python -m ocr_fusion.cli accuracy ./your-documents --profile balanced
```

`accuracy` needs no labelled data: a born-digital page is its own answer key,
so it renders the page, transcribes the image, and scores the result against
the text the file already contained. Figures are scored separately from words,
because a mistranscribed amount is the error that actually costs something.

### If it is still slow

- Check Settings › Speed. Most of the time is in options gathered there.
- Turn on **Never read the same page twice** for batches that repeat pages.
- `deepseek-ocr:3b` is pulled unquantised (F16, 6.7 GB) against
  `qwen2.5vl:3b` at 3.2 GB. With both resident on a 16 GB machine you will
  swap, and swapping is slower than not running. Leave the second engine as a
  fallback rather than a second full pass.
- A GPU changes all of this by an order of magnitude.

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
