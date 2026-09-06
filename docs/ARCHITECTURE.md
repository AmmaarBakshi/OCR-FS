# Architecture

## The shape of it

Two packages, one dependency direction:

```
app/          Streamlit UI          ──depends on──▶   ocr_fusion/   framework
```

`ocr_fusion` never imports from `app`, and it does not depend on Streamlit. You
can `pip install` it and drive the pipeline from a script, a service or a batch
job. `app` contains no OCR logic at all — it renders what the framework returns.

Inside the framework the rule is the same one repeated at each level: **the
caller depends on an interface, never on an implementation.**

```
ocr_fusion/
├── config/       settings schema, layered loading, prompt templates
├── documents/    upload → Document (pages as PNG bytes)
├── ocr/
│   ├── interface.py    OCRProvider, OCRResult, PageResult, HealthStatus
│   ├── registry.py     id → factory
│   └── providers/      qwen_vl, unlimited/, tesseract
├── pipeline/     executor, stages, comparison, fusion, result
├── export/       TXT / Markdown / JSON / CSV
└── metrics/      display-ready values
```

---

## Data flow

```
        bytes
          │
          ▼
   ┌─────────────┐   sniff format, render pages, detect PDF text layer
   │  documents  │
   └──────┬──────┘
          │  Document(pages=[DocumentPage(number, image_bytes, ...)])
          ▼
   ┌─────────────┐   for each enabled provider, in registry order
   │  pipeline   │────────────────┐
   └──────┬──────┘                │
          │                       ▼
          │              ┌─────────────────┐
          │              │  OCRProvider    │   health_check() then process()
          │              └────────┬────────┘
          │                       │  OCRResult(pages, tokens, timings)
          │◀──────────────────────┘
          ▼
   ┌─────────────┐   align lines, score agreement, flag numeric conflicts
   │ comparison  │
   └──────┬──────┘
          ▼
   ┌─────────────┐   choose or merge; never invent
   │   fusion    │
   └──────┬──────┘
          ▼
   PipelineResult ──▶ UI, exporters, metrics
```

`PipelineResult` is the single object everything downstream reads. Its
`as_dict()` is the published JSON schema.

---

## The seams

### 1. `OCRProvider` — engines are interchangeable

```python
class OCRProvider(ABC):
    provider_id: str
    provider_name: str
    processing_location: ProcessingLocation

    def process(self, document: Document) -> OCRResult: ...
    def health_check(self) -> HealthStatus: ...
    def get_metadata(self) -> dict[str, Any]: ...
```

The pipeline calls only these three methods. It has no idea whether a provider
is a vision model over HTTP, a native binary or a stub in a test. That is why
`Tesseract` — which has no prompts, no tokens and no sampling parameters — joins
a run through the same path as a VLM, and why the entire test suite can run with
no model installed.

### 2. The registry — the UI never names an engine

```python
register_provider(ProviderSpec(
    provider_id="my_engine",
    display_name="My Engine",
    factory=build_my_engine,          # lazy: called only when a run needs it
    enabled_check=lambda s: s.my_engine.enabled,
))
```

The UI asks the registry what exists and what is enabled. Add a provider and it
appears in the pipeline visualisation, the results tabs, the comparison and the
exports without a line of UI changing.

Factories are lazy on purpose: importing the registry must never import `torch`
or open a socket, because the Settings page has to *list* backends that are
unavailable in order to explain why.

### 3. Backends — one engine, several deployments

Unlimited-OCR ships as a GPU model but is deployed as a server, an in-process
model, or a CLI tool. Binding the provider to one would make the app
undemonstrable everywhere else, so:

```
UnlimitedOCRProvider          policy: page loop, retries, status, metrics
└── UnlimitedBackendBase      mechanism: how one page becomes text
    ├── HttpBackend           vLLM / SGLang / unlimited-ocr-server
    ├── TransformersBackend   in-process, needs CUDA
    ├── CliBackend            infer.py, franken_ocr, a container
    └── OllamaBackend         a local substitute model
```

A new deployment shape is one subclass. The provider does not change.

### 4. Strategy registries — policy is data

Fusion strategies, exporters and document loaders are all dictionaries mapping
an enum to a function:

```python
FUSION_STRATEGIES = {FusionStrategy.LINE_VOTE: fuse_line_vote, ...}
EXPORTERS         = {OutputFormat.JSON: export_json, ...}
LOADERS           = {DocumentKind.PDF: load_pdf_document, ...}
```

Adding a fusion strategy, an export format or a document type is one function
and one entry. DOCX support, for instance, is a loader that returns a
`Document` — nothing downstream notices.

---

## Invariants

These hold across the codebase, and the tests enforce them.

### Unknown metrics are `None`, never `0`

`TokenUsage()` with nothing set reports `total_tokens is None`. Tesseract has no
concept of a token, so its token fields stay unset and the UI renders `N/A`. A
zero would be a fabricated measurement.

### Expected failures are returned, not raised

A provider that cannot reach its runtime returns a failed `OCRResult` carrying a
plain-language message and a remedy. Raising is reserved for bugs — and the
pipeline catches those too, converting them into a failed stage so one broken
engine cannot end a run the user has already waited minutes for.

The practical effect: if Stage 2 dies, Stage 1's transcription is still on
screen, the comparison stage says *why* it was skipped, and fusion still
produces a final result from the one engine that worked.

### Fusion never invents

Three of the four strategies select between existing lines and cannot introduce
text by construction. The LLM strategy is verified afterwards by word
containment — what share of the fused text's words came from an engine — and is
discarded in favour of the deterministic merge if it drifts. Containment
measures invention directly, and it is linear where sequence similarity is
O(n·m).

### Documents stay local and in memory

Nothing is written to disk unless persistence is switched on. Backends that need
a file on disk use a temporary directory removed immediately afterwards. The log
records counts, not content. Each provider declares whether it processes locally
or in the cloud, and the HTTP backend derives that from its endpoint rather than
assuming.

### The UI states what actually ran

Every result carries `model_name` and `backend`. A substitute engine reports
`is_substitute`, and the interface labels it as a substitute everywhere it
appears rather than letting it pass as upstream Unlimited-OCR.

---

## Comparison, in detail

Naive line-by-line diffing fails on OCR output: engines pad table cells
differently and one may emit an extra line, which shifts everything after it.

1. Lines are whitespace-normalised, so cell padding is not a disagreement.
2. `SequenceMatcher` over the *line lists* finds equal runs.
3. Each replaced block is aligned by a weighted longest-common-subsequence that
   pairs lines to maximise total similarity. This is what catches
   `1,325.50` vs `1,325.60` on a row that moved.
4. A pair clearing the similarity threshold is one line transcribed two ways;
   below it, two unrelated lines.
5. Changed pairs whose digits differ are recorded as **numeric conflicts** and
   surfaced first.

Overall agreement is aggregated from the aligned rows rather than computed over
raw characters. Running `SequenceMatcher` across two full transcriptions is
O(n·m) and took ~59 s on a realistic pair — longer than the OCR itself.

---

## Performance notes

- Providers health-check once per run, not once per page. Failing forty pages
  against a server that is down wastes minutes and yields forty identical
  errors.
- `keep_alive` holds a model resident between pages. On a memory-constrained
  machine, setting it to `0` frees the first model before the second loads.
- Pages are downscaled before inference. Vision models tile large images, so a
  6000px scan multiplies latency without improving transcription.
- Comparison and fusion are pure text operations — under 0.1 s against minutes
  of inference.

---

## The UI layer

Streamlit re-executes the whole script on every interaction, which shapes three
decisions:

- **Session state is centralised** in `app/state.py`, so components cannot
  invent colliding keys.
- **Health checks are cached and manual.** Probing costs a round trip per
  engine; without caching, moving a slider would re-probe them all.
- **Uploads are keyed by name and size**, so re-parsing a 40-page PDF does not
  happen on every click.

Cards are `st.container()` plus a marker element, styled through a `:has()`
rule. HTML written with `st.markdown` cannot wrap later widgets — each call
renders into its own element, so an unclosed `<div>` is closed immediately.
Styling targets our own class names and a few stable `data-testid` hooks, never
Streamlit's generated emotion classes, which change between releases.
