# Performance: what was measured

Every number here was produced by `ocr_fusion/bench` on the documents named,
on the machine named. Nothing is estimated unless it says so.

**Machine.** Intel i5-10310U, 4 cores / 8 threads @ 1.70 GHz, 15.8 GB RAM,
Intel UHD graphics — no CUDA. Ollama drops the integrated GPU by default
(`dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1`), so every
figure below is pure CPU inference. Windows 11, Ollama 0.32.1, balanced power
plan.

**Models.** `qwen2.5vl:3b` (Q4_K_M, 3.2 GB) and `deepseek-ocr:3b` (F16,
6.7 GB — pulled unquantised, which matters below).

**Corpus.** 91 real client documents, 1,666 pages: federal tax returns, bank
statements, insurance schedules, trust deeds, pay stubs, scanned licences.

---

## 1. Where a page's time goes

One A4 page at 150 DPI through `qwen2.5vl:3b`:

| Stage | Time | Share |
|---|---:|---:|
| PDF render (PyMuPDF) | 0.18 s | 0.04% |
| PNG encode | 0.14 s | 0.03% |
| base64 for transport | 0.003 s | 0.00% |
| Model load (once, not per page) | 15.5 s | 3% |
| **Prompt eval — 2,979 image tokens** | **415.9 s** | **84.6%** |
| Generation — 283 tokens at 4.85 tok/s | 58.4 s | 12% |
| **Total** | **491 s** | |

The image is the cost, not the answer. Image decode, resize and re-encode
together are 0.07% of the run, so optimising them would have achieved
nothing measurable.

Of the 2,979 prompt tokens, exactly **265 are the text prompt** — the same in
every run, whatever the image. That is 8.9% of the prompt at 150 DPI and
19.8% at 72 DPI.

---

## 2. What the corpus actually is

| | Pages | Share |
|---|---:|---:|
| Already carry a usable text layer | 1,415 | **84.9%** |
| Genuine scans, need a model | 251 | 15.1% |

By document: 49 fully born-digital, 40 fully scanned, 2 mixed.

Reading a text layer costs ~2 ms. Recognising the same page with a vision
model costs ~491 s — about 250,000× more — and is *less* accurate, because it
replaces a stated fact with a transcription.

The gate that decides this (`documents/textlayer.py`) agrees with a manual
count exactly across all 1,666 pages and rejects nothing borderline.

---

## 3. DPI sweep

`qwen2.5vl:3b` on one real tax-form page, scored against the text layer that
page was authored with.

| DPI | Pixels | Image tokens | Prompt s | Output tokens | Wall s | Word error | Recall | px/token |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 72 | 612×792 | 1,338 | 117.5 | 913 | 282.2 | 0.1010 | 92.0% | 452 |
| 96 | 816×1056 | 1,367 | 146.3 | 963 | 325.5 | 0.0801 | 92.7% | 782 |
| **120** | 1020×1320 | 1,957 | 219.3 | 1,019 | **397.5** | **0.0279** | **99.0%** | 796 |
| 150 | 1275×1650 | 2,979 | 356.3 | 981 | 550.7 | 0.0209 | 99.7% | 775 |

**120 DPI is the default.** It costs 28% less than 150 for 0.7 points of
recall.

Below 120 is a loss, not a trade, for two reasons visible in the table. The
prompt costs a fixed 265 tokens whatever the image, and the model enforces a
minimum pixel budget — px/token is flat at ~780 across 96/120/150 but 452 at
72, which only happens if the 72 DPI render is being padded back up. So 72 DPI
costs nearly the same tokens as 96 for a blurrier page. And blurred text makes
the model ramble: output tokens *rise* as DPI falls.

---

## 4. Before and after, same document

`Trust Deed Documents.pdf`, first 3 pages, all genuine scans — the worst case
for this pipeline, because routing can save nothing.

| | Before | After |
|---|---|---|
| Configuration | every engine, every page, 150 DPI | cascade, routing on, 120 DPI |
| Engines that ran | qwen2.5vl:3b **and** deepseek-ocr:3b | qwen2.5vl:3b only |
| Total, 3 pages | **24 min 28 s** | **12 min 35 s** |
| Per page | 489 s | 252 s |
| Pages / minute | 0.123 | 0.238 |
| Extrapolated to 9 pages | **73 min** | **38 min** |

**1.94× on the case that cannot be routed around.** The measured baseline of
73 minutes for nine scanned pages matches the ~70 minutes reported from the
field, which is the check that the rest of these numbers describe the same
system.

Where routing *can* help, the difference is not 2× but three or four orders of
magnitude:

| Document | Pages | Via a model | Before | After |
|---|---:|---:|---|---|
| 2018 Adeen 1065 — Fed (born-digital) | 65 | 0 | ~8.8 h | **1.02 s** |
| Six tax returns, batched | 309 | 0 | ~42 h | **~3 s** |
| Nazila RIRA Jun–Aug (mixed) | 20 | 3 | ~2.7 h | ~20 min |

("Before" for these is the measured 489 s/page applied to every page, which is
exactly what the old pipeline did.)

---

## 5. Which model should be primary

Same three pages, same 120 DPI, scored against each page's own text layer.

| Page | Model | Wall s | Image tokens | Word error | Recall | Figure recall |
|---|---|---:|---:|---:|---:|---:|
| BofA statement p2 | deepseek-ocr:3b | 142.0 | 426 | 0.030 | 98.1% | 50.0% |
| BofA statement p2 | qwen2.5vl:3b | 443.6 | 1,957 | 0.008 | 99.7% | 88.9% |
| 1065 form p1 | deepseek-ocr:3b | 165.2 | 426 | 0.385 | 91.3% | 66.7% |
| 1065 form p1 | qwen2.5vl:3b | 295.0 | — | 0.810 | 49.6% | 19.9% |
| 1065 form p1 (b) | deepseek-ocr:3b | 88.2 | 426 | 0.771 | 50.5% | 17.7% |
| 1065 form p1 (b) | qwen2.5vl:3b | 266.7 | — | 0.818 | 50.9% | 22.4% |

Two things to read carefully here.

**DeepSeek-OCR is 3.1× faster on the clean page**, and its image cost is flat:
426 tokens regardless of DPI, against Qwen's 1,957 at 120 DPI. It compresses a
page to a fixed visual budget, which is what it was designed to do. On the
scanned deed pages at 150 DPI it averaged 159 s/page against Qwen's 330 s.

**Both models collapse on IRS form pages**, to recall around 50%. That is a
ground-truth artefact as much as a model failure: a fillable form's text layer
includes field labels and hidden text that no render shows, so the answer key
over-counts. Those two rows measure the measurement, not the models, and
should not be used to choose between them.

On the one page where the ground truth is trustworthy, Qwen is clearly more
accurate — and the difference is not a rounding error in a noisy metric. With
only 18 figures on the page the 50% could have been chance, so here are the
figures each model failed to produce:

| Model | Figures missed |
|---|---|
| deepseek-ocr:3b | `0010` `7480` `7076` `31,` `2021` `2` `10` |
| qwen2.5vl:3b | `2` `10` |

`0010 7480 7076` is the **account number**. DeepSeek did not misread it — it
omitted the header line carrying it altogether:

```
truth             FARSHID A MAZLOOM ! Account # 0010 7480 7076 ! July 31, 2021 to August 31, 2021
qwen2.5vl:3b      Account # 0010 7480 7076 | July 31, 2021 to August 31, 2021
deepseek-ocr:3b   (no account line anywhere in the output)
```

Qwen's two misses are the digits of "Page 2 of 10". One model lost the
pagination; the other lost the account the statement is about.

**Recommendation: leave Qwen primary, and keep DeepSeek as the configured
fallback.** If nine genuinely scanned pages must fit inside 25 minutes,
switching the primary to DeepSeek would do it (~15–24 min). But on this
evidence the trade costs account numbers, not merely speed, so it belongs on
document types where identifiers do not matter — never as a silent default.

---

## 6. Memory

With both models resident during the old two-engine run:

| | |
|---|---|
| deepseek-ocr:3b | 7.45 GB |
| qwen2.5vl:3b | 2.90 GB |
| Free physical memory | 1.01 GB |
| Committed | 30.5 GB against 15.8 GB physical |
| **Page file in use** | **8.49 GB** |

A swapping run is slower than no run. This is the second reason the second
engine became a fallback rather than a second full pass, and why the batch
runner refuses to start a document when there is no room for a model.

The cascade run measured 12.1 GB machine-wide in use with one model resident.

---

## 7. Throughput at scale

Projected from the measured per-page rates and the measured 84.9% text-layer
share. Arithmetic, not a second experiment — a scan-heavy batch sits closer to
the pessimistic end.

| Workload | Before | After |
|---|---|---|
| 9-page document, typical mix | 73 min | ~7 min |
| 9-page document, all scanned | 73 min | ~38 min |
| The 1,666-page corpus | 226 h (9.4 days) | 28 h (1.2 days) |
| 1,000 documents | ~104 days | ~13 days |
| 10,000 documents | ~1,037 days | ~127 days |

---

## Reproducing any of this

```bash
python -m ocr_fusion.cli bench ./documents -o report.json
python -m ocr_fusion.cli bench ./documents --no-routing --mode all_engines --dpi 150
python -m ocr_fusion.cli accuracy ./documents --profile balanced
python -m ocr_fusion.cli profiles
```

`accuracy` needs no labelled data: a born-digital page is its own answer key,
so it renders the page, transcribes the image, and scores the result against
the text the file already contained. Figures are scored separately from words.

## Open leads, not yet measured

- Ollama thread count — left at its default; 2/4/6/8 unbenchmarked on 4 cores.
- A quantised DeepSeek-OCR build. 6.7 GB unquantised is what forces the swap.
- Trimming the fixed 265-token prompt.
- Cropping to the ink bounding box before rendering.
- `OLLAMA_IGPU_ENABLE=1` on the Intel UHD — untested, and not to be reported
  as acceleration until it is.
