# TM 5-692-1 Hybrid RAG System

A hybrid Retrieval-Augmented Generation (RAG) pipeline built over **TM 5-692-1**, a U.S. Army facilities maintenance technical manual — with an integration bridge that connects it to a separately trained **LSTM Remaining Useful Life (RUL)** model, so live equipment-health predictions can automatically retrieve the correct maintenance procedure.

> **Part 2** of a two-part project. Part 1 (not in this repo) trains the LSTM RUL model on N-CMAPSS turbofan degradation data. This repo — `PetroMind-RAG-book` — is the retrieval and generation half, plus the bridge that connects the two.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
  - [1. Document Parsing](#1-document-parsing--parserpy)
  - [2. Hybrid Indexing](#2-hybrid-indexing--databasepy)
  - [3. Hybrid Retrieval & Reranking](#3-hybrid-retrieval--reranking--retrieverpy)
  - [4. Grounded Generation](#4-grounded-generation--mainpy)
  - [5. ML-to-RAG Bridge](#5-ml-to-rag-bridge--ml_to_rag_bridgepy)
- [Configuration Reference](#configuration-reference)
- [Design Decisions & Why They Were Made](#design-decisions--why-they-were-made)
- [Known Limitations](#known-limitations)
- [Glossary](#glossary)

---

## Overview

Facilities maintenance manuals like TM 5-692-1 are long, dense, and structured in ways generic chunking destroys — numbered sections, appendix cross-references, and large multi-page tables of maintenance intervals. This project parses the manual **structurally** (by chapter/section/table, not by a fixed character window), indexes it with a **hybrid dense + sparse** vector search, reranks with a **cross-encoder**, and generates answers that are **grounded and page-cited**, refusing rather than guessing when the manual doesn't cover a question.

The **bridge module** (`ml_to_rag_bridge.py`) is what makes this more than a document Q&A tool: it takes a numeric RUL forecast from a trained LSTM model, classifies it into a risk tier, translates that into a natural-language maintenance question, and runs it through the exact same retrieval and generation path — so a sensor prediction and a technician's typed question produce the same kind of trustworthy, cited answer.

### Design goals

- Every answer must be traceable to a specific page of TM 5-692-1 — no unattributed claims.
- Retrieval must recover both paraphrased procedural language *and* exact numeric/frequency tokens (e.g. "3 mos", "500 hrs").
- The system must recognize when it doesn't know an answer rather than fabricate one.
- The same retrieval path serves both a typed question and a question auto-generated from an ML prediction.

---

## Architecture

```
                     ┌───────────────────────────────────────────┐
                     │              TM 5-692-1.pdf                │
                     └───────────────────┬───────────────────────┘
                                          │
                                    1. PARSE
                                  (parser.py)
                     structural chunking by chapter/section/table
                                          │
                                    2. INDEX
                                 (database.py)
                    dense (BGE-768d) + sparse (hashed TF) → Pinecone
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │                                                       │
     typed user question                              ML RUL prediction (bridge)
              │                                                       │
              │                                          ┌────────────┴────────────┐
              │                                          │   ml_to_rag_bridge.py     │
              │                                          │  LSTM → RUL → risk tier   │
              │                                          │   → natural-language      │
              │                                          │        query              │
              │                                          └────────────┬────────────┘
              └───────────────────────────┬───────────────────────────┘
                                          │
                                  3. RETRIEVE
                                (retriever.py)
                        hybrid query → top-30 candidates
                                          │
                                   4. RERANK
                                (retriever.py)
                     cross-encoder scoring → top-5 final chunks
                                          │
                                  5. GENERATE
                                  (main.py)
                  grounded prompt → Groq Llama 3.3 70B → page-cited answer
```

---

## Repository Structure

```
PetroMind-RAG-book/
├── parser.py              # PDF → structured, semantically coherent chunks
├── config.py               # Central configuration (RAGConfig)
├── database.py              # Pinecone hybrid index construction & upload
├── retriever.py              # Hybrid retrieval + cross-encoder reranking
├── main.py                    # Prompt assembly, generation, CLI query loop
├── ml_to_rag_bridge.py          # Part 1 (RUL model) ⇄ Part 2 (RAG) integration
├── requirements.txt               # Python dependencies
└── tm_5_692_1.pdf                   # Source document (not committed — see Setup)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (loaded via `python-dotenv`) with:

```
PINECONE_API_KEY=your-pinecone-key
GROQ_API_KEY=your-groq-key
```

Place the source manual at the path configured in `config.py` (`RAGConfig.PDF_PATH`, default `tm_5_692_1.pdf`).

## Usage

```bash
python main.py
```

On first run, this will:
1. Parse the PDF into chunks (`parser.py`).
2. Connect to Pinecone and create the index if it doesn't exist (`database.py`).
3. Check whether the existing index already matches the current parse; embed and upload only if it doesn't.
4. Run a fixed set of 13 validation queries end-to-end and print each grounded, page-cited answer.

To run the ML-prediction bridge instead of a typed query:

```bash
python ml_to_rag_bridge.py
```

This loads a trained RUL checkpoint, runs inference on a real (or, if unavailable, synthetic) sensor window, and produces one grounded answer combining the live prediction with the retrieved procedure.

---

## Pipeline Stages

### 1. Document Parsing — `parser.py`

Converts the raw PDF into a list of chunk dictionaries suitable for embedding, following the manual's own structural hierarchy rather than a fixed character window.

- **Extraction** — PyMuPDF (`fitz`) pulls text per page; running headers/footers and TOC noise are stripped via `_is_noise()`.
- **Segmentation** — regex identifies structural boundaries while walking line by line:

  | Pattern | Matches | Example |
  |---|---|---|
  | `RE_CHAPTER` | Chapter / appendix headings | `CHAPTER 7.` · `APPENDIX B` |
  | `RE_SECTION` | Numbered sections, incl. split-line headings | `3-1. Purpose` · `B-2. Tool care and usage` |
  | `RE_SECTION_ALONE` | A section number alone (title on next line) | `12-1.` |
  | `RE_TABLE_NEW` | New table titles (excludes `(continued)` reprints) | `Table 3-1. Diesel engine – standby mode` |
  | `RE_SUBPARA` / `RE_NUMITEM` | Narrative subparagraph boundaries | `a. …` · `(1) …` |
  | `RE_TABLE_SUBGROUP` | ALL-CAPS sub-headers inside large tables | `COOLING TOWER` · `FANS` |

- **Noise filtering** — front matter is dropped entirely; chunks that are >35% TOC/list-of-tables lines are discarded; sub-150-character structureless fragments are dropped; Army sign-off boilerplate is trimmed from narrative tails.
- **Chunk sizing:**

  | Parameter | Value | Purpose |
  |---|---|---|
  | `MAX_CHUNK_CHARS` | 1,200 | Ceiling for a single narrative chunk |
  | `MAX_TABLE_CHARS` | 3,000 | Above this, a table is split by equipment sub-group |
  | `SUB_OVERLAP_CHARS` | 120 | Trailing chars carried into the next sub-chunk |
  | Tiny-chunk floor | 150 chars | Below this (no section/table id), dropped as noise |

- **Enrichment** — before embedding, each chunk is prefixed with a structured header (`chapter | section | table | frequency tags`), and short-but-important sections get their title repeated as a keyword boost so brevity doesn't dilute the embedding.

**Output schema per chunk:** `id`, `text` (raw, shown to the LLM), `enriched_text` (sent to the embedding model only), and `metadata` (`chapter`, `section_id`, `section_title`, `table_id`, `table_title`, `page_start`, `page_end`, `label`, `frequency_tags`).

### 2. Hybrid Indexing — `database.py`

`PineconeRAGDatabase` owns the index lifecycle: creation, embedding, upload, and a local cache.

- **Index config:** name `tm-5-692-1-v5`, dimension `768`, metric `dotproduct` (required for hybrid search — the index is auto-recreated if an existing one uses a different metric), serverless on AWS `us-east-1`.
- **Dense vectors** — `BAAI/bge-base-en-v1.5` via `SentenceTransformer`, encoding `enriched_text` (not raw text) so structural context contributes to the embedding.
- **Sparse vectors** — deterministic term-frequency hashing (`zlib.adler32`), so no separate learned vocabulary needs to be trained or shipped. A fixed stopword set filters common words and generic interval units (`mo`, `yr`, `hr`, `shift`, …) that would otherwise dominate frequency counts without adding signal.
- **Upload workflow** — chunks are cached in-memory (`chunk_store`) for fast lookup; upload proceeds in batches of 100; raw text is copied into vector metadata as `text_content` so the reranker doesn't need a second lookup; list-valued metadata is joined to a string and `None` values are stripped (Pinecone rejects nulls).

### 3. Hybrid Retrieval & Reranking — `retriever.py`

`PineconeHybridRetriever` turns a query into the final context set in two stages.

- **Query vectors** — the query is embedded the same way as a chunk; dense is scaled by `α`, sparse by `1 - α`.
- **Parameters:**

  | Parameter | Value | Meaning |
  |---|---|---|
  | `HYBRID_ALPHA` | 0.5 | Equal weighting between dense (semantic) and sparse (exact-term) similarity |
  | `TOP_K_HYBRID` | 30 | Candidates pulled from Pinecone before reranking |
  | `TOP_K_FINAL` | 5 | Chunks kept after reranking, sent to the LLM |

- **Hybrid query** — Pinecone is queried with both `vector` and `sparse_vector`; if sparse support is unavailable, it falls back to dense-only rather than failing.
- **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each `(query, chunk_text)` pair jointly — higher precision than embedding similarity alone, at a cost too high to run over the whole index, hence the two-stage design.
- **Deduplication** — reranked results are deduplicated by chunk id, since overlapping sub-chunks from large tables/narratives could otherwise occupy two of the five final slots for the same section.

### 4. Grounded Generation — `main.py`

- **Prompt anatomy** (`build_prompt`): a persona ("senior US Army facilities maintenance / infrastructure officer"), one context block per retrieved chunk (section, page range, raw text), a grounding rule (answer *only* from context), a citation rule (every statement cites its page, e.g. `(Page 256)`), a formatting rule (numbered checklist when appropriate), and a hard fallback — if zero chunks are retrieved, the prompt only permits the reply *"Technical guidelines unverified within active document boundaries."* This is the system's main defense against hallucinated procedures.
- **Inference** — Groq, `llama-3.3-70b-versatile`, `max_tokens=2048`, single-turn chat completion.
- **Index freshness guard** — before every run, `main()` compares Pinecone's `total_vector_count` against the current parse's chunk count, then probes the first and last chunk IDs. Match → skip upload, reuse index. Mismatch → re-embed and re-upload everything.
- **Validation query set** — 13 fixed domain questions covering tool care, cooling-tower and boiler maintenance, electrical safety, hot-work and vessel-entry permits, generator load testing, and bearing/coolant procedures — spanning both narrative and tabular sections of the manual.

### 5. ML-to-RAG Bridge — `ml_to_rag_bridge.py`

The integration point between Part 1 (trained LSTM RUL model over N-CMAPSS turbofan sensor data) and Part 2 (this RAG system).

1. **Load model** (`load_rul_model`) — reads a training checkpoint containing `sensor_cols` (41 feature names), `mean`/`std` normalization stats, `config` (hidden_dim, num_layers, dropout), and the trained weights; reconstructs and evaluates the model.
2. **Predict RUL** (`predict_rul`) — normalizes a raw `(window_size, n_features)` sensor window with the checkpoint's saved stats, runs it through the LSTM, and classifies risk:

   | Predicted RUL | Risk |
   |---|---|
   | < 20 cycles | **HIGH** |
   | 20–59 cycles | **MEDIUM** |
   | ≥ 60 cycles | **LOW** |

3. **Build query** (`build_maintenance_query`) — the translation layer: maps `(RUL, risk)` to a natural-language question, with a different framing per tier (imminent-failure overhaul procedure for HIGH, monitoring schedule for MEDIUM, routine preventive schedule for LOW). Defaults `equipment_type="turbine engine"` to match what the model was actually trained on — the code explicitly warns against substituting an unrelated component name like "bearing".
4. **Retrieve** — the generated query is passed unchanged into the same `PineconeHybridRetriever.retrieve()` used by `main.py`.
5. **Generate** — the same `build_prompt()` is used, but with an extra context block prepended ahead of the manual excerpts:

   ```
   ML MODEL PREDICTION (from live sensor data):
     Estimated RUL : <value> cycles
     Risk Level    : HIGH / MEDIUM / LOW
     Equipment     : <equipment_type>
   ```

   The combined prompt goes to the same `call_llm()`, so the answer is grounded in both the live prediction and the cited manual text in one generation call.

`run_full_pipeline()` chains all five steps and returns the prediction, generated query, retrieved chunks, and final answer together, so every stage of the reasoning is inspectable. When run standalone, the script locates the Part 1 checkpoint, reuses the Pinecone index if it already matches, and builds a real 30-cycle sensor window from the N-CMAPSS test split (falling back to a random window only if the test data can't be loaded, purely so the pipeline stays exercisable during development).

---

## Configuration Reference

### `config.py` — `RAGConfig`

| Setting | Value |
|---|---|
| `PDF_PATH` | `tm_5_692_1.pdf` |
| `PINECONE_INDEX_NAME` | `tm-5-692-1-v5` |
| `PINECONE_ENV` | `us-east-1` |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HYBRID_ALPHA` | `0.5` |
| `TOP_K_HYBRID` | `30` |
| `TOP_K_FINAL` | `5` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` |
| `LLM_MAX_TOKENS` | `2048` |

`PINECONE_API_KEY` and `GROQ_API_KEY` are read from environment variables via `python-dotenv` and are never hard-coded.

### `requirements.txt`

| Package | Version | Role |
|---|---|---|
| `pymupdf` | 1.24.2 | PDF text extraction |
| `sentence-transformers` | 3.0.1 | Dense embedding + cross-encoder reranking |
| `pinecone-client` | 5.0.1 | Vector index storage & hybrid query |
| `numpy` | 1.26.4 | Reranker score sorting |
| `groq` | >=0.9.0 | LLM inference client |
| `python-dotenv` | >=1.0.0 | Environment variable loading |

---

## Design Decisions & Why They Were Made

This section captures the reasoning embedded in the code — the "what happened" behind each choice, so the decisions aren't lost to future maintainers.

- **Structural parsing over fixed-window chunking.** TM 5-692-1's own numbering (chapters, `X-Y.` sections, lettered appendices) is a better semantic boundary than any fixed character count, so the parser tracks that structure explicitly rather than sliding a window across raw text. A changelog embedded in `parser.py`'s docstring records the fixes that got it there: appendix sections (`B-1`, `B-2`, …) were originally missed by the section regex, causing a real retrieval failure on "tool care" queries; split-line headings (a section number on its own line, title on the next) were silently dropped; TOC/front-matter noise leaked into early chunks; and oversized tables needed sub-group splitting rather than a blind character cut to stay retrievable.
- **Hybrid (dense + sparse) retrieval, not dense-only.** Maintenance manuals are full of short, exact tokens that matter precisely because they're exact — `"3 mos"`, `"500 hrs"`, part numbers. Pure semantic search can blur these; pure keyword search misses paraphrase. The `HYBRID_ALPHA = 0.5` split was chosen to weight both equally rather than favor one.
- **A two-stage retrieve-then-rerank pipeline.** Cross-encoders are far more accurate than embedding similarity but too slow to run over an entire index, so the system pulls a wide net of 30 hybrid candidates first and reranks only those down to 5 — precision where it's affordable, recall where it isn't.
- **A hard refusal path instead of best-effort answering.** When retrieval returns nothing relevant, `main.py` doesn't let the LLM reason from general knowledge — it constrains the prompt to a single fixed refusal string. For a manual governing facilities and safety procedures, a wrong but confident-sounding answer is worse than a refusal.
- **An index freshness check instead of unconditional re-upload.** Re-embedding hundreds of chunks on every run is slow and unnecessary once the index already matches the current parse; comparing vector counts plus a first/last-ID probe is a cheap way to skip that work safely.
- **The bridge reuses the retriever and prompt builder rather than duplicating them.** `ml_to_rag_bridge.py` deliberately calls the *same* `PineconeHybridRetriever` and `build_prompt()` that `main.py` uses, rather than building a parallel path. This keeps exactly one retrieval and grounding implementation to trust, whether the query came from a person or from a model prediction.
- **`equipment_type` defaults to "turbine engine," not something generic.** The Part 1 LSTM was trained specifically on N-CMAPSS turbofan degradation data (HPT/LPT wear patterns), and the bridge code explicitly notes that substituting a different equipment word (e.g. "bearing") would generate a query about a failure mode the model was never trained to predict — a subtle but important correctness detail.

---

## Known Limitations

- Sparse vectors use hashed term frequency rather than a learned sparse model (e.g. SPLADE), so they capture exact-term overlap but not sparse semantic expansion.
- The cross-encoder reranker adds latency; `TOP_K_HYBRID=30` is a deliberate ceiling to keep that bounded, which means a correct chunk ranked below 30 in the hybrid stage can never be recovered.
- The bridge's risk thresholds (`<20`, `<60` cycles) and query templates are fixed and specific to the N-CMAPSS-trained model; they would need re-tuning for a different RUL model or equipment class.
- The validation query set in `main.py` is illustrative, not an automated regression suite — there's no scored accuracy benchmark checked into this repo.

---

## Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — answering by first retrieving relevant source text, then generating a response constrained to it. |
| **Hybrid search** | Combining dense (semantic) and sparse (exact-term) vector similarity in a single query. |
| **Dense vector** | A fixed-length embedding capturing semantic meaning. |
| **Sparse vector** | A high-dimensional, mostly-zero vector representing exact term frequency. |
| **Cross-encoder** | A model that scores a `(query, passage)` pair jointly rather than comparing independent embeddings — higher precision, higher cost. |
| **RUL** | Remaining Useful Life — predicted operating cycles remaining before a component is expected to fail. |
| **N-CMAPSS** | NASA's turbofan engine degradation simulation dataset used to train the Part 1 LSTM model. |
| **Chunk** | A structurally coherent excerpt of the source manual, sized for embedding and retrieval. |


    
<img width="961" height="431" alt="image" src="https://github.com/user-attachments/assets/0e5c64cc-42fe-4d62-b724-37ba192a3f3e" />

<img width="1082" height="410" alt="image" src="https://github.com/user-attachments/assets/d71a6221-bc4f-4759-8156-10b31054b5aa" />

<img width="907" height="586" alt="image" src="https://github.com/user-attachments/assets/406d8f42-45af-4c10-86f7-983b54542d9a" />
