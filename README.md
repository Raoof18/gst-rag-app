# GST Export Assistant — RAG over Indian GST export/LUT regulations

A retrieval-augmented Q&A system for freelancers and service exporters navigating
Indian GST rules (export of services, LUT filing, place-of-supply). Built as a
learning project and portfolio piece.

## Architecture

**Offline indexing** (run locally, not deployed):
manuals/circulars (PDF + HTML) → structure-aware chunking → embed with
`sentence-transformers/all-MiniLM-L6-v2` (384-dim) → stored in Supabase (pgvector)

**Live query path** (deployed as a Vercel serverless function):
```
question
  → query rewriting (translates Hindi/Hinglish, cleans up vague phrasing)
  → hybrid retrieval (pgvector similarity + Postgres full-text search, merged via
    Reciprocal Rank Fusion) run for both the original and rewritten query
  → dedupe + LLM reranking of merged candidates
  → answer generation with citations, grounded strictly in retrieved context
```

## Why this stack

- **Hybrid search** — pure vector similarity misses exact-match terms common in
  regulatory text (form numbers like RFD-11, section numbers). Keyword search
  catches these; RRF combines both rankings without needing to normalize
  incompatible score scales.
- **Query rewriting + dual retrieval** — handles vague or Hinglish queries by
  translating/clarifying before search, while still searching on the original
  phrasing too, to avoid losing anything a good rewrite might drop.
- **LLM reranking** — vector/keyword ranking is a first pass; reranking against
  the full candidate set catches relevant chunks that scored lower on initial
  retrieval but are actually the best match for the specific question.
- **Guardrails in the system prompt** — the assistant is explicitly instructed
  not to state definitive tax liabilities/amounts as fact, and not to add
  details beyond what's in retrieved context, since overconfident answers on a
  tax topic carry real risk.
- **Query-time embedding via HuggingFace's hosted Inference API, not a local
  model** — Vercel serverless functions have strict size limits that a local
  `sentence-transformers`/`torch` install exceeds. The same `all-MiniLM-L6-v2`
  model is called remotely instead, keeping query embeddings in the same vector
  space as the stored chunks (critical — a different embedding model would
  silently break similarity search) while keeping the deployed function small.

## Evaluation

`evaluation/tests.jsonl` — 50 hand-built test questions across categories:
`direct_fact`, `spanning` (cross-document reasoning), `procedural`,
`classification`, `hinglish`, and `out_of_scope` (deliberately unanswerable
questions the system should hedge on rather than answer confidently).

Metrics: MRR / nDCG / keyword coverage for retrieval quality, plus an
LLM-as-judge scoring accuracy/completeness/relevance of generated answers
against reference answers.

## Local development

```bash
cp .env.example .env   # fill in real values
pip install -r requirements.txt
```

## Deployment

See deployment instructions provided alongside this project for the GitHub
push and Vercel setup steps.

## Known limitations (by design, not oversight)

- Corpus reflects a fixed set of documents indexed as of a specific date — not
  a live feed of current GST rules or thresholds.
- Deliberately does not attempt to calculate exact tax liability for a given
  transaction — explains applicable rules/conditions and defers specifics to a
  tax professional.
- New documents are added by the maintainer, not self-serve, as a deliberate
  quality-control choice.
