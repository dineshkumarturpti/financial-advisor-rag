# Financial Advisor – Secure RAG

An agentic Retrieval-Augmented Generation (RAG) system for multi-turn financial
conversations. Built with a 3-node LangGraph state machine, a fully async
FastAPI stack, pgvector semantic search, SHA-256 document deduplication,
structured audit logging, and pluggable LLM providers (OpenAI or local Ollama).

## Features

- **Agentic 3-node LangGraph state machine** — `retrieve → reason → respond`
  nodes route each turn through semantic retrieval, financial-reasoning
  synthesis, and a final guarded response step (with a re-retrieve loop when
  the reasoning node decides context is insufficient).
- **Multi-turn conversation memory** — conversation state (history + retrieved
  context) is persisted per `conversation_id` so follow-up questions retain
  context.
- **Async FastAPI stack** — all I/O (DB, vector search, LLM calls) is async
  end-to-end.
- **pgvector semantic search** — document chunks are embedded and stored in
  Postgres with the `pgvector` extension; retrieval uses cosine similarity.
- **SHA-256 deduplication** — every ingested document/chunk is hashed before
  embedding so duplicate content is never re-embedded or re-stored.
- **Audit logging** — every query, retrieved-chunk set, and model response is
  written to an `audit_log` table for compliance traceability.
- **Pluggable LLM providers** — swap between OpenAI's API and a local Ollama
  model via a single environment variable, no code changes required.

## Architecture

```
Client
  │
  ▼
FastAPI (async)
  │
  ├── /documents  → ingestion pipeline → SHA-256 dedup → embed → pgvector
  │
  └── /chat       → LangGraph state machine
                       ┌────────────┐
                       │  retrieve  │  pgvector similarity search
                       └─────┬──────┘
                             ▼
                       ┌────────────┐
                       │   reason   │  LLM drafts answer + checks
                       │            │  sufficiency of context
                       └─────┬──────┘
                     insufficient │ sufficient
                        (loop back to retrieve, max 1x)
                             ▼
                       ┌────────────┐
                       │  respond   │  final guarded answer +
                       │            │  audit log write
                       └────────────┘
```

## Project layout

```
app/
  main.py               FastAPI app + startup
  config.py             Settings (env vars)
  database.py           Async SQLAlchemy engine/session + pgvector setup
  models.py             ORM models (Document, Chunk, Conversation, AuditLog)
  schemas.py            Pydantic request/response schemas
  dedup.py              SHA-256 hashing + dedup checks
  audit.py              Audit log writer
  llm_providers/
    base.py             LLMProvider interface
    openai_provider.py  OpenAI implementation
    ollama_provider.py  Ollama implementation
  rag/
    ingestion.py         chunking + embedding + storage
    retriever.py         pgvector similarity search
    graph.py              LangGraph 3-node state machine
  routers/
    documents.py          upload/list documents
    chat.py                chat endpoint
scripts/init_db.sql      Creates pgvector extension + tables
tests/                   Unit tests for dedup + API
```

## Setup

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd financial-advisor-rag
cp .env.example .env
# edit .env: set LLM_PROVIDER=openai|ollama, OPENAI_API_KEY, DATABASE_URL, etc.
```

### 2. Run with Docker Compose (Postgres + pgvector + API)

```bash
docker compose up --build
```

The API will be live at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

### 3. Run locally without Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Postgres with pgvector must be running and reachable at DATABASE_URL
psql "$DATABASE_URL" -f scripts/init_db.sql

uvicorn app.main:app --reload
```

## Usage

### Ingest a document

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "Q1 10-K Summary", "content": "Revenue grew 12%..."}'
```

Re-submitting the same content returns `duplicate: true` and skips
re-embedding.

### Ask a question (multi-turn)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "message": "How did revenue trend last quarter?"}'
```

Send another request with the same `conversation_id` to continue the
conversation — the graph retains prior turns as context.

## Testing

```bash
pytest tests/ -v
```

## Environment variables

See `.env.example`. Key ones:

| Variable          | Description                              |
|-------------------|-------------------------------------------|
| `DATABASE_URL`    | Async Postgres DSN (`postgresql+asyncpg://...`) |
| `LLM_PROVIDER`    | `openai` or `ollama`                     |
| `OPENAI_API_KEY`  | Required if `LLM_PROVIDER=openai`        |
| `OLLAMA_BASE_URL` | Required if `LLM_PROVIDER=ollama`        |
| `EMBEDDING_MODEL` | Embedding model name                     |

## License

MIT
