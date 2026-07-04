@'
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
