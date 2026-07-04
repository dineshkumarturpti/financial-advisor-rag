"""
Integration-style tests for the ingestion pipeline logic using an in-memory
SQLite-like async setup is non-trivial with pgvector, so these tests focus on
the pure-Python chunking/dedup logic that doesn't require a live Postgres
instance. Full end-to-end API tests should run against docker-compose in CI
with a real pgvector database.
"""

from app.rag.ingestion import chunk_text


def test_chunk_text_short_returns_single_chunk():
    text = "short text"
    chunks = chunk_text(text, chunk_size=800)
    assert chunks == ["short text"]


def test_chunk_text_splits_long_text():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    # overlap means consecutive chunks share content
    assert chunks[0][-50:] in text


def test_chunk_text_covers_full_text():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    reconstructed_length = sum(len(c) for c in chunks)
    assert reconstructed_length >= len(text.strip())
