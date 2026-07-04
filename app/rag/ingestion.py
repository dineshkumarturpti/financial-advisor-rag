from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dedup import sha256_hash
from app.llm_providers.base import LLMProvider
from app.models import Chunk, Document


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple sliding-window chunker by characters (swap for a
    token-aware splitter in production)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def ingest_document(
    session: AsyncSession,
    llm: LLMProvider,
    title: str,
    content: str,
) -> tuple[Document, bool, int]:
    """Ingest a document: dedup at the document level, then chunk + dedup +
    embed each chunk. Returns (document, was_duplicate, chunks_created)."""

    doc_hash = sha256_hash(content)

    existing = await session.scalar(select(Document).where(Document.content_hash == doc_hash))
    if existing:
        return existing, True, 0

    document = Document(title=title, content_hash=doc_hash)
    session.add(document)
    await session.flush()  # get document.id

    chunks_created = 0
    for raw_chunk in chunk_text(content):
        chunk_hash = sha256_hash(raw_chunk)

        already = await session.scalar(select(Chunk).where(Chunk.chunk_hash == chunk_hash))
        if already:
            continue

        embedding = await llm.embed(raw_chunk)
        chunk = Chunk(
            document_id=document.id,
            chunk_hash=chunk_hash,
            content=raw_chunk,
            embedding=embedding,
        )
        session.add(chunk)
        chunks_created += 1

    await session.commit()
    await session.refresh(document)
    return document, False, chunks_created
