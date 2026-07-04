from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.llm_providers.base import LLMProvider
from app.models import Chunk


async def retrieve_relevant_chunks(
    session: AsyncSession,
    llm: LLMProvider,
    query: str,
    top_k: int | None = None,
) -> list[Chunk]:
    """Embed the query and run a cosine-similarity search over pgvector."""
    top_k = top_k or settings.top_k
    query_embedding = await llm.embed(query)

    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
