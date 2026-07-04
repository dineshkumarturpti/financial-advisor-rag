from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog


async def write_audit_log(
    session: AsyncSession,
    conversation_id: str,
    user_query: str,
    retrieved_chunk_ids: list[str],
    model_response: str,
    llm_provider: str,
    retrieve_attempts: int,
) -> None:
    entry = AuditLog(
        conversation_id=conversation_id,
        user_query=user_query,
        retrieved_chunk_ids=retrieved_chunk_ids,
        model_response=model_response,
        llm_provider=llm_provider,
        retrieve_attempts=retrieve_attempts,
    )
    session.add(entry)
    await session.commit()
