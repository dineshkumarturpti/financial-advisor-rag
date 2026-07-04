from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.config import settings
from app.database import get_session
from app.llm_providers import get_llm_provider
from app.models import Conversation
from app.rag.graph import run_conversation_turn
from app.schemas import ChatIn, ChatOut

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatOut)
async def chat(payload: ChatIn, session: AsyncSession = Depends(get_session)):
    llm = get_llm_provider()

    conversation = await session.scalar(
        select(Conversation).where(Conversation.id == payload.conversation_id)
    )
    if conversation is None:
        conversation = Conversation(id=payload.conversation_id, history=[])
        session.add(conversation)
        await session.flush()

    history = list(conversation.history or [])

    result = await run_conversation_turn(
        session=session,
        llm=llm,
        conversation_id=payload.conversation_id,
        query=payload.message,
        history=history,
    )

    history.append({"role": "user", "content": payload.message})
    history.append({"role": "assistant", "content": result["final_answer"]})
    conversation.history = history
    await session.commit()

    await write_audit_log(
        session=session,
        conversation_id=payload.conversation_id,
        user_query=payload.message,
        retrieved_chunk_ids=result["retrieved_chunk_ids"],
        model_response=result["final_answer"],
        llm_provider=settings.llm_provider,
        retrieve_attempts=result["retrieve_attempts"],
    )

    return ChatOut(
        conversation_id=payload.conversation_id,
        response=result["final_answer"],
        retrieved_chunk_ids=result["retrieved_chunk_ids"],
        retrieve_attempts=result["retrieve_attempts"],
    )
