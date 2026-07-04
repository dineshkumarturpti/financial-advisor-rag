from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.llm_providers import get_llm_provider
from app.rag.ingestion import ingest_document
from app.schemas import DocumentIn, DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut)
async def create_document(payload: DocumentIn, session: AsyncSession = Depends(get_session)):
    llm = get_llm_provider()
    document, duplicate, chunks_created = await ingest_document(
        session, llm, payload.title, payload.content
    )
    return DocumentOut(
        id=str(document.id),
        title=document.title,
        duplicate=duplicate,
        chunks_created=chunks_created,
    )
