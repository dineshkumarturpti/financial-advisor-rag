from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str
    content: str


class DocumentOut(BaseModel):
    id: str
    title: str
    duplicate: bool
    chunks_created: int


class ChatIn(BaseModel):
    conversation_id: str = Field(..., description="Stable ID to persist multi-turn context")
    message: str


class ChatOut(BaseModel):
    conversation_id: str
    response: str
    retrieved_chunk_ids: list[str]
    retrieve_attempts: int
