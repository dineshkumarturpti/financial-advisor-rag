from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.database import engine
from app.routers import chat, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure pgvector extension exists (tables are created via scripts/init_db.sql)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    yield


app = FastAPI(
    title="Financial Advisor - Secure RAG",
    description="Agentic RAG system for multi-turn financial conversations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
