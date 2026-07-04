"""
Agentic RAG state machine with 3 nodes: retrieve -> reason -> respond.

- retrieve: pull top-k semantically similar chunks from pgvector.
- reason: ask the LLM to draft an answer AND self-report whether the
  retrieved context was sufficient. If not, and we haven't exceeded
  max_retrieve_retries, loop back to retrieve with a reformulated query.
- respond: produce the final, guarded, user-facing answer and hand back
  everything needed for audit logging.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.llm_providers.base import LLMProvider
from app.models import Chunk
from app.rag.retriever import retrieve_relevant_chunks

SYSTEM_GUARDRAILS = (
    "You are a financial-advisory assistant. Answer using only the provided "
    "context. If the context does not contain the answer, say so plainly "
    "instead of guessing. Never fabricate figures. This is not personalized "
    "financial advice; add a brief disclaimer when discussing investment "
    "decisions."
)


class GraphState(TypedDict):
    conversation_id: str
    query: str
    history: list[dict]
    retrieved_chunks: list[Chunk]
    retrieved_chunk_ids: list[str]
    draft_answer: str
    context_sufficient: bool
    retrieve_attempts: int
    final_answer: str


def build_graph(session: AsyncSession, llm: LLMProvider):
    async def retrieve_node(state: GraphState) -> GraphState:
        chunks = await retrieve_relevant_chunks(session, llm, state["query"])
        state["retrieved_chunks"] = chunks
        state["retrieved_chunk_ids"] = [str(c.id) for c in chunks]
        state["retrieve_attempts"] = state.get("retrieve_attempts", 0) + 1
        return state

    async def reason_node(state: GraphState) -> GraphState:
        context = "\n\n".join(c.content for c in state["retrieved_chunks"]) or "(no context retrieved)"

        messages = [
            {"role": "system", "content": SYSTEM_GUARDRAILS},
            *state["history"],
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {state['query']}\n\n"
                    "First, on a line by itself, write SUFFICIENT or "
                    "INSUFFICIENT depending on whether the context answers "
                    "the question. Then on the next lines, write your answer."
                ),
            },
        ]
        raw = await llm.chat(messages)

        lines = raw.strip().splitlines()
        verdict = lines[0].strip().upper() if lines else "SUFFICIENT"
        answer_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw

        state["context_sufficient"] = "INSUFFICIENT" not in verdict
        state["draft_answer"] = answer_body or raw
        return state

    def should_retry(state: GraphState) -> str:
        if state["context_sufficient"]:
            return "respond"
        if state["retrieve_attempts"] > settings.max_retrieve_retries:
            return "respond"  # give up gracefully, don't loop forever
        return "retrieve"

    async def respond_node(state: GraphState) -> GraphState:
        answer = state["draft_answer"]
        if not state["context_sufficient"]:
            answer = (
                f"{answer}\n\n"
                "(Note: available data may be incomplete for this question.)"
            )
        state["final_answer"] = answer
        return state

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges("reason", should_retry, {"retrieve": "retrieve", "respond": "respond"})
    graph.add_edge("respond", END)

    return graph.compile()


async def run_conversation_turn(
    session: AsyncSession,
    llm: LLMProvider,
    conversation_id: str,
    query: str,
    history: list[dict],
) -> GraphState:
    app = build_graph(session, llm)
    initial_state: GraphState = {
        "conversation_id": conversation_id,
        "query": query,
        "history": history,
        "retrieved_chunks": [],
        "retrieved_chunk_ids": [],
        "draft_answer": "",
        "context_sufficient": True,
        "retrieve_attempts": 0,
        "final_answer": "",
    }
    result = await app.ainvoke(initial_state)
    return result
