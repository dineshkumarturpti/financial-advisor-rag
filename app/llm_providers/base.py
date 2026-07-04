from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface so the RAG graph never needs to know which backend
    (OpenAI, Ollama, ...) is actually serving requests."""

    name: str

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        ...
