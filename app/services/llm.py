"""
LLM service using Anthropic Claude.

Provides streaming response generation with context from
retrieved memories for RAG-based question answering.
"""

from typing import AsyncGenerator

import structlog
from anthropic import AsyncAnthropic

from app.config import get_settings
from app.models.domain import RetrievalResult
from app.utils.latency import latency_tracked, track_latency

logger = structlog.get_logger(__name__)

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are Mira, an intelligent memory assistant. Your role is to help users recall and understand information from their stored memories.

## Guidelines

1. **Use the provided context**: Base your answers on the memory context provided. If the context contains relevant information, use it to form your response.

2. **Be accurate**: Only state information that is supported by the context. If something isn't in the context, say so.

3. **Be concise**: Provide clear, focused answers. Avoid unnecessary elaboration.

4. **Handle uncertainty**: If the context doesn't fully answer the question, acknowledge what you can and cannot answer.

5. **Cite sources**: When referencing specific memories, mention the relevant details (like timestamps or topics) if available.

6. **Be conversational**: Respond naturally, as if having a conversation with the user.

## Response Format

- For factual questions: Provide direct answers based on the context
- For exploratory questions: Summarize relevant information and connections
- For unanswerable questions: Explain what information would be needed

Remember: You're helping users access their own memories, so be helpful, accurate, and respectful of their data."""


class LLMService:
    """
    Claude LLM service for RAG response generation.

    Handles context formatting and streaming responses from
    Anthropic's Claude API.
    """

    def __init__(self) -> None:
        """Initialize the LLM service."""
        self.settings = get_settings()
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    def _format_context(self, memories: list[RetrievalResult]) -> str:
        """
        Format retrieved memories into context for the LLM.

        Args:
            memories: List of retrieved memory results

        Returns:
            Formatted context string
        """
        if not memories:
            return "No relevant memories found."

        context_parts = ["## Retrieved Memories\n"]

        for i, memory in enumerate(memories, 1):
            # Format metadata
            meta_parts = []
            if memory.metadata.get("timestamp"):
                meta_parts.append(f"Time: {memory.metadata['timestamp']}")
            if memory.metadata.get("source"):
                meta_parts.append(f"Source: {memory.metadata['source']}")
            if memory.metadata.get("speaker"):
                meta_parts.append(f"Speaker: {memory.metadata['speaker']}")

            meta_str = " | ".join(meta_parts) if meta_parts else "No metadata"

            context_parts.append(
                f"### Memory {i} (Relevance: {memory.score:.0%})\n"
                f"*{meta_str}*\n\n"
                f"{memory.text}\n"
            )

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the full prompt for the LLM.

        Args:
            query: User's question
            context: Formatted context from memories

        Returns:
            Complete prompt string
        """
        return f"""{context}

---

## User Question

{query}

---

Please provide a helpful response based on the memories above. If the memories don't contain relevant information, let the user know."""

    @latency_tracked("llm_generate")
    async def generate(
        self,
        query: str,
        memories: list[RetrievalResult],
    ) -> str:
        """
        Generate a response (non-streaming).

        Args:
            query: User's question
            memories: Retrieved memory results

        Returns:
            Generated response text
        """
        client = self._get_client()

        context = self._format_context(memories)
        prompt = self._build_prompt(query, context)

        try:
            response = await client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract text from response
            result = ""
            for block in response.content:
                if block.type == "text":
                    result += block.text

            logger.info(
                "llm_response_generated",
                query_preview=query[:50],
                response_length=len(result),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return result

        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            raise

    async def generate_stream(
        self,
        query: str,
        memories: list[RetrievalResult],
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Args:
            query: User's question
            memories: Retrieved memory results

        Yields:
            Response tokens as they're generated
        """
        client = self._get_client()

        context = self._format_context(memories)
        prompt = self._build_prompt(query, context)

        try:
            async with track_latency("llm_first_token") as timing:
                first_token_received = False

                async with client.messages.stream(
                    model=self.settings.llm_model,
                    max_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                ) as stream:
                    async for text in stream.text_stream:
                        if not first_token_received:
                            first_token_received = True
                            logger.debug(
                                "llm_first_token",
                                latency_ms=timing.get("duration_ms", 0),
                            )
                        yield text

            logger.info(
                "llm_stream_completed",
                query_preview=query[:50],
            )

        except Exception as e:
            logger.error("llm_stream_failed", error=str(e))
            raise

    async def generate_with_context(
        self,
        query: str,
        memories: list[RetrievalResult],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate response with automatic streaming selection.

        Args:
            query: User's question
            memories: Retrieved memory results
            stream: Whether to stream the response

        Returns:
            Response text or async generator of tokens
        """
        if stream:
            return self.generate_stream(query, memories)
        return await self.generate(query, memories)


class ConversationManager:
    """
    Manages multi-turn conversations with memory context.

    Tracks conversation history and provides appropriate context
    for follow-up questions.
    """

    def __init__(self, llm_service: LLMService, max_history: int = 10) -> None:
        """
        Initialize conversation manager.

        Args:
            llm_service: LLM service instance
            max_history: Maximum conversation turns to keep
        """
        self.llm_service = llm_service
        self.max_history = max_history
        self._history: list[dict[str, str]] = []

    def add_turn(self, query: str, response: str) -> None:
        """Add a conversation turn to history."""
        self._history.append({
            "role": "user",
            "content": query,
        })
        self._history.append({
            "role": "assistant",
            "content": response,
        })

        # Trim history if needed
        if len(self._history) > self.max_history * 2:
            self._history = self._history[-(self.max_history * 2) :]

    async def respond(
        self,
        query: str,
        memories: list[RetrievalResult],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate response with conversation context.

        Args:
            query: User's question
            memories: Retrieved memory results
            stream: Whether to stream

        Returns:
            Response text or async generator
        """
        # For now, use simple generation
        # TODO: Incorporate conversation history
        if stream:
            return self.llm_service.generate_stream(query, memories)

        response = await self.llm_service.generate(query, memories)
        self.add_turn(query, response)
        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self._history.copy()
