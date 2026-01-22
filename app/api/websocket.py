"""
WebSocket handlers for real-time streaming.

Provides WebSocket endpoints for:
- Audio streaming and transcription
- Query streaming with real-time responses
"""

import asyncio
import base64
import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.memory import MemoryPipeline
from app.models.schemas import (
    WSAudioMessage,
    WSQueryMessage,
    WSResponseMessage,
    WSTranscriptMessage,
)
from app.services.stt import STTService, TranscriptBuffer

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """
    Manages WebSocket connections.

    Tracks active connections and provides broadcast capabilities.
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and track a new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("websocket_disconnected", total=len(self.active_connections))

    async def send_json(self, websocket: WebSocket, data: dict[str, Any]) -> bool:
        """Send JSON data to a specific connection. Returns True if successful."""
        try:
            await websocket.send_json(data)
            return True
        except Exception as e:
            logger.error("websocket_send_failed", error=str(e), exc_type=type(e).__name__, data_type=data.get("type"))
            return False

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast JSON data to all connections."""
        for connection in self.active_connections:
            await self.send_json(connection, data)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/audio")
async def audio_stream_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for audio streaming and transcription.

    Protocol:
    - Client sends: {"type": "audio", "data": "<base64>", "sample_rate": 16000}
    - Server sends: {"type": "transcript", "text": "...", "is_final": true/false}

    Audio should be PCM 16-bit, mono, 16kHz.
    """
    await manager.connect(websocket)

    # Get services from app state
    try:
        vector_db = get_vector_db(websocket)
        embedding_service = get_embedding_service(websocket)
        llm_service = get_llm_service(websocket)

        pipeline = MemoryPipeline(
            vector_db=vector_db,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": f"Service initialization failed: {str(e)}",
        })
        await websocket.close()
        return

    # Initialize STT service
    stt_service = STTService()
    transcript_buffer = TranscriptBuffer()

    # Callback for transcripts
    async def on_transcript(segment):
        """Handle incoming transcript segments."""
        logger.info("transcript_received", text=segment.text[:50] if segment.text else "", is_final=segment.is_final)
        # Send transcript to client
        await manager.send_json(
            websocket,
            WSTranscriptMessage(
                text=segment.text,
                is_final=segment.is_final,
                confidence=segment.confidence,
                timestamp=segment.start_time,
            ).model_dump(),
        )

        # Buffer and potentially ingest
        if segment.is_final:
            complete_text = transcript_buffer.add_segment(segment)
            if complete_text:
                # Ingest into memory
                result = await pipeline.ingest(
                    text=complete_text,
                    source="voice",
                )

                # Notify client of ingestion
                await manager.send_json(
                    websocket,
                    {
                        "type": "ingested",
                        "chunks": result.chunks_created,
                        "memory_ids": result.memory_ids,
                    },
                )

    # Sync callback wrapper
    def on_transcript_sync(segment):
        """Sync wrapper for async callback."""
        asyncio.create_task(on_transcript(segment))

    def on_error(error: str):
        """Handle STT errors."""
        asyncio.create_task(
            manager.send_json(
                websocket,
                {"type": "error", "error": error},
            )
        )

    # Create streaming session
    session = await stt_service.create_streaming_session(
        on_transcript=on_transcript_sync,
        on_error=on_error,
    )

    try:
        # Start the session
        await session.start()

        # Send ready message - check if it succeeded
        ready_sent = await manager.send_json(
            websocket,
            {"type": "ready", "message": "Audio streaming ready"},
        )
        if not ready_sent:
            logger.warning("ready_message_failed_client_disconnected")
            return
        
        logger.info("ready_message_sent_waiting_for_audio")

        # Process incoming messages
        while True:
            try:
                data = await websocket.receive_json()

                if data.get("type") == "audio":
                    # Validate and send audio
                    try:
                        msg = WSAudioMessage(**data)
                        logger.info("audio_chunk_received", data_length=len(msg.data))
                        await session.send_audio_base64(msg.data)
                    except ValidationError as e:
                        await manager.send_json(
                            websocket,
                            {"type": "error", "error": f"Invalid audio message: {e}"},
                        )

                elif data.get("type") == "stop":
                    # Stop streaming
                    break

                elif data.get("type") == "flush":
                    # Force flush transcript buffer
                    remaining = transcript_buffer.flush()
                    if remaining:
                        await pipeline.ingest(text=remaining, source="voice")

            except json.JSONDecodeError:
                await manager.send_json(
                    websocket,
                    {"type": "error", "error": "Invalid JSON"},
                )

    except WebSocketDisconnect:
        logger.info("audio_websocket_disconnected")

    finally:
        # Cleanup
        await session.stop()

        # Flush any remaining transcript
        remaining = transcript_buffer.flush()
        if remaining:
            await pipeline.ingest(text=remaining, source="voice")

        manager.disconnect(websocket)


@router.websocket("/ws/query")
async def query_stream_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming query responses.

    Protocol:
    - Client sends: {"type": "query", "query": "...", "top_k": 5}
    - Server sends:
        - {"type": "context", "memories": [...]}
        - {"type": "response", "content": "...", "token_index": 0}
        - {"type": "done"}

    Supports multiple queries per connection.
    """
    await manager.connect(websocket)

    # Get services
    try:
        vector_db = get_vector_db(websocket)
        embedding_service = get_embedding_service(websocket)
        llm_service = get_llm_service(websocket)

        pipeline = MemoryPipeline(
            vector_db=vector_db,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": f"Service initialization failed: {str(e)}",
        })
        await websocket.close()
        return

    # Send ready message
    await manager.send_json(
        websocket,
        {"type": "ready", "message": "Query streaming ready"},
    )

    try:
        while True:
            # Receive query
            data = await websocket.receive_json()

            if data.get("type") != "query":
                await manager.send_json(
                    websocket,
                    {"type": "error", "error": "Expected query message"},
                )
                continue

            try:
                query_msg = WSQueryMessage(**data)
            except ValidationError as e:
                await manager.send_json(
                    websocket,
                    {"type": "error", "error": f"Invalid query: {e}"},
                )
                continue

            logger.info("ws_query_received", query_preview=query_msg.query[:50])

            try:
                # Step 1: Embed and search
                query_embedding = await embedding_service.embed_text(query_msg.query)
                memories = await vector_db.search(
                    embedding=query_embedding,
                    top_k=query_msg.top_k,
                )

                # Send context
                await manager.send_json(
                    websocket,
                    {
                        "type": "context",
                        "memories": [
                            {
                                "id": m.memory_id,
                                "text": m.text[:200],  # Truncate for WS
                                "score": m.score,
                            }
                            for m in memories
                        ],
                    },
                )

                # Step 2: Stream LLM response
                token_index = 0
                async for token in llm_service.generate_stream(
                    query_msg.query, memories
                ):
                    await manager.send_json(
                        websocket,
                        WSResponseMessage(
                            type="response",
                            content=token,
                            token_index=token_index,
                        ).model_dump(),
                    )
                    token_index += 1

                # Send done
                await manager.send_json(
                    websocket,
                    WSResponseMessage(
                        type="done",
                        is_complete=True,
                    ).model_dump(),
                )

            except Exception as e:
                logger.error("ws_query_failed", error=str(e))
                await manager.send_json(
                    websocket,
                    WSResponseMessage(
                        type="error",
                        error=str(e),
                    ).model_dump(),
                )

    except WebSocketDisconnect:
        logger.info("query_websocket_disconnected")

    finally:
        manager.disconnect(websocket)


# Helper function to get services from websocket
def get_vector_db(websocket: WebSocket):
    """Get vector DB from websocket app state."""
    if not hasattr(websocket.app.state, "vector_db"):
        raise RuntimeError("Vector DB not initialized")
    return websocket.app.state.vector_db


def get_embedding_service(websocket: WebSocket):
    """Get embedding service from websocket app state."""
    if not hasattr(websocket.app.state, "embedding_service"):
        raise RuntimeError("Embedding service not initialized")
    return websocket.app.state.embedding_service


def get_llm_service(websocket: WebSocket):
    """Get LLM service from websocket app state."""
    if not hasattr(websocket.app.state, "llm_service"):
        raise RuntimeError("LLM service not initialized")
    return websocket.app.state.llm_service
