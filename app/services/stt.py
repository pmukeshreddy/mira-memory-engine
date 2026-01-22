"""
Speech-to-Text service using Deepgram.

Provides real-time streaming transcription with support for
word-level timestamps and speaker diarization.
"""

import asyncio
import base64
import json
from typing import Callable

import aiohttp
import structlog

from app.config import get_settings
from app.models.domain import TranscriptSegment
from app.utils.latency import track_latency

logger = structlog.get_logger(__name__)


class STTService:
    """
    Deepgram Speech-to-Text service for real-time transcription.

    Handles WebSocket connections to Deepgram for streaming audio
    and provides transcription results via callbacks.
    """

    def __init__(self) -> None:
        """Initialize the STT service with configuration."""
        self.settings = get_settings()

    async def transcribe_audio(self, audio_data: bytes) -> TranscriptSegment | None:
        """
        Transcribe a single audio chunk (non-streaming).

        Args:
            audio_data: Raw audio bytes (PCM 16-bit, 16kHz, mono)

        Returns:
            TranscriptSegment with transcription result
        """
        from deepgram import DeepgramClient

        async with track_latency("stt_single"):
            try:
                client = DeepgramClient(
                    api_key=self.settings.deepgram_api_key.get_secret_value()
                )

                # Use the listen API with file source
                response = await client.listen.asyncrest.v("1").transcribe_file(
                    {"buffer": audio_data},
                    {
                        "model": self.settings.stt_model,
                        "language": self.settings.stt_language,
                        "punctuate": self.settings.stt_punctuate,
                        "smart_format": self.settings.stt_smart_format,
                    },
                )

                if response.results and response.results.channels:
                    channel = response.results.channels[0]
                    if channel.alternatives:
                        alt = channel.alternatives[0]
                        return TranscriptSegment(
                            text=alt.transcript,
                            start_time=0.0,
                            end_time=response.metadata.duration if response.metadata else 0.0,
                            confidence=alt.confidence,
                            is_final=True,
                            words=[
                                {
                                    "word": w.word,
                                    "start": w.start,
                                    "end": w.end,
                                    "confidence": w.confidence,
                                }
                                for w in (alt.words or [])
                            ],
                        )
                return None

            except Exception as e:
                logger.error("stt_transcribe_failed", error=str(e))
                raise

    async def create_streaming_session(
        self,
        on_transcript: Callable[[TranscriptSegment], None],
        on_error: Callable[[str], None] | None = None,
    ) -> "StreamingSession":
        """
        Create a streaming transcription session using Deepgram Live API.

        Args:
            on_transcript: Callback for transcript results
            on_error: Callback for errors

        Returns:
            StreamingSession for sending audio
        """
        return StreamingSession(
            settings=self.settings,
            on_transcript=on_transcript,
            on_error=on_error,
        )


class StreamingSession:
    """
    Manages a real-time streaming transcription session with Deepgram.

    Uses aiohttp WebSocket client for better uvloop compatibility.
    """

    def __init__(
        self,
        settings,
        on_transcript: Callable[[TranscriptSegment], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize streaming session."""
        self.settings = settings
        self.on_transcript = on_transcript
        self.on_error = on_error or (lambda e: None)
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_active = False
        self._receive_task = None

    async def start(self) -> None:
        """Start the streaming session by connecting to Deepgram."""
        try:
            api_key = self.settings.deepgram_api_key.get_secret_value()
            if not api_key:
                raise ValueError("DEEPGRAM_API_KEY is not set")

            # Build Deepgram WebSocket URL with enhanced parameters for quality
            params = [
                f"model={self.settings.stt_model}",
                f"language={self.settings.stt_language}",
                f"punctuate={str(self.settings.stt_punctuate).lower()}",
                f"smart_format={str(self.settings.stt_smart_format).lower()}",
                "interim_results=true",  # Get partial results for real-time feedback
                "encoding=linear16",
                "sample_rate=16000",
                "channels=1",
                # Enhanced quality settings
                "endpointing=300",  # Faster endpoint detection (ms)
                "utterance_end_ms=1000",  # Detect end of speech
                "vad_events=true",  # Voice activity detection
                "filler_words=true",  # Capture um, uh, etc.
                "numerals=true",  # Convert numbers to digits
            ]

            url = f"wss://api.deepgram.com/v1/listen?{'&'.join(params)}"

            # Create aiohttp session and connect
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(
                url,
                headers={"Authorization": f"Token {api_key}"},
                heartbeat=20,
            )

            self._is_active = True

            # Start receiving transcripts in background
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info("stt_streaming_session_started", model=self.settings.stt_model)

        except Exception as e:
            logger.error("stt_session_start_failed", error=str(e))
            self.on_error(str(e))
            raise

    async def _receive_loop(self) -> None:
        """Background task to receive transcripts from Deepgram."""
        try:
            async for msg in self._ws:
                if not self._is_active:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.info("deepgram_message_received", msg_type=data.get("type"))

                        # Handle different message types
                        if data.get("type") == "Results":
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])

                            if alternatives:
                                alt = alternatives[0]
                                text = alt.get("transcript", "")

                                if text:  # Only emit if there's actual text
                                    logger.info("deepgram_transcript", text=text[:50], is_final=data.get("is_final", False))
                                    segment = TranscriptSegment(
                                        text=text,
                                        start_time=data.get("start", 0.0),
                                        end_time=data.get("start", 0.0) + data.get("duration", 0.0),
                                        confidence=alt.get("confidence", 0.0),
                                        is_final=data.get("is_final", False),
                                        words=[
                                            {
                                                "word": w.get("word", ""),
                                                "start": w.get("start", 0.0),
                                                "end": w.get("end", 0.0),
                                                "confidence": w.get("confidence", 0.0),
                                            }
                                            for w in alt.get("words", [])
                                        ],
                                    )
                                    self.on_transcript(segment)

                        elif data.get("type") == "Metadata":
                            logger.debug("deepgram_metadata", data=data)

                        elif data.get("type") == "Error":
                            error_msg = data.get("message", "Unknown Deepgram error")
                            logger.error("deepgram_error", error=error_msg)
                            self.on_error(error_msg)

                    except json.JSONDecodeError:
                        logger.warning("stt_invalid_json", message=str(msg.data)[:100])

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("stt_ws_error", error=str(self._ws.exception()))
                    self.on_error(str(self._ws.exception()))
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("deepgram_connection_closed")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("stt_receive_error", error=str(e))
            self.on_error(str(e))

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to Deepgram for transcription.

        Args:
            audio_data: Raw audio bytes (PCM 16-bit, 16kHz, mono)
        """
        if not self._is_active or not self._ws:
            raise RuntimeError("Session not active")

        try:
            logger.info("sending_audio_to_deepgram", bytes=len(audio_data))
            await self._ws.send_bytes(audio_data)
        except Exception as e:
            logger.error("stt_send_failed", error=str(e))
            self.on_error(str(e))

    async def send_audio_base64(self, audio_base64: str) -> None:
        """
        Send base64-encoded audio data.

        Args:
            audio_base64: Base64-encoded audio bytes
        """
        audio_data = base64.b64decode(audio_base64)
        await self.send_audio(audio_data)

    async def stop(self) -> None:
        """Stop the streaming session gracefully."""
        if self._is_active:
            try:
                self._is_active = False

                if self._ws and not self._ws.closed:
                    # Send close message to Deepgram
                    try:
                        await self._ws.send_str(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
                    await self._ws.close()

                if self._receive_task:
                    self._receive_task.cancel()
                    try:
                        await self._receive_task
                    except asyncio.CancelledError:
                        pass
                    self._receive_task = None

                if self._session and not self._session.closed:
                    await self._session.close()
                    self._session = None

                logger.info("stt_streaming_session_stopped")

            except Exception as e:
                logger.error("stt_session_stop_failed", error=str(e))

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._is_active


class TranscriptBuffer:
    """
    Buffer for accumulating transcript segments.

    Handles sentence boundary detection and combines partial
    transcripts into complete utterances.
    """

    def __init__(
        self,
        pause_threshold_ms: int = 1500,  # 1.5 second pause triggers flush
        min_words: int = 10,  # Accumulate at least 10 words for context
        max_words: int = 80,  # Force flush at 80 words
    ) -> None:
        """
        Initialize transcript buffer.

        Args:
            pause_threshold_ms: Pause duration to trigger flush
            min_words: Minimum words before allowing flush (more = better context)
            max_words: Maximum words before forcing flush
        """
        self.pause_threshold_ms = pause_threshold_ms
        self.min_words = min_words
        self.max_words = max_words
        self._segments: list[TranscriptSegment] = []
        self._last_end_time: float = 0.0

    def add_segment(self, segment: TranscriptSegment) -> str | None:
        """
        Add a segment to the buffer.

        Accumulates sentences to preserve conversation context.
        Flushes when: pause detected, max words reached, or enough context accumulated.

        Args:
            segment: Transcript segment to add

        Returns:
            Complete text if buffer should be flushed, None otherwise
        """
        # Only process final segments
        if not segment.is_final:
            return None

        self._segments.append(segment)
        word_count = sum(len(s.text.split()) for s in self._segments)

        # Check for pause-based flush (conversation pause)
        if self._last_end_time > 0:
            gap_ms = (segment.start_time - self._last_end_time) * 1000
            if gap_ms > self.pause_threshold_ms and word_count >= self.min_words:
                return self.flush()

        self._last_end_time = segment.end_time

        # Force flush at max words to prevent memory buildup
        if word_count >= self.max_words:
            return self.flush()

        # Flush on sentence boundary if we have enough context
        text = segment.text.strip()
        if text and text[-1] in ".!?" and word_count >= self.min_words:
            # Flush at 15+ words on sentence end for good context
            if word_count >= 15:
                return self.flush()

        return None

    def flush(self) -> str:
        """
        Flush buffer and return combined text.

        Returns:
            Combined text from all buffered segments
        """
        if not self._segments:
            return ""

        text = " ".join(s.text for s in self._segments)
        self._segments.clear()
        self._last_end_time = 0.0

        return text.strip()

    def get_current_text(self) -> str:
        """Get current buffered text without flushing."""
        return " ".join(s.text for s in self._segments).strip()

    @property
    def word_count(self) -> int:
        """Get current word count in buffer."""
        return sum(len(s.text.split()) for s in self._segments)
