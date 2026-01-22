"""
Tests for the Speech-to-Text service.
"""

import pytest

from app.models.domain import TranscriptSegment
from app.services.stt import TranscriptBuffer


class TestTranscriptBuffer:
    """Tests for the transcript buffer."""

    def test_empty_buffer(self):
        """Empty buffer should return empty text."""
        buffer = TranscriptBuffer()
        assert buffer.get_current_text() == ""
        assert buffer.word_count == 0

    def test_add_interim_segment(self):
        """Interim segments should not trigger flush."""
        buffer = TranscriptBuffer()
        
        segment = TranscriptSegment(
            text="Hello",
            start_time=0.0,
            end_time=0.5,
            is_final=False,
        )
        
        result = buffer.add_segment(segment)
        
        assert result is None  # No flush for interim
        assert buffer.get_current_text() == ""  # Interim not added to buffer

    def test_add_final_segment(self):
        """Final segments should be added to buffer."""
        buffer = TranscriptBuffer()
        
        segment = TranscriptSegment(
            text="Hello world",
            start_time=0.0,
            end_time=1.0,
            is_final=True,
        )
        
        result = buffer.add_segment(segment)
        
        # Not flushed yet (no sentence boundary)
        assert result is None
        assert buffer.get_current_text() == "Hello world"
        assert buffer.word_count == 2

    def test_sentence_boundary_flush(self):
        """Buffer should flush at sentence boundaries."""
        buffer = TranscriptBuffer(min_words=2)
        
        segment = TranscriptSegment(
            text="This is a complete sentence.",
            start_time=0.0,
            end_time=2.0,
            is_final=True,
        )
        
        result = buffer.add_segment(segment)
        
        assert result == "This is a complete sentence."
        assert buffer.get_current_text() == ""

    def test_pause_detection_flush(self):
        """Buffer should flush after pause."""
        buffer = TranscriptBuffer(pause_threshold_ms=500)
        
        # First segment
        seg1 = TranscriptSegment(
            text="First part",
            start_time=0.0,
            end_time=1.0,
            is_final=True,
        )
        buffer.add_segment(seg1)
        
        # Second segment with gap
        seg2 = TranscriptSegment(
            text="after pause",
            start_time=2.0,  # 1 second gap
            end_time=2.5,
            is_final=True,
        )
        result = buffer.add_segment(seg2)
        
        assert result == "First part"

    def test_manual_flush(self):
        """Manual flush should return all buffered text."""
        buffer = TranscriptBuffer()
        
        seg1 = TranscriptSegment(
            text="Part one",
            start_time=0.0,
            end_time=0.5,
            is_final=True,
        )
        seg2 = TranscriptSegment(
            text="part two",
            start_time=0.5,
            end_time=1.0,
            is_final=True,
        )
        
        buffer.add_segment(seg1)
        buffer.add_segment(seg2)
        
        result = buffer.flush()
        
        assert result == "Part one part two"
        assert buffer.get_current_text() == ""

    def test_min_words_requirement(self):
        """Buffer should not flush below minimum word count."""
        buffer = TranscriptBuffer(min_words=5)
        
        segment = TranscriptSegment(
            text="Hi.",
            start_time=0.0,
            end_time=0.5,
            is_final=True,
        )
        
        result = buffer.add_segment(segment)
        
        # Sentence boundary but below min words
        assert result is None
        assert buffer.word_count == 1

    def test_accumulation(self):
        """Multiple segments should accumulate."""
        buffer = TranscriptBuffer()
        
        segments = [
            TranscriptSegment(text="First", start_time=0.0, end_time=0.5, is_final=True),
            TranscriptSegment(text="second", start_time=0.5, end_time=1.0, is_final=True),
            TranscriptSegment(text="third", start_time=1.0, end_time=1.5, is_final=True),
        ]
        
        for seg in segments:
            buffer.add_segment(seg)
        
        assert buffer.get_current_text() == "First second third"
        assert buffer.word_count == 3
