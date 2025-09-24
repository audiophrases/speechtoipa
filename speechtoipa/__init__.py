"""Speech-to-IPA prototype package."""

__all__ = ["transcribe_audio_to_ipa", "TranscriptionResult", "SegmentResult"]

from .pipeline import transcribe_audio_to_ipa, TranscriptionResult, SegmentResult
