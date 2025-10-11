"""Core transcription pipeline for converting speech audio to IPA."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from functools import lru_cache

from faster_whisper import WhisperModel
from phonemizer import phonemize

__all__ = [
    "SegmentResult",
    "TranscriptionResult",
    "load_model",
    "transcribe_audio_to_ipa",
]

# Minimal mapping between Whisper language hints and espeak-ng codes.
_ESPEAK_LANG_MAP = {
    "af": "af",
    "ar": "ar",
    "bn": "bn",
    "cs": "cs",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "fi": "fi",
    "fr": "fr-fr",
    "gu": "gu",
    "hi": "hi",
    "hu": "hu",
    "id": "id",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "mr": "mr",
    "nb": "nb",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt-br",
    "pt-br": "pt-br",
    "pt-pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sv": "sv",
    "ta": "ta",
    "te": "te",
    "th": "th",
    "tr": "tr",
    "uk": "uk",
    "ur": "ur",
    "vi": "vi",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "ca": "ca",
}

_PUNCTUATION = ";:,.!?¡¿—…\"«»“”()[]{}"


def _normalise_lang_code(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    return code.lower().strip().replace("_", "-")


def _resolve_espeak_language(language_hint: Optional[str]) -> str:
    """Map Whisper language hints to espeak-ng language codes."""
    if not language_hint:
        return "en-us"

    normalised = _normalise_lang_code(language_hint)
    if not normalised:
        return "en-us"

    if normalised in _ESPEAK_LANG_MAP:
        return _ESPEAK_LANG_MAP[normalised]

    # Try stripping regional subtags (e.g., "pt-br" -> "pt").
    if "-" in normalised:
        base = normalised.split("-", 1)[0]
        if base in _ESPEAK_LANG_MAP:
            return _ESPEAK_LANG_MAP[base]

    # Fall back to the normalised hint itself. espeak-ng is permissive with many aliases.
    return normalised


@dataclass
class SegmentResult:
    """Stores the timing, recognised text, and IPA for a single segment."""

    start: float
    end: float
    text: str
    ipa: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Complete transcription result including segments and metadata."""

    audio_path: Path
    text: str
    ipa: str
    segments: List[SegmentResult]
    language: str
    language_confidence: Optional[float]
    duration: Optional[float]
    model_size: str
    ipa_language: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["audio_path"] = str(self.audio_path)
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _phonemize_text(text: str, ipa_language: str) -> str:
    if not text.strip():
        return ""

    try:
        return phonemize(
            text,
            language=ipa_language,
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
            punctuation_marks=_PUNCTUATION,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to phonemize text. Ensure espeak-ng is installed and the language "
            f"'{ipa_language}' is supported."
        ) from exc


@lru_cache(maxsize=8)
def _load_model(model_size: str, device: str, compute_type: Optional[str]) -> WhisperModel:
    """Load and cache ``WhisperModel`` instances for reuse."""

    resolved_compute_type = compute_type
    if resolved_compute_type is None:
        resolved_compute_type = "int8" if device == "cpu" else "auto"

    return WhisperModel(model_size, device=device, compute_type=resolved_compute_type)


def load_model(
    model_size: str = "large-v2",
    *,
    device: str = "cpu",
    compute_type: Optional[str] = None,
) -> WhisperModel:
    """Public helper to warm and retrieve cached Whisper models."""

    return _load_model(model_size, device, compute_type)


def transcribe_audio_to_ipa(
    audio_path: Path | str,
    *,
    model_size: str = "large-v2",
    language: Optional[str] = None,
    ipa_language: Optional[str] = None,
    device: str = "cpu",
    compute_type: Optional[str] = None,
    vad_filter: bool = True,
) -> TranscriptionResult:
    """Transcribes ``audio_path`` and returns IPA-rich metadata."""

    resolved_audio_path = Path(audio_path)
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    model = _load_model(model_size, device, compute_type)
    # Use the model defaults for decoding parameters such as ``beam_size`` and
    # ``temperature``.  The previous hard-coded greedy configuration traded a
    # little speed for a noticeable drop in recognition quality, especially for
    # short phrases (e.g., "hello" -> "how long").  Relying on the tuned
    # defaults restores beam search with automatic temperature fallback which
    # produces far more faithful transcripts while keeping VAD filtering
    # configurable.
    segments_iter, info = model.transcribe(
        str(resolved_audio_path),
        language=language,
        vad_filter=vad_filter,
    )

    recognised_language = language or info.language or ""
    ipa_code = ipa_language or _resolve_espeak_language(recognised_language)

    segments: List[SegmentResult] = []
    full_text_parts: List[str] = []

    for segment in segments_iter:
        segment_text = segment.text.strip()
        if not segment_text:
            continue
        full_text_parts.append(segment_text)
        segment_ipa = _phonemize_text(segment_text, ipa_code)
        segments.append(
            SegmentResult(
                start=segment.start,
                end=segment.end,
                text=segment_text,
                ipa=segment_ipa,
                avg_logprob=getattr(segment, "avg_logprob", None),
                no_speech_prob=getattr(segment, "no_speech_prob", None),
            )
        )

    full_text = " ".join(full_text_parts).strip()
    full_ipa = _phonemize_text(full_text, ipa_code) if full_text else ""

    return TranscriptionResult(
        audio_path=resolved_audio_path,
        text=full_text,
        ipa=full_ipa,
        segments=segments,
        language=recognised_language,
        language_confidence=getattr(info, "language_probability", None),
        duration=getattr(info, "duration", None),
        model_size=model_size,
        ipa_language=ipa_code,
    )
