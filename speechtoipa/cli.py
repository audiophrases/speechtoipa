"""Command-line interface for the speech-to-IPA prototype."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .pipeline import TranscriptionResult, transcribe_audio_to_ipa


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio file with faster-whisper and convert the transcript to IPA using espeak-ng."
        )
    )
    parser.add_argument("audio", type=Path, help="Path to the input audio file (any ffmpeg-compatible format).")
    parser.add_argument(
        "--model",
        default="large-v2",
        help="Whisper model size to download/use (tiny, base, small, medium, large-v2, etc.).",
    )
    parser.add_argument(
        "--language",
        help="Override language detection by providing a two-letter language hint understood by Whisper.",
    )
    parser.add_argument(
        "--ipa-language",
        help="Override the espeak-ng language code used for phonemization (defaults to the detected language).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on. Use 'cuda' for NVIDIA GPUs if available.",
    )
    parser.add_argument(
        "--compute-type",
        help="Override faster-whisper compute type (e.g. int8, float16, float32). Auto-selected by default.",
    )
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        help="Disable the built-in voice-activity detector (keep long silences).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the transcription result as JSON (UTF-8).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Do not use ANSI colors in the terminal summary output.",
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Print each segment with timestamps alongside the aggregate IPA transcript.",
    )
    return parser


def _format_summary(result: TranscriptionResult, *, color: bool, segments: bool) -> str:
    def style(text: str, code: str) -> str:
        if not color:
            return text
        return f"\033[{code}m{text}\033[0m"

    if result.language_confidence is None:
        confidence_str = "n/a"
    else:
        confidence_str = f"{result.language_confidence:.3f}"

    lines = [
        style("Speech-to-IPA prototype", "1;36"),
        f"Audio: {result.audio_path}",
        f"Model: {result.model_size}",
        f"Detected language: {result.language or 'unknown'} (confidence={confidence_str})",
        f"IPA backend language: {result.ipa_language}",
        f"Duration: {result.duration:.2f}s" if result.duration is not None else "Duration: n/a",
        "",
        style("Transcript:", "1;37"),
        result.text or "<empty>",
        "",
        style("IPA:", "1;37"),
        result.ipa or "<empty>",
    ]

    if segments and result.segments:
        lines.append("")
        lines.append(style("Segments:", "1;37"))
        for seg in result.segments:
            lines.append(
                f"[{seg.start:6.2f}s – {seg.end:6.2f}s] {seg.text}\n    {style(seg.ipa, '36')}"
            )

    return "\n".join(lines)


def run_cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = transcribe_audio_to_ipa(
        args.audio,
        model_size=args.model,
        language=args.language,
        ipa_language=args.ipa_language,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=not args.disable_vad,
    )

    if args.output:
        args.output.write_text(result.to_json(indent=2), encoding="utf-8")

    summary = _format_summary(result, color=not args.no_color, segments=args.segments)
    print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
