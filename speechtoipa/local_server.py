"""Local FastAPI server for offline speech-to-IPA transcription."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .pipeline import load_model, transcribe_audio_to_ipa

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Speech to IPA Local Server",
    description=(
        "Accepts uploaded audio snippets, transcribes them with faster-whisper, and "
        "returns IPA-rich results without contacting external services."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _log_startup() -> None:
    LOGGER.info("Speech to IPA local server initialised")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    """Serve the demo interface bundled with the repository."""

    index_path = ROOT_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")

    return HTMLResponse(index_path.read_text(encoding="utf-8"))


static_dir = ROOT_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.post("/api/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    model_size: str = Form("large-v2"),
    language: Optional[str] = Form(None),
    ipa_language: Optional[str] = Form(None),
    device: str = Form("cpu"),
    compute_type: Optional[str] = Form(None),
    vad_filter: Optional[str] = Form("true"),
) -> dict:
    """Persist the uploaded audio temporarily and return the transcription result."""

    try:
        audio_bytes = await audio.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles response
        raise HTTPException(status_code=400, detail="Unable to read uploaded audio") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio was empty")

    suffix = Path(audio.filename or "recording.wav").suffix or ".wav"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(audio_bytes)

    resolved_compute_type = compute_type or None
    resolved_language = (language or "").strip() or None
    if isinstance(resolved_language, str) and resolved_language.lower() == "auto":
        resolved_language = None
    resolved_ipa_language = (ipa_language or "").strip() or None
    if (
        isinstance(resolved_ipa_language, str)
        and resolved_ipa_language.lower() == "auto"
    ):
        resolved_ipa_language = None

    vad_filter_flag = True
    if isinstance(vad_filter, str):
        value = vad_filter.strip().lower()
        if value in {"false", "0", "no", "off"}:
            vad_filter_flag = False
        elif value in {"true", "1", "yes", "on"}:
            vad_filter_flag = True
    elif isinstance(vad_filter, bool):
        vad_filter_flag = vad_filter

    try:
        load_model(model_size, device=device, compute_type=resolved_compute_type)
        result = transcribe_audio_to_ipa(
            tmp_path,
            model_size=model_size,
            language=resolved_language,
            ipa_language=resolved_ipa_language,
            device=device,
            compute_type=resolved_compute_type,
            vad_filter=vad_filter_flag,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            LOGGER.warning("Failed to delete temporary audio file %s", tmp_path)

    payload = result.to_dict()
    payload.pop("audio_path", None)
    return payload


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_size: str = "large-v2",
    device: str = "cpu",
    compute_type: Optional[str] = None,
) -> None:
    """Warm the requested model and start the FastAPI application."""

    LOGGER.info(
        "Starting local server (model_size=%s, device=%s, compute_type=%s)",
        model_size,
        device,
        compute_type or "auto",
    )

    load_model(model_size, device=device, compute_type=compute_type)

    uvicorn.run(
        "speechtoipa.local_server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Speech to IPA offline server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument(
        "--model-size",
        default="large-v2",
        help="faster-whisper model size to load (e.g. tiny, base, small, medium)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device for faster-whisper (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Optional compute type override for faster-whisper",
    )
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()

