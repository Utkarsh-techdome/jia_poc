"""
Text-to-Speech using Kokoro ONNX.
Generates audio per sentence for streaming-style playback.
"""
import os
import io
import re
import logging
from pathlib import Path
from queue import Queue
import numpy as np
import soundfile as sf
import httpx
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

# Download URLs (official Kokoro ONNX releases)
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_if_missing(url: str, dest: Path):
    """Download model files on first run."""
    if dest.exists():
        return
    logger.info(f"Downloading {url} -> {dest} (this happens once)...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    logger.info(f"Downloaded {dest.name}")


# Ensure models exist
_download_if_missing(MODEL_URL, MODEL_PATH)
_download_if_missing(VOICES_URL, VOICES_PATH)

TTS_POOL_SIZE = int(os.getenv("TTS_POOL_SIZE", "2"))

logger.info(f"Loading {TTS_POOL_SIZE}x Kokoro TTS...")
_pool: Queue[Kokoro] = Queue()
for _ in range(TTS_POOL_SIZE):
    _pool.put(Kokoro(str(MODEL_PATH), str(VOICES_PATH)))
logger.info(f"Kokoro pool ready ({TTS_POOL_SIZE} instances).")


def _clean_for_tts(text: str) -> str:
    """
    Normalize text before phonemization.

    The espeak/phonemizer backend raises 'number of lines in input and output
    must be equal' when the input contains newlines or certain characters that
    desync its internal line counting. Collapsing whitespace to single spaces
    and dropping control chars avoids that.
    """
    # Collapse all whitespace (incl. newlines/tabs) to single spaces
    text = re.sub(r"\s+", " ", text)
    # Drop characters that tend to break espeak line counting
    text = text.replace("*", " ").replace("`", " ").replace("|", " ")
    return text.strip()


def synthesize(text: str) -> bytes:
    """
    Synthesize text to a WAV byte string (mono, 24kHz).
    Returns WAV-formatted bytes ready to send to the browser.

    Never raises: if phonemization or encoding fails, returns b"" so a
    single bad sentence can't take down the whole turn.
    """
    text = _clean_for_tts(text)
    if not text:
        return b""

    kokoro = _pool.get()
    samples = None
    sample_rate = None
    try:
        try:
            samples, sample_rate = kokoro.create(
                text, voice=KOKORO_VOICE, speed=1.0, lang="en-us",
            )
        except Exception as e:
            # Known failure: phonemizer line-count mismatch on odd punctuation.
            # Retry once with a more aggressive cleanup (letters/digits/basic punct).
            logger.warning(f"TTS first attempt failed for {text[:50]!r}: {e}; retrying cleaned.")
            safe = re.sub(r"[^A-Za-z0-9 .,!?'\-]", " ", text)
            safe = re.sub(r"\s+", " ", safe).strip()
            if not safe:
                return b""
            try:
                samples, sample_rate = kokoro.create(
                    safe, voice=KOKORO_VOICE, speed=1.0, lang="en-us",
                )
            except Exception:
                logger.exception(f"TTS gave up on chunk: {text[:50]!r}")
                return b""
    finally:
        _pool.put(kokoro)

    if samples is None or (hasattr(samples, '__len__') and len(samples) == 0):
        logger.warning(f"TTS produced empty samples for: {text[:50]!r}")
        return b""

    # Convert to 16-bit PCM WAV
    try:
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        wav = buf.getvalue()
        if not wav:
            logger.warning(f"sf.write produced empty bytes for: {text[:50]!r}")
        return wav
    except Exception:
        logger.exception(f"WAV encoding failed for: {text[:50]!r}")
        return b""
