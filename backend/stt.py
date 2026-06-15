"""
Speech-to-Text using faster-whisper.
Loaded once at startup, reused per request.
"""
import os
import io
import glob
import shutil
import tempfile
import logging
from queue import Queue
from faster_whisper import WhisperModel
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def _configure_ffmpeg():
    """
    Point pydub at ffmpeg/ffprobe explicitly so STT works regardless of how
    the process was launched (a terminal opened before ffmpeg was added to
    PATH won't see it otherwise).

    Resolution order:
      1. FFMPEG_BIN env var (a directory containing ffmpeg.exe), if set
      2. ffmpeg already on PATH
      3. Common winget install location (Gyan.FFmpeg)
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if not ffmpeg:
        candidates = []
        env_dir = os.getenv("FFMPEG_BIN")
        if env_dir:
            candidates.append(env_dir)
        # winget default install path for Gyan.FFmpeg
        local = os.getenv("LOCALAPPDATA", "")
        if local:
            candidates += glob.glob(
                os.path.join(local, "Microsoft", "WinGet", "Packages",
                             "Gyan.FFmpeg_*", "ffmpeg-*", "bin")
            )
        for d in candidates:
            exe = os.path.join(d, "ffmpeg.exe")
            if os.path.isfile(exe):
                ffmpeg = exe
                ffprobe = os.path.join(d, "ffprobe.exe")
                # Also add to PATH so child processes can find it
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                break

    if ffmpeg:
        AudioSegment.converter = ffmpeg
        if ffprobe and os.path.isfile(ffprobe):
            AudioSegment.ffprobe = ffprobe
        logger.info(f"ffmpeg configured: {ffmpeg}")
    else:
        logger.warning(
            "ffmpeg not found. STT will fail. Install it (winget install Gyan.FFmpeg) "
            "or set FFMPEG_BIN to its bin directory."
        )


_configure_ffmpeg()


def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) based on CUDA availability."""
    try:
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False
    if has_cuda:
        return "cuda", "float16"
    return "cpu", "int8"


# Load model once at module import
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_POOL_SIZE = int(os.getenv("WHISPER_POOL_SIZE", "2"))

_auto_device, _auto_compute = _detect_device()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE") or _auto_device
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE") or _auto_compute

logger.info(
    f"Loading {WHISPER_POOL_SIZE}x Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} "
    f"(compute_type={WHISPER_COMPUTE_TYPE})"
    + (" [auto-detected]" if not os.getenv("WHISPER_DEVICE") else " [env override]")
)
_pool: Queue[WhisperModel] = Queue()
for _ in range(WHISPER_POOL_SIZE):
    _pool.put(WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE))
logger.info(f"Whisper pool ready ({WHISPER_POOL_SIZE} instances).")


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes (webm/ogg/wav format from browser) to text.
    Returns the transcribed text.
    """
    # Browser sends webm/ogg from MediaRecorder. Convert to wav for whisper.
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # Whisper expects mono 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Write to temp file (faster-whisper accepts file paths or numpy arrays)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio.export(tmp.name, format="wav")
        tmp_path = tmp.name

    model = _pool.get()
    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,           # Greedy decoding = faster
            vad_filter=True,        # Skip silence
            vad_parameters={"min_silence_duration_ms": 500},
            language="en",          # Set to None for auto-detect (slower)
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"Transcribed: {text!r}")
        return text
    finally:
        _pool.put(model)
        os.unlink(tmp_path)
