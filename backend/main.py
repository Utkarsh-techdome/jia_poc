"""
FastAPI server orchestrating the AI interview pipeline.

Flow per turn:
  1. Browser records audio, sends as binary WebSocket message
  2. STT transcribes (faster-whisper)
  3. LLM streams response tokens (OpenRouter)
  4. Each completed sentence -> TTS (Kokoro) -> sent as binary to browser
  5. Browser plays audio chunks in order, animates avatar from amplitude
"""
import os
import re
import json
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load env before importing modules that read it
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import stt
import llm
import tts
from prompts import build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AI Interview POC backend ready.")
    backend = os.getenv("LLM_BACKEND", "openrouter").lower()
    if backend == "llamacpp":
        logger.info(f"LLM backend: llamacpp @ {os.getenv('LLAMA_BASE_URL', 'https://assetid-65.tail55f76c.ts.net/v1')}  model: {os.getenv('LLAMA_MODEL', 'LFM2.5-8B-A1B-Q5_K_M.gguf')}")
    else:
        logger.info(f"LLM backend: openrouter  model: {os.getenv('LLM_MODEL', 'meta-llama/llama-3.3-70b-instruct')}")
    logger.info(f"Whisper: {os.getenv('WHISPER_MODEL')} on {os.getenv('WHISPER_DEVICE')}")
    logger.info(f"Kokoro voice: {os.getenv('KOKORO_VOICE')}")
    logger.info("=" * 60)
    yield


app = FastAPI(lifespan=lifespan)

# Allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# Split text into sentences for incremental TTS
SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])$')


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Pull complete sentences out of a streaming buffer.
    Returns (complete_sentences, remaining_buffer).
    """
    parts = SENTENCE_END_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    # All except last are complete
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remaining = parts[-1]
    return complete, remaining


async def _safe_send_text(ws: WebSocket, payload: dict) -> bool:
    """Send JSON, swallowing errors if the client has gone away. Returns ok."""
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError covers 'Cannot call "send" once a close message has been
        # sent' — i.e. the client disconnected mid-turn. Nothing to do.
        return False


async def _safe_send_bytes(ws: WebSocket, data: bytes) -> bool:
    try:
        await ws.send_bytes(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _synthesize_and_send(ws: WebSocket, sentence: str):
    """Synthesize one sentence and send to client. Never raises."""
    if not sentence.strip():
        return
    t0 = time.perf_counter()
    wav_bytes = await asyncio.to_thread(tts.synthesize, sentence)
    if not wav_bytes:
        logger.warning(f"TTS returned empty for: {sentence[:60]!r}")
        return
    dt = (time.perf_counter() - t0) * 1000
    logger.info(f"TTS '{sentence[:40]}...' -> {len(wav_bytes)} bytes in {dt:.0f}ms")

    # Marker first so the frontend knows audio is coming, then the audio.
    if await _safe_send_text(ws, {"type": "tts_chunk_start", "text": sentence}):
        await _safe_send_bytes(ws, wav_bytes)


@app.websocket("/ws")
async def interview_socket(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected.")

    history: list[dict] = []
    session_system_prompt = None
    greeting = "Hi! Thanks for joining today. To start, could you tell me a bit about yourself and your background?"
    pending_audio = None

    # Peek at the first message.  JIA sends {"type":"start"} immediately before
    # any audio so we can pick up the system prompt and greeting it provides.
    # The PoC's own React frontend sends audio straight away, so we time-out
    # after 2 s and fall back to defaults in that case.
    try:
        first_msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
        if first_msg.get("type") != "websocket.disconnect":
            if "text" in first_msg and first_msg["text"]:
                try:
                    ctrl = json.loads(first_msg["text"])
                    if ctrl.get("type") == "start":
                        session_system_prompt = ctrl.get("system_prompt") or None
                        greeting = ctrl.get("greeting") or greeting
                        logger.info(
                            f"Session start received: system_prompt={'yes' if session_system_prompt else 'no'}, "
                            f"greeting={greeting[:60]!r}"
                        )
                except json.JSONDecodeError:
                    pass
            elif "bytes" in first_msg and first_msg["bytes"]:
                pending_audio = first_msg["bytes"]
    except asyncio.TimeoutError:
        logger.info("No start message within 2 s — using defaults")

    # Send initial greeting
    history.append({"role": "assistant", "content": greeting})
    await _safe_send_text(ws, {"type": "ai_text", "text": greeting})
    await _synthesize_and_send(ws, greeting)
    await _safe_send_text(ws, {"type": "turn_end"})

    async def _handle_audio(audio_bytes: bytes):
        logger.info(f"Received {len(audio_bytes)} bytes of user audio")

        t0 = time.perf_counter()
        try:
            user_text = await asyncio.to_thread(stt.transcribe_audio, audio_bytes)
        except Exception as e:
            logger.exception("STT failed")
            await _safe_send_text(ws, {"type": "error", "message": f"STT error: {e}"})
            return
        stt_ms = (time.perf_counter() - t0) * 1000

        if not user_text:
            logger.info("STT returned empty — skipping turn silently")
            return  # don't flood the client with "I didn't catch that"

        await _safe_send_text(ws, {
            "type": "user_text",
            "text": user_text,
            "stt_ms": round(stt_ms),
        })
        history.append({"role": "user", "content": user_text})

        t0 = time.perf_counter()
        first_token_ms = None
        buffer = ""
        full_response = ""
        tts_tasks: list[asyncio.Task] = []

        try:
            async for token in llm.stream_completion(
                build_messages(history[:-1], user_text, system_prompt=session_system_prompt)
            ):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM TTFT: {first_token_ms:.0f}ms")

                buffer += token
                full_response += token

                sentences, buffer = _split_sentences(buffer)
                for sent in sentences:
                    task = asyncio.create_task(_synthesize_and_send(ws, sent))
                    tts_tasks.append(task)

            if buffer.strip():
                task = asyncio.create_task(_synthesize_and_send(ws, buffer.strip()))
                tts_tasks.append(task)

        except Exception as e:
            logger.exception("LLM failed")
            await _safe_send_text(ws, {"type": "error", "message": f"LLM error: {e}"})
            return

        # Wait for all in-flight TTS sends to complete before moving on.
        if tts_tasks:
            results = await asyncio.gather(*tts_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"TTS task raised: {r!r}")

        history.append({"role": "assistant", "content": full_response})
        logger.info(f"AI: {full_response!r}")
        await _safe_send_text(ws, {"type": "ai_text_final", "text": full_response})
        await _safe_send_text(ws, {"type": "turn_end"})

    # Process any audio that arrived before the greeting (PoC frontend only)
    if pending_audio:
        await _handle_audio(pending_audio)

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                await _handle_audio(msg["bytes"])

            elif "text" in msg and msg["text"] is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if ctrl.get("type") == "reset":
                    history.clear()
                    history.append({"role": "assistant", "content": greeting})
                    await _safe_send_text(ws, {"type": "ai_text", "text": greeting})
                    await _synthesize_and_send(ws, greeting)
                    await _safe_send_text(ws, {"type": "turn_end"})
                elif ctrl.get("type") == "start":
                    # Late-arriving start — update the system prompt for next turn
                    session_system_prompt = ctrl.get("system_prompt") or session_system_prompt

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.exception(f"Unhandled error in websocket handler: {e}")
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
