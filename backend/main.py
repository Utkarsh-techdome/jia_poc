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
    logger.info(f"Whisper: {stt.WHISPER_MODEL} on {stt.WHISPER_DEVICE} (compute_type={stt.WHISPER_COMPUTE_TYPE}, pool={stt.WHISPER_POOL_SIZE})")
    logger.info(f"Kokoro TTS: voice={os.getenv('KOKORO_VOICE', 'af_heart')}, pool={tts.TTS_POOL_SIZE}")
    logger.info("=" * 60)
    yield


app = FastAPI(lifespan=lifespan)

# Allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://crawford-protection-specialists-paso.trycloudflare.com",
    ],
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
    # Run TTS in a thread so we don't block the event loop
    wav_bytes = await asyncio.to_thread(tts.synthesize, sentence)
    if not wav_bytes:
        return  # TTS skipped this chunk (e.g. unspeakable text)
    dt = (time.perf_counter() - t0) * 1000
    logger.info(f"TTS '{sentence[:40]}...' -> {len(wav_bytes)} bytes in {dt:.0f}ms")

    # Marker first so the frontend knows audio is coming, then the audio.
    if await _safe_send_text(ws, {"type": "tts_chunk_start", "text": sentence}):
        await _safe_send_bytes(ws, wav_bytes)


@app.websocket("/ws")
async def interview_socket(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected.")

    # Conversation history (just for this session, in memory)
    history: list[dict] = []

    GREETING = "Hi! Thanks for joining today. To start, could you tell me a bit about yourself and your background?"

    async def send_greeting():
        history.clear()
        history.append({"role": "assistant", "content": GREETING})
        await _safe_send_text(ws, {"type": "ai_text", "text": GREETING})
        await _synthesize_and_send(ws, GREETING)
        await _safe_send_text(ws, {"type": "turn_end"})

    try:
        while True:
            # Wait for a message - could be text (control) or bytes (audio)
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            # Audio chunk from user
            if "bytes" in msg and msg["bytes"] is not None:
                audio_bytes = msg["bytes"]
                logger.info(f"Received {len(audio_bytes)} bytes of user audio")

                # --- STT ---
                t0 = time.perf_counter()
                try:
                    user_text = await asyncio.to_thread(stt.transcribe_audio, audio_bytes)
                except Exception as e:
                    logger.exception("STT failed")
                    await _safe_send_text(ws, {"type": "error", "message": f"STT error: {e}"})
                    continue
                stt_ms = (time.perf_counter() - t0) * 1000

                if not user_text:
                    await _safe_send_text(ws, {
                        "type": "error",
                        "message": "I didn't catch that, could you try again?",
                    })
                    continue

                await _safe_send_text(ws, {
                    "type": "user_text",
                    "text": user_text,
                    "stt_ms": round(stt_ms),
                })
                history.append({"role": "user", "content": user_text})

                # --- LLM streaming + sentence-level TTS ---
                t0 = time.perf_counter()
                first_token_ms = None
                buffer = ""
                full_response = ""
                tts_tasks = []

                try:
                    async for token in llm.stream_completion(build_messages(history[:-1], user_text)):
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000
                            logger.info(f"LLM TTFT: {first_token_ms:.0f}ms")

                        buffer += token
                        full_response += token

                        # Pull out any complete sentences and start TTS on each
                        sentences, buffer = _split_sentences(buffer)
                        for sent in sentences:
                            # Fire TTS but don't await - let it run in background
                            # so the next LLM tokens can keep arriving
                            task = asyncio.create_task(_synthesize_and_send(ws, sent))
                            tts_tasks.append(task)

                    # Anything left in buffer is the last partial sentence
                    if buffer.strip():
                        task = asyncio.create_task(_synthesize_and_send(ws, buffer.strip()))
                        tts_tasks.append(task)

                except Exception as e:
                    logger.exception("LLM failed")
                    await _safe_send_text(ws, {"type": "error", "message": f"LLM error: {e}"})
                    continue

                # Wait for all TTS chunks to finish sending. A chunk task never
                # raises (errors are swallowed inside), but guard anyway.
                if tts_tasks:
                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                history.append({"role": "assistant", "content": full_response})
                logger.info(f"AI: {full_response!r}")

                await _safe_send_text(ws, {"type": "ai_text_final", "text": full_response})
                await _safe_send_text(ws, {"type": "turn_end"})

            # Control messages from frontend (e.g. "reset")
            elif "text" in msg and msg["text"] is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if ctrl.get("type") in ("start", "reset"):
                    await send_greeting()

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
