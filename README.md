# AI Interview POC

A minimal end-to-end AI voice interview proof-of-concept.

**Stack:** React (Vite) + Python (FastAPI) + Whisper (STT) + OpenRouter (LLM) + Kokoro (TTS)

The user holds a button, speaks, releases. The AI transcribes, generates a streaming
response, synthesizes each sentence with TTS, and an SVG avatar lip-syncs to the
audio amplitude in real time.

---

## Architecture

```
┌─────────── Browser (React) ───────────┐         ┌─────── Backend (FastAPI) ───────┐
│                                        │         │                                  │
│  MediaRecorder ── webm audio ─────────►│  WS    │  faster-whisper (STT)            │
│                                        ├────────┤      │                            │
│  WAV chunks ◄──────── sentence audio ──│        │      ▼                            │
│      │                                 │        │  OpenRouter stream tokens        │
│      ▼                                 │        │      │                            │
│  Queued playback + amplitude analyser  │        │      ▼ (per sentence)            │
│      │                                 │        │  Kokoro TTS → WAV bytes          │
│      ▼                                 │        │                                  │
│  SVG Avatar (mouth from amplitude)     │        └──────────────────────────────────┘
└────────────────────────────────────────┘
```

The key trick: TTS fires **per sentence** as the LLM streams. The user hears
sentence 1 while the LLM is still generating sentence 3. That's how we keep
perceived latency low even with the cascaded pipeline.

---

## Prerequisites

- **Python 3.10+**
- **Node 18+**
- **ffmpeg** (required by `pydub` to decode browser audio)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from https://ffmpeg.org/download.html and add to PATH
- **OpenRouter API key** — sign up free at https://openrouter.ai/keys

GPU is optional. CPU works for the POC (latency ~1.2–1.5s end-to-end).
With a GPU, set `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE_TYPE=float16`.

---

## Setup

### 1. Backend

```bash
cd backend

# Create venv
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and paste your OPENROUTER_API_KEY
```

**First run downloads ~500 MB:** the Whisper `small` model (~250 MB) and the
Kokoro ONNX model + voices (~330 MB total). Subsequent runs use the cache.

```bash
python main.py
```

You should see:
```
[INFO] Loading Whisper model: small on cpu
[INFO] Whisper model loaded.
[INFO] Loading Kokoro TTS...
[INFO] Kokoro loaded.
[INFO] AI Interview POC backend ready.
[INFO] Uvicorn running on http://0.0.0.0:8000
```

### 2. Frontend

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser. Grant microphone permission.

---

## Using it

1. The AI greets you and asks you to introduce yourself.
2. **Press and hold** the "Hold to speak" button while talking.
3. **Release** when done — your audio uploads, transcribes, and the AI responds.
4. Watch the avatar's mouth move while the AI speaks.
5. Hit **Restart** to clear the conversation and start over.

---

## File layout

```
ai-interview-poc/
├── backend/
│   ├── main.py            # FastAPI app + WebSocket orchestration
│   ├── stt.py             # faster-whisper wrapper
│   ├── llm.py             # OpenRouter streaming client
│   ├── tts.py             # Kokoro wrapper + model downloader
│   ├── prompts.py         # Interviewer system prompt
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   ├── styles.css
    │   ├── components/
    │   │   ├── Avatar.jsx       # SVG avatar with mouth animation
    │   │   ├── Controls.jsx     # Push-to-talk button
    │   │   └── Transcript.jsx   # Conversation log
    │   ├── hooks/
    │   │   └── useInterview.js  # WS + state machine
    │   └── utils/
    │       └── audio.js          # Recorder + queued WAV player
    ├── index.html
    ├── package.json
    └── vite.config.js
```

Total: ~900 lines of code.

---

## Latency expectations

Measured on a recent MacBook (CPU only, `small` whisper, llama-3.3-70b via OpenRouter):

| Stage | Time |
|---|---|
| STT (1–3 second utterance) | 250–450 ms |
| LLM TTFT (OpenRouter) | 200–500 ms |
| First sentence ready for TTS | +200–400 ms |
| Kokoro first audio chunk | 200–350 ms |
| **Total to first audio** | **~1.1–1.7 s** |

To hit sub-1s reliably:
- Use GPU for Whisper (`WHISPER_DEVICE=cuda`)
- Use a faster LLM (Groq-hosted Llama, Gemini Flash)
- Switch Kokoro to GPU
- Drop Whisper to `tiny` if accuracy allows

---

## How the streaming pipeline works (the important part)

In `backend/main.py`, the LLM is consumed token-by-token. A regex splits the
running buffer on sentence boundaries (`.`, `!`, `?` followed by whitespace + capital).
Each completed sentence is fired off as an `asyncio.create_task` that calls Kokoro
and `ws.send_bytes`. The LLM keeps streaming while TTS runs in the background.

The frontend (`utils/audio.js`) maintains a queue of `ArrayBuffer` WAV chunks.
Each chunk plays via `AudioBufferSourceNode`. An `AnalyserNode` taps the signal
to compute RMS amplitude every frame, which drives the avatar's mouth via React
state.

---

## Next steps (intentionally out of scope for the POC)

- **VAD-based turn-taking** — replace push-to-talk with `silero-vad` so the user
  doesn't have to click. Detect end-of-speech, auto-submit.
- **Barge-in** — let the user interrupt the AI mid-response. Stop TTS playback,
  clear the queue, start listening.
- **Reconnection logic** — preserve session state in Redis so a dropped WS can
  resume mid-conversation.
- **Concurrent sessions** — current code holds Whisper + Kokoro models in memory
  per process. For multi-user, use a separate worker pool.
- **Real 3D avatar** — swap the SVG component for [TalkingHead.js](https://github.com/met4citizen/TalkingHead)
  + a Ready Player Me avatar. Wire Kokoro's word timings to ARKit blendshapes
  for proper lip-sync.
- **Self-hosted LLM** — replace OpenRouter with a local vLLM serving Qwen 3.5 or
  Llama 4. The `llm.py` interface stays the same.
- **Screen monitoring** — add a second WebRTC stream for screen capture, feed
  periodic frames to a vision model for "what is the candidate doing" follow-ups.

---

## Troubleshooting

**`OPENROUTER_API_KEY is not set`** — copy `.env.example` to `.env` and add your key.

**`Could not load codec` / pydub errors** — ffmpeg is missing. Install it and
make sure it's on your PATH.

**Whisper model download is slow** — it's a one-time download (~250 MB). Cached
in `~/.cache/huggingface/`.

**Kokoro download stuck** — files are ~330 MB total. Check `backend/models/` —
if the download partially completed, delete the folder and restart.

**"Could not connect to backend"** — make sure the backend is running on port
8000 and the frontend is using the Vite dev server (which proxies `/ws`).

**No audio playback** — most browsers require a user gesture before audio.
Click the page once before pressing the talk button.
