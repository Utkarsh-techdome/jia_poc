"""
LLM client with two backends:
  - "openrouter"  — cloud API via OpenRouter (default)
  - "llamacpp"    — local llama.cpp OpenAI-compatible endpoint

Select via LLM_BACKEND env var.
"""
import os
import json
import logging
from typing import AsyncIterator
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter").lower()

# --- OpenRouter config ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# --- llama.cpp config ---
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "https://assetid-65.tail55f76c.ts.net/v1")
LLAMA_MODEL    = os.getenv("LLAMA_MODEL",    "LFM2.5-8B-A1B-Q5_K_M.gguf")

_llama_client = AsyncOpenAI(base_url=LLAMA_BASE_URL, api_key="none")


async def _stream_openrouter(messages: list) -> AsyncIterator[str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "AI Interview POC",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 250,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"OpenRouter error {resp.status_code}: {error_body.decode()}")
                raise RuntimeError(f"OpenRouter returned {resp.status_code}")

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse SSE line: {data_str!r}")


async def _stream_llamacpp(messages: list) -> AsyncIterator[str]:
    # max_tokens must be large enough for the reasoning model to finish its
    # chain-of-thought (reasoning_content) before producing content tokens.
    stream = await _llama_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=4096,
    )
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


async def stream_completion(messages: list) -> AsyncIterator[str]:
    """
    Stream tokens from the configured LLM backend.
    Set LLM_BACKEND=llamacpp to use the local llama.cpp endpoint,
    or LLM_BACKEND=openrouter (default) for the OpenRouter cloud API.
    """
    if LLM_BACKEND == "llamacpp":
        async for token in _stream_llamacpp(messages):
            yield token
    else:
        async for token in _stream_openrouter(messages):
            yield token
