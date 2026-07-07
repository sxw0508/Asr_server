"""
ASR gateway for pure vLLM upstream.

- Upstream (HTTP):    OpenAI-compatible /v1/audio/transcriptions on port 23310.
- Upstream (WS):      vLLM /v1/realtime WebSocket on port 23310 (requires
                      ASR_REALTIME_PATCH=1 when starting upstream).
- This gateway is the only external API (default port 23311).
- Forces generation params for deterministic robot usage.
- WebSocket /ws/asr/upload: true incremental KV-cache decoding via /v1/realtime.
  NOTE: hotwords are injected only for HTTP endpoint; /v1/realtime uses no system
  prompt by default (vLLM limitation).
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
import websockets
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASR_BASE_URL = os.environ.get("ASR_BASE_URL", "http://127.0.0.1:23310").rstrip("/")
ASR_MODEL_DEFAULT = os.environ.get("ASR_MODEL", "qwen3-asr")
ASR_API_KEY = os.environ.get("ASR_API_KEY", "")
ASR_GATEWAY_PORT = int(os.environ.get("ASR_GATEWAY_PORT", "23311"))

FORCED_MAX_TOKENS = int(os.environ.get("ASR_FORCED_MAX_TOKENS", "64"))
FORCED_TEMPERATURE = float(os.environ.get("ASR_FORCED_TEMPERATURE", "0"))

ASR_HOTWORDS_PATH = os.environ.get("ASR_HOTWORDS_PATH", "").strip()
ASR_HOTWORDS_MAX = int(os.environ.get("ASR_HOTWORDS_MAX", "100"))

ASR_TEXT_MARKER = "<asr_text>"
MAX_LOG_RESPONSE_CHARS = int(os.environ.get("ASR_MAX_LOG_RESPONSE_CHARS", "4000"))
STREAM_MAX_BUFFER_BYTES = int(os.environ.get("ASR_STREAM_MAX_BUFFER_BYTES", str(32 * 1024 * 1024)))

# Derive realtime WebSocket URL from HTTP base URL
_REALTIME_WS_URL = (
    ASR_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    + "/v1/realtime"
)

# ---------------------------------------------------------------------------
# Hotword helpers
# ---------------------------------------------------------------------------


def build_hotwords_context(hotwords_list: list[str], max_words: int = 100) -> str:
    selected = hotwords_list[:max_words]
    if not selected:
        return ""
    words_str = "、".join(selected)
    return f"你正在处理一段中文语音转写任务。请特别注意以下专业术语或药品名称：{words_str}。"


def load_hotwords_from_path(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        logger.warning("hotwords file does not exist: %s", path)
        return []

    ext = p.suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            import pandas as pd

            df = pd.read_excel(str(p))
            return df.iloc[:, 0].dropna().astype(str).tolist()
        if ext == ".csv":
            out: list[str] = []
            with p.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and str(row[0]).strip():
                        out.append(str(row[0]).strip())
            return out

        lines: list[str] = []
        with p.open(encoding="utf-8-sig") as f:
            for line in f:
                w = line.strip()
                if w:
                    lines.append(w)
        return lines
    except Exception:
        logger.exception("failed to load hotwords from %s", path)
        return []


def init_hotword_prompt() -> str:
    if not ASR_HOTWORDS_PATH:
        return ""
    words = load_hotwords_from_path(ASR_HOTWORDS_PATH)
    if not words:
        logger.warning("ASR_HOTWORDS_PATH set but no words loaded: %s", ASR_HOTWORDS_PATH)
        return ""
    prompt = build_hotwords_context(words, max_words=ASR_HOTWORDS_MAX)
    logger.info(
        "hotwords loaded: file=%s count=%d max=%d prompt_len=%d",
        ASR_HOTWORDS_PATH,
        len(words),
        ASR_HOTWORDS_MAX,
        len(prompt),
    )
    return prompt


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def plain_text_after_marker(raw: str) -> str:
    """Strip <asr_text> marker prefix if present."""
    if ASR_TEXT_MARKER in raw:
        return raw.split(ASR_TEXT_MARKER, 1)[1].strip()
    return raw.strip()


def postprocess_transcription_json(data: dict[str, Any]) -> dict[str, Any]:
    text = data.get("text")
    if isinstance(text, str):
        return {**data, "text": plain_text_after_marker(text)}
    return data


def log_transcription_response(payload: Any) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    if len(text) > MAX_LOG_RESPONSE_CHARS:
        text = f"{text[:MAX_LOG_RESPONSE_CHARS]}...(truncated)"
    logger.info("transcription response: %s", text)


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
HOTWORD_PROMPT = init_hotword_prompt()

app = FastAPI(title="ASR Gateway (vLLM built-in qwen3_asr)", version="1.0.0")


# ---------------------------------------------------------------------------
# HTTP: non-streaming transcription
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(..., description="audio file"),
    model: str | None = Form(None, description="upstream model name"),
    language: str | None = Form(None, description="optional language code, e.g. zh/en"),
    response_format: str = Form("json"),
) -> dict[str, Any]:
    upstream = f"{ASR_BASE_URL}/v1/audio/transcriptions"
    body = await file.read()
    filename = file.filename or "audio.wav"
    mime = file.content_type or "application/octet-stream"

    headers: dict[str, str] = {}
    if ASR_API_KEY:
        headers["Authorization"] = f"Bearer {ASR_API_KEY}"

    form: dict[str, str] = {
        "model": model or ASR_MODEL_DEFAULT,
        "response_format": response_format,
        "stream": "false",
        "max_tokens": str(FORCED_MAX_TOKENS),
        "max_completion_tokens": str(FORCED_MAX_TOKENS),
        "temperature": str(FORCED_TEMPERATURE),
    }
    if language is not None:
        form["language"] = language
    if HOTWORD_PROMPT:
        form["prompt"] = HOTWORD_PROMPT

    files = {"file": (filename, body, mime)}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.post(upstream, data=form, files=files, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"upstream unreachable: {e}") from e
        local_infer_ms = (time.perf_counter() - t0) * 1000.0

    ct = resp.headers.get("content-type", "")
    if resp.status_code != 200:
        detail = resp.text[:8000] if "json" not in ct.lower() else None
        try:
            detail = resp.json() if detail is None else {"raw": detail}
        except Exception:
            detail = {"raw": resp.text[:8000]}
        raise HTTPException(status_code=resp.status_code, detail=detail)

    if "application/json" in ct:
        payload = resp.json()
        if isinstance(payload, dict):
            out = postprocess_transcription_json(payload)
            out["local_infer_time_cost"] = round(local_infer_ms, 1)
            log_transcription_response(out)
            return out
        log_transcription_response(payload)
        return payload

    raise HTTPException(status_code=500, detail="upstream returned non-json success body")


# ---------------------------------------------------------------------------
# WebSocket: streaming upload → vLLM /v1/realtime incremental decoding
# ---------------------------------------------------------------------------


@app.websocket("/ws/asr/upload")
async def ws_asr_upload(websocket: WebSocket) -> None:
    """
    Streaming-upload ASR with true incremental KV-cache decoding.

    Bridges the client's simple binary-upload protocol to vLLM's /v1/realtime
    WebSocket protocol for genuine streaming decode.

    Prerequisites:
      - Upstream vLLM server must be started with ASR_REALTIME_PATCH=1
        (the default in start_upstream.sh).

    Client protocol:
      ① text JSON: {"type":"start", "sample_rate":16000, "format":"pcm_s16le",
                     "language":"zh"}          — open a session
      ② binary frames: raw PCM16-LE mono bytes at 16 kHz              — stream audio
      ③ text JSON: {"type":"finish"}                                   — end + get result

      Other control commands (text JSON):
        {"type":"commit"}   — query current partial text
        {"type":"reset"}    — abort current session (keep connection)
        {"type":"ping"}     — keepalive

    Server responses (text JSON):
      {"type":"ready",   ...}               — connection established
      {"type":"started", ...}               — session accepted
      {"type":"partial", "text":"...", "delta":"..."}  — incremental result
      {"type":"final",   "text":"...", "total_time_ms":..., "finish_time_ms":...}
      {"type":"pong"}
      {"type":"reset_ok"}
      {"type":"error",   "code":"...", "message":"..."}

    Note: hotwords are NOT injected in realtime mode (vLLM /v1/realtime does not
    accept a system prompt). Use HTTP /v1/audio/transcriptions for hotword support.
    """
    await websocket.accept()

    # Try connecting to upstream realtime endpoint
    try:
        upstream_ws = await websockets.connect(
            _REALTIME_WS_URL,
            open_timeout=10,
            ping_interval=None,       # disable auto-ping; we handle the session ourselves
            max_size=50 * 1024 * 1024,
        )
    except Exception as exc:
        logger.error(
            "ws/asr/upload: cannot connect to upstream realtime %s: %s",
            _REALTIME_WS_URL,
            exc,
        )
        await websocket.send_json({
            "type": "error",
            "code": "upstream_unavailable",
            "message": (
                f"realtime upstream unavailable ({exc}). "
                "Make sure upstream is started with ASR_REALTIME_PATCH=1."
            ),
        })
        await websocket.close(code=1011)
        return

    # Per-session state
    started = False
    language: str | None = None
    sample_rate = 16000
    pcm_format = "pcm_s16le"
    chunk_count = 0
    bytes_received = 0
    accumulated_text = ""
    t_start: float | None = None

    # Synchronisation for transcription.done
    done_event = asyncio.Event()
    done_text_ref: list[str] = [""]

    await websocket.send_json({
        "type": "ready",
        "message": "stream upload ready (vLLM realtime incremental decoding)",
        "decode_enabled": True,
        "max_buffer_bytes": STREAM_MAX_BUFFER_BYTES,
        "required_format": "pcm_s16le",
        "required_sample_rate": 16000,
        "mode": "realtime_incremental",
        "upstream_url": _REALTIME_WS_URL,
    })

    # Background task: receive upstream events and forward deltas to client
    async def upstream_receiver() -> None:
        nonlocal accumulated_text
        try:
            async for raw_msg in upstream_ws:
                try:
                    msg = json.loads(raw_msg)
                except Exception:
                    continue
                msg_type = msg.get("type")

                if msg_type == "session.created":
                    # expected immediately after connect; nothing to do
                    pass

                elif msg_type == "transcription.delta":
                    delta = msg.get("delta", "")
                    delta = plain_text_after_marker(delta)
                    accumulated_text += delta
                    try:
                        await websocket.send_json({
                            "type": "partial",
                            "text": accumulated_text,
                            "delta": delta,
                        })
                    except Exception:
                        pass

                elif msg_type == "transcription.done":
                    final_text = plain_text_after_marker(msg.get("text", ""))
                    done_text_ref[0] = final_text
                    accumulated_text = final_text
                    done_event.set()

                elif msg_type == "error":
                    logger.warning("ws/asr/upload upstream error: %s", msg)
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "code": msg.get("code", "upstream_error"),
                            "message": msg.get("error", "upstream error"),
                        })
                    except Exception:
                        pass
                    done_event.set()  # unblock any waiting finish handler

        except Exception as exc:
            logger.debug("ws/asr/upload upstream_receiver stopped: %s", exc)
            done_event.set()

    recv_task = asyncio.create_task(upstream_receiver())

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            # ── Binary frame: raw PCM audio ──────────────────────────────
            chunk = msg.get("bytes")
            if chunk is not None:
                if not started:
                    await websocket.send_json({
                        "type": "error",
                        "code": "session_not_started",
                        "message": "send {type:start} before sending audio",
                    })
                    continue

                chunk_count += 1
                bytes_received += len(chunk)

                if bytes_received > STREAM_MAX_BUFFER_BYTES:
                    await websocket.send_json({
                        "type": "error",
                        "code": "buffer_overflow",
                        "message": f"accumulated audio exceeds {STREAM_MAX_BUFFER_BYTES} bytes",
                    })
                    await websocket.close(code=1009)
                    break

                # Forward PCM16-LE bytes as base64 to upstream
                b64_audio = base64.b64encode(bytes(chunk)).decode("ascii")
                try:
                    await upstream_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64_audio,
                    }))
                except Exception as exc:
                    logger.warning("ws/asr/upload: send audio to upstream failed: %s", exc)
                    await websocket.send_json({
                        "type": "error",
                        "code": "upstream_send_failed",
                        "message": str(exc),
                    })
                continue

            # ── Text frame: control JSON ──────────────────────────────────
            text_frame = msg.get("text")
            if text_frame is None:
                continue

            try:
                data = json.loads(text_frame)
            except Exception:
                await websocket.send_json({
                    "type": "error",
                    "code": "invalid_json",
                    "message": "text frame must be valid JSON",
                })
                continue

            cmd = str(data.get("type", "")).lower().strip()

            # ── start ────────────────────────────────────────────────────
            if cmd == "start":
                sample_rate = int(data.get("sample_rate", 16000))
                pcm_format = str(data.get("format", "pcm_s16le")).strip().lower()
                language = data.get("language") or None

                if sample_rate != 16000:
                    await websocket.send_json({
                        "type": "error",
                        "code": "unsupported_sample_rate",
                        "message": "only 16000 Hz PCM is supported for streaming",
                    })
                    continue

                # Reset session state
                chunk_count = 0
                bytes_received = 0
                accumulated_text = ""
                done_event.clear()
                done_text_ref[0] = ""
                t_start = time.perf_counter()
                started = True

                # Tell upstream to:
                # 1. Validate model via session.update
                # 2. Start generation task via non-final commit
                #    (generation task will consume audio as it arrives)
                try:
                    await upstream_ws.send(json.dumps({
                        "type": "session.update",
                        "model": ASR_MODEL_DEFAULT,
                    }))
                    await upstream_ws.send(json.dumps({
                        "type": "input_audio_buffer.commit",
                        "final": False,
                    }))
                except Exception as exc:
                    logger.warning("ws/asr/upload: upstream session init failed: %s", exc)
                    await websocket.send_json({
                        "type": "error",
                        "code": "upstream_init_failed",
                        "message": str(exc),
                    })
                    started = False
                    continue

                await websocket.send_json({
                    "type": "started",
                    "sample_rate": sample_rate,
                    "format": pcm_format,
                    "language": language,
                    "decode_enabled": True,
                    "mode": "realtime_incremental",
                })

            # ── commit: query partial state ───────────────────────────────
            elif cmd == "commit":
                await websocket.send_json({
                    "type": "partial",
                    "text": accumulated_text,
                    "chunks": chunk_count,
                    "bytes_received": bytes_received,
                })

            # ── finish / stop / end ───────────────────────────────────────
            elif cmd in ("finish", "stop", "end"):
                t0_finish = time.perf_counter()

                # Tell upstream the audio stream is complete
                try:
                    await upstream_ws.send(json.dumps({
                        "type": "input_audio_buffer.commit",
                        "final": True,
                    }))
                except Exception as exc:
                    logger.warning("ws/asr/upload: final commit to upstream failed: %s", exc)

                # Wait for transcription.done from upstream
                try:
                    await asyncio.wait_for(done_event.wait(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning("ws/asr/upload: timed out waiting for transcription.done")

                total_ms = (time.perf_counter() - t_start) * 1000.0 if t_start else 0.0
                finish_ms = (time.perf_counter() - t0_finish) * 1000.0
                final_text = done_text_ref[0] or accumulated_text

                logger.info(
                    "ws/asr/upload final: bytes=%d chunks=%d total_ms=%.1f finish_ms=%.1f text=%r",
                    bytes_received,
                    chunk_count,
                    total_ms,
                    finish_ms,
                    final_text[:120],
                )

                await websocket.send_json({
                    "type": "final",
                    "text": final_text,
                    "language": language or "",
                    "chunks": chunk_count,
                    "bytes_received": bytes_received,
                    "total_time_ms": round(total_ms, 1),
                    "finish_time_ms": round(finish_ms, 1),
                })
                await websocket.close(code=1000)
                break

            # ── reset ─────────────────────────────────────────────────────
            elif cmd == "reset":
                started = False
                chunk_count = 0
                bytes_received = 0
                accumulated_text = ""
                done_event.clear()
                done_text_ref[0] = ""
                t_start = None
                await websocket.send_json({"type": "reset_ok"})

            # ── ping ──────────────────────────────────────────────────────
            elif cmd == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "code": "unknown_command",
                    "message": f"unsupported command type: {cmd!r}",
                })

    except WebSocketDisconnect:
        logger.info("ws/asr/upload: client disconnected")
    except Exception as exc:
        logger.exception("ws/asr/upload: unexpected error: %s", exc)
    finally:
        recv_task.cancel()
        try:
            await upstream_ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("asr_gateway_fastapi:app", host="0.0.0.0", port=ASR_GATEWAY_PORT, reload=False)
