"""
Pure vLLM 0.19 launcher for Qwen3-ASR (no qwen-asr package dependency).

What this file does:
1) Uses vLLM built-in qwen3_asr model implementation.
2) Optionally patches request prompt into system prompt for ASR hotwords/context.
3) Optionally logs wall-clock latency for each transcription request.
4) Delegates to vLLM CLI "serve".

Usage:
  python asr_serve_vllm_builtin.py serve /path/to/Qwen3-ASR-1.7B \
      --host 0.0.0.0 --port 23310 --served-model-name qwen3-asr \
      --dtype bfloat16
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from typing import Any


def _register_transformers_qwen3_asr_compat() -> None:
    """
    Make transformers recognize qwen3_asr config/tokenizer mapping.

    vLLM can parse Qwen3-ASR config via its own config classes, but when
    transformers AutoTokenizer sees `Qwen3ASRConfig` without a registered
    tokenizer mapping, it raises:
      KeyError: 'Qwen3ASRConfig'
    """
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import Qwen2Tokenizer, Qwen2TokenizerFast
    from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except Exception:
        pass

    try:
        AutoTokenizer.register(
            Qwen3ASRConfig,
            slow_tokenizer_class=Qwen2Tokenizer,
            fast_tokenizer_class=Qwen2TokenizerFast,
        )
    except Exception:
        pass

    print(
        "[asr_serve_vllm_builtin] registered transformers compat for qwen3_asr",
        flush=True,
    )


def _patch_prompt_as_system() -> None:
    """
    Patch built-in qwen3_asr get_generation_prompt:
    - default vLLM ignores request_prompt for Qwen3-ASR transcription.
    - this patch injects it as system message, aligned with context behavior.
    """
    enabled = os.environ.get("ASR_SERVE_PATCH_PROMPT_AS_SYSTEM", "1").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return

    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models.qwen3_asr import (
        Qwen3ASRForConditionalGeneration as VLLMQwen3ASR,
    )
    from vllm.tokenizers import cached_tokenizer_from_config

    @classmethod
    def get_generation_prompt(  # noqa: ANN001
        cls,
        audio,
        model_config,
        stt_config,
        language,
        task_type,
        request_prompt,
        to_language,
    ):
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_placeholder = cls.get_placeholder_str("audio", 0)
        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )
        full_lang_name_to = cls.supported_languages.get(to_language, to_language)

        ctx = (request_prompt or "").strip()
        system_prefix = f"<|im_start|>system\n{ctx}<|im_end|>\n" if ctx else ""

        if to_language is None:
            prompt = (
                f"{system_prefix}"
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"{system_prefix}"
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\nlanguage {full_lang_name_to}<asr_text>"
            )

        return TokensPrompt(
            prompt_token_ids=tokenizer.encode(prompt),
            multi_modal_data={"audio": audio},
        )

    VLLMQwen3ASR.get_generation_prompt = get_generation_prompt
    print(
        "[asr_serve_vllm_builtin] enabled prompt->system patch "
        "(disable: ASR_SERVE_PATCH_PROMPT_AS_SYSTEM=0)",
        flush=True,
    )


def _patch_inference_timing_log() -> None:
    """
    Patch OpenAISpeechToText._create_speech_to_text for timing logs.
    Compatible with vLLM 0.19 path layout.
    """
    enabled = os.environ.get("ASR_SERVE_LOG_INFERENCE_TIME", "1").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return

    from vllm.logger import init_logger

    OpenAISpeechToText = None
    for mod_path in (
        "vllm.entrypoints.openai.speech_to_text.speech_to_text",
        "vllm.entrypoints.openai.speech_to_text",
    ):
        try:
            mod = __import__(mod_path, fromlist=["OpenAISpeechToText"])
            OpenAISpeechToText = getattr(mod, "OpenAISpeechToText", None)
            if OpenAISpeechToText is not None:
                break
        except Exception:
            continue

    if OpenAISpeechToText is None:
        print(
            "[asr_serve_vllm_builtin] OpenAISpeechToText not found; skip ASR_TIMING patch",
            flush=True,
        )
        return

    log = init_logger("vllm.entrypoints.openai.asr_timing")
    _orig = OpenAISpeechToText._create_speech_to_text

    async def _wrapped(self, *args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        out = None
        err: BaseException | None = None

        raw_request = kwargs.get("raw_request")
        audio_data = kwargs.get("audio_data")
        if len(args) >= 3:
            audio_data = args[0]
            raw_request = args[2]

        try:
            out = await _orig(self, *args, **kwargs)
            return out
        except BaseException as e:
            err = e
            raise
        finally:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            nbyte = len(audio_data) if isinstance(audio_data, (bytes, bytearray)) else -1
            is_stream = out is not None and inspect.isasyncgen(out)
            rid = ""
            try:
                md = getattr(raw_request.state, "request_metadata", None)
                rid = getattr(md, "request_id", "") or ""
            except Exception:
                pass

            if err is None:
                log.info(
                    "[ASR_TIMING] wall_ms=%.1f audio_bytes=%d stream=%s request_id=%s",
                    wall_ms,
                    nbyte,
                    is_stream,
                    rid,
                )
            else:
                log.warning(
                    "[ASR_TIMING] wall_ms=%.1f audio_bytes=%d stream=%s request_id=%s err=%s: %s",
                    wall_ms,
                    nbyte,
                    is_stream,
                    rid,
                    type(err).__name__,
                    err,
                )

    OpenAISpeechToText._create_speech_to_text = _wrapped
    print(
        "[asr_serve_vllm_builtin] enabled ASR_TIMING patch "
        "(disable: ASR_SERVE_LOG_INFERENCE_TIME=0)",
        flush=True,
    )


def _patch_register_realtime_model() -> None:
    """
    Register Qwen3ASRRealtimeGeneration for the 'Qwen3ASRForConditionalGeneration'
    architecture so vLLM exposes the /v1/realtime WebSocket endpoint.

    Qwen3ASRRealtimeGeneration is a subclass that adds SupportsRealtime, enabling
    true incremental KV-cache decoding. It uses a different multimodal processor
    (Qwen3ASRRealtimeMultiModalProcessor) optimised for streaming audio chunks.

    NOTE: HTTP /v1/audio/transcriptions still works after this patch.

    Controlled by env var ASR_REALTIME_PATCH (default "1", set to "0" to disable).
    """
    enabled = os.environ.get("ASR_REALTIME_PATCH", "1").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        print(
            "[asr_serve_vllm_builtin] ASR_REALTIME_PATCH=0; "
            "realtime model registration skipped, /v1/realtime will NOT be available",
            flush=True,
        )
        return

    try:
        from vllm import ModelRegistry
        from vllm.model_executor.models.qwen3_asr_realtime import (
            Qwen3ASRRealtimeGeneration,
        )

        ModelRegistry.register_model(
            "Qwen3ASRForConditionalGeneration",
            Qwen3ASRRealtimeGeneration,
        )
        print(
            "[asr_serve_vllm_builtin] registered Qwen3ASRRealtimeGeneration "
            "for arch='Qwen3ASRForConditionalGeneration'; /v1/realtime enabled "
            "(disable: ASR_REALTIME_PATCH=0)",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[asr_serve_vllm_builtin] WARNING: failed to register realtime model: {exc}; "
            "/v1/realtime will NOT be available",
            flush=True,
        )


def main() -> None:
    _register_transformers_qwen3_asr_compat()
    _patch_register_realtime_model()
    _patch_prompt_as_system()
    _patch_inference_timing_log()

    from vllm.entrypoints.cli.main import main as vllm_main

    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")
    vllm_main()


if __name__ == "__main__":
    main()
