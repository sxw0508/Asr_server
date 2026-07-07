"""vLLM Qwen3-ASR 补丁（原生内置或手动插件）。"""

from __future__ import annotations

import importlib
import inspect
import os
import time
from typing import Any

from asr_backend import detect_backend, vllm_version_tuple


def get_qwen3_asr_model_class() -> type:
    mode = detect_backend()
    if mode == "native_vllm":
        from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

        return Qwen3ASRForConditionalGeneration
    if mode == "manual_vllm":
        from qwen3_asr_plugin.vllm_backend import Qwen3ASRForConditionalGeneration

        return Qwen3ASRForConditionalGeneration
    raise RuntimeError("无可用 vLLM Qwen3-ASR 后端，应使用 Transformers 模式")


def patch_prompt_as_system() -> None:
    if os.environ.get("ASR_SERVE_PATCH_PROMPT_AS_SYSTEM", "1") != "1":
        return

    cls = get_qwen3_asr_model_class()
    sig = inspect.signature(cls.get_generation_prompt)
    params = list(sig.parameters)

    if len(params) >= 2 and params[1].name in ("stt_params", "speech_to_text_params"):
        _patch_prompt_v017_style(cls)
    else:
        _patch_prompt_v016_style(cls)

    print(
        "[asr_serve] 已启用补丁：OpenAI prompt → system（关闭：ASR_SERVE_PATCH_PROMPT_AS_SYSTEM=0）",
        flush=True,
    )


def _patch_prompt_v016_style(cls: type) -> None:
    from vllm.tokenizers import cached_tokenizer_from_config

    @classmethod
    def get_generation_prompt(  # type: ignore[no-untyped-def]
        cls_,
        audio,
        model_config,
        stt_config,
        language,
        task_type,
        request_prompt,
        to_language,
    ):
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_placeholder = cls_.get_placeholder_str("audio", 0)
        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )
        full_lang_name_to = cls_.supported_languages.get(to_language, to_language)
        im_end = "<|" + "im_end" + "|>"
        ctx = (request_prompt or "").strip()
        system_prefix = (
            f"<|im_start|>system\n{ctx}{im_end}\n" if ctx else ""
        )
        if to_language is None:
            prompt = (
                f"{system_prefix}"
                f"<|im_start|>user\n{audio_placeholder}{im_end}\n"
                f"<|im_start|>assistant\n"
            )
        else:
            asr_tag = getattr(cls_, "_ASR_TEXT_TAG", "<asr_text>")
            prompt = (
                f"{system_prefix}"
                f"<|im_start|>user\n{audio_placeholder}{im_end}\n"
                f"<|im_start|>assistant\nlanguage {full_lang_name_to}{asr_tag}"
            )
        prompt_token_ids = tokenizer.encode(prompt)
        try:
            from vllm.inputs.data import TokensPrompt

            return TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": audio},
            )
        except ImportError:
            return {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"audio": audio},
            }

    cls.get_generation_prompt = get_generation_prompt  # type: ignore[method-assign]


def _patch_prompt_v017_style(cls: type) -> None:
    orig = cls.get_generation_prompt

    @classmethod
    def get_generation_prompt(cls_, stt_params):  # type: ignore[no-untyped-def]
        ctx = (getattr(stt_params, "request_prompt", None) or "").strip()
        if not ctx:
            return orig(stt_params)
        stt_params.request_prompt = ""
        out = orig(stt_params)
        from vllm.tokenizers import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        im_end = "<|" + "im_end" + "|>"
        system_prefix = f"<|im_start|>system\n{ctx}{im_end}\n"
        if hasattr(out, "prompt_token_ids"):
            user_part = tokenizer.decode(out.prompt_token_ids)
            out.prompt_token_ids = tokenizer.encode(system_prefix + user_part)
            return out
        if isinstance(out, dict) and "prompt_token_ids" in out:
            user_part = tokenizer.decode(out["prompt_token_ids"])
            out["prompt_token_ids"] = tokenizer.encode(system_prefix + user_part)
        return out

    cls.get_generation_prompt = get_generation_prompt  # type: ignore[method-assign]


def _load_openai_speech_to_text() -> type | None:
    for path in (
        "vllm.entrypoints.openai.speech_to_text.speech_to_text",
        "vllm.entrypoints.openai.speech_to_text",
        "vllm.entrypoints.openai.speech_to_text.serving",
    ):
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "OpenAISpeechToText"):
                return mod.OpenAISpeechToText
        except ImportError:
            continue
    return None


def patch_inference_timing_log() -> None:
    v = os.environ.get("ASR_SERVE_LOG_INFERENCE_TIME", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return

    stt_cls = _load_openai_speech_to_text()
    if stt_cls is None:
        print(
            "[asr_serve] 跳过 [ASR_TIMING] 补丁：当前 vLLM 无 OpenAISpeechToText 模块",
            flush=True,
        )
        return

    from vllm.logger import init_logger

    log = init_logger("vllm.entrypoints.openai.asr_timing")
    _orig = stt_cls._create_speech_to_text

    async def _wrapped(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        out = None
        err: BaseException | None = None
        try:
            out = await _orig(self, *args, **kwargs)
            return out
        except BaseException as e:
            err = e
            raise
        finally:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            audio_data = args[0] if args else kwargs.get("audio_data")
            nbyte = (
                len(audio_data)
                if isinstance(audio_data, (bytes, bytearray))
                else -1
            )
            is_stream = out is not None and inspect.isasyncgen(out)
            if err is not None:
                log.warning(
                    "[ASR_TIMING] wall_ms=%.1f audio_bytes=%d stream=%s err=%s: %s",
                    wall_ms,
                    nbyte,
                    is_stream,
                    type(err).__name__,
                    err,
                )
            else:
                log.info(
                    "[ASR_TIMING] wall_ms=%.1f audio_bytes=%d stream=%s",
                    wall_ms,
                    nbyte,
                    is_stream,
                )

    stt_cls._create_speech_to_text = _wrapped  # type: ignore[method-assign]
    print(
        "[asr_serve] 已启用推理耗时日志 [ASR_TIMING]（关闭：ASR_SERVE_LOG_INFERENCE_TIME=0）",
        flush=True,
    )
