"""Register Qwen3-ASR with transformers / vLLM ModelRegistry."""

from __future__ import annotations


def register_transformers() -> None:
    from qwen3_asr_plugin.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from transformers import AutoConfig, AutoModel, AutoProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def register_vllm_manual() -> None:
    from qwen3_asr_plugin.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration
    )


def vllm_has_speech_to_text() -> bool:
    try:
        from vllm.config import SpeechToTextConfig  # noqa: F401

        return True
    except ImportError:
        return False
