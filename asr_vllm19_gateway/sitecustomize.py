"""
Auto-imported by Python in every process when PYTHONPATH includes this directory.

Registers Qwen3ASRRealtimeGeneration so vLLM worker subprocesses also expose
the /v1/realtime endpoint (ModelRegistry patch in asr_serve_vllm_builtin.py
only runs in the API-server parent process).
"""

from __future__ import annotations

import os


def _patch_register_realtime_model() -> None:
    enabled = os.environ.get("ASR_REALTIME_PATCH", "1").strip().lower()
    if enabled in ("0", "false", "no", "off"):
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
    except Exception:
        pass


_patch_register_realtime_model()
