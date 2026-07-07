"""ASR 后端：仅 vLLM（原生内置或手动插件）。"""

from __future__ import annotations

import importlib
from typing import Literal

from qwen3_asr_plugin.register import register_transformers, register_vllm_manual, vllm_has_speech_to_text

BackendMode = Literal["native_vllm", "manual_vllm"]

_VLLM_MIN_MANUAL = (0, 14, 0)
_VLLM_MIN_NATIVE = (0, 16, 0)


def vllm_version_tuple() -> tuple[int, ...]:
    try:
        import vllm
    except ImportError as e:
        raise RuntimeError("未安装 vLLM，本服务仅支持 vLLM 部署。") from e
    parts: list[int] = []
    for piece in vllm.__version__.split("+")[0].split("."):
        num = ""
        for ch in piece:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            parts.append(int(num))
    return tuple(parts)


def detect_backend() -> BackendMode:
    ver = vllm_version_tuple()
    if ver >= _VLLM_MIN_NATIVE:
        try:
            importlib.import_module("vllm.model_executor.models.qwen3_asr")
            return "native_vllm"
        except ImportError:
            pass
    if vllm_has_speech_to_text():
        return "manual_vllm"
    raise RuntimeError(
        f"当前 vLLM {'.'.join(map(str, ver))} 无法部署 Qwen3-ASR。\n"
        f"需要其一：\n"
        f"  • vLLM >= {_VLLM_MIN_NATIVE[0]}.{_VLLM_MIN_NATIVE[1]}（内置 qwen3_asr）\n"
        f"  • vLLM >= {_VLLM_MIN_MANUAL[0]}.{_VLLM_MIN_MANUAL[1]} + 内置 qwen3_asr_plugin（手动注册）\n"
        "请升级 Jetson 基础镜像，例如更新 dustynv/nvidia-ai-iot vLLM 标签。"
    )


def setup_backend(mode: BackendMode | None = None) -> BackendMode:
    mode = mode or detect_backend()
    register_transformers()
    if mode == "manual_vllm":
        register_vllm_manual()
    print(f"[asr_backend] vLLM 模式={mode}", flush=True)
    return mode
