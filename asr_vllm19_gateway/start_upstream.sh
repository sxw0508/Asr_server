#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <QWEN3_ASR_MODEL_PATH> [--foreground] [额外 vLLM 参数...]"
  exit 1
fi

MODEL_PATH="$1"
shift || true

FOREGROUND=0
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground|-f)
      FOREGROUND=1
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$(dirname "$0")"
LOG_DIR="${PWD}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/asr_upstream_$(date +%Y%m%d_%H%M%S).log"

# Ensure sitecustomize.py (realtime model patch) loads in API server AND worker subprocesses
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

ENABLE_PREFIX_CACHING="${ASR_ENABLE_PREFIX_CACHING:-1}"
PREFIX_CACHE_ARGS=()
case "${ENABLE_PREFIX_CACHING,,}" in
  1|true|yes|on)
    PREFIX_CACHE_ARGS+=(--enable-prefix-caching)
    echo "前缀缓存: 开启（ASR_ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}）"
    ;;
  0|false|no|off)
    echo "前缀缓存: 关闭（ASR_ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}）"
    ;;
  *)
    echo "无效 ASR_ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}，默认开启。"
    PREFIX_CACHE_ARGS+=(--enable-prefix-caching)
    ;;
esac

TOKENIZER_ARGS=()
if [[ -n "${ASR_TOKENIZER:-}" ]]; then
  TOKENIZER_ARGS+=(--tokenizer "${ASR_TOKENIZER}")
  echo "Tokenizer: 使用 ASR_TOKENIZER=${ASR_TOKENIZER}"
else
  if [[ ! -f "${MODEL_PATH}/tokenizer.json" && ! -f "${MODEL_PATH}/vocab.json" ]]; then
    echo "警告: 模型目录缺少 tokenizer.json/vocab.json。若启动失败，请设置 ASR_TOKENIZER（如 HF repo）。"
  fi
fi

ASR_GPU_MEMORY_UTILIZATION="${ASR_GPU_MEMORY_UTILIZATION:-0.3}"
echo "显存利用率: ${ASR_GPU_MEMORY_UTILIZATION}（可通过 ASR_GPU_MEMORY_UTILIZATION 覆盖）"

# ASR_REALTIME_PATCH: 注册 Qwen3ASRRealtimeGeneration，启用 /v1/realtime WebSocket 端点
# 默认开启（"1"）。设为 "0" 可回退到仅支持 HTTP 非流式 transcription 模式。
ASR_REALTIME_PATCH="${ASR_REALTIME_PATCH:-1}"
export ASR_REALTIME_PATCH
echo "Realtime 端点注册: ${ASR_REALTIME_PATCH}（可通过 ASR_REALTIME_PATCH 控制）"
CMD=(python asr_serve_vllm_builtin.py serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 23310 \
  --served-model-name qwen3-asr \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization "${ASR_GPU_MEMORY_UTILIZATION}" \
  "${PREFIX_CACHE_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  "${PASSTHROUGH_ARGS[@]}")

echo "日志文件: ${LOG_FILE}"

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "运行模式: 前台（--foreground）"
  exec "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  echo "运行模式: 默认 nohup 后台"
  nohup "${CMD[@]}" >>"${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > "${LOG_DIR}/asr_upstream.pid"
  disown "${PID}" 2>/dev/null || true
  echo "已启动上游服务，PID=${PID}"
  echo "PID 文件: ${LOG_DIR}/asr_upstream.pid"
  echo "查看日志: tail -f ${LOG_FILE}"
fi
