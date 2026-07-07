#!/bin/bash
# 单容器入口：先启动上游 vLLM，就绪后前台启动网关（保持容器存活）
set -euo pipefail

MODEL_PATH="${ASR_MODEL_PATH:-/workspace/model/Qwen3-ASR-1.7B}"
GATEWAY_DIR="/workspace/asr_vllm19_gateway"
LOG_DIR="/workspace/logs"

mkdir -p "${LOG_DIR}" "${GATEWAY_DIR}/logs"

export PYTHONPATH="${GATEWAY_DIR}:${PYTHONPATH:-}"
export ASR_REALTIME_PATCH="${ASR_REALTIME_PATCH:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export ASR_GPU_MEMORY_UTILIZATION="${ASR_GPU_MEMORY_UTILIZATION:-0.3}"

cleanup() {
  echo "[start.sh] 收到退出信号，停止服务..."
  if [[ -f "${GATEWAY_DIR}/logs/asr_upstream.pid" ]]; then
    kill "$(cat "${GATEWAY_DIR}/logs/asr_upstream.pid")" 2>/dev/null || true
  fi
  if [[ -f "${GATEWAY_DIR}/logs/asr_gateway.pid" ]]; then
    kill "$(cat "${GATEWAY_DIR}/logs/asr_gateway.pid")" 2>/dev/null || true
  fi
}
trap cleanup SIGTERM SIGINT

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[start.sh] 错误: 模型目录不存在: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${GATEWAY_DIR}"

echo "===== 启动上游 vLLM ====="
./start_upstream.sh "${MODEL_PATH}"

echo "===== 等待上游就绪 ====="
until curl -sf http://127.0.0.1:23310/v1/models >/dev/null 2>&1; do
  if [[ -f logs/asr_upstream.pid ]] && ! kill -0 "$(cat logs/asr_upstream.pid)" 2>/dev/null; then
    echo "[start.sh] 上游进程已退出，最近日志:" >&2
    tail -n 30 logs/asr_upstream_*.log 2>/dev/null | tail -n 30 >&2 || true
    exit 1
  fi
  sleep 2
done
echo "上游 vLLM 就绪"

echo "===== 启动网关（前台） ====="
exec ./start_gateway.sh --hotwords true --foreground
