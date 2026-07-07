#!/usr/bin/env bash
# 容器入口：挂载 /workspace=asr_server 后自动加载模型并启动网关
set -euo pipefail

MODEL_PATH="${ASR_MODEL_PATH:-/workspace/model/Qwen3-ASR-1.7B}"
GATEWAY_DIR="/workspace/asr_vllm19_gateway"
STARTUP_TIMEOUT="${ASR_STARTUP_TIMEOUT_SEC:-300}"

mkdir -p /workspace/logs "${GATEWAY_DIR}/logs"

export PYTHONPATH="${GATEWAY_DIR}:${PYTHONPATH:-}"
export ASR_REALTIME_PATCH="${ASR_REALTIME_PATCH:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export ASR_GPU_MEMORY_UTILIZATION="${ASR_GPU_MEMORY_UTILIZATION:-0.3}"

cleanup() {
  echo "[entrypoint] 收到退出信号，停止服务..."
  if [[ -f "${GATEWAY_DIR}/logs/asr_upstream.pid" ]]; then
    kill "$(cat "${GATEWAY_DIR}/logs/asr_upstream.pid")" 2>/dev/null || true
  fi
}
trap cleanup SIGTERM SIGINT

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[entrypoint] 错误: 模型目录不存在: ${MODEL_PATH}" >&2
  echo "[entrypoint] 请确认 asr_server 已挂载到 /workspace，且含 model/Qwen3-ASR-1.7B" >&2
  exit 1
fi

cd "${GATEWAY_DIR}"

echo "===== [1/3] 启动上游 vLLM (port 23310) ====="
./start_upstream.sh "${MODEL_PATH}"

echo "===== [2/3] 等待上游就绪 (timeout=${STARTUP_TIMEOUT}s) ====="
ready=0
for ((i = 1; i <= STARTUP_TIMEOUT; i++)); do
  if [[ -f logs/asr_upstream.pid ]] \
    && ! kill -0 "$(cat logs/asr_upstream.pid)" 2>/dev/null; then
    echo "[entrypoint] 上游进程已退出，最近日志:" >&2
    tail -n 40 logs/asr_upstream_*.log 2>/dev/null | tail -n 40 >&2 || true
    exit 1
  fi
  if curl -sf http://127.0.0.1:23310/v1/models >/dev/null 2>&1; then
    ready=1
    echo "[entrypoint] 上游 vLLM 就绪 (${i}s)"
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[entrypoint] 超时: 上游未在 ${STARTUP_TIMEOUT}s 内就绪" >&2
  tail -n 40 logs/asr_upstream_*.log 2>/dev/null | tail -n 40 >&2 || true
  exit 1
fi

echo "===== [3/3] 启动网关 (port 23311, 前台) ====="
exec ./start_gateway.sh --hotwords true --foreground
