#!/usr/bin/env bash
# 一键启动 qwen3-asr 容器：挂载整个 asr_server，自动加载模型+网关，崩溃后自动重启
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE="${ASR_IMAGE:-qwen3-asr:latest}"
CONTAINER_NAME="${ASR_CONTAINER_NAME:-qwen3-asr}"
HOST_PORT="${ASR_GATEWAY_HOST_PORT:-23311}"
GPU_MEM="${ASR_GPU_MEMORY_UTILIZATION:-0.3}"
STARTUP_TIMEOUT="${ASR_STARTUP_TIMEOUT_SEC:-300}"

chmod +x "${SCRIPT_DIR}/entrypoint.sh"

if [[ ! -d "${SCRIPT_DIR}/model/Qwen3-ASR-1.7B" ]]; then
  echo "错误: 模型目录不存在: ${SCRIPT_DIR}/model/Qwen3-ASR-1.7B" >&2
  exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "错误: 镜像不存在: ${IMAGE}" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "容器 ${CONTAINER_NAME} 已存在，执行 docker start..."
  docker start "${CONTAINER_NAME}"
  exit 0
fi

echo "创建并启动容器 ${CONTAINER_NAME}"
echo "  镜像:   ${IMAGE}"
echo "  挂载:   ${SCRIPT_DIR} -> /workspace"
echo "  策略:   restart=always"
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart=always \
  --runtime nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e ASR_MODEL_PATH=/workspace/model/Qwen3-ASR-1.7B \
  -e ASR_GPU_MEMORY_UTILIZATION="${GPU_MEM}" \
  -e ASR_STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT}" \
  -e ASR_REALTIME_PATCH="${ASR_REALTIME_PATCH:-1}" \
  -e VLLM_NO_USAGE_STATS=1 \
  -p "${HOST_PORT}:23311" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${IMAGE}" \
  /bin/bash /workspace/entrypoint.sh

echo ""
echo "容器已启动: ${CONTAINER_NAME}"
echo "  健康检查: curl http://localhost:${HOST_PORT}/health"
echo "  流式 WS:  ws://localhost:${HOST_PORT}/ws/asr/upload"
echo ""
echo "查看日志: docker logs -f ${CONTAINER_NAME}"
echo "停止:     docker stop ${CONTAINER_NAME}"
echo "删除:     docker rm -f ${CONTAINER_NAME}"
