#!/usr/bin/env bash
# 登录镜像仓库 → 拉取最新镜像 → docker compose up -d
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
else
  echo "错误: 未找到 .env，请先执行: cp .env.example .env 并填写 ACR 账号密码" >&2
  exit 1
fi

ACR_REGISTRY="${ACR_REGISTRY:-autoflux-registry.cn-hangzhou.cr.aliyuncs.com}"
ASR_IMAGE="${ASR_IMAGE:-${ACR_REGISTRY}/autobio/qwen3-asr-jetson:latest}"
LOCAL_TAG="${ASR_LOCAL_TAG:-qwen3-asr:latest}"

if [[ -z "${ACR_USERNAME:-}" || -z "${ACR_PASSWORD:-}" ]]; then
  echo "错误: .env 中需设置 ACR_USERNAME / ACR_PASSWORD" >&2
  exit 1
fi

if [[ ! -d "${SCRIPT_DIR}/model/Qwen3-ASR-1.7B" ]]; then
  echo "错误: 模型目录不存在: ${SCRIPT_DIR}/model/Qwen3-ASR-1.7B" >&2
  exit 1
fi

chmod +x "${SCRIPT_DIR}/entrypoint.sh"

echo "===== [1/3] 登录镜像仓库: ${ACR_REGISTRY} ====="
echo "${ACR_PASSWORD}" | docker login --username "${ACR_USERNAME}" --password-stdin "${ACR_REGISTRY}"

echo "===== [2/3] 拉取镜像: ${ASR_IMAGE} ====="
docker compose pull
docker tag "${ASR_IMAGE}" "${LOCAL_TAG}" 2>/dev/null || true

echo "===== [3/3] 启动服务 ====="
docker compose up -d

echo ""
echo "已启动: qwen3-asr"
echo "  镜像:   ${ASR_IMAGE}"
echo "  健康检查: curl http://localhost:${ASR_GATEWAY_HOST_PORT:-23311}/health"
echo "  日志:   docker compose logs -f"
echo "  停止:   docker compose down"
