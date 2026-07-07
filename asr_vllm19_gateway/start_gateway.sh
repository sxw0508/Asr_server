#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/asr_gateway_$(date +%Y%m%d_%H%M%S).log"

USE_HOTWORDS="true"
FOREGROUND=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hotwords)
      if [[ $# -lt 2 ]]; then
        echo "用法: $0 [--hotwords true|false] [--foreground]"
        exit 2
      fi
      case "${2,,}" in
        true|1|yes|on) USE_HOTWORDS="true" ;;
        false|0|no|off) USE_HOTWORDS="false" ;;
        *)
          echo "无效 --hotwords 值: $2（请用 true 或 false）"
          exit 2
          ;;
      esac
      shift 2
      ;;
    --foreground|-f)
      FOREGROUND=1
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--hotwords true|false] [--foreground]"
      exit 2
      ;;
  esac
done

export ASR_BASE_URL="${ASR_BASE_URL:-http://127.0.0.1:23310}"
export ASR_GATEWAY_PORT="${ASR_GATEWAY_PORT:-23311}"

if [[ "${USE_HOTWORDS}" == "true" ]]; then
  if [[ -z "${ASR_HOTWORDS_PATH:-}" ]]; then
    _default_hotwords="${PWD}/ASR_Top100_HighQuality_Drugs_utf8.csv"
    if [[ -f "${_default_hotwords}" ]]; then
      export ASR_HOTWORDS_PATH="${_default_hotwords}"
    fi
  fi
  if [[ -n "${ASR_HOTWORDS_PATH:-}" ]]; then
    echo "热词: 开启  ASR_HOTWORDS_PATH=${ASR_HOTWORDS_PATH}"
  else
    echo "热词: 开启  但未设置 ASR_HOTWORDS_PATH 且无默认 CSV，将不注入热词 prompt。"
  fi
else
  unset ASR_HOTWORDS_PATH
  echo "热词: 关闭（已取消 ASR_HOTWORDS_PATH）"
fi

UVICORN_CMD=(
  uvicorn asr_gateway_fastapi:app
  --host 0.0.0.0
  --port "${ASR_GATEWAY_PORT}"
  --log-config "${SCRIPT_DIR}/uvicorn_log_config.json"
)

write_banner() {
  echo "========================================"
  echo "启动时间: $(date -Iseconds)"
  echo "ASR Gateway (uvicorn asr_gateway_fastapi:app)"
  echo "ASR_BASE_URL=${ASR_BASE_URL}"
  echo "监听: 0.0.0.0:${ASR_GATEWAY_PORT}"
  echo "热词: USE_HOTWORDS=${USE_HOTWORDS} ASR_HOTWORDS_PATH=${ASR_HOTWORDS_PATH:-<未设置>}"
  echo "日志文件: ${LOG_FILE}"
  echo "========================================"
}

if [[ "${FOREGROUND}" -eq 1 ]]; then
  {
    write_banner
    echo "[foreground] exec uvicorn ..."
  } >>"${LOG_FILE}" 2>&1
  exec >>"${LOG_FILE}" 2>&1
  exec "${UVICORN_CMD[@]}"
fi

{
  write_banner
  echo "[nohup] 启动 uvicorn 子进程..."
} >>"${LOG_FILE}"

nohup "${UVICORN_CMD[@]}" >>"${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/asr_gateway.pid"
disown "${PID}" 2>/dev/null || true

echo "已用 nohup 后台启动 ASR Gateway，PID=${PID}"
echo "PID 已写入: ${LOG_DIR}/asr_gateway.pid"
echo "日志: ${LOG_FILE}"
echo "跟踪日志: tail -f ${LOG_FILE}"
