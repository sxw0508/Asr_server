# asr_vllm19_gateway

纯 `vLLM 0.19` 内置 `qwen3_asr` 的 ASR 服务方案，不依赖 `qwen-asr` Python 包。

目标链路：

`Audio -> vLLM(OpenAI transcription) -> Text -> FastAPI Gateway -> HTTP(23311)`

## 目录

- `asr_serve_vllm_builtin.py`：上游启动入口（调用 vLLM CLI `serve`），可选：
  - `prompt -> system` 补丁（热词/上下文可生效）
  - 每请求 ASR 耗时日志
- `asr_gateway_fastapi.py`：对外网关（默认 `23311`）
- `requirements.txt`：网关依赖

## 前提

- 已有可用环境（例如 NVIDIA Jetson 官方 vLLM Docker）
- `vllm==0.19.x` 可导入
- 模型路径可访问（本地路径或 HF）

## 启动步骤

### 1) 启动上游 vLLM ASR（23310）

```bash
cd /workspace/asr_vllm19_gateway
./start_upstream.sh /path/to/Qwen3-ASR-1.7B
```

可选环境变量：

- `ASR_SERVE_PATCH_PROMPT_AS_SYSTEM=1`（默认 1）
- `ASR_SERVE_LOG_INFERENCE_TIME=1`（默认 1）
- `ASR_ENABLE_PREFIX_CACHING=1`（默认 1，传 `--enable-prefix-caching`）
  - 关闭：`ASR_ENABLE_PREFIX_CACHING=0 ./start_upstream.sh /path/to/Qwen3-ASR-1.7B`
- `ASR_TOKENIZER`（可选）
  - 当本地模型目录缺少 `tokenizer.json` / `vocab.json` 时，设置 tokenizer 来源。
  - 示例：`ASR_TOKENIZER=Qwen/Qwen3-ASR-1.7B ./start_upstream.sh /path/to/Qwen3-ASR-1.7B`

### 2) 启动网关（23311）

```bash
cd /workspace/asr_vllm19_gateway
pip install -r requirements.txt
export ASR_BASE_URL=http://127.0.0.1:23310
./start_gateway.sh
```

网关热词逻辑（与原项目保持一致）：

- `./start_gateway.sh --hotwords true`（默认）
  - 若已设置 `ASR_HOTWORDS_PATH`，直接使用该文件。
  - 若未设置，则自动尝试当前目录下 `ASR_Top100_HighQuality_Drugs_utf8.csv`。
- `./start_gateway.sh --hotwords false`
  - 强制取消 `ASR_HOTWORDS_PATH`，不注入热词 prompt。
- 网关不接受客户端自定义 `prompt` 字段，避免覆盖服务端热词策略。

## 接口

- 健康检查：`GET /health`
- 识别接口：`POST /v1/audio/transcriptions`（multipart/form-data）

网关固定参数（客户端不可覆盖）：

- `max_tokens=max_completion_tokens=64`
- `temperature=0`
- `stream=false`

## 调用示例

```bash
curl -sS -X POST "http://127.0.0.1:23311/v1/audio/transcriptions" \
  -F "file=@/path/to/audio.wav" \
  -F "model=qwen3-asr" \
  -F "response_format=json"
```

## 说明

- 本工程避免了 `qwen-asr` 包带来的 `librosa/numba/nagisa/soynlp` 依赖链问题。
- 若你不需要热词/上下文注入，可设置 `ASR_SERVE_PATCH_PROMPT_AS_SYSTEM=0`，完全贴近 vLLM 原生行为。
