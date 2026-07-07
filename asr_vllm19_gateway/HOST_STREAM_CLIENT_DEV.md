# Jetson Qwen3-ASR 流式客户端开发文档（宿主机 / VAD 驱动）

## 背景

服务端已在 Jetson 上部署完成（vLLM + FastAPI 网关），提供两种 ASR 接口：

- **流式 WebSocket**：麦克风分段上传，增量解码，尾延迟低
- **非流式 HTTP**：整段上传，支持热词，精度高

在宿主机（Linux）用 Python 开发**流式 ASR 客户端**：

- 端侧已有 **VAD 算法**（语音活动检测），用于判断「开始说话 / 说完」
- **不要**用「按回车开始/结束」交互
- **不需要**实时展示 partial 结果，只要 VAD 判定说完后拿到最终文本

---

## 服务地址

| 用途 | 地址 |
|------|------|
| 网关（唯一入口） | `<JETSON_IP>:23311` |
| 流式 WebSocket | `ws://<JETSON_IP>:23311/ws/asr/upload` |
| 非流式 HTTP（备用） | `POST http://<JETSON_IP>:23311/v1/audio/transcriptions` |

将 `<JETSON_IP>` 替换为 Jetson 实际 IP，并确认防火墙放行 **23311**。

服务健康检查：

```bash
curl http://<JETSON_IP>:23311/health
```

---

## 一、两个「分段」概念（必须分清）

### 1. 上传分段（Upload Chunking）—— 发给服务端的粒度

目的：把麦克风 PCM **按固定时长切成小块**，通过 WebSocket **binary frame** 持续上传。

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 采样率 | 16000 Hz | 硬性要求 |
| 声道 | mono | 硬性要求 |
| 编码 | pcm_s16le | int16 小端 |
| 上传块时长 | 0.5s ~ 1.0s | 推荐 **1.0s**（32000 bytes） |
| 上传块字节数 | `16000 * 2 * chunk_sec` | 1s = **32000 bytes** |

**操作流程：**

```
麦克风连续采集（小 buffer，如 20~50ms）
    ↓ 写入 ring buffer
ring buffer 凑满 CHUNK_BYTES（如 32000）
    ↓ 且当前 VAD 状态 == SPEAKING
ws.send(binary_pcm_chunk)    # 不要加 WAV 头
```

要点：

- 上传分段与 VAD **无关**，是固定字节切分
- VAD 只决定「什么时候开始往 WS 送」和「什么时候 finish」
- 若 VAD 已进入 SPEAKING 但 buffer 不足 1 块，先攒着，凑满再发

### 2. 话轮分段（Utterance Segmentation）—— VAD 决定的一轮说话

目的：判断用户**什么时候开始说、什么时候说完**，从而控制 WS 会话生命周期。

```
VAD: SILENCE → SPEECH_START
    → 建立 WS 会话，发 start
    → 进入 SPEAKING，开始按上传分段发 binary

VAD: SPEECH_END（尾部静音达到阈值）
    → 把 ring buffer 剩余 PCM 全部发出（最后一块可 < 32000 bytes）
    → 发 finish
    → 等待 final，清洗文本，返回结果
    → 关闭 WS，等待下一轮
```

---

## 二、VAD 与 WebSocket 状态机

```text
[IDLE]
  VAD 检测到 speech_start
    → [CONNECTING] 建立 ws，收 ready，发 start，收 started
    → [SPEAKING]

[SPEAKING]
  每凑满 CHUNK_BYTES → ws.send(binary)
  VAD 持续 speech（忽略 partial，可仅 debug 日志）

  VAD 检测到 speech_end
    → 发送 buffer 中剩余 PCM（若有）
    → ws.send({"type":"finish"})
    → [WAIT_FINAL]

[WAIT_FINAL]
  收到 {"type":"final"} → 清洗 text → 业务回调
    → 关闭连接
    → [IDLE]

错误：
  {"type":"error"} → 记录日志，关闭连接，回 IDLE
```

### VAD 接口约定（按项目实际 VAD 适配）

假设端侧 VAD 提供以下之一，客户端做薄封装：

```python
from enum import Enum

class VadEvent(Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SPEECH_ONGOING = "speech_ongoing"  # 可选

# 方式 A：回调
vad.on_speech_start(callback)
vad.on_speech_end(callback)

# 方式 B：每帧返回状态
state = vad.process_frame(pcm_frame)  # -> SILENCE | SPEECH
```

客户端不关心 VAD 内部算法，只消费 `speech_start` / `speech_end` 事件。

### VAD 参数建议（可调）

| 参数 | 建议 | 说明 |
|------|------|------|
| `speech_end_silence_ms` | 300 ~ 800 ms | 尾部静音多久判定说完 |
| `speech_start_trigger_ms` | 100 ~ 300 ms | 连续有声多久判定开说 |
| `pre_speech_pad_ms` | 100 ~ 300 ms | 防止丢字头，start 后补发前导 PCM |
| `post_speech_pad_ms` | 0 ~ 200 ms | end 后再多收一点再 finish |

---

## 三、音频格式与采集

### 硬性要求

| 参数 | 值 |
|------|-----|
| 采样率 | **16000 Hz** |
| 声道 | **mono** |
| 编码 | **PCM signed 16-bit little-endian**（pcm_s16le） |
| 流式上传 | WebSocket **binary frame**，**原始 PCM 字节**（不要 WAV 头） |

### 采集要求

1. 麦克风**连续采集**（不要等用户按键）
2. 采集帧建议 **20ms**（640 samples @16k）或 **30ms**
3. 若设备原生 **48kHz**，必须在端侧**重采样到 16kHz** 再进入 VAD 和上传链路
   - 可用 `scipy.signal.resample_poly(audio, 16000, 48000)` 或等效方案
4. **禁止**把 48k PCM 直接当 16k 上传

---

## 四、WebSocket 协议

连接：`ws://<JETSON_IP>:23311/ws/asr/upload`

### 客户端 → 服务端

| 时机 | 消息 | 格式 |
|------|------|------|
| 连接后 | （收 ready） | text JSON |
| VAD speech_start 后 | `{"type":"start","sample_rate":16000,"format":"pcm_s16le"}` | text JSON |
| SPEAKING 期间每凑满一块 | 原始 PCM 字节 | **binary** |
| VAD speech_end 后 | 剩余 PCM（若有，可不足 32000 字节） | **binary** |
| 发完最后 PCM 后 | `{"type":"finish"}` | text JSON |

`start` 可选字段：

```json
{
  "type": "start",
  "sample_rate": 16000,
  "format": "pcm_s16le",
  "language": "zh"
}
```

其他控制命令（text JSON，可选）：

| 命令 | 作用 |
|------|------|
| `{"type":"commit"}` | 查询当前 partial 文本 |
| `{"type":"reset"}` | 重置会话（不断开连接） |
| `{"type":"ping"}` | 保活，返回 `pong` |

### 服务端 → 客户端

| type | 处理 |
|------|------|
| `ready` | 连接建立，含 `mode: realtime_incremental` |
| `started` | 会话开始确认 |
| `partial` | **忽略**（不展示，可写 debug 日志） |
| `final` | **取 `text`**，作为本轮最终结果 |
| `error` | 错误处理 |

### `final` 响应示例

```json
{
  "type": "final",
  "text": "肠炎宁片，我要肠炎宁片。",
  "language": "",
  "chunks": 12,
  "bytes_received": 354180,
  "total_time_ms": 12000.5,
  "finish_time_ms": 656.3
}
```

| 字段 | 含义 |
|------|------|
| `text` | 最终识别文本（需客户端清洗） |
| `finish_time_ms` | 服务端：收到 finish → 返回 final 耗时 |
| `total_time_ms` | 服务端：会话 start → final 总耗时 |

---

## 五、文本清洗（流式必须做）

`final.text` 可能包含原始标记，例如：

```
language Chinese<asr_text>肠炎宁片，我要肠炎宁片。
```

清洗函数：

```python
ASR_TEXT_MARKER = "<asr_text>"

def clean_asr_text(raw: str) -> str:
    s = raw or ""
    if ASR_TEXT_MARKER in s:
        s = s.split(ASR_TEXT_MARKER)[-1]
    for prefix in ("languageChinese", "language Chinese", "language"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()
```

---

## 六、非流式 HTTP 接口（备用，支持热词）

```
POST http://<JETSON_IP>:23311/v1/audio/transcriptions
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | 文件 | 是 | 音频文件，推荐 16kHz mono WAV |
| `model` | string | 否 | 默认 `qwen3-asr` |
| `language` | string | 否 | 如 `zh` |
| `response_format` | string | 否 | 默认 `json` |

响应示例：

```json
{
  "text": "肠炎宁片，我要肠炎宁片。",
  "usage": {"type": "duration", "seconds": 12},
  "local_infer_time_cost": 918.1
}
```

特点：`text` 已由网关清洗；支持热词（药品名等）。流式模式**不支持热词**。

---

## 七、性能参考

测试音频：11.07s，1s 上传块，1s 上传间隔

| 指标 | 约值 |
|------|------|
| 流式 finish → final（说完到出结果） | **~650 ms** |
| 非流式整段推理 | **~980 ms** |

流式优势：上传过程中服务端已在增量解码，用户说完后尾延迟更短。

---

## 八、模块划分

```
asr_client/
├── audio_capture.py        # 连续麦克风采集 + 重采样 → 16k int16 小帧
├── vad_adapter.py          # 封装现有 VAD，输出 speech_start / speech_end
├── upload_chunker.py       # ring buffer，凑满 CHUNK_BYTES 产出上传块
├── ws_stream_asr.py          # WS 会话：start → 发块 → finish → 收 final
├── text_utils.py             # clean_asr_text()
├── utterance_pipeline.py     # 状态机：IDLE → SPEAKING → WAIT_FINAL
└── main.py                   # 启动采集 + VAD + 打印最终结果
```

---

## 九、核心伪代码

```python
import asyncio
import json
import time
import websockets

CHUNK_SEC = 1.0
CHUNK_BYTES = int(16000 * 2 * CHUNK_SEC)  # 32000
WS_URL = "ws://<JETSON_IP>:23311/ws/asr/upload"


class UtterancePipeline:
    def __init__(self):
        self.state = "IDLE"
        self.ring = bytearray()
        self.ws = None
        self._final_task = None

    def on_audio_frame(self, pcm16_frame: bytes):
        """每 20~50ms 被 audio_capture 调用一次"""
        vad_state = self.vad.process(pcm16_frame)

        if self.state == "IDLE" and vad_state == "SPEECH_START":
            asyncio.create_task(self._begin_utterance())

        if self.state == "SPEAKING":
            self.ring.extend(pcm16_frame)
            while len(self.ring) >= CHUNK_BYTES:
                chunk = bytes(self.ring[:CHUNK_BYTES])
                del self.ring[:CHUNK_BYTES]
                asyncio.create_task(self.ws.send(chunk))  # binary

            if vad_state == "SPEECH_END":
                asyncio.create_task(self._end_utterance())

    async def _begin_utterance(self):
        self.state = "CONNECTING"
        self.ws = await websockets.connect(WS_URL, max_size=50 * 1024 * 1024)
        await self.ws.recv()  # ready
        await self.ws.send(json.dumps({
            "type": "start",
            "sample_rate": 16000,
            "format": "pcm_s16le",
        }))
        await self.ws.recv()  # started
        self.state = "SPEAKING"
        self._final_task = asyncio.create_task(self._wait_final())

    async def _end_utterance(self):
        if self.ring:
            await self.ws.send(bytes(self.ring))
            self.ring.clear()
        t0 = time.perf_counter()
        await self.ws.send(json.dumps({"type": "finish"}))
        final = await self._final_task
        text = clean_asr_text(final.get("text", ""))
        client_ms = (time.perf_counter() - t0) * 1000
        print(f"识别结果: {text}")
        print(f"finish_time_ms={final.get('finish_time_ms')} client={client_ms:.0f}ms")
        await self.ws.close()
        self.state = "IDLE"

    async def _wait_final(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg["type"] == "final":
                return msg
            if msg["type"] == "error":
                raise RuntimeError(msg)
```

---

## 十、依赖

```
websockets
numpy
scipy          # 重采样（若麦克风非 16kHz）
sounddevice    # 或 pyaudio
# VAD：使用项目已有 VAD 库/模块，不在此重新实现
```

安装：

```bash
pip install websockets numpy scipy sounddevice
```

---

## 十一、常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `upstream_unavailable` | 上游未启或 realtime 未注册 | 重启上游，日志需含 `Supported tasks: [..., 'realtime']` |
| `session_not_started` | 未发 `start` 就发 binary | VAD start 后再上传 |
| `unsupported_sample_rate` | 非 16kHz | 端侧重采样 |
| 识别乱码/错词 | 48k 当 16k 发 | 检查采样率 |
| 药品名识别错 | 流式无热词 | 关键场景用 HTTP 非流式 |
| `buffer_overflow` | 单次会话超 32MB | 及时 finish，开新会话 |

Jetson 日志：

```bash
tail -f asr_vllm19_gateway/logs/asr_gateway_*.log
tail -f asr_vllm19_gateway/logs/asr_upstream_*.log
```

---

## 十二、验收标准

- [ ] 麦克风连续采集，VAD 自动检测开说/说完，无需按键
- [ ] 说话期间按 1s（可配置）上传 PCM binary
- [ ] VAD end 后自动 finish，拿到清洗后的最终文本
- [ ] 打印 `finish_time_ms` 和客户端尾延迟
- [ ] 48kHz 麦克风能正确重采样到 16kHz
- [ ] 连续多轮：每轮 end 后回 IDLE，下一轮 VAD start 自动新建会话

---

## 十三、一句话总结

- **上传分段**：PCM 攒满 1 秒（32000 字节）发一块 binary，与 VAD 无关。
- **话轮分段**：VAD `speech_start` → `start`；VAD `speech_end` → 发尾块 + `finish` → 取 `final` 并清洗。
