[English](../architecture.md) · **繁體中文**

# 架構

presence-demo 是一個 Python 套件：核心是純邏輯（沒有 I/O，可以注入時鐘做單元測試），外面包一層 FastAPI + WebSocket 網頁 demo。
瀏覽器負責擷取鏡頭和麥克風、繪製介面；所有分析都在伺服器記憶體中完成。

![架構](../img/architecture.zh-TW.png)

## 程式結構

```
presence_demo/
  config.py          Config 資料類別、RANGES（驗證範圍）、State 列舉
  tracker.py         PresenceTracker：追蹤、意圖、在場、狀態機（純邏輯）
  kalman.py          等速 Kalman 濾波、匈牙利演算法
  presence_model.py  HMM 對數勝算（Layer 2）
  intent.py          意圖線索與平滑（Layer 3）
  appearance.py      軀幹 HSV 直方圖、Bhattacharyya 相似度
  body.py            姿勢朝向、臉與身體配對、距離與校正
  vision.py          YuNet / SFace / Haar 備援、MoveNet 姿勢、記憶體內解碼
  embed_backends.py  SFace fp32 / int8 / TFLite（+ delegate）後端
  optimizer.py       移動閘門、embedding 節流、ROI 裁切、自適應 fps
  audio.py           PureSound 串流銜接、乾淨 VAD、影像交叉確認
  asr.py             AssemblyAI 串流用戶端（選用）
  reid.py            跨 session re-ID 管理與雲端後端（選用）
  diag.py            結構化日誌、遮蔽、事件環形緩衝、watchdog 工具
  stats.py           資源與延遲統計
  server.py          FastAPI 應用：路由、權杖中介層、靜態檔、進入點
  web/
    runtime.py       程序共用的模型、鎖、執行緒池
    outbox.py        每個連線的傳送佇列（含背壓）
    session.py       每個連線的 WebSocket 主迴圈
  static/
    index.html       只有版面（文字用 data-i18n 屬性）
    app.css          在 PureSound CSS 之上的 presence 專用版面
    js/              i18n · core · settings · app · audio · panels · main（一般 script，載入順序有意義）
    i18n/            en.js、zh-TW.js 字串表
    vendor/puresound PureSound 設計系統（未修改）
scripts/             基準測試、評估、模型匯出／下載、圖片、設定文件
tests/               單元、情境、WebSocket、隱私、介面和 i18n 測試
```

## 一個連線

每個瀏覽器分頁開一個 WebSocket（`/ws`），由 `web/session.py` 處理：

```mermaid
flowchart LR
  WS[WebSocket 接收] -->|frame JSON| SLOT[最新畫面槽]
  WS -->|binary 0x01 音訊| AQ[音訊佇列 ≤ 1 s]
  WS -->|config、vad、start、stop…| CTRL[控制]
  SLOT --> VW[影像 worker<br/>執行緒池] --> TR[PresenceTracker]
  AQ --> AW[音訊 worker] --> TR
  TICK[ticker 0.5 s] --> TR
  TR --> OB[outbox] --> OUT[WebSocket 傳送]
  TR --> RE[re-ID 工作] --> OB
```

- **只處理最新的畫面。** 前一張還在處理時又來了新畫面，較舊的等待畫面會被丟掉（並計數），所以瀏覽器端不會累積延遲。
- 偵測器由所有連線共用，用鎖保護（`runtime.py`）；每個連線有自己的追蹤器、音訊狀態和 re-ID 參考臉。
- 輸入持續進來但 worker 5 秒沒有進展時，watchdog 會重啟它並記錄 stack dump。
- 處理從不等待網路：outbox 有上限，瀏覽器或網路慢時，會丟掉新的音訊和會被取代的狀態訊息，而不是卡住 worker。

## WebSocket 協定

瀏覽器 → 伺服器（除非註明，都是 JSON）：

| 訊息 | 用途 |
|---|---|
| `frame` | `{data: base64 JPEG, speech, t}`，一張鏡頭畫面 |
| `vad` | 瀏覽器 VAD 改變（energy 管線） |
| `config` | 設定變更；驗證並限制範圍後回傳警告 |
| `start` / `stop` / `reset` | session 控制 |
| `calibrate` | 在已知距離做距離校正 |
| `record` | 要求錄音串流（`0x03`） |
| `reid_wipe` | 清除參考臉 |
| `client` | 瀏覽器診斷（可見性、重新連線…） |
| `ping` | 量測往返時間（回 `pong`） |
| binary `0x01` | 音訊區塊：時間戳 + 20 ms、16 kHz PCM16 |

伺服器 → 瀏覽器：

| 訊息 | 用途 |
|---|---|
| `state` | track、狀態、原因、`p_group`、建議 fps、耗時 |
| `audio` | 交叉確認判斷、VAD 旗標、增益、延遲（每 100 ms） |
| `asr` | 逐字稿和狀態 |
| `reid`、`handoff`、`reid_view`、`reid_view_clear` | re-ID 狀態、判斷、選用縮圖 |
| `event`、`error` | 診斷事件和錯誤橫幅 |
| binary `0x02` / `0x03` | 監聽用的乾淨音軌／錄音音軌 |

## HTTP 端點

| 端點 | 用途 |
|---|---|
| `GET /` | demo 頁面 |
| `GET /api/config` | 預設值和範圍 |
| `GET /api/asr_status`、`POST /api/asr_key` | ASR key 狀態／設定（從不回傳 key） |
| `GET /api/reid_status`、`POST /api/reid_key`、`POST /api/reid_reset` | re-ID key 和參考臉 |
| `GET /debug/events` | 最近的診斷事件（見[日誌](logging.md)） |

設定 `PRESENCE_TOKEN` 後，所有路由和 WebSocket 都需要 `?token=`。只有 vendored CSS、script 和字串表是公開的（不含任何資料）。

## 擴充

- **換偵測器或 embedder**：實作 `vision.Detector` / `vision.Embedder`，傳給 `VisionPipeline(detector=…, embedder=…)`。
- **換 re-ID 服務**：實作 `CloudReid.compare`。
- **新增參數**：見 [CONTRIBUTING](../../CONTRIBUTING.md#adding-a-parameter)。
