# 架構

## 元件

| 模組 | 說明 |
|---|---|
| `presence_demo/tracker.py` | `Config`、`PresenceTracker`：群組追蹤、遲滯、聲音／影像融合、狀態機。只依賴 numpy，時鐘可注入，完全可單元測試。 |
| `presence_demo/vision.py` | `Detector`／`Embedder` 介面；`YuNetDetector` + `SFaceEmbedder`（預設），`HaarDetector` + `HistEmbedder`（備援）；`VisionPipeline` 負責解碼、偵測、embedding 並計時。 |
| `presence_demo/optimizer.py` | `FrameProcessor`：移動偵測閘門、ROI 裁切＋縮小、embedding 節流；`recommend_fps()` 自適應 fps。全部在記憶體中處理。 |
| `presence_demo/stats.py` | `ConnStats`（每個連線的延遲 avg/p95、fps、丟棄張數、頻寬、RTT）和 `system_stats()`（psutil）。 |
| `presence_demo/server.py` | FastAPI：靜態頁、`/api/config`、`/ws`；權杖中介層；每個 WebSocket 一個 tracker。 |
| `presence_demo/static/index.html` | 單一檔案網頁（vanilla JS/CSS，不需建置）。 |

## 資料流

```mermaid
flowchart LR
  subgraph Browser
    CAM[getUserMedia 鏡頭] -->|每 1/fps 秒| JPG[canvas → JPEG]
    MIC[麥克風] --> VAD[能量 VAD<br/>自適應噪音底 + hangover]
    UI[畫面：ROI/框/狀態/倒數/監控/設定]
  end
  JPG -->|WS frame| Q[最新一張槽位<br/>舊的計為 dropped]
  VAD -->|WS vad 狀態改變時| TR
  Q --> W[worker thread]
  W --> DEC[記憶體內解碼] --> GATE{移動閘門} -->|有變化或強制| CROP[ROI 裁切＋縮小] --> DET[YuNet 偵測] --> EMB[SFace embedding<br/>節流]
  GATE -->|靜止| REUSE[沿用上次結果] --> TR
  EMB --> TR[PresenceTracker.update]
  TK[ticker 0.5s] --> TR
  TR -->|WS state| UI
  ST[ConnStats + psutil] -->|WS stats 每秒，僅執行中| UI
```

WebSocket 訊息：

| 方向 | type | 內容 |
|---|---|---|
| → | `start` / `stop` | 開始／停止（停止會重設 tracker、清除 embedding、忽略後續畫面） |
| → | `frame` | `data`（JPEG data URL）、`speech` |
| → | `vad` | `speech` |
| → | `config` | 部分設定；回覆 `config_ack` + `warnings` |
| → | `ping` | `t`、上次 `rtt`；回覆 `pong` |
| ← | `state` | 狀態、計時、tracks、log、config、`running` |
| ← | `stats` | `system`、`conn`（latency/fps/frames/bandwidth/rtt）、`info` |

## 追蹤

1. 只保留中心點在 ROI 裡的偵測；conf < `min_conf` 的算低信心。
2. 高信心偵測和既有 track 做 greedy 配對：分數 = cosine + 0.5·IoU。cosine ≥ `cos_match`，或 IoU ≥ `iou_match` 且 cosine ≥ `cos_weak`，就算同一個人。
3. 配對成功就更新 bbox、lastSeen，embedding 用 EMA 更新後再做 L2 正規化。沒配到的偵測建立新 track。
4. 連續在 ROI 裡 `join_seconds`（或在 ROI 裡說話）就成為參與者；消失超過 `grace_seconds` 就移除。
5. 有人／ROI 空採遲滯判定：`present_frames`／`absent_frames`，低信心畫面不影響這兩個計數；連續低信心則切換成「不確定」。

## 狀態機

```mermaid
stateDiagram-v2
  [*] --> ENDED
  ENDED --> ACTIVE: 有人進入 ROI / 說話（vision 模式只看有人）
  ACTIVE --> IDLE: 靜默 ≥ idle_after
  IDLE --> ACTIVE: 說話
  IDLE --> MAYBE_AWAY: 計時 ≥ maybe_away_frac × 逾時
  MAYBE_AWAY --> ACTIVE: 說話 / 人回來
  MAYBE_AWAY --> AWAY: 計時 ≥ 逾時
  AWAY --> ACTIVE: 人回來（balanced/audio 也接受說話）
  AWAY --> ENDED: 持續 end_after_away（清除 embedding）
  ACTIVE --> ACTIVE: NEW_GROUP（結束舊 session，開新 session）
```

逾時和計時依 `priority_mode` 與影像狀態決定（詳見 CONFIGURATION.md）：

| 模式 | 有人 | ROI 空 | 不確定 |
|---|---|---|---|
| vision | 不因靜默離開（`present_max_silence` 上限，0＝關） | 空了 `empty_timeout` 就 AWAY，不管聲音 | `base_timeout`（靜默） |
| balanced | `present_timeout`（靜默） | `empty_timeout`，計時取 min(靜默, 空的秒數) | `base_timeout` |
| audio | 1.5 × `base_timeout` | 0.5 × `base_timeout` | `base_timeout` |

### NEW_GROUP

原本的參與者全部不在畫面上，而 ROI 裡有沒配對到的新 track 待滿 `join_seconds`，再持續 `new_group_seconds`，就觸發：記錄 `NEW_GROUP`、結束舊 session（清除 embedding）、用新的人開新 session。

## 隱私

- 影像只在記憶體解碼，處理完就丟棄。
- Embedding 只存在 `PresenceTracker.tracks`，ENDED、停止或斷線時清除。
- 停止時瀏覽器會關閉所有 media track 和 AudioContext。
