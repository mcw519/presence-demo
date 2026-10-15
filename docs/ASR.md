# 雲端語音辨識（選用）：AssemblyAI Universal-Streaming v3

> ⚠ **外部雲端服務：開啟後聲音會離開裝置**，送到 AssemblyAI。預設關閉（`asr_provider = none`）。
> 使用前請確認使用者同意、當地個資法規，以及 AssemblyAI 的服務條款與資料保留政策。

API 文件（2026-10-15 查證）：https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api 、
模型選擇：https://www.assemblyai.com/docs/streaming/select-the-speech-model

## 運作方式

```
麥克風 → PureSound + 影像交叉確認 → 修正後乾淨音軌 ──(只在 session 進行中)──▶ wss://streaming.assemblyai.com/v3/ws
                                                                                   │ Turn（partial / final）
瀏覽器「語音辨識」面板 ◀──── WS {"type":"asr"} ◀──── 伺服器（只轉送，不儲存） ◀────┘
```

- 需要 `audio_pipeline = puresound`（音訊串流到伺服器）。連線由**伺服器**建立，用 `Authorization` header 帶 API key；
  key 不會到瀏覽器，所以不需要臨時 token（臨時 token 是給瀏覽器直連用的）。
- **只在 session 進行中**（狀態不是 ENDED）連線；ENDED、停止、斷線或換新 session 時送 `Terminate` 並關閉。
- `asr_source`：`clean`（預設，修正後的乾淨音軌，漏音／超出範圍已被壓低）或 `raw`。
- `asr_feed`：`speech`（預設，只有「校正後算說話」的 20 ms 塊送真正的聲音，其他送全零，保持即時節奏但內容不離開裝置）
  或 `session`（session 中全部送出）。
- 20 ms 塊累積成 100 ms 一則訊息（AssemblyAI 要求 50–1000 ms）；網路卡住時最多排隊 5 s，超過丟最舊的。
- 錯誤或伺服器關閉時自動重連（指數退避，最多 3 次；收到 `Begin` 後重設次數）。
- 時間：每個 Turn 的第一個／最後一個字（`words[].start/end`，相對於串流開頭）換算成**相對 session 開始**的秒數
  （`start_s`、`end_s`）。重連會開新的串流，時間仍以 session 開始為準（重連期間的聲音不會被辨識）。
- 逐字稿只存在瀏覽器頁面記憶體（最多 100 句），伺服器不保存、不寫 log。

## API key

- 伺服器環境變數 `ASSEMBLYAI_API_KEY`，或在 UI「設定 → 語音辨識」的密碼欄位輸入（`POST /api/asr_key`，受 `PRESENCE_TOKEN` 保護）。
- key **只存在伺服器程序記憶體**（重啟即消失）：不寫檔、不寫 log（websockets 的 DEBUG log 會印出 header，因此另給它一個只記 WARNING 的 logger）、
  不回傳給瀏覽器；`GET /api/asr_status` 只回 `{"set": true/false, "source": "env"|"ui"}`。
- key 是程序層級的設定，所有連線共用。對外開放 demo 時務必設 `PRESENCE_TOKEN`。

## 參數

| 參數 | 預設 | 送出的 query 參數 | 適用模型 |
|---|---|---|---|
| `asr_speech_model` | `universal-3-6-pro` | `speech_model` | 全部 |
| `asr_language_codes` | `zh,en` | `language_codes`（JSON 陣列），並開 `language_detection` | Universal-3.6 / 3.5 Pro |
| `asr_keyterms` | 空 | `keyterms_prompt`（JSON 陣列，最多 100 個） | 全部 |
| `asr_format_turns` | 開 | `format_turns` | Universal-Streaming English / Multilingual |
| `asr_eot_confidence` | 0.4 | `end_of_turn_confidence_threshold` | Universal-Streaming English / Multilingual |
| `asr_min_turn_silence` | 400 ms | `min_turn_silence` | 全部 |
| `asr_max_turn_silence` | 1280 ms | `max_turn_silence` | 全部 |
| `asr_region` | `global` | 端點：global / `streaming.us` / `streaming.eu` | 全部 |
| — | — | `sample_rate=16000`、`encoding=pcm_s16le` | 全部 |

只送所選模型支援的參數（不支援的參數 AssemblyAI 會靜默忽略）。

## 語言支援（依官方文件，未實測）

| 模型 | 語言 | 中文 |
|---|---|---|
| Universal-3.6 Pro（預設、官方推薦） | 32 種，原生混語 | ✅ `zh`（普通話）、`yue`（粵語） |
| Universal-3.5 Pro | 19 種 | ✅ `zh`；❌ `yue` |
| Universal-Streaming English | 英文 | ❌ |
| Universal-Streaming Multilingual | 英、西、德、法、葡、義 | ❌ |

文件沒有說明繁體／簡體的輸出字形；`zh` 的實際辨識品質（台灣口音、繁體輸出）**沒有實測**。

## 測試

`tests/test_asr.py`：本機 mock WebSocket 伺服器（檢查 `Authorization` header、query 參數、100 ms 訊息、`Terminate`、
partial/final 解析）、API key 端點不回傳 key、log 中沒有 key、連不上時的錯誤處理。實際連線測試只在有 `ASSEMBLYAI_API_KEY` 時執行（本次未執行）。

## 費用與授權

AssemblyAI 是付費服務，依串流時間計費（請見官方價目）。這不是開源元件；本專案只透過公開 API 使用。
