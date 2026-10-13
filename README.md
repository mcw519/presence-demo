# Presence Demo：偵測對話中的使用者是否離開

> 私人專案 · All rights reserved（未授權任何使用、複製或散布）

用**影像（人臉偵測 + 人臉 embedding 群組追蹤）**加上**聲音（VAD 靜默）**，判斷原本正在跟機器人對話的人（或一群人）是不是已經離開。內含 Python 核心模組（純邏輯、可單元測試）和瀏覽器網頁 demo：開鏡頭與麥克風，即時看到偵測框、狀態機、倒數、資源監控，並可在線調整參數。

![demo](docs/screenshot.png)

<sub>截圖使用 Chrome 的假鏡頭畫面，所以沒有人臉。</sub>

## 功能

- 只看中央 **ROI** 框裡的人，可以在畫面上直接拖曳、縮放。
- **群組模式**：多人圍觀時追蹤每個人（IoU + cosine 比對），任何一位參與者還在就不算離開；被擋住有寬限期；換了一群陌生人時觸發 **NEW_GROUP**，開新對話。
- 狀態機 `ACTIVE → IDLE → MAYBE_AWAY → AWAY → ENDED`，每次轉換都記錄原因。
- **判斷優先**：`vision`（預設，影像優先，適合吵雜環境）／`balanced`／`audio`。
- 防閃爍遲滯、低信心降權、可注入時鐘。
- 模型：OpenCV **YuNet**（偵測）+ **SFace**（embedding），CPU 即可。下載失敗會自動改用 Haar + 色彩直方圖備援。
- Embedding 只存在記憶體，對話結束（ENDED）時清除；影像不存檔。
- **效能／省電**：移動偵測閘門、embedding 節流、自適應 fps、ROI 裁切＋縮小，每項都能個別開關。實測 CPU 約減少 60–70%，見 [docs/PERFORMANCE.md](docs/PERFORMANCE.md)。
- 網頁：開始／停止（真正關閉鏡頭與麥克風，切換分頁自動暫停）、設定抽屜（預設組合、localStorage 自動保存、JSON 匯出入）、資源監控（CPU、記憶體、各階段延遲 avg/p95、fps、頻寬、RTT）。

## 快速開始（uv）

需要 [uv](https://docs.astral.sh/uv/) 和 Python ≥ 3.10。

```bash
uv sync                       # 建立 .venv 並安裝相依套件（依 uv.lock）
uv run presence-demo          # 預設 http://127.0.0.1:8000
uv run presence-demo --host 0.0.0.0 --port 8000
```

用瀏覽器打開 http://localhost:8000 ，按「▶ 開始」並允許鏡頭與麥克風。

- 瀏覽器只有在 **localhost 或 HTTPS** 下才允許使用鏡頭和麥克風。
- 第一次執行會把模型下載到 `models/`（YuNet 約 230 KB、SFace 約 37 MB）。可以用 `PRESENCE_MODEL_DIR` 指定位置。

### 存取權杖（token）

伺服器本身沒有帳號系統。對外開放時**一定要**設定權杖：

```bash
export PRESENCE_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(16))')"
uv run presence-demo
# 開啟 http://localhost:8000/?token=<PRESENCE_TOKEN>
```

設定後，頁面、API 和 WebSocket 都必須帶 `?token=`，否則回 403。權杖只從環境變數讀取，請勿寫進任何檔案或提交到 repo。

### 用 Tailscale 在其他裝置（例如 iPhone）使用

```bash
tailscale up --hostname=presence-demo
tailscale serve --bg 8000          # tailnet 內 HTTPS → 127.0.0.1:8000
tailscale serve status
```

需要在 Tailscale 管理後台啟用 MagicDNS 和 HTTPS 憑證。之後在 tailnet 裡的裝置打開 `https://presence-demo.<tailnet>.ts.net/?token=<PRESENCE_TOKEN>` 即可（只有 tailnet 內可存取）。如果機器沒有 systemd，可以用 userspace 模式啟動 `tailscaled --tun=userspace-networking --state=... --socket=...`，`tailscale` 指令再加上 `--socket=...`。

## 設定

所有參數都能在網頁的「⚙ 設定」即時調整（只套用到目前連線，伺服器會檢查並限制範圍）。完整說明見 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)，調參方法見 [docs/TUNING.md](docs/TUNING.md)。

## 文件

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：架構、資料流、狀態機（mermaid）
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md)：所有參數、範圍、預設值、預設組合、判斷優先模式
- [docs/TUNING.md](docs/TUNING.md)：如何用實際資料調整門檻
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md)：效能最佳化、基準測試、Android 下一步
- [CONTRIBUTING.md](CONTRIBUTING.md)：開發流程（uv、ruff、pytest）
- [CHANGELOG.md](CHANGELOG.md)

## 注意事項

- 預設門檻和時間都只是起點，**一定要用實際場景資料調整**。
- 只偵測人臉：側身、背對、戴口罩時效果會變差。可以實作 `Detector`／`Embedder` 換成 YOLO person + re-ID。
- 瀏覽器端是簡單的能量 VAD，分不出人聲和電視；正式產品建議改用 webrtcvad 或 Silero VAD。也因此預設是影像優先。
- EmbeddingGemma 是**文字**向量模型，不能處理影像，所以這裡用人臉辨識 embedding（SFace）。
- 需要 OpenCV 4.x（5.x 移除了 Haar，備援會失效），相依套件已鎖定 `<5`。
- 測試圖片 `tests/data/face.jpg` 取自 OpenCV samples（lena.jpg）。

## 授權

私人專案，All rights reserved。未經作者書面同意，不得使用、複製、修改或散布。
