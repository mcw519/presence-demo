# Changelog

## 0.1.0 — 2026-10-13

首個版本。

- 核心：`PresenceTracker` 群組追蹤（IoU + cosine greedy 配對、EMA 模板、參與者加入與寬限期）、遲滯與低信心降權、狀態機 ACTIVE/IDLE/MAYBE_AWAY/AWAY/ENDED、NEW_GROUP。
- 判斷優先 `priority_mode`：vision（預設）／balanced／audio，以及 `present_max_silence`。
- 影像：OpenCV YuNet + SFace，備援 Haar + HSV 直方圖；模型自動下載。
- 伺服器：FastAPI + WebSocket，每個連線獨立 tracker；設定驗證與範圍限制；`PRESENCE_TOKEN` 權杖；最新畫面槽位（丟棄舊畫面）；資源與延遲統計。
- 網頁：開始／停止（關閉鏡頭與麥克風、切換分頁自動暫停）、ROI 拖曳、設定抽屜（分組、說明、預設組合、localStorage、JSON 匯出入）、狀態機、倒數、參與者、轉換紀錄、資源監控與 sparkline；支援手機。
- 預設值：`present_timeout` 15 s、`empty_timeout` 5 s。
- 工具：uv（pyproject + uv.lock）、ruff、pytest。
