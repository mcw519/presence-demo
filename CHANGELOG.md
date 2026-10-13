# Changelog

## 0.2.0 — 2026-10-13

- 新增效能／省電最佳化（`presence_demo/optimizer.py`），每項都可在設定的「效能」分組個別開關：
  - 移動偵測閘門（`motion_gate`、`motion_threshold`、`force_detect_seconds`）
  - Embedding 節流（`embed_throttle`、`embed_interval`、`embed_move_thresh`）
  - 自適應 fps（`adaptive_fps`、`fps_low`、`fps_high`），伺服器在 state 訊息中回傳 `recommended_fps`
  - ROI 裁切＋縮小（`roi_crop`、`crop_margin`、`max_width`）
- 資源監控新增：跳過比例、embedding 計算／重用次數、建議 fps、移動偵測耗時。
- 新增 `scripts/benchmark.py`（記憶體中合成畫面，不寫入影像）和 `docs/PERFORMANCE.md`。全部開啟時 CPU 約減少 63–69%。
- 設定驗證支援布林值；`fps_low` 不會大於 `fps_high`。
- 隱私：解碼改用 `decode_jpeg()`，在記憶體中完成，處理後立即釋放。

## 0.1.0 — 2026-10-13

首個版本。

- 核心：`PresenceTracker` 群組追蹤（IoU + cosine greedy 配對、EMA 模板、參與者加入與寬限期）、遲滯與低信心降權、狀態機 ACTIVE/IDLE/MAYBE_AWAY/AWAY/ENDED、NEW_GROUP。
- 判斷優先 `priority_mode`：vision（預設）／balanced／audio，以及 `present_max_silence`。
- 影像：OpenCV YuNet + SFace，備援 Haar + HSV 直方圖；模型自動下載。
- 伺服器：FastAPI + WebSocket，每個連線獨立 tracker；設定驗證與範圍限制；`PRESENCE_TOKEN` 權杖；最新畫面槽位（丟棄舊畫面）；資源與延遲統計。
- 網頁：開始／停止（關閉鏡頭與麥克風、切換分頁自動暫停）、ROI 拖曳、設定抽屜（分組、說明、預設組合、localStorage、JSON 匯出入）、狀態機、倒數、參與者、轉換紀錄、資源監控與 sparkline；支援手機。
- 預設值：`present_timeout` 15 s、`empty_timeout` 5 s。
- 工具：uv（pyproject + uv.lock）、ruff、pytest。
