# Changelog

## 0.3.0 — 2026-10-13

- **姿勢偵測（選用，`pose_enabled`，預設關閉）**：可替換的 `PoseDetector` 介面；預設是 YOLOv8n-pose（ONNX，OpenCV DNN CPU，第一次使用時自動下載）。輸出身體框、關鍵點（鼻、眼、耳、肩、髖）、朝向（正面／側身／背對）和信心值。
- **融合**：臉和身體配對；看不到臉時用身體判斷有人，並用 IoU（含身體框）延續身分；轉身而且正在遠離時，提早進入 MAYBE_AWAY（`turned_away_early`、`turned_away_seconds`、`recede_ratio`）。
- **距離估計（選用，`distance_enabled`）**：用臉寬或肩寬加上鏡頭視角估計距離，EMA 平滑；對話區（`zone_min`／`zone_max`）、離開距離 `leave_distance`、靠近／遠離趨勢；網頁一鍵校正（WS `calibrate`）。
- **分開的 ROI**：`face_roi`（取代 `roi`，舊設定自動轉換）和 `pose_roi`，判定規則 `roi_rule`；兩個 ROI 都能在畫面上拖曳（藍色／紫色），也能各自隱藏。
- 網頁：新增「姿勢」「距離」設定分組；畫面上畫出骨架、身體框、距離和趨勢標籤、對話區提示；資源監控顯示姿勢耗時、呼叫次數和模型。
- 基準測試新增姿勢和距離的比較；更新 `docs/PERFORMANCE.md`。
- 新增 `presence_demo/body.py` 和 `tests/test_pose_distance.py`。

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
