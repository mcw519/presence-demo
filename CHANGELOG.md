# Changelog

## 0.6.0 — 2026-10-14

- **Layer 2 在場機率**：新增 `presence_model`（`rules`／`hmm`，預設 `rules`，行為不變）。
  - `hmm`：每個 track 一個兩狀態（在場／離開）Bayes 濾波器，用對數勝算加「每秒證據 × 實際 dt」計算，和 fps 無關。
  - 證據來源：偵測信心、kalman 預測（較弱）、低信心幀（接近中立）、轉身＋遠離、超出距離。
  - 聲音是全域的弱證據，權重依 `priority_mode` 不同，不會歸到任何人身上。
  - 群組 P 取參與者中的最大值，再用遲滯門檻決定 ACTIVE／MAYBE_AWAY／AWAY。
  - hmm 模式的 NEW_GROUP 條件：舊參與者 P < `p_away`，而且新候選人的 P 穩定在 `p_active` 以上。
  - 新增參數：`leave_halflife`、`detect_gain`、`lowconf_factor`、`p_active`、`p_maybe`、`p_away`、`away_hold_seconds`。
- 新增 `presence_demo/presence_model.py`。
- state 新增 `p_group`、`p_tracks`；每個 track 新增 `p`。
- 網頁：新增「在場機率」卡片（群組 P 曲線和門檻線）、每個人的機率條，以及「在場機率」設定分組。
- `scripts/eval_tracking.py`（別名 `scripts/eval_presence.py`）：
  - 改為比較 legacy+rules、kalman+rules、kalman+hmm 三個版本；
  - 新增 6 個情境：低信心閃爍、ROI 空＋電視聲、長時間安靜、慢慢離開、短暫走開、遠處路人；
  - 新增「狀態切換/分」指標。
- 新增 [`docs/PRESENCE_MODEL.md`](docs/PRESENCE_MODEL.md)。

## 0.5.0 — 2026-10-14

- **Layer 4 追蹤**：新增 `tracker_mode`（`legacy`／`kalman`，預設 `legacy`，行為不變）。
  - `kalman`：每個 track 一個等速 Kalman 濾波器（用實際時間戳記計算 dt），Mahalanobis 門檻篩選，綜合 IoU、移動、臉部 cosine 和衣服顏色的分數，用匈牙利演算法配對。
  - 短暫遮擋時沿預測位置延續（`coast_seconds`、`coast_present_conf`）。
  - 新增參數：`gate_mahalanobis`、`appearance_min_sim`、`appearance_ema`。
- 新增 `presence_demo/kalman.py`（Kalman、匈牙利演算法、Bhattacharyya 相似度，純 numpy，不需要 SciPy）和 `presence_demo/appearance.py`（軀幹 HSV 直方圖，不需要模型權重，只在記憶體中計算）。
- snapshot 每個 track 新增 `visible_kind`、`vis_conf`、`dwell_seconds`、`velocity`。
- 網頁：新增「追蹤」設定分組；預測中的框用虛線畫，並顯示可見信心、停留時間和速度。
- 新增 `scripts/eval_tracking.py`（在記憶體中產生合成情境，比較兩種模式）和 [`docs/TRACKING.md`](docs/TRACKING.md)。
- 新增 `tests/test_kalman.py`；更新 ALGORITHM、CONFIGURATION。

## 0.4.1 — 2026-10-14

修正 [`docs/ALGORITHM.md`](docs/ALGORITHM.md) 稽核中發現的問題（每項都有測試，見 `tests/test_fixes_041.py`）：

- `empty_since` 記錄第一張空幀的時間，不再假設 1 fps（自適應 fps 下 `empty_timeout` 不會再提早或延後）。
- 新增 `body_min_conf`（預設 0.4）：只有身體的偵測用這個門檻判斷「有信心」，看不到臉時也能算有人。
- 超出 `leave_distance` 的人不會從 ENDED 開始 session，也不會觸發 NEW_GROUP（修正每 10 秒重新開始的循環）。
- 新增 `speech_join`（`auto`／`off`／`single_candidate`／`all`）：vision 模式下說話預設不會讓人立即加入；balanced／audio 只在唯一候選人（臉在 ROI、正面、在對話區內）時立即加入。
- vision 模式：單靠說話不會從 AWAY 恢復（包括影像不確定時），和文件一致。
- 修正 `present_timeout` 的說明（有人時的靜默逾時；預設 15 s 比 `base_timeout` 20 s 短）。
- 小修正：新 session 重設低信心計數；只有低信心的幀不算「舊群組消失」；snapshot 新增 `face_roi`／`pose_roi`（保留 `roi` 別名）。
- 新增 `docs/ALGORITHM.md`：依程式碼整理的完整偵測邏輯。

## 0.4.0 — 2026-10-13

- **授權稽核**：新增 [`docs/LICENSES.md`](docs/LICENSES.md)，列出每個模型、套件和測試資料的授權、來源和商用風險。
- **姿勢模型改成 MoveNet MultiPose Lightning**（Google，Apache-2.0，TFLite float16，用 `ai-edge-litert` 執行），最多 6 人；移除 YOLOv8n-pose（AGPL-3.0）和相關下載程式。姿勢成本約為原來的 1/5。
- **測試圖片**：`tests/data/face.jpg` 從 lena.jpg（授權不明）換成 Cleveland Museum of Art 釋出的 CC0 圖片（Julia Margaret Cameron，1867）。
- 新增環境變數 `PRESENCE_FACE_EMBEDDER=hist`：不載入 SFace，改用 HSV 直方圖（SFace 訓練資料的商用風險見 LICENSES.md）。
- 新增 `tests/test_movenet.py`；重新量測基準測試。

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
