# Changelog

## 0.8.2 — 2026-10-15

- **修正**：balanced／audio priority + rules 在場模型下，在場但沉默的人讓狀態每 0.5 s 在 AWAY 和 ACTIVE 之間來回跳、session 永遠不會結束。
  - 從 AWAY 回來現在需要**新的證據**：說話開始、確認沒人後再出現，或進入 AWAY 時不在畫面上的新人。人一直待著不算。回來時重設靜默計時。
  - 因 AWAY 逾時而 ENDED 後，同一個人一直待著不會立刻開始新的 session（rules 和 hmm 都適用；hmm 以前會這樣不斷重開）。要說話，或離開後再出現。
  - 新的時間（有人在場但沉默，預設值）：balanced rules 7.5 s MAYBE／15 s AWAY／20 s ENDED；audio rules 15／30／35 s；balanced hmm 約 14.5／20.5／25.5 s；audio hmm 約 11.5／16.5／21.5 s。
  - vision 模式行為不變。
- 新增 `tests/test_away_return.py`：3 種 priority × 2 種在場模型的回歸測試，加上 score 意圖模式。

## 未發布（文件）

- `docs/ALGORITHM.md` 改寫成演算法架構總文件（v0.8.1）：端到端資料流、各層的公式和預設值、priority_mode 的影響、模式矩陣與建議預設、時間表、效能／隱私／授權、已知限制。所有數字都已經和程式碼預設值核對，時間表用模擬實測。
- 新增 mermaid 原始檔和 PNG：`docs/img/dataflow`、`state_rules`、`state_hmm`。
- 修正 `docs/PRESENCE_MODEL.md` 中 balanced／audio 靜默時的時間（依實測：audio 約 16.5 s AWAY，不是 20 s）。
- 記錄已知問題：balanced／audio + rules 模式下，在場但沉默的人會讓狀態在 AWAY 和 ACTIVE 之間來回跳（程式碼尚未修改）。

## 0.8.1 — 2026-10-14

- 網頁預覽新增「鏡像顯示」（預設開啟，存在 localStorage）：影像與疊圖（人臉框、身體框、關鍵點、ROI）一起左右翻轉成自拍視角；疊圖是換算座標後再畫，文字不會反過來。
- 送到伺服器的影像不翻轉，和機器人相機一致；ROI 一律以原始相機座標儲存，拖曳時由網頁換算。
- 已用 headless Chrome 以合成框驗證疊圖與影像對齊、拖曳座標換算正確。

## 0.8.0 — 2026-10-14

- **SFace 嵌入後端**：新增 `embed_backend`（`fp32`（預設）／`int8`／`npu`）、`npu_precision`（`int8`／`fp16`）和 `cos_match_int8`。
  - `int8`：opencv_zoo 官方的 `face_recognition_sface_2021dec_int8.onnx`，下載時驗證 sha256。
  - `npu`：用 LiteRT 執行 TFLite 模型，可以掛外部 delegate。
    - delegate 路徑只能用環境變數 `PRESENCE_NPU_DELEGATE` 設定，WebSocket 無法修改。
    - delegate 無法使用時改用 CPU XNNPACK，並在狀態中顯示實際使用的加速器。
- 新增 `presence_demo/embed_backends.py`（Umeyama 5 點對齊、OpenCV 和 TFLite 兩種後端）。
- `ensure_model`：支援 sha256 驗證，先下載到 `.part` 暫存檔再換名。
- `VisionPipeline(allow_download=False)` 會使用已經下載的模型。
- 新增 `scripts/export_sface_tflite.py`：可重現的 fp16／全 INT8 TFLite 匯出。
  - PReLU 合併成單一 op；
  - 用 CC0 人像在記憶體中校正；
  - onnx2tf 會改寫輸入檔，所以轉換複本。
- 新增 `scripts/faces_cc0.py`（只在記憶體中處理的 CC0 人臉）和 `scripts/eval_embed_backends.py`（各後端和 fp32 的精度一致性）。
- `benchmark.py --latency`：新增嵌入後端比較和模型本身的延遲。
- 網頁：新增「嵌入後端」設定；「資源監控」顯示實際使用的後端和加速器。
- 文件：新增 [`docs/NPU.md`](docs/NPU.md)；更新 LICENSES、CONFIGURATION、PERFORMANCE。

### 先前未發布的變更

- `scripts/benchmark.py` 新增兩種模式：
  - `--latency`：各階段 p50／p95／max（設定 × 輸入）；
  - `--ws`：本機 WebSocket 端到端延遲（暫時啟動伺服器，量完自動關閉）。
- `PresenceTracker.timings`：記錄最近一次 update 的 tracker／intent／presence 耗時（ms）。
- PERFORMANCE.md 新增 v0.7.0 的每幀延遲結果。

## 0.7.0 — 2026-10-14

- **Layer 3 互動意圖**：新增 `intent_model`（`rules`／`score`，預設 `rules`，行為不變）。
  - `score`：每個 track 有一個 0–1 的意圖分數，用實際 dt 做 EMA。線索有：
    - 面向（YuNet 五點的正臉分數＋肩膀方向）；
    - 對話距離（距離估計關閉時自動略過）；
    - 停留時間；
    - 走近／走遠；
    - 是否在橫向路過；
    - 是否接近 ROI 中央；
    - 可歸屬的說話（弱，權重依 priority_mode）。
  - 分數還會經過一個「面向」注意力閘門。
  - 加入：意圖 ≥ `join_threshold` 持續 `join_seconds`。session 也要有人達到這個條件才開始。
  - 降為旁觀者：意圖 < `drop_threshold` 持續 `drop_seconds`；最後一位參與者不會被降級。
  - NEW_GROUP 的新候選人也要達到加入條件。
  - hmm 的群組 P 只看參與者；可以選擇依意圖加權（`intent_weighted_presence`）。
- 新增 `presence_demo/intent.py`。
- `Detection.frontal`：視覺管線從 YuNet 五點計算正臉分數。
- state 每個 track 新增 `intent`、`intent_cues`、`role`（`participant`／`bystander`）。
- 網頁：每個人的意圖條和「參與者／旁觀者」標籤；新增「互動意圖」設定分組。
- 評估：新增 6 個情境，以及「誤加入」「加入時間」指標；比較版本加入 `kalman+hmm+score`。
- 新增 [`docs/INTENT_MODEL.md`](docs/INTENT_MODEL.md)。

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
