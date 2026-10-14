# 設定參數

伺服器端參數在 `presence_demo/tracker.py` 的 `Config`，範圍在 `RANGES`。網頁送來的值會先檢查、限制在範圍內再套用（只套用到該連線），並回傳警告。`GET /api/config` 會回傳預設值和範圍。

## 判斷優先

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `priority_mode` | `vision` | `vision`／`balanced`／`audio` | 判斷優先（見下方說明） |
| `present_max_silence` | 0 | 0–3600 s | vision 模式下，人在畫面裡時最長允許的靜默；0＝不限 |

- **vision（影像優先）**：影像有信心時，畫面有參與者就不會因為靜默判定離開；ROI 確定空了 `empty_timeout` 秒就 AWAY，不管有沒有聲音；單靠聲音不會開始 session，也**不會從 AWAY 恢復**（即使影像不確定也一樣，只有影像確認有人才恢復）。影像不確定時，離開計時才退回用聲音（`base_timeout`）。說話預設**不會**讓人立即加入參與者（`speech_join=auto` → 關閉），要等 `join_seconds`。適合吵雜環境。
- **balanced（平衡）**：有人時用 `present_timeout`、ROI 空時用 `empty_timeout`（計時取 min(靜默, 空的秒數)）、不確定時用 `base_timeout`；說話一定能維持對話，也能讓唯一的候選人立即加入（`speech_join=auto` → `single_candidate`）。
- **audio（聲音優先）**：只看靜默計時，影像只調整逾時（有人 ×1.5、ROI 空 ×0.5 倍 `base_timeout`）。

## 偵測範圍

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `face_roi` | (0.2, 0.05, 0.8, 0.95) | 各值 0–1；寬、高至少 0.05 | 臉部 ROI（藍色），正規化座標 x0,y0,x1,y1；臉的中心點要在框內才算。會自動排序。**舊的 `roi` 會自動轉成 `face_roi`**（伺服器 API、localStorage、匯入 JSON 都相容） |
| `min_conf` | 0.6 | 0–1 | 低於這個值的偵測視為低信心 |

## 身分比對

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `cos_match` | 0.40 | 0–1 | cosine 相似度門檻，達到就算同一人（SFace 參考值約 0.363） |
| `cos_weak` | 0.20 | 0–1，≤ `cos_match` | 位置重疊時使用的較低門檻 |
| `iou_match` | 0.30 | 0–1 | 位置重疊門檻 |
| `embed_ema` | 0.15 | 0–1 | 模板更新速度 |

## 嵌入後端

說明、匯出方式和實機注意事項見 [NPU.md](NPU.md)。

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `embed_backend` | `fp32` | `fp32`／`int8`／`npu` | SFace 的計算方式：fp32 ONNX、官方 int8 ONNX（OpenCV DNN），或 TFLite（LiteRT，可以掛 NPU delegate） |
| `npu_precision` | `int8` | `int8`／`fp16` | npu 後端使用的 TFLite 模型。沒有 delegate 時，int8 會自動改用 fp16（CPU 上比較快） |
| `cos_match_int8` | 0.40 | 0–1 | 使用 int8 或 npu int8 時的「同一人」門檻（實測不需要調整） |
| `npu_delegate_path` | 空 | — | 外部 delegate `.so`。**只能用環境變數 `PRESENCE_NPU_DELEGATE` 設定**，WebSocket 無法修改 |

網頁「資源監控 → Embedding」會顯示實際使用的後端和加速器。

## 加入與防閃爍

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `join_seconds` | 2.0 | 0–30 s | 在 ROI 裡待滿這麼久才成為參與者 |
| `speech_join` | `auto` | `auto`／`off`／`single_candidate`／`all` | 有人說話時是否不等 `join_seconds` 就立即加入（聲音沒有位置資訊）。`auto`：vision 模式為 `off`，其他模式為 `single_candidate`。`single_candidate`：只有一位可見、非 far 的候選人，臉在 `face_roi` 內、朝向為正面，而且開啟距離時在 `zone_max` 以內，才立即加入。`all`：所有可見候選人立即加入（v0.4.0 以前的行為） |
| `grace_seconds` | 15 | 0–120 s | 消失多久才移除 track |
| `present_frames` | 2 | 1–20（整數） | 連續幾張有人才算「有人」 |
| `absent_frames` | 3 | 1–20（整數） | 連續幾張沒人才算「ROI 空」 |

## 逾時

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `idle_after` | 3 | 0–60 s | 靜默多久從 ACTIVE 轉成 IDLE |
| `present_timeout` | 15 | 1–600 s | balanced 模式：有參與者在畫面中但都不說話時的靜默逾時。注意預設 15 s 比 `base_timeout`（20 s）短，所以有人在場時反而比影像不確定時更快判定離開；需要的話請調高 |
| `empty_timeout` | 5 | 1–120 s | ROI 空時的逾時 |
| `base_timeout` | 20 | 1–300 s | 影像不確定時的逾時；audio 模式的基準 |
| `maybe_away_frac` | 0.5 | 0.05–0.95 | 計時到逾時的這個比例就進入 MAYBE_AWAY |
| `end_after_away` | 5 | 0–120 s | AWAY 持續多久後 ENDED |
| `new_group_seconds` | 3 | 0–60 s | 換組確認時間 |

## 追蹤（Layer 4）

比較結果和建議見 [TRACKING.md](TRACKING.md)。除了 `tracker_mode`，其他參數只在 `kalman` 模式下有作用。

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `tracker_mode` | `legacy` | `legacy`／`kalman` | `legacy`：IoU + 臉部 cosine，逐一貪婪配對（v0.4 行為）。`kalman`：Kalman 等速預測、門檻篩選、IoU + 移動 + 臉 + 衣服顏色的綜合分數、匈牙利演算法配對、短暫遮擋時沿預測位置延續 |
| `coast_seconds` | 2 | 0–10 s | 沒偵測到時沿預測位置延續多久（`visible_kind=predicted`），之後變成不可見；寬限期照常計算 |
| `coast_present_conf` | 0.4 | 0–1 | 預測中的 track，`vis_conf`（1 → 0 線性下降）≥ 這個值時仍然算有人，也會延後 ROI 空的判定。預設約延續 1.2 s；調高可以更快判定離開 |
| `gate_mahalanobis` | 9.21 | 1–100 | 偵測中心和預測中心的 Mahalanobis² 門檻（2 自由度 99%）；超過而且 IoU 也很低時不配對，除非臉部相似度 ≥ `cos_match` |
| `appearance_min_sim` | 0.6 | 0–1 | 看不到臉時，衣服顏色（軀幹 HSV 直方圖，Bhattacharyya 係數）低於這個值就視為不同的人 |
| `appearance_ema` | 0.2 | 0–1 | 衣服顏色模板的更新率 |

## 在場機率（Layer 2）

數學說明和評估見 [PRESENCE_MODEL.md](PRESENCE_MODEL.md)。除了 `presence_model`，其他參數只在 `hmm` 模式下有作用；`hmm` 模式不使用 `present_frames`／`absent_frames` 和各模式的逾時。

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `presence_model` | `rules` | `rules`／`hmm` | 判斷模型。rules：計數＋逾時；hmm：每個人的在場機率 |
| `leave_halflife` | 0.6 | 0.1–30 s | 完全沒看到人時，在場勝算減半的時間。預設約 5 s 判定 AWAY |
| `detect_gain` | 4.0 | 0.1–20 | 偵測證據強度（每秒，乘上信心） |
| `lowconf_factor` | 0.15 | 0–1 | 只有低信心偵測時，下降速度是「完全沒看到」的幾倍 |
| `p_active` | 0.8 | 0.05–0.99 | 群組 P ≥ 這個值才回到 ACTIVE |
| `p_maybe` | 0.6 | 0.02–0.98 | 群組 P < 這個值進入 MAYBE_AWAY |
| `p_away` | 0.3 | 0.01–0.95 | 群組 P < 這個值並持續 `away_hold_seconds`，判定 AWAY |
| `away_hold_seconds` | 1.0 | 0–30 s | AWAY 確認時間 |

## 互動意圖（Layer 3）

說明和評估見 [INTENT_MODEL.md](INTENT_MODEL.md)。除了 `intent_model`，其他參數只在 `score` 模式下有作用；`score` 模式不使用 `speech_join`。

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `intent_model` | `rules` | `rules`／`score` | rules：待滿 `join_seconds` 就加入；score：意圖分數 |
| `join_threshold` | 0.6 | 0.05–0.99 | 意圖 ≥ 這個值並持續 `join_seconds` 才加入或開始 session |
| `drop_threshold` | 0.35 | 0–0.95 | 參與者意圖 < 這個值並持續 `drop_seconds` → 旁觀者（不會降級最後一位） |
| `drop_seconds` | 8 | 0–120 s | 降級時間 |
| `intent_tau` | 1.0 | 0.1–20 s | EMA 時間常數 |
| `intent_weighted_presence` | 關 | 開／關 | hmm 的群組 P 依意圖加權 |
| `w_facing`／`w_distance`／`w_dwell`／`w_approach`／`w_still`／`w_center` | 0.45／0.1／0.1／0.05／0.15／0.15 | 0–1 | 各線索權重（會自動正規化；沒有的線索會略過） |

## 姿勢（選用，預設關閉）

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `pose_enabled` | 關 | 開/關 | 啟用 MoveNet MultiPose Lightning 身體與關鍵點偵測 |
| `pose_conf` | 0.4 | 0.05–1 | 身體偵測信心門檻（低於這個值的身體直接丟掉） |
| `body_min_conf` | 0.4 | 0.05–1 | 只有身體、沒有臉的偵測算「有信心」（可以開始 session、算有人）的門檻；有臉的偵測仍用 `min_conf` |
| `pose_roi` | (0, 0, 1, 1) | 同 ROI | 身體 ROI（紫色）；身體框中心點要在框內 |
| `roi_rule` | `either` | `either`／`face`／`body` | 參與者判定規則：臉在臉部 ROI **或**身體在身體 ROI／只看臉／只看身體。姿勢關閉時一律只看臉 |
| `turned_away_early` | 開 | 開/關 | 所有參與者轉身（側面或背對）**且**正在遠離時，提早進入 MAYBE_AWAY |
| `turned_away_seconds` | 2 | 0–30 s | 上述狀態要持續多久 |
| `recede_ratio` | 0.1 | 0.01–0.9 | 約 2 秒內身形（沒有身體時用臉寬）縮小超過這個比例就視為遠離（靠近則是放大超過這個比例） |

身體朝向依據：鼻子和雙眼是否可見、左右肩在影像中的左右關係（正面時人的左肩在影像右側），以及肩寬和軀幹高度的比例。結果分成 `facing`（正面）、`side`（側身）、`away`（背對）。

網頁上的「顯示臉部 ROI」和「顯示身體 ROI」只影響畫面顯示，不會送到伺服器。

## 距離（選用，預設關閉）

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `distance_enabled` | 關 | 開/關 | 啟用距離估計、對話區和離開距離 |
| `camera_hfov` | 60 | 20–170° | 鏡頭水平視角 |
| `face_width_m` | 0.15 | 0.08–0.3 m | 平均臉寬 |
| `shoulder_width_m` | 0.38 | 0.2–0.6 m | 平均肩寬（只在正面時使用） |
| `distance_calib` | 1.0 | 0.2–5 | 校正係數 |
| `distance_ema` | 0.3 | 0.01–1 | 平滑時新估計值的權重 |
| `zone_min` / `zone_max` | 0.4 / 2.0 | 0–10 / 0.1–20 m | 對話區（`zone_min` 不會大於 `zone_max`） |
| `leave_distance` | 2.5 | 0.2–30 m | 超過這個距離，即使還在畫面裡也視為已離開（不算「有人」，也不會加入參與者） |

公式（針孔模型）：`距離 = 實際寬度 / (正規化寬度 × 2 × tan(HFOV/2)) × 校正係數`。同時有臉寬和肩寬時取平均。

**校正**：站在已知距離（例如 1.0 m），在設定的「距離」分組輸入距離，按「我站在這個距離」。伺服器會用目前最近那個人的未校正估計值計算 `distance_calib = 已知距離 / 估計值`（WebSocket 訊息 `{"type":"calibrate","meters":1.0}`）。

每個人的對話區標記：`in`（對話區內）、`near`（太近）、`far`（偏遠）、`left`（超過離開距離）。趨勢：`approaching`、`receding`、`steady`。

## 效能

細節和基準測試結果見 [PERFORMANCE.md](PERFORMANCE.md)。

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `motion_gate` | 開 | 開/關 | 移動偵測閘門：畫面幾乎沒變而且狀態穩定時跳過偵測 |
| `motion_threshold` | 0.015 | 0–0.2 | 64×48 灰階縮圖的平均差異門檻（0–1） |
| `force_detect_seconds` | 3 | 0.5–30 s | 至少每隔這麼久完整偵測一次 |
| `embed_throttle` | 開 | 開/關 | Embedding 節流 |
| `embed_interval` | 3 | 0–30 s | 每張臉至少每隔這麼久重算一次 |
| `embed_move_thresh` | 0.15 | 0–1 | 中心位移或大小變化（相對臉框）超過這個值就重算 |
| `adaptive_fps` | 開 | 開/關 | 由伺服器建議 fps，開啟時取代瀏覽器的 `fps` |
| `fps_low` | 0.5 | 0.2–5（≤ `fps_high`） | 穩定時的 fps |
| `fps_high` | 2.0 | 0.2–5 | 有變化或 MAYBE_AWAY 時的 fps |
| `roi_crop` | 開 | 開/關 | 偵測前先裁切 ROI 並縮小 |
| `crop_margin` | 0.1 | 0–0.5 | 裁切時 ROI 四周多保留的範圍 |
| `max_width` | 320 | 96–1920 px（整數） | 送進偵測器的最大寬度 |

布林值接受 `true`/`false`、`1`/`0`、`on`/`off`。

## 瀏覽器端（不送到伺服器）

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `fps` | 1 | 0.5–5 | 每秒送出幾張畫面（開啟 `adaptive_fps` 時不使用） |
| `vad_k` | 2.5 | 1.2–8 | 說話門檻＝噪音底 × 這個倍數 |
| `vad_min` | 0.005 | 0.001–0.05 | 最低 RMS 音量門檻 |
| `vad_hang` | 400 | 100–1500 ms | 音量變小後維持「說話中」多久 |

## 預設組合

預設組合是套用在預設值之上的差異：

| 組合 | 內容 |
|---|---|
| 單人 | face_roi (0.25,0.05,0.75,0.95)、join 1.5 s、grace 10 s、present_timeout 45、empty_timeout 5 |
| 多人圍觀 | face_roi (0.08,0.02,0.92,0.98)、join 3 s、grace 20 s、absent_frames 4、present_timeout 90、empty_timeout 8、new_group 5 s、cos_match 0.42、fps 2 |
| 吵雜環境 | priority_mode vision、vad_k 3.5、vad_min 0.02、vad_hang 300 ms、min_conf 0.7、base_timeout 15、present_timeout 45 |
| 重設預設值 | 回到上表預設值 |

設定會存在瀏覽器的 localStorage（`presence-cfg-v1`），每次連線都會重送；也可以匯出或匯入 JSON。注意：存過的設定會蓋過新版的預設值，需要時請按「重設預設值」。

## 環境變數

| 變數 | 說明 |
|---|---|
| `PRESENCE_TOKEN` | 設定後，頁面、API、WS 都需要 `?token=` |
| `PRESENCE_MODEL_DIR` | 模型存放目錄（預設 `./models`） |
| `PRESENCE_NPU_DELEGATE` | npu 後端的外部 TFLite delegate 函式庫路徑（例如 NNAPI／NeuroPilot）；沒有設定時用 CPU |
