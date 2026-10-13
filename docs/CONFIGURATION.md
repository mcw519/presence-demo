# 設定參數

伺服器端參數在 `presence_demo/tracker.py` 的 `Config`，範圍在 `RANGES`。網頁送來的值會先檢查、限制在範圍內再套用（只套用到該連線），並回傳警告。`GET /api/config` 會回傳預設值和範圍。

## 判斷優先

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `priority_mode` | `vision` | `vision`／`balanced`／`audio` | 判斷優先（見下方說明） |
| `present_max_silence` | 0 | 0–3600 s | vision 模式下，人在畫面裡時最長允許的靜默；0＝不限 |

- **vision（影像優先）**：影像有信心時，畫面有參與者就不會因為靜默判定離開；ROI 確定空了 `empty_timeout` 秒就 AWAY，不管有沒有聲音；單靠聲音不會開始 session，也不會從 AWAY 恢復。影像不確定時才退回用聲音（`base_timeout`）。在 ROI 裡說話仍會立即加入參與者。適合吵雜環境。
- **balanced（平衡）**：有人時用 `present_timeout`、ROI 空時用 `empty_timeout`（計時取 min(靜默, 空的秒數)）、不確定時用 `base_timeout`；說話一定能維持對話。
- **audio（聲音優先）**：只看靜默計時，影像只調整逾時（有人 ×1.5、ROI 空 ×0.5 倍 `base_timeout`）。

## 偵測範圍

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `roi` | (0.2, 0.05, 0.8, 0.95) | 各值 0–1；寬、高至少 0.05 | 正規化座標 x0,y0,x1,y1；臉的中心點要在框內才算。會自動排序 |
| `min_conf` | 0.6 | 0–1 | 低於這個值的偵測視為低信心 |

## 身分比對

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `cos_match` | 0.40 | 0–1 | cosine 相似度門檻，達到就算同一人（SFace 參考值約 0.363） |
| `cos_weak` | 0.20 | 0–1，≤ `cos_match` | 位置重疊時使用的較低門檻 |
| `iou_match` | 0.30 | 0–1 | 位置重疊門檻 |
| `embed_ema` | 0.15 | 0–1 | 模板更新速度 |

## 加入與防閃爍

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `join_seconds` | 2.0 | 0–30 s | 在 ROI 裡待滿這麼久才成為參與者 |
| `grace_seconds` | 15 | 0–120 s | 消失多久才移除 track |
| `present_frames` | 2 | 1–20（整數） | 連續幾張有人才算「有人」 |
| `absent_frames` | 3 | 1–20（整數） | 連續幾張沒人才算「ROI 空」 |

## 逾時

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `idle_after` | 3 | 0–60 s | 靜默多久從 ACTIVE 轉成 IDLE |
| `present_timeout` | 15 | 1–600 s | balanced 模式下有人時的逾時 |
| `empty_timeout` | 5 | 1–120 s | ROI 空時的逾時 |
| `base_timeout` | 20 | 1–300 s | 影像不確定時的逾時；audio 模式的基準 |
| `maybe_away_frac` | 0.5 | 0.05–0.95 | 計時到逾時的這個比例就進入 MAYBE_AWAY |
| `end_after_away` | 5 | 0–120 s | AWAY 持續多久後 ENDED |
| `new_group_seconds` | 3 | 0–60 s | 換組確認時間 |

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
| 單人 | roi (0.25,0.05,0.75,0.95)、join 1.5 s、grace 10 s、present_timeout 45、empty_timeout 5 |
| 多人圍觀 | roi (0.08,0.02,0.92,0.98)、join 3 s、grace 20 s、absent_frames 4、present_timeout 90、empty_timeout 8、new_group 5 s、cos_match 0.42、fps 2 |
| 吵雜環境 | priority_mode vision、vad_k 3.5、vad_min 0.02、vad_hang 300 ms、min_conf 0.7、base_timeout 15、present_timeout 45 |
| 重設預設值 | 回到上表預設值 |

設定會存在瀏覽器的 localStorage（`presence-cfg-v1`），每次連線都會重送；也可以匯出或匯入 JSON。注意：存過的設定會蓋過新版的預設值，需要時請按「重設預設值」。

## 環境變數

| 變數 | 說明 |
|---|---|
| `PRESENCE_TOKEN` | 設定後，頁面、API、WS 都需要 `?token=` |
| `PRESENCE_MODEL_DIR` | 模型存放目錄（預設 `./models`） |
