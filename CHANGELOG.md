# Changelog

## 0.10.4 — 2026-10-16

- **狀態機改成軸上的節點**：ACTIVE → IDLE → MAYBE AWAY → AWAY → ENDED 排成一條軸，目前狀態的節點高亮（狀態色），最近一次轉換的來源節點以虛線標示、經過的線段以強調色加箭頭（往回轉換箭頭反向）。軸下方列出最近 6 次轉換（時間、來源 → 目標，最後一筆附原因）。
- **音軌依訊號分組**：「原始 noisy」一組＝波形＋頻譜，「輸出 output」一組＝波形＋頻譜，兩組分開並以 mint／periwinkle 色條區分；頻譜改為全寬。
- **臉部與身體 ROI 都能直接在影像上拖曳設定**：相機上方新增「編輯 ROI：臉部／身體／鎖定」選擇與「重設此 ROI」。只有選中的 ROI 可拖曳（避免重疊時抓錯框）：框內拖曳移動、四角縮放、框外拖曳畫新框；正在編輯的身體 ROI 即使姿勢偵測關閉也會顯示並提示。鏡像顯示時座標一律換算回原始相機座標儲存（新增純函式 `roiMath` 與測試）。數字輸入仍保留在設定中。
- 修正 `app.css` 裡 `--line:var(--line)` 的自我參照（讓 `--line` 變成無效值，邊框退回 currentColor），現在直接使用上游 token。

## 0.10.3 — 2026-10-15

- **真正貼齊 PureSound playground 的設計，不只是字體與色票。** HTML 改用 vendored 的 PureSound 元件 class（`.button` / `.button-primary` / `.panel.panel-pad` / `.panel-title` / `.metric` / `.segmented` / `.chip` / `.stat` / `.table` / `.toast` / `.toggle-row` / `.control` / `.range` / `.inspector` / `.eyebrow` / `.empty-state` / `.status-dot`），並載入完整的 `shell.css` 與 `screens-shared.css`。自訂樣式收到 `app.css`，只留相機、狀態機、倒數、scope 底色、追蹤名單、設定模組手風琴等 presence 專用佈局，不再重寫 button/panel/token。
- 版面：playground 式 page-header（標題＋說明＋actions）、workbench（相機 panel ＋右側 panel 堆疊）、設定改為 Inspector 抽屜（eyebrow 分節、segmented 預設、toggle 進階、field/control/range）。監聽改 segmented；指標改 metric；連線改 status-dot；toast / 錯誤橫幅用 playground toast 樣式。
- `docs/DESIGN.md` 改寫為元件對照表。測試確認 index 使用共用 class、且 `app.css` 不重新定義上游 token／component。

## 0.10.2 — 2026-10-15

- **UI 改用 PureSound playground 的設計系統**：直接載入 PureSound web 的 `tokens.css`、`base.css`、`components.css` 與 IBM Plex 字型（釘在與 `puresound-streaming` 相同的 revision `121bf62`，放在 `presence_demo/static/vendor/puresound/`，Apache-2.0／OFL，授權檔一併附上），不再維護自己的一份色票。App 自己的 CSS 只把 presence 的角色對應到上游 token。
- 版面改成 playground 的 shell：深色側欄（導覽、連線狀態、主題 自動／淺色／深色、版本）＋頁首（標題、說明、主要動作）＋右側面板欄。主要按鈕用 ink 色、強調色（橘）只用於焦點、選取和進行中；原始音訊＝mint、乾淨音訊＝periwinkle，與 playground 同義；波形、頻譜、時間軸、sparkline 放在常駐深色底上（同 playground 播放器）。設定抽屜、事件紀錄、錯誤橫幅、toast 套用 Inspector／toast 樣式。
- 主題預設改為「自動」（跟隨系統），仍可固定淺色／深色。
- `/static/vendor/` 下的公開 CSS／字型不需 token，其餘路徑仍受 token 保護；首頁顯示版本號。
- `docs/DESIGN.md` 改為記錄與 PureSound token 的對應；NOTICE 補上第三方 CSS／字型。
- 新增 `tests/test_design_tokens.py`（vendored 檔案與釘選 checksum、未重新定義上游 token、vendor 公開但 app 受保護）。
- 無頭 Chrome：深／淺色主畫面與設定頁沒有 JS 錯誤，文字對比全部符合 WCAG AA。

## 0.10.1 — 2026-10-15

- **修正：啟用姿勢後臉部追蹤與音訊停止**。伺服器原本在影像／音訊處理工作中直接 `await` 傳送給瀏覽器；慢的或經過轉送的連線（每個連線約 50 kB/s，兩個分頁加倍）會讓處理工作卡在傳送上，最後連線因 keepalive 被關閉；工作中的例外也不會被記錄。現在所有傳送改走每連線一個非阻塞的傳送佇列（Outbox：過舊的音訊與可丟的訊息會被丟棄並計數），state 訊息只在設定變更時附上設定。
- 工作（ticker、影像、音訊）由監督器執行：例外會記錄完整 traceback 並自動重啟；watchdog 偵測處理停滯，記錄所有執行緒的 stack dump、重啟工作，並在 UI 顯示錯誤橫幅。
- 共用偵測器鎖加上逾時（`PRESENCE_VISION_LOCK_TIMEOUT`），逾時會記錄 stack dump。
- **結構化日誌**（`docs/LOGGING.md`）：連線生命週期、設定變更（密鑰遮罩）、模組載入時間、每 10 秒心跳計數、狀態轉換與 session 事件、交叉確認計數、門檻警告。`PRESENCE_LOG_LEVEL`、`PRESENCE_LOG_FILE`（輪替）。不記錄影像、音訊、逐字稿或密鑰。
- UI 新增「事件紀錄」面板；`/debug/events`（需 token）。
- 新增 `tests/test_stall_0101.py`：音訊串流中途啟用姿勢、兩個同時連線、工作崩潰後重啟、慢連線不阻塞、`/debug/events` 保護、日誌遮罩。

## 0.10.0 — 2026-10-15

- **選用的錄音（評估效果用）**：「乾淨音軌」面板的錄音按鈕，預設關閉。三軌 raw（已補償 PureSound 延遲）／vi（PureSound 輸出）／out（影像交叉確認後）逐樣本對齊、同長度，16 kHz mono PCM16 WAV，加上 JSON（每 100 ms 判斷、狀態時間軸、設定快照），打包成一個 zip 下載到使用者的電腦。最長 10 分鐘，標題列有錄音中指示。錄音只在瀏覽器記憶體；伺服器只在錄音時多串流 vi 與對齊的 raw（WS `0x03`），不寫任何檔案。影像從不錄製。
- **設定頁重新設計**：依資料流分成 12 個模組（輸入 → 視覺偵測 → 姿勢 → 距離 → 追蹤 → 互動意圖 → 在場判斷 → 狀態機／逾時 → 音訊 → 語音辨識 → 效能 → 顯示）。上方是可點擊的資料流狀態條；每個模組最上面是啟用／模式，相依參數縮排且只在相關時顯示（例如 Kalman 參數只在 Kalman 模式），不適用的參數變灰並說明原因（例如距離關閉時跳過情況 3）。基本／進階切換（進階預設隱藏，改過的仍顯示）、每個模組的重設、新的預設組合「推薦新組合」（Kalman + 機率 + 意圖分數）與「低功耗」。設定鍵與匯出／匯入格式不變。
- **整體 UI 改版**：新的設計規範 `docs/DESIGN.md`（DESIGN.md 格式）；深色為主、可切換淺色；所有顏色改用 CSS 變數 token，圖表也讀同一組 token；數字用等寬字與 tabular figures；焦點框、減少動態效果。無頭 Chrome 檢查深／淺色的主畫面與設定頁：沒有 JS 錯誤、文字對比全部符合 WCAG AA。
- `docs/CONFIGURATION.md` 依模組重新整理，加上模組相依表。
- 新增 `tests/test_recording.py`、`tests/test_settings_ui.py`。


## 0.9.1 — 2026-10-15

修正遠端（Tailscale Serve）實測回報的兩個問題。

- **監聽延遲很高**：根因在瀏覽器的播放器。舊的播放佇列最多累積 25 塊（500 ms）才砍回 5 塊，網路一次卡頓後湧入的聲音就變成**永久延遲**（之後送達與播放同速，佇列不會消化），麥克風與喇叭的時脈漂移也會慢慢累積。
  - 新播放器：固定小 jitter buffer（累積 60 ms 才開始、目標 80 ms），超過 160 ms 立刻丟最舊的聲音回到目標，超過目標 + 20 ms 持續 0.5 s 就丟一塊（慢速追趕）；斷音時重新預先緩衝；線性內插重取樣；`AudioContext({latencyHint:'interactive'})`。
  - 伺服器：人聲分離改在專用的音訊執行緒池（不再和影像共用預設 thread pool），每塊只跳一次執行緒；積壓時一次處理整批；佇列上限從 1 s 降到 500 ms。
  - UI 新增「監聽延遲（擷取→播放）」：擷取 20 + 演算法 + 往返 p50/p95 + 播放緩衝 + 輸出裝置延遲，以及斷音次數、追趕丟棄量、伺服器佇列。
  - 模擬（60 ms 單程網路、每 20 s 一次 400 ms 卡頓、0.2 % 時脈差）：擷取→播放（不含演算法／輸出）p50 479 → 179 ms，最大 480 → 180 ms；在 node 中執行真正的 worklet 程式碼，卡頓後緩衝最多 117 ms（舊版 ≥ 400 ms）。伺服器端（本機 WS）往返 p50 6.5 → 6.9 ms、最大 23.8 → 11.6 ms，原本就不是瓶頸。
- **常常偵測不到臉**，三個根因：
  1. **`max_width` 預設 320 太小**：ROI 裁切後（640×480 畫面約 512 px 寬）縮到 320 px，臉高約 80 px 以下（約 1.5 m 以外的筆電鏡頭）就偵測不到。預設改成 640（實際不縮小，偵測 2.4 → 5 ms）；瀏覽器裡存過的 320 會自動升級一次。
  2. **多個連線共用同一個 YuNet 偵測器但沒有同步**：`cv2.FaceDetectorYN` 的輸入尺寸是可變狀態、不是 thread-safe，兩個分頁／裝置同時偵測會得到錯誤結果，甚至讓程序崩潰（重現：兩個執行緒同時偵測不同尺寸 → `double free or corruption`）。所有影像推論改成經過同一把鎖。
  3. **移動閘門會沿用「沒有臉」的結果**：畫面變化小（小臉或慢慢走進來）時最多 3 s 都不重新偵測。現在只有上一次**有**偵測到臉時才會跳過。
  - 已確認**不是**鏡像問題：送出的畫面不翻轉（鏡像只在 CSS），ROI 一律以原始相機座標儲存，加了回歸測試。
  - 資源監控新增「偵測率」（有臉的偵測次數／偵測次數，記憶體中的計數器）。
- 安全：uvicorn 的 WebSocket 握手 log 會印出 `?token=`，現在會遮成 `token=***`。
- 新增 `tests/test_face_misses_091.py`、`tests/test_player_js.py`（有 node 才執行）。


## 0.9.0 — 2026-10-15

- **新增 PureSound 串流人聲分離**（`audio_pipeline = puresound`；預設仍是 `energy_vad`，行為不變）。詳見 `docs/AUDIO.md`。
  - PureSound 可攜式 SDK（`sdk/python`，Apache-2.0）以**選用相依**安裝：`uv sync --extra puresound`（uv git 來源，公開 repo github.com/mcw519/puresound，固定 commit `121bf62`）。沒有複製任何 PureSound 程式碼，只有 `presence_demo/audio.py` 的 glue。
  - 模型 `dpcrn_curriculum_v2`（ONNX + manifest）用 `scripts/fetch_puresound_model.py` 從 PureSound clone 複製到 gitignored `models/`，並檢查 sha256。
  - 瀏覽器 AudioWorklet 擷取麥克風 → 16 kHz PCM16 → 每 20 ms 一個 WS binary 訊框；伺服器回傳修正後的乾淨音軌（binary）和每 100 ms 的 `audio` 中繼資料。
  - 乾淨 VAD（v2 沒有 VAD 輸出頭，用分離後訊號的能量 + 尾音）。
  - **影像交叉確認**每 20 ms 一次：漏音（有乾淨說話但沒人）、聲學誤判-超出範圍（所有人都 > `near_range` 1.0 m）→ 壓低、不算說話；聲學誤判-漏收（近距離有人、原始有說話、模型沒聲音）→ 多放原聲、算說話；影像不確定時只看聲音。
  - **進場規則**：乾淨音源下，session 只在「修正後說話開頭 + 近距離有人」時開始（`onset_entry_only`）。
  - NEW_GROUP／新 session／參與者換人時重置分離串流狀態（重新啟動 onset guard）。
  - UI：「乾淨音軌」面板（監聽 不監聽／原始／乾淨，預設靜音，耳機警告；原始與乾淨波形、電平；判斷標籤與次數；RTF、延遲），設定群組「乾淨音軌（PureSound）」。
- **選用的雲端語音辨識**（`asr_provider = assemblyai`，預設 `none`）：AssemblyAI Universal-Streaming v3 WebSocket，由伺服器連線；只在 session 進行中、預設只送校正後算說話的片段（其餘送靜音）；ENDED 時斷線；自動重連。API key 來自環境變數或 UI 密碼欄位，只存在伺服器記憶體、不寫 log、不回傳瀏覽器。逐字稿（partial / final、相對 session 的時間）只顯示在 UI。詳見 `docs/ASR.md`。**聲音會離開裝置**。
- UI：原始／乾淨頻譜、30 秒時間軸（原始 VAD／乾淨 VAD／算說話／判斷／狀態 + P(group)）、閘門狀態、影像與音訊的時間對齊（畫面訊息帶 `performance.now()`）、資源監控中的人聲分離 RTF／延遲。
- **開源準備**：改為 Apache-2.0（新增 `LICENSE`、`NOTICE`），文件改成中性用語，移除個人／私有環境的描述；PureSound 描述為公開的外部相依。
- 效能（x86 測試伺服器，單執行緒）：分離 RTF 0.22–0.27，演算法延遲 52–62 ms，WS + 計算 p50 6 ms；伺服器 CPU 影像 3.8 % → 影像 + 音訊 31.4 %。
- 輸出驗證：glue vs SDK、SDK vs PureSound `StreamingOrt` 最大差 0.0。
- 新增 `tests/test_asr.py`（mock AssemblyAI 伺服器、key 不外洩）。
- 新增 `tests/test_audio.py`（SDK smoke 與分塊不變性〔沒有 SDK 時跳過〕、四種交叉確認情況、影像不確定的退回、距離關閉、保持時間、NEW_GROUP 重置、乾淨 VAD、進場規則、WS 音訊來回不寫檔）。
- 新文件 `docs/AUDIO.md`；更新 ALGORITHM、CONFIGURATION、PERFORMANCE、LICENSES。

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
