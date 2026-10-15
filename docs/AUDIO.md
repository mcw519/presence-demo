# 音訊：PureSound 人聲分離 + 影像交叉確認（v0.9.0）

`audio_pipeline = puresound` 時，麥克風聲音送到伺服器，用 PureSound 的串流人聲分離（`dpcrn_curriculum_v2`）一直過濾，
收音範圍大約 1 m。畫面再逐塊（20 ms）確認過濾的結果，修正聲學模型的錯誤，最後產生：

- 修正後的**乾淨音軌**（回傳瀏覽器，可監聽）；
- 修正後的**說話旗標**，取代瀏覽器 VAD 餵給狀態機；
- 每塊的**判斷標籤**與計數（UI「乾淨音軌」面板）。

預設 `audio_pipeline = energy_vad`：行為和 v0.8.2 完全一樣（瀏覽器音量 VAD），不需要安裝 PureSound。

![音訊資料流](img/audio.png)（原始檔 `img/audio.mmd`）

## 1. 安裝

PureSound 不是 vendored 的程式碼，而是**選用相依套件**：PureSound repo 裡的獨立 SDK `sdk/python`（`puresound-streaming`，
Apache-2.0，只依賴 numpy + onnxruntime），用 uv git 來源，固定在 commit `121bf62`。

```bash
uv sync --extra puresound          # 從公開 repo github.com/mcw519/puresound 安裝（需要 git 與網路）
uv run python scripts/fetch_puresound_model.py --clone   # 或 --from /path/to/puresound（已有 clone 時）
```

- 模型（`dpcrn_curriculum_v2.onnx` 4.9 MB + `.json` manifest）複製到 gitignored 的 `models/`，sha256 和 PureSound
  `model_zoo/catalog.yaml` 一致才會放進去。SDK 本身沒有模型下載功能，所以用這個腳本。
- PureSound 是公開的開源專案（Apache-2.0）：https://github.com/mcw519/puresound 。
- 需要 Python ≥ 3.12（SDK 的要求）。更舊的 Python 只能用 `energy_vad`。
- 注意：`uv sync`（不帶 `--extra puresound`）會把 SDK 移除；`uv run` 不會。
- 沒裝或沒模型時，選 `puresound` 會在面板顯示原因，狀態機繼續用瀏覽器 VAD。

## 2. 串流

| 方向 | 格式 |
|---|---|
| 瀏覽器 → 伺服器 | WS binary：`0x01` + float64 LE（`performance.now()` ms）+ PCM16 LE，16 kHz mono，320 樣本（20 ms） |
| 伺服器 → 瀏覽器 | WS binary：`0x02` + 回傳最近一塊輸入的時間戳 + 修正後 PCM16 |
| 伺服器 → 瀏覽器 | JSON `{"type":"audio"}` 每 100 ms：`t_vision_client`（決策所用畫面的瀏覽器時間，畫面訊息帶 `t`＝`performance.now()`，與音訊同一時間基準）、`asr` 狀態、`decision`、`why`、`counts`、`speech`、`clean_vad`、`raw_vad`、`gain`、`mix`、`target_track`、`distance`、`p_group`、`state`、`rtf`、`proc_ms`、`queue`、`dropped`、`resets`、`algo_latency_ms`、`t_server`、`t_client` |

- 瀏覽器：`AudioWorklet` 取原生取樣率（例如 48 kHz），線性內插重取樣到 16 kHz（降取樣時先做 2 點平均低通），每 320 樣本送出一塊。
- 伺服器：每個連線一個佇列（最多 50 塊 = 1 s，滿了丟最舊的並計數），依序處理；ONNX session 全程序共用（單執行緒），
  每個連線有自己的串流狀態。
- 隱私：音訊只在記憶體（worklet 緩衝、WS、佇列、numpy 陣列），不寫檔、不記 log。`tests/test_audio.py` 檢查整段 WS 來回不寫任何檔案。

## 3. PureSound 部分（與上游完全相同）

直接呼叫 SDK 的 `StftFrameOrtProcessor`：STFT 512 / hop 160（10 ms）/ 16 kHz、前瞻 3 幀（30 ms）、
manifest 的 `dry_blend = 0.9`（`0.9 × 增強 + 0.1 × 時間對齊的原聲`），可選 onset guard（`iso_onset_guard`，預設關，
和上游 catalog 一樣）。演算法延遲 52–62 ms（manifest `algorithmic_latency`）。

驗證（x86 測試伺服器，合成類語音訊號 6 s，記憶體內）：
- 本專案的 glue（mix = 0.9）vs SDK `PureSoundStreamingRuntime`：最大絕對差 **0.0**；20 ms vs 517 樣本分塊：**0.0**。
- SDK vs PureSound 自己的 `puresound.streaming.StreamingOrt`（另一個 venv）：**0.0**；開 onset guard 時也是 **0.0**。

v2 匯出**沒有 VAD 輸出頭**（`extra_output_names = []`），所以「乾淨 VAD」是分離後訊號的能量 VAD：
10 ms 一幀，自適應噪音底（下降立即、上升約 3 dB/s），超過 `max(clean_vad_min_db, 噪音底 + clean_vad_margin_db)`
持續 60 ms 算開頭，`clean_vad_hang` 秒的尾音。原始 VAD 用同樣的演算法跑在輸入上（只給「漏收」判斷用）。

## 4. 影像交叉確認（`xcheck_enabled`，預設開）

影像每秒 0.5–2 張，聲音 20 ms 一塊。對齊方式：每塊都讀目前的追蹤快照；「ROI 內有人」「有人在 `near_range` 內」
「所有人都在範圍外」三個條件各自**保持 `xcheck_hold`（0.75 s）**；「範圍外」只有在保持時間內沒有任何近距離的人時才成立
（保護近距離說話者的句首）。

| 乾淨 VAD | 原始 VAD | 影像 | 判斷 `decision` | 輸出 | 算說話 |
|---|---|---|---|---|---|
| 有 | – | ROI 內沒人 | `leak` 漏音 | 壓到 `xcheck_floor_db`（-40 dB） | 否 |
| 有 | – | 所有人 > `near_range` | `false_accept` 聲學誤判-超出範圍 | 壓低 | 否 |
| 有 | – | 有人 ≤ `near_range` | `agree` 一致 | 不變 | 是 |
| 無 | 有 | 有人 ≤ `near_range` | `false_reject` 聲學誤判-漏收 | 增強比例降到 `xcheck_recover_mix`（0.3），其餘是時間對齊的原聲 | 是 |
| 其他 | | | `silence` 安靜 | 不變 | 否 |
| 任何 | | 影像不確定 | `acoustic_only` 只看聲音 | PureSound 原樣 | = 乾淨 VAD |

- **影像不確定**：沒有開始、超過 `xcheck_stale`（2 s）沒有處理過的畫面、追蹤路徑是 `uncertain`（低信心偵測）、或關閉交叉確認。
- **距離**：只有開啟距離估計時才用；關閉時「近距離」= ROI 內有人，`false_accept` 不會發生。
  `xcheck_require_calib = true` 時，還要先做過距離校正。
- 增益淡入 `xcheck_attack`（10 ms）、淡出 `xcheck_release`（50 ms）；增強比例 50 ms 平滑。
- 判斷用的是**前一塊**為止的 VAD（最多 20 ms 落後）；乾淨 VAD 本身在演算法延遲之後。
- 「次數」是進入每個判斷的次數（不是塊數）。

## 5. 進場（開始對話）規則

`audio_pipeline = puresound` 且有音訊串流時（`speech_source = clean`）：

- `onset_entry_only = true`（預設）：**只有**「修正後的說話開頭」+「最近 1.5 s 內 ROI 內有一位信心足夠、不在 leave_distance 外的人」
  同時成立才開始 session（所有 priority_mode）。只有人經過、只有聲音（電視、遠處的人、空房間）都不會開始。
  修正後的說話在影像可用時本來就要求 `near_range` 內有人，所以實際上是「近距離有人 + 說話開頭」。
- `onset_needs_vision = true`：從 AWAY 用說話回來也要畫面有人（影像不確定時例外）。
- `speech_join = auto` 在乾淨音源下一律用「唯一候選人」（乾淨的近距離說話夠可靠，vision 模式也一樣）。
- 修正後的說話取代瀏覽器 VAD（`vad` 訊息和畫面裡的 `speech` 欄位會被忽略）；音訊停止超過 1 s 自動回到瀏覽器 VAD。

## 6. 換人時重置

NEW_GROUP、session 改變、或參與者集合換成另一組人時（`iso_reset_on_change`，預設開），重置 PureSound 串流狀態
（SDK `reset()`，也會重新啟動 onset guard）。重置後約 32 ms 沒有輸出（重新填滿分析視窗）。

## 7. 效能（x86 測試伺服器，單執行緒 ORT）

| 項目 | 數值 |
|---|---|
| 分離 RTF（單執行緒，含 glue） | 0.22–0.27（每 20 ms 塊約 5 ms） |
| 演算法延遲 | 52–62 ms（30 ms 前瞻 + 32 ms 視窗） |
| WS + 佇列 + 計算（localhost，p50 / p95） | 6.0 / 7.8 ms |
| 端到端（估計） | 擷取 20 ms + 演算法 52–62 ms + 傳輸/計算 6–8 ms + 瀏覽器播放緩衝 ≈ 85–100 ms |
| 伺服器 CPU（1 張臉、2 fps 影像） | 只有影像 3.8 % → 影像 + 音訊 31.4 %（一顆核心 = 100 %） |

## 8. 需要實機驗證

- `near_range`、`xcheck_hold`、`clean_vad_*` 的預設值都還沒在實際房間調過；乾淨 VAD 的 -50 dBFS 門檻和麥克風增益有關。
- 瀏覽器 `echoCancellation/noiseSuppression` 開著，PureSound 收到的是已經處理過的聲音（和 Kebbi 的麥克風鏈不同）。
- 低位置的 Collibot 鏡頭、距離估計誤差 → `false_accept` 誤殺；建議先校正距離。
- 電視在 ROI 內近距離（< 1 m）時無法分辨。
- 多人同時說話時只看「有沒有近距離的人」，不判斷是誰在說話（沒有 AV 主動說話者偵測）。
- 4 麥克風陣列 / 波束成形不在這個版本。

## 9. 需要 PureSound 上游提供（未修改 PureSound）

1. **執行時可調的 `dry_blend`**（例如 `set_dry_blend()`）：現在 glue 用 `a·SDK輸出 + (1−a)·對齊原聲` 模擬較低比例（數學上等價，但重複保存一份原聲）。
2. **VAD／語音機率輸出頭**（`extra_output_names`），取代分離後能量 VAD。
3. **公開的 `streaming_delay` / 對齊原聲取用介面**，glue 不必自己維護原聲緩衝。
4. **模型取得**：SDK 沒有模型下載／校驗功能（現在用 `scripts/fetch_puresound_model.py`）；建議上游提供下載＋校驗（例如 release asset + sha256）。
5. **Python 3.10/3.11 支援**（SDK 目前要求 ≥ 3.12）。

## 9a. 監聽延遲

擷取→播放 ≈ 擷取 20 ms + 演算法 52–62 ms + 網路往返（Tailscale：同一區域直連約 10–40 ms，經 DERP relay 可能 100 ms 以上）+ 伺服器處理約 5 ms + 播放緩衝 60–100 ms + 輸出裝置延遲（瀏覽器回報的 `baseLatency + outputLatency`，Mac 內建喇叭／有線耳機約 10–30 ms，藍牙耳機 150–250 ms）。
因此遠端監聽的實際下限大約是 **150–250 ms**（有線耳機、直連）；這是監聽用的延遲，不影響 VAD／判斷（判斷在伺服器上，不經過播放緩衝）。
播放器用固定的小 jitter buffer（目標 80 ms），網路卡頓後多出來的聲音會被丟掉，不會累積。UI「監聽延遲」會列出各段。

## 9c. 錄音（選用，只在瀏覽器）

用來評估效果：在「乾淨音軌」面板按「⏺ 錄音（存到這台電腦）」，再按一次停止。**預設關閉，必須使用者自己按**；需要 PureSound 管線。

- 三軌，16 kHz mono PCM16 WAV、逐樣本對齊、長度相同：
  - `*_raw.wav`：麥克風輸入，**已補償 PureSound 的串流延遲**（`streaming_delay_frames × hop` = 480 樣本），和 vi／out 對齊；
  - `*_vi.wav`：PureSound 人聲分離輸出（manifest 的 dry_blend 0.9，未經影像交叉確認）；
  - `*_out.wav`：最終輸出（影像交叉確認後的增益／混合，也就是監聽的「乾淨」音軌）。
- 另附 `*.json`：每 100 ms 的判斷（decision、why、speech、乾淨／原始 VAD、gain、mix、state、P(group)、目標與距離）、狀態變化時間軸、錄音當下的完整設定、演算法延遲、伺服器丟棄的區塊數。時間 `t` 以樣本數計（秒）。
- 四個檔案打包成一個 `presence-YYYYMMDD-HHMMSS.zip`，下載到**使用者自己的電腦**。最長 10 分鐘（自動停止並下載）；錄音中標題列有紅色「● 錄音中 m:ss / 10:00」。停止、關閉鏡頭或切走分頁都會結束錄音並下載。
- 錄音只存在瀏覽器記憶體。伺服器只在錄音開啟時多送一種 WebSocket 訊框（`0x03`：時間、樣本數 n、out、vi、raw 各 n 個樣本），**伺服器不寫任何聲音或影像檔**（`tests/test_recording.py` 檢查）。影像從不錄製。
- 伺服器因為積壓丟掉的區塊不在檔案中（JSON 有記數）。
- 開源用途：錄下的是使用者自己的聲音，請在說明／同意流程中告知被錄的人。

## 9b. 雲端語音辨識（選用）

修正後的乾淨音軌可以再送到 AssemblyAI 做即時語音辨識（`asr_provider = assemblyai`），只在 session 進行中、預設只送「算說話」的片段。詳見 [ASR.md](ASR.md)。

## 10. 授權

PureSound 原始碼、SDK、模型權重：Apache-2.0（Copyright 2026 Milo Wu and PureSound contributors）。
**資料集不在 Apache-2.0 授權範圍內**（PureSound NOTICE），訓練資料的授權需另外確認後才能商用。詳見 `docs/LICENSES.md`。
