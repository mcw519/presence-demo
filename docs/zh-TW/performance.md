[English](../performance.md) · **繁體中文**

# 效能

目標是小型 ARM 裝置，所以管線設計成盡量少做事。以下數字都在 8 核 x86 測試伺服器上量測（OpenCV DNN、CPU）。
請在自己的硬體上重新量測。

## 四個省電機制

| 機制 | 作法 | 參數 |
|---|---|---|
| **移動閘門** | 把畫面縮成 64×48 灰階，和上一張比較。低於門檻且狀態穩定時跳過偵測、沿用上次結果。每 `force_detect_seconds` 仍會完整偵測；MAYBE_AWAY / AWAY 時一定完整偵測。 | `motion_gate`、`motion_threshold`、`force_detect_seconds` |
| **Embedding 節流** | 除非是新的臉、移動或大小變化超過 `embed_move_thresh`，或超過 `embed_interval`，否則沿用上次的 embedding。 | `embed_throttle`、`embed_interval`、`embed_move_thresh` |
| **自適應 fps** | 伺服器建議 fps：穩定時 `fps_low`；有臉出現、參與者消失、track 剛變動，或 MAYBE_AWAY / AWAY 時 `fps_high`。 | `adaptive_fps`、`fps_low`、`fps_high` |
| **ROI 裁切** | 裁下 ROI 加 `crop_margin`，縮到最寬 `max_width`，偵測後再把框換算回去。 | `roi_crop`、`crop_margin`、`max_width` |

### 基準測試

`uv run python scripts/benchmark.py` 以 1 fps 模擬 60 秒：有人靜止（0–15 s）、移動（15–30 s）、ROI 空（30–45 s）、
人回來（45–60 s）。畫面在記憶體中合成和 JPEG 編解碼，不寫任何影像檔。每組跑 3 次取最好的一次。

| 設定 | 每張平均 ms | p95 | 總 CPU | 相對全部關閉 |
|---|---|---|---|---|
| 全部關閉 | 21.2 | 27.2 | 6.27 s | — |
| 只開 ROI 裁切 | 16.4 | 23.0 | 4.43 s | −29% |
| 只開移動閘門 | 10.1 | 25.9 | 2.94 s | −53% |
| 只開 embedding 節流 | 14.4 | 25.5 | 4.50 s | −28% |
| 全部開啟，固定 fps | 7.9 | 22.1 | 2.06 s | −67% |
| 全部開啟 + 自適應 fps | 6.9 | 18.2 | 1.81 s | **−71%** |
| 姿勢，省電機制關閉 | 31.1 | 37.9 | 7.93 s | +27% |
| 全部開啟 + 姿勢 | 13.7 | 31.9 | 2.97 s | −53% |
| 全部開啟 + 姿勢 + 距離 | 14.0 | 32.4 | 3.09 s | −51% |

每種設定都正確判定出 AWAY。p95 只小幅下降：需要完整偵測的畫面成本一樣，節省來自「不用做」的畫面。

## 時間花在哪裡

各階段延遲，2 fps，640×480 的移動人臉，省電機制全開（p50 ms）：

| 階段 | legacy + rules | kalman + hmm + score |
|---|---|---|
| JPEG 解碼 | 0.7 | 0.6 |
| 移動閘門 | 0.2 | 0.2 |
| 人臉偵測（YuNet，裁切後） | 2.4 | 2.4 |
| **Embedding（SFace）** | **9.9** | **10.0** |
| 姿勢（MoveNet，開啟時） | 6.3 | 6.1 |
| 衣服直方圖 | — | 0.1 |
| 追蹤 + 在場 + 意圖 | < 0.1 | 0.3 |
| **總計**（無姿勢 / 有姿勢） | 13.7 / 19.9 | 14.0 / 20.2 |

- 臉部 embedding 是最大的成本，其次是姿勢，再來是偵測。
- 進階的幾層（Kalman、HMM、意圖）加起來不到 0.5 ms。
- 1280×720 輸入只多約 1 ms（因為有 ROI 裁切，只有解碼變慢）。
- 空畫面或靜止畫面大多在 1 ms 內完成。
- WebSocket 端到端 p50：沒有姿勢約 17–18 ms，有姿勢約 25 ms。

`max_width` 會影響小臉：320 px 時，高度約 80 px 以下的臉會漏掉，所以預設是 640（偵測約 2.4 → 5 ms）。
縮小後臉的寬度最好至少 40–50 px。

## Embedding 後端與 NPU

`embed_backend` 選擇 SFace 的計算方式。模型相同，只有執行環境不同。

| 後端 | 模型 | 執行方式 | 用途 |
|---|---|---|---|
| `fp32`（預設） | opencv_zoo SFace ONNX | OpenCV DNN，CPU | 參考基準，最準 |
| `int8` | opencv_zoo 官方 INT8 ONNX | OpenCV DNN，CPU | 小 4 倍；ARM CPU 上常比較快 |
| `npu` | 本專案匯出的 TFLite（`int8` 或 `fp16`） | LiteRT，可掛外部 delegate | Android NPU / GPU；伺服器上用 XNNPACK |

每張都計算 embedding 的整條管線（p50 / p95 ms，640×480）：

| 後端 | Embed | 總計 |
|---|---|---|
| fp32，OpenCV DNN | 10.5 / 12.7 | 14.7 / 17.2 |
| int8，OpenCV DNN | 12.2 / 14.6 | 16.6 / 19.4 |
| npu fp16，LiteRT XNNPACK | 3.6 / 4.1 | 7.7 / 8.4 |

和 fp32 的一致性（`scripts/eval_embed_backends.py`：31 張 CC0 人像，每張 6 種增強，651 對，門檻 0.40）：

| 後端 | 和 fp32 的平均 cosine | 判定和 fp32 一致 |
|---|---|---|
| int8 ONNX | 0.972 | 99.85% |
| npu int8 TFLite | 0.991 | 100% |
| npu fp16 TFLite | 1.000 | 100% |

所以 `cos_match_int8` 預設也是 0.40。這裡比較的是量化模型和 fp32 是否一致，不是辨識準確率。

### 取得 TFLite 模型

模型不放在 git（10 MB 和 19 MB），請在獨立環境中匯出：

```bash
uv venv -p 3.12 /tmp/.venv-sface-export
VIRTUAL_ENV=/tmp/.venv-sface-export uv pip install onnx2tf tensorflow tf-keras onnx onnxruntime \
    onnx-graphsurgeon onnxsim sng4onnx psutil ai-edge-litert opencv-python-headless
/tmp/.venv-sface-export/bin/python scripts/export_sface_tflite.py
```

腳本在 ONNX 的複本上轉換，把 PReLU 改寫成單一的 `PRELU` op（對 NPU 友善），產生 fp16 和全 INT8 模型
（只在記憶體中用 CC0 人像校正），並和 fp32 比對。

### 使用 delegate

- 在伺服器上設定 `PRESENCE_NPU_DELEGATE=/path/to/delegate.so`。基於安全，只能用環境變數設定，不能透過 WebSocket：
  載入共享函式庫等於執行原生程式碼。
- delegate 載入失敗時，LiteRT 會改用 CPU。資源面板會顯示實際使用的後端和原因。
- 沒有 delegate 時，`npu_precision = int8` 會自動改用 fp16 模型，在 x86 CPU 上快約 10 倍。
- 在 Android 上，可以在 App 裡用同一個 `.tflite` 搭配 NNAPI 或廠商的 delegate。輸入：int8 `[1, 112, 112, 3]`，
  RGB − 128，用相同的 5 點相似變換對齊。用 TensorFlow 的 `benchmark_model`（`--use_nnapi`、`--external_delegate_path`、
  `--enable_op_profiling`）確認哪些 op 真的在 NPU 上執行。這些步驟都還沒在實機上驗證。

## 音訊成本

PureSound 管線在一顆核心上每 20 ms 一塊約 5 ms（即時率約 0.25），比 1–2 fps 的影像貴得多。請先在目標裝置上量測。
見[音訊](audio.md#效能)。

## 移植到嵌入式裝置的想法

- 偵測和 embedding 放在 NPU，閘門、追蹤器、狀態機留在 CPU。
- 追蹤器和狀態機很小、沒有相依套件，可以移植（例如 Rust 或 Kotlin），以這個 Python 版本和情境測試作為參考。
- 直接用相機的 Y 平面做移動閘門，省掉 JPEG 編解碼。
- 沒有臉也沒有聲音時降到 0.2 fps，用語音活動喚醒。
