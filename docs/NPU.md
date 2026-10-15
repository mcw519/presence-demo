# SFace 嵌入後端與 NPU（`embed_backend`）

本文說明 SFace 人臉特徵（embedding）的三種計算方式，以及怎麼在 Kebbi（MediaTek G520）／Android 上使用 NPU。

| `embed_backend` | 模型 | 執行方式 | 用途 |
|---|---|---|---|
| `fp32`（預設） | `face_recognition_sface_2021dec.onnx`（opencv_zoo） | OpenCV DNN，CPU | 參考基準，最準 |
| `int8` | `face_recognition_sface_2021dec_int8.onnx`（opencv_zoo **官方**量化版） | OpenCV DNN，CPU | 檔案小 4 倍；ARM CPU 上通常比較快 |
| `npu` | 本專案匯出的 TFLite：`..._int8.tflite`（全整數）或 `..._fp16.tflite` | LiteRT（ai-edge-litert），可以掛外部 delegate | Android NPU／GPU；伺服器上用 CPU XNNPACK |

## 怎麼取得模型

- **int8 ONNX**：第一次使用時自動下載到 `models/`，並檢查 sha256（`2b0e941e…d78a`，等於 opencv_zoo 的 git-LFS oid）。fp32 模型也會檢查 sha256（`0ba9fbfa…4e79`）。
- **TFLite**：不會放進 git（10 MB 和 19 MB，而且 SFace 有授權疑慮，見 LICENSES.md），請用下面的腳本重新產生。

  ```bash
  uv venv -p 3.12 /tmp/.venv-sface-export
  VIRTUAL_ENV=/tmp/.venv-sface-export uv pip install onnx2tf tensorflow tf-keras onnx onnxruntime \
      onnx-graphsurgeon onnxsim sng4onnx psutil ai-edge-litert opencv-python-headless
  /tmp/.venv-sface-export/bin/python scripts/export_sface_tflite.py
  ```

### 匯出流程（`scripts/export_sface_tflite.py`）

1. 確認 fp32 ONNX 的 sha256。
2. 把 ONNX **複製一份**再交給 onnx2tf 轉換。onnx2tf 會直接改寫輸入檔，曾經把 fp32 模型改壞，所以一定要用複本。
3. 轉成 TF SavedModel（NHWC、batch 固定為 1）。
4. PReLU 改用 Keras 的寫法 `relu(x) + (-a)·relu(-x)`，TFLite 轉換器才會把它合併成**單一 PRELU op**。
   - onnx2tf 預設的 ABS 拆解，會變成 27 組 ABS／SUB／MUL 元素運算。
   - 這些運算 NPU 不友善；在 x86 上 int8 版本還慢了約 2 倍（64 ms 和 34 ms）。
5. 產生兩個 TFLite：
   - **fp16**：float16 權重，float32 輸入輸出。
   - **全 INT8**：`TFLITE_BUILTINS_INT8`，輸入和輸出都是 int8；輸入是 `[1,112,112,3]` NHWC 的 RGB 0–255，scale 1.0、zero point −128。
6. INT8 校正資料：30 張 CC0 人像（Cleveland Museum of Art Open Access），每張加 3 種增強，共 120 張臉，**全部只在記憶體中處理**。
7. 用 LiteRT CPU 直譯器和 fp32 ONNX 比對 cosine 相似度。

匯出的 INT8 模型使用的 op：

| op | 數量 |
|---|---|
| CONV_2D | 14 |
| DEPTHWISE_CONV_2D | 13 |
| PRELU | 27 |
| PAD | 4 |
| FULLY_CONNECTED | 1 |
| SUB、MUL、ADD、TRANSPOSE、RESHAPE | 各 1 |

這些都是 NNAPI 和 NeuroPilot 常見有支援的 op，**但實際是否全部下放到 NPU，要在裝置上確認**（見下方「在裝置上量測」）。

本次產生的檔案：

| 檔案 | 大小 | sha256 |
|---|---|---|
| `face_recognition_sface_2021dec_int8.tflite` | 10.0 MB | `8b289b43547623fac919ecca9641cfee13eff9edaeef87f3bbc09f9a39df1a8b` |
| `face_recognition_sface_2021dec_fp16.tflite` | 19.3 MB | `036f99661285d77d996feaea2db379655bb7b79b9f9b7c38bc129c842397d153` |

TF 轉換不保證每次產生完全相同的位元組；如果 sha 不同，以腳本最後印出的 cosine 驗證為準。

## Python 後端（`presence_demo/embed_backends.py`）

- **對齊**：`TFLiteSFace` 用自己實作的 Umeyama 相似變換做 5 點對齊。和 OpenCV `alignCrop` 的像素差最多 1，平均約 0.0003，有測試覆蓋。
- **delegate**：設定了 `npu_delegate_path` 時，會先用 `load_delegate(path)` 載入外部 delegate。
  - 載入失敗或無法準備計算圖時，改用 CPU（LiteRT 預設 XNNPACK）。
  - 失敗原因記在 `delegate_error`。
- **安全**：`npu_delegate_path` **只能在伺服器端用環境變數 `PRESENCE_NPU_DELEGATE` 設定**，WebSocket 的 config 無法修改。原因是載入 .so 等於執行原生程式碼。
- **實際狀態**：`pipe.embed_status` 記錄要求的後端、實際使用的後端、模型、加速器和備註。網頁「資源監控 → Embedding」會顯示，例如：`npu（要求 npu）· SFace TFLite fp16 · cpu (LiteRT XNNPACK) · NPU delegate not active (no npu_delegate_path set); running on CPU …`。
- **沒有 delegate 時**：`npu_precision=int8` 會自動改用 fp16 模型，因為 LiteRT 在 x86 CPU 上跑這個 int8 圖很慢（約 33 ms，fp16 約 3 ms）。備註欄會寫明這一點。
- **模型不存在**（還沒執行匯出）：退回 fp32，備註寫「unavailable」。

## 在 Kebbi／Android 上使用

> 以下步驟**都沒有在實機上驗證過**。x86 測試伺服器沒有 ARM，也沒有 NPU。

Kebbi 上比較實際的做法，是在 Android App（Kotlin／Java）裡用同一個 `.tflite`，Python 伺服器的邏輯則移植過去或繼續在伺服器上執行。

1. **NNAPI delegate**（Android 8.1 以上內建；Android 15 起 Google 已宣告 NNAPI 停止維護，但舊系統仍可以用）：

   ```kotlin
   val nnapi = NnApiDelegate(NnApiDelegate.Options().apply {
       setAcceleratorName("<裝置回報的 NPU 名稱>")   // 先用 benchmark 工具列出可用的 accelerator
       setAllowFp16(true)
       setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
   })
   val tflite = Interpreter(modelBuffer, Interpreter.Options().addDelegate(nnapi))
   // 輸入：int8 [1,112,112,3]，值 = RGB(0..255) − 128；輸出：int8 [1,128]，(q − zp)·scale 後再做 L2 正規化（scale／zp 從模型的輸出 tensor 讀取）
   ```

2. **MediaTek NeuroPilot（Neuron Delegate）**：需要 MediaTek 提供的 delegate 函式庫和 SDK。
   - SDK 屬於 MediaTek 的專有授權，商用條款要向 MediaTek 或機器人廠商確認。
   - 如果有可以用 `dlopen` 的 delegate `.so`，在能跑 Python 的 ARM Linux 上，可以設定 `PRESENCE_NPU_DELEGATE=/path/libneuron_delegate.so`。
3. **前處理**要和 Python 版完全一致：
   - YuNet 五點，用 SFace 樣板做相似變換，裁出 112×112；
   - BGR 轉 RGB；
   - 輸入 int8 = 像素 − 128。
4. **G520 的硬體能力**：G520 是否有可用的 APU／NPU、NNAPI 驅動支援哪些 op，**需要和廠商確認**。如果沒有 NPU，NNAPI 可能交給 GPU 或 CPU 參考實作，反而比 XNNPACK 慢。這時建議改用 CPU XNNPACK 跑 fp16（或 int8）模型。

### 在裝置上量測

用 TensorFlow 官方的 `benchmark_model`（Apache-2.0，可以從 TF release 或自行編譯 Android arm64 版）：

```bash
adb push face_recognition_sface_2021dec_int8.tflite benchmark_model /data/local/tmp/
adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/face_recognition_sface_2021dec_int8.tflite \
    --num_threads=4 --use_xnnpack=true                                 # CPU 基準
adb shell /data/local/tmp/benchmark_model --graph=... --use_nnapi=true --nnapi_accelerator_name=<名稱> \
    --enable_op_profiling=true                                        # 看哪些 op 真的下放到 NPU
adb shell /data/local/tmp/benchmark_model --graph=... --external_delegate_path=/vendor/lib64/<neuron delegate>.so
```

建議量測項目：
- int8 和 fp16 在 CPU、NNAPI（NPU）、GPU delegate 上的平均和 p95 延遲；
- 各 op 實際由誰執行（op profiling）；
- 持續 10 分鐘的溫度和降頻情況；
- 精度：在裝置上算同一批臉的 embedding，和伺服器的 fp32 比較 cosine，應該 ≥ 0.95。

## 驗證狀態

| 項目 | 狀態 |
|---|---|
| int8 ONNX 下載和 sha256 驗證 | ✅ 伺服器上驗證 |
| int8 ONNX 和 fp32 的精度（cosine、判定一致率） | ✅ 伺服器上驗證（見下表） |
| TFLite fp16／INT8 匯出、op 清單、LiteRT CPU 推論和精度 | ✅ 伺服器上驗證 |
| 外部 delegate 載入失敗時退回 CPU，並顯示實際加速器 | ✅ 伺服器上驗證（用不存在的 .so 測試） |
| delegate 真的在 NPU 上執行、哪些 op 被下放、延遲和耗電 | ❌ **需要在實機上測試** |
| G520 是否有 NPU／NNAPI 驅動、NeuroPilot 的取得和授權 | ❌ **需要向廠商確認** |
| 實機上的 int8 精度（NPU 的量化實作可能和 LiteRT CPU 略有不同） | ❌ **需要在實機上測試** |

## 精度（伺服器）

```bash
uv run python scripts/eval_embed_backends.py
```

方法：
- 從 Cleveland Museum of Art Open Access（CC0）取 31 張人像，和 INT8 校正資料不重疊；加上測試用的 face.jpg。
- 每張做 6 種增強：亮、暗、模糊、旋轉縮放、JPEG q35、雜訊。
- 同一人：原圖和增強圖，共 186 對。不同人：不同人像兩兩配對，共 465 對。
- 判定門檻是目前的 `cos_match` 0.40。
- 全部在記憶體中處理，不寫入任何影像。

| 後端 | 與 fp32 cos 平均 | 最小 | 同一人 cos 平均 | 不同人 cos 最大 | 同一人通過率 | 不同人誤配率 | 與 fp32 判定一致 |
|---|---|---|---|---|---|---|---|
| fp32 (ONNX, OpenCV) | 1.0000 | 1.0000 | 0.962 | 0.940 | 100.0% | 1.29% | 100.00% |
| int8 (ONNX, OpenCV) | 0.9721 | 0.9342 | 0.949 | 0.916 | 100.0% | 1.51% | 99.85% |
| npu int8 (TFLite) | 0.9906 | 0.9556 | 0.953 | 0.932 | 100.0% | 1.29% | 100.00% |
| npu fp16 (TFLite) | 1.0000 | 1.0000 | 0.962 | 0.940 | 100.0% | 1.29% | 100.00% |

- **int8 ONNX**：和 fp32 的平均 cosine 0.972（最低 0.934），判定一致率 99.85%（651 對中只有 1 對不同）。
- **本專案匯出的 TFLite INT8**：比官方 int8 更接近 fp32（平均 0.991），判定和 fp32 **完全一致**。
- **fp16**：和 fp32 相同。
- **門檻**：各後端的最佳門檻只差約 0.01–0.02，所以 `cos_match_int8` 預設和 fp32 相同（0.40），**不需要調整**；需要時可以另外設定。
- **不同人誤配率**：fp32 也有 1.29%（6 對）。這些多半是博物館收藏中同一位被攝者的不同照片或不同印樣，沒有逐一確認。
- **限制**：「同一人」只用了增強圖，不是不同時間拍的照片，所以這裡**比較的是「量化後和 fp32 是否一致」**，不能當作辨識準確率。opencv_zoo 官方的 LFW 準確率：fp32 0.9940、int8 0.9932。

## 延遲（伺服器：Intel Xeon 8 vCPU，不是 Kebbi）

整條管線，每張都計算 embedding：

| 後端 | 實際使用 | 加速器 | 輸入 | embed | total |
|---|---|---|---|---|---|
| fp32 | SFace | cpu (OpenCV DNN) | 640×480 | 10.5 / 12.7 / 13.4 | 14.7 / 17.2 / 18.0 |
| fp32 | SFace | cpu (OpenCV DNN) | 1280×720 | 10.9 / 13.5 / 14.0 | 16.8 / 19.6 / 20.5 |
| int8 | SFace int8 | cpu (OpenCV DNN) | 640×480 | 12.2 / 14.6 / 15.6 | 16.6 / 19.4 / 20.4 |
| int8 | SFace int8 | cpu (OpenCV DNN) | 1280×720 | 13.9 / 15.5 / 21.9 | 20.2 / 22.1 / 31.9 |
| npu (int8) | SFace TFLite fp16 | cpu (LiteRT XNNPACK) | 640×480 | 4.0 / 4.6 / 5.0 | 8.6 / 10.0 / 10.3 |
| npu (int8) | SFace TFLite fp16 | cpu (LiteRT XNNPACK) | 1280×720 | 4.7 / 6.1 / 6.9 | 10.7 / 13.2 / 16.5 |
|  | 註：NPU delegate not active (no npu_delegate_path set); running on CPU with the fp16 model (faster than int8 on CPU) | | | | |
| npu (fp16) | SFace TFLite fp16 | cpu (LiteRT XNNPACK) | 640×480 | 3.6 / 4.1 / 4.1 | 7.7 / 8.4 / 8.6 |
| npu (fp16) | SFace TFLite fp16 | cpu (LiteRT XNNPACK) | 1280×720 | 4.1 / 4.9 / 5.1 | 9.4 / 10.5 / 12.0 |
|  | 註：NPU delegate not active (no npu_delegate_path set); running on CPU | | | | |

只算模型本身：

| 模型 | p50 | p95 | max |
|---|---|---|---|
| fp32 ONNX · OpenCV DNN | 7.47 | 10.83 | 25.30 |
| int8 ONNX · OpenCV DNN | 12.51 | 13.95 | 15.01 |
| int8 TFLite · LiteRT CPU 4 執行緒 | 32.59 | 35.36 | 37.59 |
| int8 TFLite · LiteRT CPU 1 執行緒 | 34.01 | 36.95 | 40.21 |
| fp16 TFLite · LiteRT CPU 4 執行緒 | 2.81 | 3.23 | 3.73 |
| fp16 TFLite · LiteRT CPU 1 執行緒 | 8.58 | 10.53 | 11.86 |

- **x86 測試伺服器**：官方 int8 ONNX 在 OpenCV DNN 上比 fp32 **慢**（約 12.5 ms 和 7.5 ms）。本專案匯出的 INT8 TFLite 在 LiteRT CPU 上也很慢（約 33 ms），可能是 XNNPACK 沒有涵蓋這個 int8 圖的部分 op（推測），所以 npu 後端在沒有 delegate 時會改用 fp16。fp16 TFLite 用 4 個執行緒最快（約 2.8 ms）。
- **ARM 和 NPU**：情況會不同。INT8 在 ARM NEON 和 NPU 上通常比較快，但**要在 Kebbi 上實測**。
