# 效能與省電

所有最佳化都可以在網頁「⚙ 設定 → 效能」個別開關（對應 `Config` 參數，見 [CONFIGURATION.md](CONFIGURATION.md#效能)），實作在 `presence_demo/optimizer.py`。資源監控面板會顯示跳過比例、embedding 計算／重用次數，以及伺服器建議的 fps。

## 四項最佳化

| 項目 | 作法 | 主要參數 |
|---|---|---|
| 移動偵測閘門 | 把畫面縮成 64×48 灰階，和上一張比平均差異。低於門檻且狀態穩定（非 MAYBE_AWAY/AWAY）時跳過偵測，沿用上次結果；最久每 `force_detect_seconds` 秒強制完整偵測一次。 | `motion_gate`、`motion_threshold`、`force_detect_seconds` |
| Embedding 節流 | 每張臉用上次的 embedding，除非：是新的臉、中心位移或大小變化超過 `embed_move_thresh`、上次重算時的自我相似度低於 `cos_match`，或超過 `embed_interval`。 | `embed_throttle`、`embed_interval`、`embed_move_thresh` |
| 自適應 fps | 伺服器依狀態建議 fps：穩定時用 `fps_low`；有候選人臉、參與者暫時消失、track 剛變動（3 秒內）、MAYBE_AWAY/AWAY 時用 `fps_high`。開啟時瀏覽器會照建議送幀。 | `adaptive_fps`、`fps_low`、`fps_high` |
| ROI 裁切＋縮小 | 只把 ROI 加 `crop_margin` 的範圍裁下來，縮到最大寬度 `max_width` 再偵測；框和 5 個特徵點換算回原畫面座標。Embedding 也在縮小後的影像上計算。 | `roi_crop`、`crop_margin`、`max_width` |

## 基準測試

```bash
uv run python scripts/benchmark.py            # --repeat 3 --fps 1
```

情境是模擬 60 秒：0–15 s 有人靜止、15–30 s 有人左右移動、30–45 s ROI 空、45–60 s 有人回來（靜止），每張加上輕微的感光雜訊。畫面在記憶體中由 `tests/data/face.jpg` 合成，並在記憶體中做 JPEG 編碼和解碼，**不寫入任何影像檔**。關閉自適應 fps 時固定 1 fps。CPU 時間只計算解碼、影像處理和 tracker，不含畫面合成；每組跑 3 次，取 CPU 最低的一次。

量測環境：本機 box，8 核 x86_64 CPU，OpenCV 4.14.0，YuNet + SFace（OpenCV DNN、CPU、8 threads）。

| 設定 | 處理張數 | 每張 avg (ms) | 每張 p95 (ms) | 總 CPU (s) | detect 次數 | embed 計算/重用 | 跳過 | AWAY |
|---|---|---|---|---|---|---|---|---|
| 全部關閉 | 60 | 19.1 | 26.1 | 5.78 | 60 | 45 / 0 | 0 | 是 |
| 只開 ROI 裁切+縮小 | 60 | 15.2 | 21.1 | 4.15 | 60 | 45 / 0 | 0 | 是 |
| 只開移動偵測閘門 | 60 | 11.1 | 23.5 | 3.27 | 35 | 25 / 0 | 25 | 是 |
| 只開 embedding 節流 | 60 | 13.5 | 25.0 | 4.27 | 60 | 23 / 22 | 0 | 是 |
| 全部開啟（固定 fps） | 60 | 8.1 | 19.7 | 2.14 | 35 | 20 / 5 | 25 | 是 |
| 全部開啟 + 自適應 fps | 57 | 7.0 | 19.3 | 1.77 | 33 | 16 / 0 | 24 | 是 |

觀察：

- 全部開啟（固定 1 fps）時，總 CPU 從 5.78 s 降到 2.14 s（約 **−63%**），每張平均延遲從 19.1 ms 降到 8.1 ms。加上自適應 fps 後降到 1.77 s（約 **−69%**），平均 7.0 ms。
- p95 只小幅下降（26 → 19 ms），因為仍需要完整偵測的畫面成本差不多，主要節省來自「不用做」的畫面。
- 移動偵測閘門在靜止時段效果最大；embedding 節流在小幅移動時也有效。
- 自適應 fps 在這個情境只少送 3 張：ROI 空的期間參與者還在寬限期（被視為「暫時消失」），會用高 fps 確認離開，這是刻意的設計。長時間穩定對話時，節省會接近 `fps_low / fps`（預設 0.5×）。
- 所有設定都正確判定出 AWAY。
- 實際結果取決於硬體、鏡頭雜訊（會影響 `motion_threshold`）和人臉大小（會影響 `max_width`）。請用自己的裝置重新量測。

## 注意事項

- `max_width` 太小時，遠處的小臉會偵測不到；臉在縮小後的影像中寬度最好至少 40–50 px。
- 鏡頭雜訊大或自動曝光時，畫面差異會變大，閘門比較少觸發，可以稍微調高 `motion_threshold`。
- 閘門只在狀態穩定時生效；MAYBE_AWAY/AWAY 一定完整偵測，避免漏判有人回來。

## 下一步（Android 嵌入式裝置）

1. **TFLite / LiteRT INT8**：把 YuNet、SFace（或 MobileFaceNet）轉成 INT8 量化模型，用 GPU / NNAPI / 廠商 NPU delegate。偵測和 embedding 預期可以再快 2–5 倍，也比較省電。
2. **NPU 排程**：偵測和 embedding 放在 NPU，CPU 只跑閘門、tracker 和狀態機。
3. **核心邏輯移植**：tracker、狀態機、閘門邏輯都很小，可以改寫成 **Rust**，透過 JNI（或 UniFFI）給 Android 使用，Python 版當作參考實作和測試基準（同一組情境測試可以移植過去）。
4. **相機端最佳化**：直接從 CameraX `ImageAnalysis` 取 YUV，用 Y 平面做閘門，省掉 JPEG 編解碼；ROI 裁切在 YUV 上完成。
5. **事件驅動**：沒有人臉而且沒有聲音時，降到 0.2 fps，再搭配 VAD 喚醒。
