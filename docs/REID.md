# 跨 session 使用者判斷（雲端 re-ID，選用）

> 預設**關閉**。開啟後，人臉裁切會**離開這台裝置**送到第三方雲端服務，而且伺服器會在**記憶體中短暫保存**上一位使用者的臉部裁切。使用前請先閱讀本文的「隱私」與「待確認事項」。

## 用途

presence 開啟一個**新的 session** 時，判斷「這是不是上一位（或最近幾位）使用者回來了」：

- **同一人** → 接續上一段對話（沿用同一個 `conversation_id`）
- **不同人、不確定，或任何失敗** → 開新對話（新的 `conversation_id`）

session 進行中**不會呼叫**雲端。沒有保存中的參考臉（第一次使用，或已超過保存時間）時，直接開新對話，**也不呼叫**。

## 流程

```
session 開始（tracker 的 session 編號增加）
 ├─ 沒有未過期的參考臉 → 新對話（reason=no_reference，不呼叫）
 ├─ 距離上次判斷 < reid_cooldown → 新對話（reason=cooldown，不呼叫）
 └─ 等第一位參與者出現清楚的臉（正面或側面、寬度 ≥ reid_min_face）
      （MAYBE_AWAY／AWAY 期間暫停計時）
      → 依序和參考臉比對（最新的先，最多 reid_max_refs 位）
          分數 ≥ reid_threshold（0.9）       → 接續（reason=match）
          reid_uncertain_low ≤ 分數 < 門檻    → 再取一張新畫面比對一次；仍未達門檻 → 新對話（reason=uncertain）
          分數 < reid_uncertain_low          → 新對話（reason=below_threshold）
          逾時／錯誤／沒有 key／達上限       → 新對話（reason=error / no_key / cap / breaker）
      參與者出現後 reid_face_wait（3 s）內沒有清楚的臉 → 新對話（reason=no_face_timeout）
      整體超過 reid_decision_timeout（10 s）→ 新對話（reason=decision_timeout）
      結束等待時記錄 `reid_wait`（等了多久、有沒有看到人、為什麼沒有可用的臉）
```

判斷在背景執行，**不會延遲** session 開啟、影像或聲音處理；結果出來後才送出 `handoff` 事件。session 進行中，伺服器會保留這位使用者品質最好的一張臉部裁切（只在記憶體），session 結束時成為下一次比對用的參考臉。

## 「接續對話」的意思

presence-demo **沒有對話記憶**。「接續」只代表輸出相同的交接 ID，由下游系統（對話引擎等）決定要不要接上之前的內容。WebSocket 會送出：

```json
{"type":"handoff","session":12,"conversation_id":"3f…","decision":"resume","resumed_from_session":11,
 "score":0.94,"reason":"match","calls":3}
```

- `conversation_id`：隨機 ID（UUID hex），和人的身分沒有關聯，不含任何個人資料。
- 每秒還會送一次 `{"type":"reid", …}` 狀態：階段、參考臉數量、剩餘保存時間、呼叫數、tokens、估計成本。這份狀態**不含影像**。

## 隱私：記憶體中的參考臉（「伺服器不保存影像」原則的例外）

本專案的原則是伺服器不保存任何影像。這個功能需要記住上一位使用者的臉，因此有一個**明文例外**，範圍如下：

| 項目 | 規則 |
|---|---|
| 保存什麼 | 每位使用者**一張**臉部裁切（320×320 JPEG）。不保存整張畫面 |
| 保存在哪 | 伺服器 **process 記憶體**，綁定在該 WebSocket 連線。**不寫磁碟、暫存檔、log、`/debug`，也不傳回瀏覽器** |
| 保存多久 | `reid_ttl`，預設也是最長 **600 秒（10 分鐘）**，到期自動刪除；最多 `reid_max_refs` 位 |
| 何時清除 | 到期、關閉功能、設定頁「清除參考臉」、`reset`、`POST /api/reid_reset`、連線結束、伺服器關閉 |
| 送給誰 | 只在比對時，把參考臉和目前的臉（兩張裁切）送到所選的雲端服務 |
| 啟用條件 | `reid_enabled` **和** `reid_consent` 都要開啟（預設都關閉） |

人臉屬於生物特徵資料，許多地區（例如 GDPR 第 9 條、台灣個人資料保護法）有特別規範。部署者應該告知現場使用者並取得同意，確認雲端服務的資料保存政策，並視需要設定更短的 `reid_ttl`。

## 雲端後端（可抽換）

- 介面：`presence_demo.reid.CloudReid.compare(reference_jpeg, query_jpeg) -> (score 0..1, meta)`。新增其他服務只需要實作這個方法。
- 目前內建：**Jev（Typesafe）**，`POST https://api.typesafe.ai/v1/systemone`。
  - 兩張影像放在 `assets`，以 `state.reference`、`state.query` 參照，問題型別為 `noul`（0～1 分數）。
  - header 帶 `TypeSafe-Danger-Zone-Images: true`。
  - 影像太小會被拒絕（`Invalid image dimensions`），所以裁切一律縮放成 320×320。
- API key：環境變數 `TYPESAFE_API_KEY`，或在設定頁輸入（`POST /api/reid_key`）。只存在記憶體，不寫入 log 或檔案，也不會回傳。
- 測試或 proxy 可以用 `REID_JEV_URL` 改端點。

## 評測（參考）

LFW test pairs 隨機 100 對（50 同人／50 不同人），影像 320 px，jev-1.13.0：

- AUC 0.992
- 門檻 0.7：準確率 94%，FAR 4%，FRR 8%
- 門檻 0.9：FAR 0%，FRR 14%
- 延遲 p50 0.16 s，p90 0.19 s；每次約 950 input tokens

LFW 是清楚的正面照片，和低角度機器人鏡頭差異很大，正式使用前請用實際場景重新校正 `reid_threshold` 與 `reid_uncertain_low`。預設 0.9 偏向「寧可開新對話，也不要把別人的對話接給新使用者」。

## 成本控制

- 只在「新 session 開始，而且有參考臉」時呼叫。每次判斷最多比對 `reid_max_refs` 位，加上不確定時重試一次。
- `reid_cooldown`、`reid_cap_per_hour`、`reid_cap_total`。達上限一律開新對話。
- 連續 3 次錯誤時，暫停 60 秒（斷路器）。
- 呼叫數、tokens、估計成本（呼叫數 × `reid_cost_per_call`）顯示在狀態機卡片，並寫進每 10 秒的 heartbeat。

## 記錄事件

`reid_start`、`reid_call`（分數、延遲、模型、tokens、累計呼叫、估計成本）、`reid_retry`、`reid_decision`、`reid_skip`、`reid_error`、`reid_breaker`、`reid_store`（add／expire／wipe 與數量）。這些事件**不記影像、base64、key**。

## 待確認事項（尚未驗證）

- **商用條款**：本專案只採用可商用的元件。Jev（Typesafe）服務是否允許商業使用、是否允許人臉比對用途、價格、SLA、資料保存與地區，**尚未確認**。部署前請自行查閱並遵守服務條款。
- **`TypeSafe-Danger-Zone-Images` header**：這個 header 是送出影像所必需的，但它的正式意義（例如實驗性功能、額外條款或不保證穩定）**尚未有官方說明**。在確認之前，請把這個後端視為實驗性、選用的功能。
- **實際場景準確度**：見「評測」。

## 關閉

設定頁關閉「跨 session 使用者判斷」，或清除 key。關閉時會立即刪除記憶體中的參考臉。
