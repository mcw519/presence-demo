# 在場機率模型（Layer 2，`presence_model=hmm`）

`presence_model` 選擇「有沒有人」的判斷方式：

| 值 | 說明 |
|---|---|
| `rules` | **預設**，和 v0.5 行為相同。用連續幀計數（`present_frames`／`absent_frames`）判斷有人或沒人，再加上各模式的逾時。 |
| `hmm` | 每個 track 有一個「在場機率」P，整群的機率再用門檻決定狀態。legacy 和 kalman 追蹤都能用，搭配 `tracker_mode=kalman` 效果最好（預測中的框會提供較弱的證據）。 |

程式在 `presence_demo/presence_model.py`（純數學，可單元測試）和 `tracker.py` 的 `_hmm_update`／`_evaluate_hmm`。

## 數學（簡單版）

每個人只有兩種狀態：**在場**或**已離開**。模型不直接存機率 P，而是存它的「對數勝算」：

  L = ln( P / (1 − P) )

- P = 0.5 時 L = 0；P = 0.98 時 L ≈ 3.9；P = 0.02 時 L ≈ −3.9。
- L 加 ln 2（約 0.69），代表「在場的勝算變成兩倍」。

每處理一張幀，就把這張幀看到的證據當作「每秒的變化量」（rate），乘上和上一張幀的**實際時間差 dt**：

  L ← clip( L + rate × dt, −3.9, +3.9 )

因為證據是「每秒」計算，所以 2 fps（每次 dt=0.5）和 0.5 fps（每次 dt=2）在同樣時間內累積的結果相同，**和 fps 無關**。上下限把 P 限制在 0.02–0.98，所以不會因為看了很久就變得很難判定離開。

這就是「兩狀態 HMM／Bayes 濾波」在連續時間下的寫法：
- 「沒看到人時勝算按半衰期下降」對應狀態轉移（人可能已經走了）；
- 「看到人時勝算上升」對應觀測似然比（likelihood ratio）。

### 每個 track 的證據（每秒）

令 `leave_rate = ln2 / leave_halflife`（預設 0.6 s → 約 1.16／s）。

| 這張幀的觀測 | rate | 意思 |
|---|---|---|
| 偵測到（臉或身體） | `+detect_gain × conf`（4 × 0.9 = +3.6） | 看到人，機率快速上升 |
| 預測中（kalman 延續） | `−leave_rate × (1 − vis_conf)` | 較弱；剛消失時幾乎不變，之後越來越接近「沒看到」 |
| 沒看到，但這張幀有低信心偵測 | `−leave_rate × lowconf_factor`（0.15） | 接近中立：光線差、偵測閃爍時不輕易判定離開 |
| 完全沒看到 | `−leave_rate` | 勝算每 `leave_halflife` 秒減半 |
| 轉身（側面或背對）**而且**正在遠離 | `−leave_rate`（即使有偵測到） | 往「離開」推 |
| 超出 `leave_distance` | `−3 × leave_rate` | 強烈往「離開」推 |

新的 track 從 P = 0.5 開始，再加上半秒的證據。

### 群組機率

1. **影像部分**：`P_影像` = 參與者中最大的 P。
   - 還沒有參與者時，用所有 track 中最大的 P。
   - 舊參與者都低於 `p_away`，但已經有穩定的新候選人（P ≥ `p_active`、待滿 `join_seconds`）時，也把他算進來，避免在 NEW_GROUP 成立前先跳到 AWAY。
2. **聲音部分**：全域的弱證據 A，加在群組的對數勝算上。聲音**不會**歸到任何一個人身上。
   - 說話時：A = `w_說話 × 5`；
   - 靜默時：A = `−w_靜默 × 6 × min(1, 靜默秒數／base_timeout)`。

   | priority_mode | w_說話 | w_靜默 | 效果 |
   |---|---|---|---|
   | vision | 0.05 | 0 | 聲音幾乎不影響；安靜的人在畫面裡不會被判定離開 |
   | balanced | 0.8 | 0.8 | 說話可以撐住；有人在場但靜默時，約 14.5 s MAYBE、20.5 s AWAY、25.5 s ENDED；之後人一直待著不會重新開始（v0.8.2），要說話或離開後再出現 |
   | audio | 1.0 | 1.0 | 主要看說話；有人在場但靜默時，約 11.5 s MAYBE、16.5 s AWAY、21.5 s ENDED；之後人一直待著不會重新開始（v0.8.2），要說話或離開後再出現 |

3. **群組機率**：`P_群組 = sigmoid( logit(P_影像) + A )`。

### 狀態（含遲滯）

| 條件 | 結果 |
|---|---|
| `P_群組 < p_away`（0.3）持續 `away_hold_seconds`（1 s） | **AWAY** |
| `P_群組 < p_maybe`（0.6），或已經在 MAYBE_AWAY 而且 `P_群組 < p_active`（0.8） | **MAYBE_AWAY** |
| 其他 | 靜默 < `idle_after` → **ACTIVE**，否則 **IDLE** |
| 在 AWAY 時 `P ≥ p_active` | 回到 **ACTIVE**。vision 模式只看 `P_影像`，聲音不能讓人「回來」 |
| 在 AWAY 持續 `end_after_away` | **ENDED**（和 rules 相同） |

- 0.6 和 0.8 之間是遲滯區：進入 MAYBE_AWAY 之後，要回到 0.8 以上才算恢復，避免狀態在門檻附近來回跳。
- `present_max_silence`（vision 模式）仍然有效。

預設值的時間：ROI 完全空時，P 從 0.98 降到 0.6 約 3 s（MAYBE_AWAY），降到 0.3 約 4 s，加上 1 s 確認 → 約 **5 s 判定 AWAY**，和 rules 的 vision 模式相同。

### NEW_GROUP（hmm）

成立條件（持續 `new_group_seconds`）：
- 所有舊參與者的 P < `p_away`；
- **而且**有新候選人：不是參與者、這一幀有偵測到、不是 far、待滿 `join_seconds`，而且 P ≥ `p_active`。

加入規則維持原樣（Layer 3 之後會取代）：舊參與者都不可見時，新的人不會加入舊群組。

### 取代的東西

hmm 模式不使用 `present_frames`／`absent_frames` 計數，也不使用各模式的逾時（`present_timeout`、`empty_timeout`、`base_timeout` 只用在聲音的靜默比例）。`present`／`empty` 旗標改由 `P_影像` 決定：
- `present`：P ≥ `p_active`，或已經 present 而且 P ≥ `p_maybe`；
- `empty`：P < `p_away`。

## 參數

| 參數 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `presence_model` | `rules` | `rules`／`hmm` | 判斷模型 |
| `leave_halflife` | 0.6 | 0.1–30 s | 完全沒看到人時，在場勝算減半的時間。調大＝更慢判定離開 |
| `detect_gain` | 4.0 | 0.1–20 | 偵測證據強度（每秒，乘上信心）。調大＝更快回到 ACTIVE |
| `lowconf_factor` | 0.15 | 0–1 | 只有低信心偵測時，下降速度是「完全沒看到」的幾倍 |
| `p_active` | 0.8 | 0.05–0.99 | 回到 ACTIVE 的門檻 |
| `p_maybe` | 0.6 | 0.02–0.98 | 進入 MAYBE_AWAY 的門檻 |
| `p_away` | 0.3 | 0.01–0.95 | AWAY 門檻（要持續 `away_hold_seconds`） |
| `away_hold_seconds` | 1.0 | 0–30 s | AWAY 確認時間 |

設定時會強制 `p_away ≤ p_maybe ≤ p_active`。

state 訊息新增：
- `p_group`：群組機率，含聲音；
- `p_tracks`：只有影像；
- 每個 track 的 `p`（rules 模式下是 null）。

網頁「在場機率」卡片：
- 畫出最近 2 分鐘的群組 P；
- 三條門檻線：綠＝ACTIVE、黃＝MAYBE_AWAY、紅＝AWAY；
- 每個追蹤對象下方有自己的機率條。

## 評估

```bash
uv run python scripts/eval_tracking.py        # 或 scripts/eval_presence.py
```

比較三個版本：
- `legacy+rules`：v0.4 行為，目前預設；
- `kalman+rules`：v0.5 選項；
- `kalman+hmm`：本版。

所有情境都在記憶體中產生，沒有任何影像。每一幀之間每 0.5 s 執行一次 ticker，預設 `priority_mode=vision`。

指標：

| 指標 | 定義 |
|---|---|
| 誤判 MAYBE_AWAY／AWAY | 場景裡還有人時（對「離開」情境是開始離開之前；「短暫走開」情境則是走開期間出現 AWAY） |
| 離開判定 | 從離開（或開始轉身離開）到 AWAY 的秒數 |
| 誤判 NEW_GROUP | 不該換組的情境卻換組 |
| 狀態切換/分 | 每分鐘狀態轉換次數，越少越穩定 |

### 結果（2026-10-14，20 個種子）

| 情境 | 版本 | ID 互換 | 斷裂 | 誤判 MAYBE_AWAY | 誤判 AWAY | 誤判 NEW_GROUP | 離開判定 (s) | 狀態切換/分 |
|---|---|---|---|---|---|---|---|---|
| 遮擋 1–3 s（1 fps） | legacy+rules | 0 | 0 | 20 | 0 | 0 | — | 8.0 |
| 遮擋 1–3 s（1 fps） | kalman+rules | 0 | 0 | 20 | 0 | 0 | — | 8.0 |
| 遮擋 1–3 s（1 fps） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| 遮擋 1–3 s（2 fps） | legacy+rules | 0 | 0 | 20 | 0 | 0 | — | 8.0 |
| 遮擋 1–3 s（2 fps） | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| 遮擋 1–3 s（2 fps） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| 兩人交錯走過 | legacy+rules | 0 | 0 | 0 | 0 | 0 | — | 5.0 |
| 兩人交錯走過 | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 5.0 |
| 兩人交錯走過 | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 5.0 |
| 側身看不到臉 | legacy+rules | 0 | 0 | 0 | 0 | 0 | — | 5.5 |
| 側身看不到臉 | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 5.5 |
| 側身看不到臉 | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 5.5 |
| fps 變動 0.5–2 | legacy+rules | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| fps 變動 0.5–2 | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| fps 變動 0.5–2 | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| 兩人側身互換位置 | legacy+rules | 38 | 38 | 0 | 0 | 0 | — | 6.0 |
| 兩人側身互換位置 | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 6.0 |
| 兩人側身互換位置 | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 6.0 |
| 離開後新的人站到同一位置 | legacy+rules | 0 | 0 | 0 | 0 | 0 | — | 12.0 |
| 離開後新的人站到同一位置 | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 12.0 |
| 離開後新的人站到同一位置 | kalman+hmm | 0 | 0 | 20 | 0 | 0 | — | 16.8 |
| 離開（1 fps） | legacy+rules | 0 | 0 | 0 | 0 | 0 | 5.0 | 12.0 |
| 離開（1 fps） | kalman+rules | 0 | 0 | 0 | 0 | 0 | 6.0 | 12.0 |
| 離開（1 fps） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | 5.0 | 12.0 |
| 離開（2 fps） | legacy+rules | 0 | 0 | 0 | 0 | 0 | 5.0 | 12.0 |
| 離開（2 fps） | kalman+rules | 0 | 0 | 0 | 0 | 0 | 6.0 | 12.0 |
| 離開（2 fps） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | 5.5 | 12.0 |
| 低信心閃爍（40% 幀 conf 0.4） | legacy+rules | 0 | 7 | 13 | 27 | 0 | — | 7.5 |
| 低信心閃爍（40% 幀 conf 0.4） | kalman+rules | 0 | 7 | 13 | 27 | 0 | — | 7.5 |
| 低信心閃爍（40% 幀 conf 0.4） | kalman+hmm | 0 | 0 | 5 | 0 | 0 | — | 2.2 |
| ROI 空 + 電視聲 | legacy+rules | 0 | 0 | 0 | 0 | 0 | 5.0 | 12.0 |
| ROI 空 + 電視聲 | kalman+rules | 0 | 0 | 0 | 0 | 0 | 6.0 | 12.0 |
| ROI 空 + 電視聲 | kalman+hmm | 0 | 0 | 0 | 0 | 0 | 5.0 | 12.0 |
| 長時間安靜在場（120 s） | legacy+rules | 0 | 0 | 0 | 0 | 0 | — | 1.0 |
| 長時間安靜在場（120 s） | kalman+rules | 0 | 0 | 0 | 0 | 0 | — | 1.0 |
| 長時間安靜在場（120 s） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 1.0 |
| 慢慢離開（轉身、變小 6 s） | legacy+rules | 0 | 0 | 0 | 0 | 0 | 11.0 | 14.0 |
| 慢慢離開（轉身、變小 6 s） | kalman+rules | 0 | 0 | 0 | 0 | 0 | 12.0 | 14.0 |
| 慢慢離開（轉身、變小 6 s） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | 7.9 | 10.0 |
| 短暫走開又回來（3.5 s） | legacy+rules | 0 | 0 | 0 | 0 | 0 | — | 8.0 |
| 短暫走開又回來（3.5 s） | kalman+rules | 0 | 0 | 20 | 0 | 0 | — | 8.0 |
| 短暫走開又回來（3.5 s） | kalman+hmm | 0 | 0 | 0 | 0 | 0 | — | 4.0 |
| 遠處路人（3.5 m）+ 參與者離開 | legacy+rules | 0 | 20 | 0 | 0 | 0 | 5.0 | 8.6 |
| 遠處路人（3.5 m）+ 參與者離開 | kalman+rules | 0 | 20 | 0 | 0 | 0 | 6.0 | 8.6 |
| 遠處路人（3.5 m）+ 參與者離開 | kalman+hmm | 0 | 20 | 0 | 0 | 0 | 5.0 | 8.6 |
| **合計** | legacy+rules | **38** | **65** | **53** | **27** | **0** | | |
| **合計** | kalman+rules | **0** | **27** | **53** | **27** | **0** | | |
| **合計** | kalman+hmm | **0** | **20** | **25** | **0** | **0** | | |

### 解讀

- **hmm 比 rules 好的地方**：
  - 低信心閃爍：誤判 AWAY 從 27 降到 0；誤判 MAYBE_AWAY 從 13 降到 5；斷裂從 7 降到 0（rules 會誤判 AWAY → ENDED，清空 track 再重建）；狀態切換從 7.5 降到 2.2 次/分。
  - 1 fps、3 s 遮擋：誤判 MAYBE_AWAY 從 20 降到 0。
  - 短暫走開 3.5 s：kalman+rules 在人回來時會短暫誤判 MAYBE_AWAY，hmm 沒有；狀態切換也較少。
  - 慢慢離開：hmm 7.9 s（約在人走出畫面後 1.9 s），rules 11–12 s。因為「轉身＋遠離」和「變遠」會持續往離開推。
- **和 rules 相同的地方**：
  - ROI 空時判定離開約 5 s（2 fps 時 5.5 s），比 kalman+rules 的 6 s 快。
  - 電視聲（vision 模式）不會延後離開判定。
  - 長時間安靜在場 120 s 不會離開。
  - 遠處路人不會讓參與者「還在」。
  - 沒有誤判 NEW_GROUP。
- **hmm 比較差的地方**：「離開後新的人站到同一位置」。舊參與者離開 2 s 後新的人才出現，hmm 會先短暫進入 MAYBE_AWAY（每個種子 1 次），等新的人穩定後再 NEW_GROUP。
  - rules 只要有任何可見的人就算「有人」，所以不會 MAYBE。
  - 這次 MAYBE 其實反映了「原本的對話對象離開了」，可以說是正確的訊號，但依照指標定義算作誤判。
- 「遠處路人」的斷裂在三個版本都是 20：參與者離開後 session 結束（ENDED 會清空 track），路人的 track 重建。這是設計如此，不是追蹤錯誤。

## 建議

- 在合成資料上，`kalman+hmm` 的總誤判（MAYBE 25、AWAY 0）明顯少於 `legacy+rules`（53、27），離開判定一樣快或更快，狀態也更穩定。
- 建議先在實際場景用網頁切換「在場判斷模型＝機率」＋「追蹤模式＝Kalman」試用，觀察「在場機率」曲線。沒有問題後，把兩個預設都改成新版本。
- 如果覺得離開判定太快或太慢，只要調 `leave_halflife`。
