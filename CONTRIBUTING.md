# 開發指南

## 環境

```bash
uv sync                 # 安裝執行期與 dev 相依套件（pytest、httpx、ruff、websockets）
```

## 執行

```bash
uv run presence-demo --port 8000
PRESENCE_TOKEN=xxx uv run presence-demo
```

## 測試

```bash
uv run pytest -q
```

- `tests/test_tracker.py`：狀態機情境（注入時鐘與合成 embedding），不需要模型。
- `tests/test_config.py`、`tests/test_stats.py`：設定驗證、統計。
- `tests/test_ws_pipeline.py`：WebSocket 端到端。會嘗試下載模型；下載失敗時自動改用備援偵測器。

## 程式風格

```bash
uv run ruff check .          # lint
uv run ruff format .         # 格式化
uv run ruff check . --fix
```

提交前請確認 `ruff check`、`ruff format --check` 和 `pytest` 都通過。

## 擴充

- 換偵測器或 embedding：實作 `vision.Detector`／`vision.Embedder`，傳入 `VisionPipeline(detector=..., embedder=...)`。
- 新增參數：在 `Config` 加欄位、在 `RANGES` 加範圍，在 `static/index.html` 的 `DEF`／`GROUPS` 加 UI，並更新 `docs/CONFIGURATION.md` 和測試（`test_ranges_cover_all_fields` 會檢查 RANGES 是否完整）。

## 安全

- 不要提交權杖、`.env`、`models/`、`logs/`（已列入 `.gitignore`）。
- 權杖只從 `PRESENCE_TOKEN` 環境變數讀取。
