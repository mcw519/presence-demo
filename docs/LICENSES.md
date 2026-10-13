# 授權稽核（商用可行性）

> **這不是法律意見。** 本文件是工程上的整理，依據 2026-10-13 從官方儲存庫、模型卡和套件中繼資料讀到的授權文字。正式商用前，請由法務確認，尤其是「模型權重的訓練資料」這類授權文字沒有明確處理的部分。

圖例：✅ 授權文字允許商用　⚠️ 授權允許，但有未確認或潛在風險　❌ 不建議商用

## 模型

| 元件 | 授權 | 來源 | 商用？ | 說明／風險 |
|---|---|---|---|---|
| **YuNet** 人臉偵測（`face_detection_yunet_2023mar.onnx`） | MIT（opencv_zoo 目錄 LICENSE，Copyright 2020 Shiqi Yu）；訓練程式 libfacedetection.train 為 BSD-3-Clause | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet ・ https://github.com/ShiqiYu/libfacedetection.train | ⚠️ 大致可以 | 權重本身是 MIT。README 說明是用 **WIDER FACE** 訓練；WIDER FACE 資料集本身的使用條款沒有在 opencv_zoo 寫明，我們也沒能從官方網站確認（網站連線逾時）。「用研究資料集訓練的權重能否商用」在法律上沒有定論。風險比 SFace 低。 |
| **SFace** 人臉 embedding（`face_recognition_sface_2021dec.onnx`） | Apache-2.0（opencv_zoo 目錄 LICENSE：「All files in this directory are licensed under Apache 2.0」） | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface ・ 原始程式 https://github.com/zhongyy/SFace | ⚠️ **高風險** | 權重檔標示 Apache-2.0，但 opencv_zoo **沒有寫出這個權重用哪個資料集訓練**。SFace 論文和原始 repo 的實驗是用 **CASIA-WebFace、VGGFace2、MS-Celeb-1M** 訓練。這些資料集是從網路蒐集的名人照片：MS-Celeb-1M 已被微軟撤下，VGGFace2 和 CASIA-WebFace 也都只限非商業研究使用（請自行查證各資料集現行條款）。如果這個權重是用其中之一訓練的，商用有風險，而且臉部 embedding 屬於生物特徵資料，另有個資法規（GDPR、BIPA 等）問題。**建議產品不要依賴這個模型**（見下方替代方案）。 |
| **MoveNet MultiPose Lightning**（`movenet_multipose_lightning_fp16.tflite`，本版預設姿勢模型） | Apache-2.0（Kaggle 模型頁：「This model follows Apache 2.0」，每個 variation 標示 License: Apache 2.0） | https://www.kaggle.com/models/google/movenet （variation：`tfLite/multipose-lightning-tflite-float16`） | ✅ | Google 官方釋出。模型卡把「任何形式的監控或身分辨識」列為**非預期用途（out-of-scope）**；本專案只用它判斷有沒有人、朝向和距離，不用來辨識身分。官方說明主要針對 3–6 呎距離、健身動作，對話場景請自行驗證準確度。模型卡的訓練資料（COCO 等）沒有在授權中另外限制。 |
| ~~YOLOv8n-pose~~（v0.3.0） | AGPL-3.0（Ultralytics） | https://github.com/ultralytics/ultralytics | ❌ 已移除 | AGPL 對網路服務也要求開放原始碼；商用需要 Ultralytics 企業授權。v0.4.0 已刪除程式碼和下載。 |
| Haar cascade（備援，`haarcascade_frontalface_default.xml`，OpenCV 內附） | Intel License Agreement（BSD 類，寫在 XML 檔頭） | OpenCV 套件內 `cv2/data/` | ✅ | 只在沒有 YuNet 時使用。 |
| HSV 直方圖 embedding（備援） | 本專案程式碼，沒有模型權重 | `presence_demo/vision.py` | ✅ | 沒有訓練資料問題，但辨識力很弱（主要靠衣服顏色），換衣服或光線變化就不準。 |

## Python 套件（執行時）

由 `pip-licenses` 從安裝的套件中繼資料讀出，並對照各專案 repo。

| 套件 | 版本 | 授權 | 商用？ | 說明 |
|---|---|---|---|---|
| fastapi | 0.143.0 | MIT | ✅ | |
| starlette | 1.8.0 | BSD-3-Clause | ✅ | |
| uvicorn（含 standard：uvloop、httptools、watchfiles、websockets、python-dotenv、PyYAML） | 0.54.0 | BSD-3-Clause（uvloop：MIT/Apache-2.0；httptools、watchfiles、PyYAML：MIT；websockets、python-dotenv：BSD-3） | ✅ | |
| pydantic / pydantic-core | 2.14 / 2.50 | MIT | ✅ | |
| opencv-python-headless | 4.14.0.94 | Apache-2.0 | ⚠️ 可以，需附授權 | wheel 內附 **FFmpeg（LGPL-2.1）**、OpenBLAS、libvpx、libaom、OpenSSL 等（見套件內 `LICENSE-3RD-PARTY.txt`）。LGPL 是動態連結的 .so，一般可以商用，但散布時要附上授權並允許替換。Android 產品通常不會用這個 wheel。 |
| numpy | 2.5.4 | BSD-3-Clause（另含 0BSD、MIT、Zlib、CC0 元件） | ✅ | |
| psutil | 7.2.2 | BSD-3-Clause | ✅ | |
| ai-edge-litert（LiteRT） | 2.3.0 | Apache-2.0 | ✅ | 依賴 flatbuffers（Apache-2.0）、protobuf（BSD-3）、ml-dtypes（Apache-2.0）、tqdm（MPL-2.0 AND MIT）、backports.strenum（MIT）。 |
| anyio、h11、idna、click、annotated-types、typing-inspection、annotated-doc | — | MIT／BSD-3 | ✅ | |
| typing-extensions | 4.16 | PSF-2.0 | ✅ | |
| tqdm、certifi | — | MPL-2.0（tqdm 為 MPL-2.0 AND MIT） | ✅ | MPL 是檔案層級的 copyleft，只有修改這些檔案本身才需要公開。 |
| opentelemetry-api | 1.45 | Apache-2.0 | ✅ | |

開發用（不隨產品散布）：pytest（MIT）、httpx（BSD-3）、ruff（MIT）、websockets（BSD-3）。

前端：`static/index.html` 是本專案自寫，沒有使用第三方 JS 函式庫或 CDN。

## 測試資料

| 檔案 | 授權 | 來源 | 商用？ | 說明 |
|---|---|---|---|---|
| `tests/data/face.jpg` | **CC0 1.0** | Julia Margaret Cameron，〈Julia Jackson Duckworth〉，1867，Cleveland Museum of Art 1996.343 ・ https://clevelandart.org/art/1996.343 ・ https://commons.wikimedia.org/wiki/File:Julia_Margaret_Cameron_-_Julia_Jackson_Duckworth_(1846-1895)_-_1996.343_-_Cleveland_Museum_of_Art.tif | ✅ | 500 px 縮圖。19 世紀照片，被攝者 1895 年過世，沒有肖像權問題。只用在測試和基準測試，不隨產品散布。 |
| ~~lena.jpg~~（v0.3.0 以前） | 不明（原圖是 1972 年 Playboy 照片，版權歸屬有爭議） | OpenCV samples | ❌ 已替換 | |

## 有問題的元件與替代方案

1. **SFace（人臉 embedding）— 最大風險。** 可以考慮：
   - 用 `PRESENCE_FACE_EMBEDDER=hist` 關掉 SFace，只用 HSV 直方圖；不用任何人臉辨識權重，但重新辨識能力會變差。這個產品只需要「同一群人還在不在」，不需要知道是誰，所以這可能就夠用。
   - 用身體外觀或位置追蹤取代人臉身分：MoveNet 骨架加 IoU 追蹤，或加上授權乾淨的 person re-ID。注意很多 re-ID 權重是用 Market-1501／DukeMTMC 訓練的，DukeMTMC 已撤下。
   - 自己用有商用授權的資料（自家蒐集並取得同意，或合成人臉資料集）訓練或微調 MobileFaceNet。
   - 向供應商購買有商用授權的人臉辨識 SDK。
   - 注意：常見的公開人臉辨識權重（InsightFace／ArcFace 系列、FaceNet 系列等）幾乎都是用 MS1M、VGGFace2、Glint360K、WebFace 等資料訓練的。InsightFace 的預訓練模型明文只限非商業研究使用，**不要假設「Apache／MIT 程式碼」就代表權重可以商用**。
2. **YuNet（人臉偵測）— 中低風險。** 替代方案：MediaPipe **BlazeFace**（Face Detector，Google AI Edge）。MediaPipe 程式碼是 Apache-2.0；模型的授權寫在個別模型卡（PDF），**本次沒有逐份確認**，採用前請閱讀模型卡。在這台環境，MediaPipe Python 套件需要 `libEGL`，但 `.tflite` 模型本身可以用 ai-edge-litert 執行，和 MoveNet 一樣。也可以直接用 MoveNet 的鼻子、眼睛、耳朵關鍵點估計頭部位置，完全不需要人臉偵測器。
3. **姿勢模型的其他候選（沒有採用）：** RTMPose／RTMO（MMPose，程式碼 Apache-2.0，權重用 COCO 等資料訓練，部分版本混入 AIC／CrowdPose 等資料集，各有條款）；MediaPipe Pose Landmarker（單人為主，需要另外加人物偵測器）。MoveNet MultiPose 是唯一由官方直接以 Apache-2.0 發布、原生支援多人、而且能在這個環境（Python 3.13、Linux、無 GPU）用 pip 套件執行的選項。
