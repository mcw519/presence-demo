"""Pluggable Detector / Embedder. Default: OpenCV YuNet + SFace (ONNX, CPU).
Fallback (no model files): Haar cascade (bundled with OpenCV) + HSV color-histogram embedding."""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .tracker import Detection

log = logging.getLogger(__name__)
MODEL_DIR = Path(os.environ.get("PRESENCE_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
YUNET = ("face_detection_yunet_2023mar.onnx", ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = (
    "face_recognition_sface_2021dec.onnx",
    ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
)


YOLO_POSE = (
    "yolov8n-pose.onnx",
    "https://huggingface.co/Xenova/yolov8n-pose/resolve/main/onnx/model.onnx",
)
COCO_KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder"]
COCO_KP_IDX = {**{n: i for i, n in enumerate(COCO_KP)}, "l_hip": 11, "r_hip": 12}


class PoseDetector(Protocol):
    """Returns bodies in pixel coordinates of the given image: (bbox_xyxy, conf, {name: (x, y, c)})."""

    name: str

    def detect(self, img: np.ndarray) -> list[tuple[tuple, float, dict]]: ...


class YoloPoseDetector:
    """YOLOv8n-pose (ONNX, COCO-17 keypoints) via OpenCV DNN on CPU. Fixed 640x640 letterboxed input."""

    name = "YOLOv8n-pose"
    SIZE = 640

    def __init__(self, path, conf: float = 0.4):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.conf = conf

    def detect(self, img):
        h, w = img.shape[:2]
        s = self.SIZE / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.SIZE, self.SIZE), swapRB=True))
        out = self.net.forward()[0].T  # (8400, 56)
        keep = out[:, 4] >= self.conf
        out = out[keep]
        if not len(out):
            return []
        boxes = [[float(c[0] - c[2] / 2), float(c[1] - c[3] / 2), float(c[2]), float(c[3])] for c in out]
        idx = cv2.dnn.NMSBoxes(boxes, out[:, 4].tolist(), self.conf, 0.45)
        res = []
        for i in np.array(idx).ravel():
            r = out[i]
            x, y, bw, bh = boxes[i]
            bbox = (x / s, y / s, (x + bw) / s, (y + bh) / s)
            kps = r[5:].reshape(17, 3)
            kp = {n: (float(kps[j, 0] / s), float(kps[j, 1] / s), float(kps[j, 2])) for n, j in COCO_KP_IDX.items()}
            res.append((bbox, float(r[4]), kp))
        return res


def load_pose_detector(allow_download: bool = True) -> PoseDetector | None:
    """Load the default pose detector, downloading the model on first use. None if unavailable."""
    p = ensure_model(*YOLO_POSE) if allow_download else (MODEL_DIR / YOLO_POSE[0])
    if not p or not Path(p).exists():
        return None
    try:
        return YoloPoseDetector(p)
    except Exception as e:  # noqa: BLE001
        log.warning("pose model failed to load: %s", e)
        return None


class Detector(Protocol):
    name: str

    def detect(self, img: np.ndarray) -> list[tuple[np.ndarray, float]]: ...  # (raw row, conf)


class Embedder(Protocol):
    name: str

    def embed(self, img: np.ndarray, raw: np.ndarray) -> np.ndarray: ...


def ensure_model(fname, url) -> Path | None:
    p = MODEL_DIR / fname
    if p.exists() and p.stat().st_size > 1000:
        return p
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, p)
        return p
    except Exception as e:  # noqa: BLE001
        log.warning("model download failed (%s): %s", fname, e)
        if p.exists():
            p.unlink()
        return None


class YuNetDetector:
    name = "YuNet"

    def __init__(self, path, score=0.5):
        self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 50)

    def detect(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        return [] if faces is None else [(f, float(f[14])) for f in faces]


class HaarDetector:
    name = "Haar(fallback)"

    def __init__(self):
        self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects, _, weights = self.det.detectMultiScale3(g, 1.1, 5, minSize=(40, 40), outputRejectLevels=True)
        out = []
        for (x, y, w, h), wt in zip(rects, np.ravel(weights) if len(rects) else [], strict=False):
            row = np.zeros(15, np.float32)
            row[:4] = (x, y, w, h)
            out.append((row, float(min(1.0, max(0.0, wt / 5.0)))))  # rough confidence
        return out


class SFaceEmbedder:
    name = "SFace"

    def __init__(self, path):
        self.rec = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, img, raw):
        return self.rec.feature(self.rec.alignCrop(img, raw)).ravel()


class HistEmbedder:
    """Weak fallback: HSV histogram of face + upper body region. Tune cos thresholds higher."""

    name = "HSV-hist(fallback)"

    def embed(self, img, raw):
        x, y, w, h = [int(v) for v in raw[:4]]
        H, W = img.shape[:2]
        crop = img[max(0, y) : min(H, y + int(h * 2.2)), max(0, x - w // 4) : min(W, x + w + w // 4)]
        if crop.size == 0:
            return np.zeros(96, np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 6], [0, 180, 0, 256]).ravel()
        return hist - hist.mean()


def decode_jpeg(data: bytes) -> tuple[np.ndarray | None, float]:
    """Decode JPEG bytes in memory (PRIVACY: never written to disk). Returns (image, ms)."""
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img, (time.perf_counter() - t0) * 1000


class VisionPipeline:
    def __init__(self, detector: Detector | None = None, embedder: Embedder | None = None, allow_download: bool = True):
        if detector is None:
            p = ensure_model(*YUNET) if allow_download else None
            detector = YuNetDetector(p) if p else HaarDetector()
        if embedder is None:
            p = ensure_model(*SFACE) if allow_download else None
            embedder = SFaceEmbedder(p) if p and isinstance(detector, YuNetDetector) else HistEmbedder()
        self.detector, self.embedder = detector, embedder

    @property
    def name(self):
        return f"{self.detector.name} + {self.embedder.name}"

    def process(self, img: np.ndarray) -> list[Detection]:
        return self.process_timed(img)[0]

    def process_timed(self, img: np.ndarray) -> tuple[list[Detection], dict]:
        """Returns detections and timings in ms: decode/detect/embed."""
        h, w = img.shape[:2]
        out = []
        t0 = time.perf_counter()
        raws = self.detector.detect(img)
        t1 = time.perf_counter()
        for raw, conf in raws:
            x, y, bw, bh = [float(v) for v in raw[:4]]
            bbox = (max(0, x / w), max(0, y / h), min(1, (x + bw) / w), min(1, (y + bh) / h))
            try:
                e = self.embedder.embed(img, raw)
            except Exception:  # noqa: BLE001
                e = None
            out.append(Detection(bbox, conf, e))
        t2 = time.perf_counter()
        return out, {"detect": (t1 - t0) * 1000, "embed": (t2 - t1) * 1000}

    def process_jpeg(self, data: bytes) -> list[Detection]:
        return self.process_jpeg_timed(data)[0]

    def process_jpeg_timed(self, data: bytes) -> tuple[list[Detection], dict]:
        """Decode + process a JPEG entirely in memory.

        PRIVACY GUARD: frames must never touch the disk. Do not add cv2.imwrite, debug dumps,
        temp files or logging of `data` here; the image buffer is dropped when this returns.
        tests/test_privacy.py enforces this.
        """
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        dec = (time.perf_counter() - t0) * 1000
        if img is None:
            return [], {"decode": dec, "detect": 0.0, "embed": 0.0, "w": 0, "h": 0}
        dets, t = self.process_timed(img)
        return dets, {"decode": dec, **t, "w": img.shape[1], "h": img.shape[0]}

    def info(self) -> dict:
        return {
            "pipeline": self.name,
            "opencv": cv2.__version__,
            "cv_threads": cv2.getNumThreads(),
            "dnn_backend": "OpenCV (CPU)",
            "dnn_target": "CPU",
            "cpu_count": os.cpu_count(),
        }
