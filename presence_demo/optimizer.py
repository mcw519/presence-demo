"""Performance / power optimizations around the vision pipeline.

All work happens on in-memory numpy arrays. PRIVACY: never write frames to disk here.

- Motion gate: compare a tiny grayscale thumbnail with the previous frame; if nearly identical and the
  state is stable, reuse the previous detections instead of running the detector.
- Embedding throttle: reuse a face's embedding while it stays put, recompute when it is new, moved or
  resized, had low self-similarity, or every ``embed_interval`` seconds.
- ROI crop + downscale: run detection on the ROI (+margin) resized to ``max_width``; boxes and
  landmarks are mapped back to full-frame coordinates.
- Adaptive fps: :func:`recommend_fps` suggests a client frame rate from the tracker state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .appearance import descriptor, torso_box
from .body import Body, associate, head_box
from .intent import frontal_from_landmarks
from .tracker import Config, Detection, iou
from .vision import PoseDetector, VisionPipeline

THUMB = (64, 48)


def crop_box(roi: tuple, margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    """ROI (normalized) + margin -> pixel crop (x0, y0, x1, y1), clamped to the frame."""
    x0 = int(max(0.0, roi[0] - margin) * w)
    y0 = int(max(0.0, roi[1] - margin) * h)
    x1 = int(np.ceil(min(1.0, roi[2] + margin) * w))
    y1 = int(np.ceil(min(1.0, roi[3] + margin) * h))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def map_raw_to_frame(raw: np.ndarray, scale: float, ox: int, oy: int) -> np.ndarray:
    """Map a YuNet-style row (x, y, w, h, 5 landmark pairs, score) from crop space to frame space."""
    out = np.array(raw, dtype=np.float32, copy=True)
    out[:4] = out[:4] / scale
    out[0] += ox
    out[1] += oy
    for i in range(4, 14, 2):
        out[i] = out[i] / scale + ox
        out[i + 1] = out[i + 1] / scale + oy
    return out


def moved(a: tuple, b: tuple, thresh: float) -> bool:
    """True if box b's center moved or its size changed by more than ``thresh`` relative to box a."""
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if aw <= 0 or ah <= 0:
        return True
    dx = abs((a[0] + a[2]) - (b[0] + b[2])) / 2 / aw
    dy = abs((a[1] + a[3]) - (b[1] + b[3])) / 2 / ah
    ds = max(abs(bw - aw) / aw, abs(bh - ah) / ah)
    return max(dx, dy, ds) > thresh


@dataclass
class _Cached:
    bbox: tuple
    embedding: np.ndarray | None
    t: float
    self_sim: float = 1.0


def recommend_fps(snapshot: dict, cfg: Config, tracks_changed: bool) -> float:
    """Low fps when stable (ACTIVE/IDLE/ENDED, no candidates), high when things change."""
    if not cfg.adaptive_fps:
        return 0.0  # 0 = client uses its own fps setting
    state = snapshot.get("state")
    candidates = any(t["visible"] and not t["participant"] for t in snapshot.get("tracks", []))
    missing = any(t["participant"] and not t["visible"] for t in snapshot.get("tracks", []))
    if state in ("MAYBE_AWAY", "AWAY") or tracks_changed or candidates or missing:
        return cfg.fps_high
    return cfg.fps_low


def fuse(faces: list[Detection], bodies: list[Body]) -> list[Detection]:
    """Attach bodies to faces (face center in the body's head region); body-only people get a head box."""
    out = []
    for fi, bi in associate([f.bbox for f in faces], bodies):
        if fi is not None:
            f = faces[fi]
            out.append(
                Detection(
                    f.bbox,
                    f.conf,
                    f.embedding,
                    face_bbox=f.bbox,
                    body=bodies[bi] if bi is not None else None,
                    frontal=f.frontal,
                )
            )
        else:
            b = bodies[bi]
            out.append(Detection(head_box(b), b.conf, None, face_bbox=None, body=b))
    return out


class FrameProcessor:
    """Per-connection wrapper around :class:`VisionPipeline` applying the optimizations."""

    def __init__(
        self,
        pipe: VisionPipeline,
        clock=time.monotonic,
        pose_loader: Callable[[], PoseDetector | None] | None = None,
    ):
        self.pipe = pipe
        self.clock = clock
        self.pose_loader = pose_loader
        self._pose: PoseDetector | None = None
        self.pose_available: bool | None = None  # None = not tried yet
        self.reset()

    def _get_pose(self) -> PoseDetector | None:
        if self._pose is None and self.pose_available is None and self.pose_loader is not None:
            self._pose = self.pose_loader()
            self.pose_available = self._pose is not None
        return self._pose

    def _detect_bodies(self, img: np.ndarray, cfg: Config) -> list[Body]:
        """Run the pose detector on the pose ROI (+margin); returns bodies in normalized frame coordinates."""
        pose = self._get_pose()
        if pose is None:
            return []
        h, w = img.shape[:2]
        x0, y0, x1, y1 = crop_box(cfg.pose_roi, cfg.crop_margin if cfg.roi_crop else 0.0, w, h)
        sub = img[y0:y1, x0:x1]
        self.counts["pose_calls"] += 1
        bodies = []
        for (bx0, by0, bx1, by1), conf, kp in pose.detect(sub):
            if conf < cfg.pose_conf:
                continue
            bbox = (
                max(0.0, (bx0 + x0) / w),
                max(0.0, (by0 + y0) / h),
                min(1.0, (bx1 + x0) / w),
                min(1.0, (by1 + y0) / h),
            )
            kps = {n: ((p[0] + x0) / w, (p[1] + y0) / h, p[2]) for n, p in kp.items()}
            bodies.append(Body(bbox, conf, kps))
        return bodies

    def reset(self) -> None:
        self.last_thumb: np.ndarray | None = None
        self.last_dets: list[Detection] | None = None
        self.last_full = -1e9
        self.cache: list[_Cached] = []
        self.counts = {
            "frames": 0,
            "skipped": 0,
            "detect_calls": 0,
            "embed_computed": 0,
            "embed_reused": 0,
            "pose_calls": 0,
        }

    def process(self, img: np.ndarray, cfg: Config, stable: bool = True) -> tuple[list[Detection], dict]:
        now = self.clock()
        h, w = img.shape[:2]
        self.counts["frames"] += 1
        if hasattr(self.pipe, "select_embedder"):
            self.pipe.select_embedder(cfg.embed_backend, cfg.npu_delegate_path, cfg.npu_precision)
        t0 = time.perf_counter()
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)
        diff = 1.0 if self.last_thumb is None else float(np.mean(cv2.absdiff(thumb, self.last_thumb))) / 255.0
        self.last_thumb = thumb
        tm = {"motion": (time.perf_counter() - t0) * 1000, "detect": 0.0, "embed": 0.0, "diff": diff}

        if (
            cfg.motion_gate
            and stable
            and self.last_dets is not None
            and diff < cfg.motion_threshold
            and now - self.last_full < cfg.force_detect_seconds
        ):
            self.counts["skipped"] += 1
            tm.update(skipped=True, embed_computed=0, embed_reused=0)
            tm["pose"] = 0.0
            return [replace(d) for d in self.last_dets], tm

        # --- detection (optionally on cropped + downscaled ROI) ---
        if cfg.roi_crop:
            x0, y0, x1, y1 = crop_box(cfg.face_roi, cfg.crop_margin, w, h)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        sub = img[y0:y1, x0:x1]
        scale = min(1.0, cfg.max_width / max(1, sub.shape[1]))
        if scale < 1.0:
            sub = cv2.resize(sub, (max(1, int(sub.shape[1] * scale)), max(1, int(sub.shape[0] * scale))))
        t1 = time.perf_counter()
        raws = self.pipe.detector.detect(sub)
        self.counts["detect_calls"] += 1
        self.last_full = now
        t2 = time.perf_counter()
        tm["detect"] = (t2 - t1) * 1000

        # --- embeddings (throttled) ---
        out: list[Detection] = []
        new_cache: list[_Cached] = []
        computed = reused = 0
        for raw, conf in raws:
            full = map_raw_to_frame(raw, scale, x0, y0)
            x, y, bw, bh = (float(v) for v in full[:4])
            bbox = (max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w), min(1.0, (y + bh) / h))
            prev = max(self.cache, key=lambda c: iou(c.bbox, bbox), default=None)
            reuse = (
                cfg.embed_throttle
                and prev is not None
                and prev.embedding is not None
                and iou(prev.bbox, bbox) > 0.3
                and not moved(prev.bbox, bbox, cfg.embed_move_thresh)
                and now - prev.t < cfg.embed_interval
                and prev.self_sim >= cfg.cos_match
            )
            if reuse:
                emb, t_emb, ss, cb = prev.embedding, prev.t, prev.self_sim, prev.bbox
                reused += 1
            else:
                try:
                    emb = self.pipe.embedder.embed(sub, raw)
                except Exception:  # noqa: BLE001
                    emb = None
                computed += 1
                ss = 1.0
                if emb is not None and prev is not None and prev.embedding is not None and iou(prev.bbox, bbox) > 0.3:
                    a, b = np.ravel(emb), np.ravel(prev.embedding)
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    ss = float(a @ b / (na * nb)) if na > 0 and nb > 0 and a.shape == b.shape else 0.0
                t_emb, cb = now, bbox
            new_cache.append(_Cached(cb, emb, t_emb, ss))
            out.append(Detection(bbox, conf, emb, frontal=frontal_from_landmarks(full)))
        self.cache = new_cache
        tm["embed"] = (time.perf_counter() - t2) * 1000
        tm["pose"] = 0.0
        if cfg.pose_enabled:
            t3 = time.perf_counter()
            bodies = self._detect_bodies(img, cfg)
            tm["pose"] = (time.perf_counter() - t3) * 1000
            out = fuse(out, bodies)
        if cfg.tracker_mode == "kalman":  # torso appearance for re-identification (in memory only)
            t4 = time.perf_counter()
            for d in out:
                d.appearance = descriptor(img, torso_box(d.face_bbox, d.body))
            tm["appearance"] = (time.perf_counter() - t4) * 1000
        self.last_dets = [replace(d) for d in out]
        self.counts["embed_computed"] += computed
        self.counts["embed_reused"] += reused
        tm.update(
            skipped=False,
            embed_computed=computed,
            embed_reused=reused,
            det_w=sub.shape[1],
            det_h=sub.shape[0],
        )
        return out, tm
