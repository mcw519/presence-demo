import cv2
import numpy as np
import pytest

from presence_demo.optimizer import FrameProcessor, crop_box, map_raw_to_frame, moved, recommend_fps
from presence_demo.tracker import Config


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class RectDetector:
    """Finds the white rectangle in the (possibly cropped/scaled) image; records input sizes."""

    name = "rect"

    def __init__(self):
        self.calls = []

    def detect(self, img):
        self.calls.append(img.shape[:2])
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts = cv2.findNonZero((g > 200).astype(np.uint8))
        if pts is None:
            return []
        x, y, w, h = cv2.boundingRect(pts)
        row = np.zeros(15, np.float32)
        row[:4] = (x, y, w, h)
        row[4:14] = [x, y] * 5
        return [(row, 0.9)]


class CountEmbedder:
    name = "count"

    def __init__(self):
        self.calls = 0

    def embed(self, img, raw):
        self.calls += 1
        return np.ones(8, np.float32)


class Pipe:
    def __init__(self):
        self.detector, self.embedder = RectDetector(), CountEmbedder()


def frame(x=300, y=200, w=60, h=80, noise=0):
    img = np.full((480, 640, 3), 40, np.uint8)
    if w:
        img[y : y + h, x : x + w] = 255
    if noise:
        img[:50, :50] = noise
    return img


@pytest.fixture
def setup():
    clk = Clock()
    pipe = Pipe()
    return FrameProcessor(pipe, clock=clk), pipe, clk


def test_crop_box_and_mapping():
    assert crop_box((0.2, 0.1, 0.8, 0.9), 0.1, 640, 480) == (64, 0, 576, 480)
    raw = np.zeros(15, np.float32)
    raw[:4] = (10, 20, 30, 40)
    raw[4:6] = (15, 25)
    m = map_raw_to_frame(raw, 0.5, 64, 0)
    assert list(m[:4]) == [84, 40, 60, 80] and list(m[4:6]) == [94, 50]


def test_crop_downscale_boxes_map_back(setup):
    proc, pipe, clk = setup
    cfg = Config(roi_crop=True, max_width=200, motion_gate=False)
    dets, tm = proc.process(frame(300, 200, 60, 80), cfg)
    assert pipe.detector.calls[-1][1] <= 200  # detector saw a downscaled crop
    x0, y0, x1, y1 = dets[0].bbox
    assert abs(x0 * 640 - 300) < 4 and abs(y0 * 480 - 200) < 4
    assert abs(x1 * 640 - 360) < 4 and abs(y1 * 480 - 280) < 4
    cfg2 = Config(roi_crop=False, max_width=1920, motion_gate=False)
    dets2, _ = proc.process(frame(300, 200, 60, 80), cfg2)
    assert pipe.detector.calls[-1] == (480, 640)
    assert np.allclose(dets2[0].bbox, (300 / 640, 200 / 480, 360 / 640, 280 / 480), atol=1e-3)


def test_motion_gate_skips_static_and_forces_periodic(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=True, force_detect_seconds=3.0)
    proc.process(frame(), cfg)
    clk.t += 1
    dets, tm = proc.process(frame(), cfg)
    assert tm["skipped"] and len(pipe.detector.calls) == 1 and len(dets) == 1
    clk.t += 1
    proc.process(frame(), cfg, stable=False)  # unstable state -> always detect
    assert len(pipe.detector.calls) == 2
    clk.t += 3.5
    _, tm = proc.process(frame(), cfg)  # forced detection after force_detect_seconds
    assert not tm["skipped"] and len(pipe.detector.calls) == 3
    clk.t += 1
    _, tm = proc.process(frame(x=100), cfg)  # motion -> detect
    assert not tm["skipped"]
    assert proc.counts["skipped"] == 1


def test_motion_gate_off(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=False)
    for _ in range(3):
        clk.t += 1
        proc.process(frame(), cfg)
    assert len(pipe.detector.calls) == 3 and proc.counts["skipped"] == 0


def test_embedding_throttle(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=False, embed_throttle=True, embed_interval=3.0, embed_move_thresh=0.15)
    proc.process(frame(), cfg)
    clk.t += 1
    proc.process(frame(x=302), cfg)  # tiny move -> reuse
    clk.t += 1
    proc.process(frame(), cfg)
    assert pipe.embedder.calls == 1 and proc.counts["embed_reused"] == 2
    clk.t += 0.5
    proc.process(frame(x=330), cfg)  # moved half a face width -> recompute
    assert pipe.embedder.calls == 2
    clk.t += 3.1
    proc.process(frame(x=330), cfg)  # interval elapsed -> recompute
    assert pipe.embedder.calls == 3


def test_embedding_throttle_off(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=False, embed_throttle=False)
    for _ in range(3):
        clk.t += 1
        proc.process(frame(), cfg)
    assert pipe.embedder.calls == 3


def test_moved():
    a = (0.4, 0.4, 0.5, 0.5)
    assert not moved(a, (0.405, 0.4, 0.505, 0.5), 0.15)
    assert moved(a, (0.43, 0.4, 0.53, 0.5), 0.15)
    assert moved(a, (0.4, 0.4, 0.55, 0.55), 0.15)


def _snap(state, tracks=()):
    return {"state": state, "tracks": list(tracks)}


def test_adaptive_fps():
    cfg = Config(fps_low=0.5, fps_high=2.0)
    p = {"id": 1, "participant": True, "visible": True}
    assert recommend_fps(_snap("ACTIVE", [p]), cfg, False) == 0.5
    assert recommend_fps(_snap("IDLE", [p]), cfg, False) == 0.5
    assert recommend_fps(_snap("MAYBE_AWAY", [p]), cfg, False) == 2.0
    assert recommend_fps(_snap("ACTIVE", [p]), cfg, True) == 2.0  # tracks changing
    cand = {"id": 2, "participant": False, "visible": True}
    assert recommend_fps(_snap("ACTIVE", [p, cand]), cfg, False) == 2.0
    gone = {"id": 1, "participant": True, "visible": False}
    assert recommend_fps(_snap("IDLE", [gone]), cfg, False) == 2.0
    cfg.adaptive_fps = False
    assert recommend_fps(_snap("MAYBE_AWAY"), cfg, True) == 0.0


def test_bool_config_validation():
    c = Config()
    c.update({"motion_gate": "false", "roi_crop": 0, "adaptive_fps": "on"})
    assert c.motion_gate is False and c.roi_crop is False and c.adaptive_fps is True
    w = c.update({"embed_throttle": "maybe", "fps_low": 4, "fps_high": 1})
    assert c.embed_throttle is True and w and c.fps_low == 1


class FakePose:
    name = "fake-pose"

    def __init__(self):
        self.shapes = []

    def detect(self, img):
        self.shapes.append(img.shape[:2])
        h, w = img.shape[:2]
        kp = {
            "nose": (w / 2, h * 0.2, 0.9),
            "l_shoulder": (w * 0.6, h * 0.4, 0.9),
            "r_shoulder": (w * 0.4, h * 0.4, 0.9),
        }
        return [((w * 0.25, h * 0.1, w * 0.75, h * 0.9), 0.8, kp)]


def test_pose_runs_on_pose_roi_and_maps_back():
    clk = Clock()
    fake = FakePose()
    proc = FrameProcessor(Pipe(), clock=clk, pose_loader=lambda: fake)
    cfg = Config(motion_gate=False, pose_enabled=True, pose_roi=(0.5, 0.0, 1.0, 1.0), crop_margin=0.0)
    dets, tm = proc.process(frame(w=0), cfg)  # no face, one body
    assert fake.shapes[-1] == (480, 320) and proc.counts["pose_calls"] == 1 and "pose" in tm
    (d,) = dets
    assert d.face_bbox is None and d.body is not None and d.embedding is None
    x0, y0, x1, y1 = d.body.bbox
    assert abs(x0 - (0.5 + 0.25 * 0.5)) < 1e-3 and abs(x1 - (0.5 + 0.75 * 0.5)) < 1e-3
    assert abs(d.body.keypoints["nose"][0] - 0.75) < 1e-3
    cfg.pose_enabled = False
    proc.process(frame(), cfg)
    assert proc.counts["pose_calls"] == 1
