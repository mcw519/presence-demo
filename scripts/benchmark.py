"""Benchmark optimizations off vs on over a synthetic 60 s scenario.

PRIVACY: frames are synthesized in memory from tests/data/face.jpg and JPEG-encoded/decoded in memory.
Nothing is written to disk.

Scenario (simulated time): 0-15 s person static, 15-30 s person shifting, 30-45 s empty ROI,
45-60 s person back (static). Frames get light sensor noise.

    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from presence_demo.optimizer import FrameProcessor, recommend_fps
from presence_demo.tracker import Config, PresenceTracker
from presence_demo.vision import VisionPipeline, decode_jpeg

FACE = Path(__file__).resolve().parent.parent / "tests" / "data" / "face.jpg"
DURATION = 60.0


class SimClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_frame(t: float, face: np.ndarray, bg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = bg.copy()
    if not (30 <= t < 45):
        dx = 0
        if 15 <= t < 30:  # walking around a bit
            dx = int(40 * np.sin((t - 15) * 0.8))
        x = 200 + dx
        img[60:420, x : x + 240] = face
    noise = rng.normal(0, 2.0, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def run(pipe: VisionPipeline, cfg: Config, fixed_fps: float, label: str) -> dict:
    clk = SimClock()
    proc = FrameProcessor(pipe, clock=clk)
    tr = PresenceTracker(cfg, clock=clk)
    face = cv2.resize(cv2.imread(str(FACE)), (240, 360))
    bg = np.full((480, 640, 3), 90, np.uint8)
    bg[:, :, 0] = np.linspace(60, 140, 640, dtype=np.uint8)
    rng = np.random.default_rng(0)
    lat, sig, changed_at, states = [], (), -10.0, []
    cpu, wall0 = 0.0, time.perf_counter()
    while clk.t < DURATION:
        jpg = cv2.imencode(".jpg", make_frame(clk.t, face, bg, rng), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        t0, c0 = time.perf_counter(), time.process_time()
        img, _ = decode_jpeg(jpg)
        stable = tr.state.value not in ("MAYBE_AWAY", "AWAY")
        dets, _ = proc.process(img, cfg, stable)
        snap = tr.update(dets, speaking=(clk.t < 3 or 45 <= clk.t < 47))
        lat.append((time.perf_counter() - t0) * 1000)
        cpu += time.process_time() - c0  # CPU of decode + vision + tracker only (not frame synthesis)
        states.append(snap["state"])
        s = tuple(sorted((x["id"], x["participant"], x["visible"]) for x in snap["tracks"]))
        if s != sig:
            sig, changed_at = s, clk.t
        rec = recommend_fps(snap, cfg, clk.t - changed_at < 3.0)
        clk.t += 1.0 / (rec if rec > 0 else fixed_fps)
    a = np.array(lat)
    return {
        "label": label,
        "frames": len(lat),
        "avg": a.mean(),
        "p95": np.percentile(a, 95),
        "sum_ms": a.sum(),
        "cpu_s": cpu,
        "wall_s": time.perf_counter() - wall0,
        "detect": proc.counts["detect_calls"],
        "embed": proc.counts["embed_computed"],
        "reused": proc.counts["embed_reused"],
        "skipped": proc.counts["skipped"],
        "away_seen": "AWAY" in states,
    }


OFF = dict(motion_gate=False, embed_throttle=False, adaptive_fps=False, roi_crop=False, max_width=1920)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0, help="fixed fps when adaptive fps is off")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    cv2.setNumThreads(cv2.getNumThreads())
    pipe = VisionPipeline()
    variants = [
        ("全部關閉", Config(**OFF)),
        ("只開 ROI 裁切+縮小", Config(**{**OFF, "roi_crop": True, "max_width": 320})),
        ("只開移動偵測閘門", Config(**{**OFF, "motion_gate": True})),
        ("只開 embedding 節流", Config(**{**OFF, "embed_throttle": True})),
        ("全部開啟（固定 fps）", Config(adaptive_fps=False)),
        ("全部開啟 + 自適應 fps", Config()),
    ]
    run(pipe, Config(**OFF), a.fps, "warmup")
    rows = []
    for label, cfg in variants:
        rs = [run(pipe, cfg, a.fps, label) for _ in range(a.repeat)]
        best = min(rs, key=lambda r: r["cpu_s"])
        rows.append(best)
    print(f"Pipeline: {pipe.name} · OpenCV {cv2.__version__} · {a.repeat} 次取 CPU 最低 · 模擬 {DURATION:.0f} s")
    print(
        "| 設定 | 處理張數 | 每張 avg (ms) | 每張 p95 (ms) | 總 CPU (s) | detect 次數 | embed 計算/重用 | 跳過 | AWAY |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['label']} | {r['frames']} | {r['avg']:.1f} | {r['p95']:.1f} | {r['cpu_s']:.2f} | {r['detect']} | "
            f"{r['embed']} / {r['reused']} | {r['skipped']} | {'是' if r['away_seen'] else '否'} |"
        )


if __name__ == "__main__":
    main()
