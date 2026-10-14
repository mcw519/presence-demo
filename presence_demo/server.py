"""FastAPI server: serves the demo page and runs one PresenceTracker per WebSocket."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from .body import calibration_factor
from .optimizer import FrameProcessor, recommend_fps
from .stats import ConnStats, build_stats
from .tracker import RANGES, Config, PresenceTracker
from .vision import VisionPipeline, decode_jpeg, load_pose_detector

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Presence demo")
_pipeline: VisionPipeline | None = None
TOKEN = os.environ.get("PRESENCE_TOKEN", "")  # empty = no auth (local use)


@app.middleware("http")
async def require_token(request: Request, call_next):
    if TOKEN and not secrets.compare_digest(request.query_params.get("token", ""), TOKEN):
        return PlainTextResponse("forbidden", status_code=403)
    return await call_next(request)


_pose_cache: dict = {}


def pose_detector():
    """Load the pose detector once per process (downloads MoveNet MultiPose Lightning on first use)."""
    if "det" not in _pose_cache:
        _pose_cache["det"] = load_pose_detector()
        logging.info("pose detector: %s", _pose_cache["det"].name if _pose_cache["det"] else "unavailable")
    return _pose_cache["det"]


def calibrate(tr: PresenceTracker, meters) -> list[str]:
    """Set distance_calib so the closest visible person's raw distance maps to ``meters``."""
    try:
        known = float(meters)
    except (TypeError, ValueError):
        return ["calibration: invalid distance"]
    cands = [t for t in tr.tracks.values() if t.visible and t.dist_raw]
    if not cands or known <= 0:
        return ["calibration: no visible person with a distance estimate"]
    t = min(cands, key=lambda x: x.dist_raw)
    warns = tr.cfg.update({"distance_calib": calibration_factor(t.dist_raw, known)})
    for x in tr.tracks.values():
        x.distance = None  # restart EMA with the new factor
    return warns + [f"calibrated: factor {tr.cfg.distance_calib:.3f} (track #{t.id})"]


def pipeline() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
        logging.info("vision pipeline: %s", _pipeline.name)
    return _pipeline


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def default_config():
    return {
        "defaults": asdict(Config()),
        "ranges": {k: [lo, hi, t.__name__ if t else "choice"] for k, (lo, hi, t) in RANGES.items()},
    }


@app.websocket("/ws")
async def ws(sock: WebSocket):
    if TOKEN and not secrets.compare_digest(sock.query_params.get("token", ""), TOKEN):
        await sock.close(code=1008)
        return
    await sock.accept()
    pipe = await asyncio.to_thread(pipeline)
    st = {"tr": PresenceTracker(Config()), "running": False}
    stats = ConnStats()
    proc = FrameProcessor(pipe, pose_loader=pose_detector)
    stats.counts = proc.counts
    fpsstate = {"sig": (), "changed_at": 0.0, "rec": 0.0}
    lock = asyncio.Lock()
    slot: dict = {"frame": None}
    wake = asyncio.Event()

    async def send(snap, extra=None):
        snap.update({"type": "state", "running": st["running"], "pipeline": pipe.name, "config": asdict(st["tr"].cfg)})
        if extra:
            snap.update(extra)
        await sock.send_json(snap)

    async def ticker():  # keep timers moving; stats ~1/s while running
        n = 0
        while True:
            await asyncio.sleep(0.5)
            n += 1
            async with lock:
                tr = st["tr"]
                await send(tr.tick() if st["running"] else tr.snapshot())
                if st["running"] and n % 2 == 0:
                    await sock.send_json(
                        build_stats(
                            stats,
                            {
                                **pipe.info(),
                                "pose_model": (
                                    _pose_cache.get("det").name
                                    if _pose_cache.get("det")
                                    else ("unavailable" if "det" in _pose_cache else "not loaded")
                                ),
                            },
                        )
                    )

    async def worker():  # processes only the latest frame; older queued frames are dropped
        while True:
            await wake.wait()
            wake.clear()
            msg = slot["frame"]
            slot["frame"] = None
            if msg is None or not st["running"]:
                continue
            t0 = time.perf_counter()
            data = base64.b64decode(msg["data"].split(",")[-1])
            stable = st["tr"].state.value not in ("MAYBE_AWAY", "AWAY")
            img, dec_ms = decode_jpeg(data)  # in memory only
            del data
            if img is None:
                continue
            dets, tm = await asyncio.to_thread(proc.process, img, st["tr"].cfg, stable)
            tm.update(decode=dec_ms, w=img.shape[1], h=img.shape[0])
            del img
            async with lock:
                if not st["running"]:
                    continue
                t1 = time.perf_counter()
                tr = st["tr"]
                snap = tr.update(dets, speaking=msg.get("speech"))
                tm["tracker"] = (time.perf_counter() - t1) * 1000
                tm["total"] = (time.perf_counter() - t0) * 1000
                sig = tuple(sorted((t["id"], t["participant"], t["visible"]) for t in snap["tracks"]))
                now = time.monotonic()
                if sig != fpsstate["sig"]:
                    fpsstate["sig"], fpsstate["changed_at"] = sig, now
                fpsstate["rec"] = recommend_fps(snap, tr.cfg, now - fpsstate["changed_at"] < 3.0)
                stats.on_processed(tm)
                stats.recommended_fps = fpsstate["rec"]
                await send(
                    snap,
                    {
                        "detections": [{"bbox": d.bbox, "conf": d.conf} for d in dets],
                        "skipped": tm.get("skipped", False),
                    },
                )

    tasks = [asyncio.create_task(ticker()), asyncio.create_task(worker())]
    try:
        while True:
            msg = await sock.receive_json()
            kind = msg.get("type")
            if kind == "frame":
                # PRIVACY: the frame (base64 JPEG) is kept in memory only, never logged or written to disk.
                if not st["running"]:
                    continue
                stats.on_received(len(msg.get("data", "")))
                if slot["frame"] is not None:
                    stats.dropped += 1
                slot["frame"] = msg
                wake.set()
                continue
            async with lock:
                tr = st["tr"]
                if kind == "ping":
                    if msg.get("rtt") is not None:
                        stats.rtt.append(float(msg["rtt"]))
                    await sock.send_json({"type": "pong", "t": msg.get("t")})
                elif kind == "vad":
                    if st["running"]:
                        await send(tr.set_speech(bool(msg.get("speech"))))
                elif kind == "config":
                    warns = tr.cfg.update(msg.get("config", {}))
                    await send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns})
                elif kind == "calibrate":
                    warns = calibrate(tr, msg.get("meters"))
                    await send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns})
                elif kind == "start":
                    st["running"] = True
                    await send(tr.snapshot())
                elif kind in ("stop", "reset"):
                    proc.reset()
                    stats.counts = proc.counts
                    st["running"] = kind == "reset" and st["running"]
                    st["tr"] = PresenceTracker(tr.cfg)  # clears tracks + embeddings
                    slot["frame"] = None
                    await send(st["tr"].snapshot(), {"detections": []})
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    # access_log=False: request lines would contain the ?token= query string. Frames travel over the
    # WebSocket and are never logged; only startup/model messages are written to the log.
    uvicorn.run(app, host=a.host, port=a.port, access_log=False, log_level="info", ws_max_size=8 * 1024 * 1024)


if __name__ == "__main__":
    main()
