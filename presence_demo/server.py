"""FastAPI server: serves the demo page and runs one PresenceTracker per WebSocket."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import secrets
import struct
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from . import asr as asrmod
from .audio import AudioStream, puresound_available
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
        "ranges": {
            k: [lo, hi, t if isinstance(t, str) else (t.__name__ if t else "choice")]
            for k, (lo, hi, t) in RANGES.items()
        },
        "puresound": dict(zip(("available", "reason"), puresound_available(), strict=True)),
    }


@app.get("/api/asr_status")
def asr_status():
    return asrmod.key_status()  # never the key itself


@app.post("/api/asr_key")
async def asr_key(request: Request):
    """Set/clear the AssemblyAI key in server memory (token-protected by the middleware). Not logged/stored."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    return asrmod.set_key(str(body.get("key") or "")[:256])


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
    # audio (audio_pipeline=puresound): binary PCM16 chunks in, clean PCM16 + "audio" meta out; memory only
    aud: dict = {
        "stream": None,
        "q": asyncio.Queue(maxsize=50),
        "last_rx": 0.0,
        "meta_t": 0.0,
        "err": None,
        "dropped": 0,
        "lat": None,
        "vis_t": -1e9,  # monotonic time of the last processed frame (vision freshness for the cross-check)
        "calibrated": False,
        "vis_tc": None,  # browser performance.now() of the last processed frame (shared audio/video timebase)
        "asr": None,  # AssemblyAIStream while a session is active (asr_provider=assemblyai)
        "asr_sess": None,
        "sess": None,
        "sess_t0": 0.0,
    }

    def clean_active() -> bool:
        return aud["stream"] is not None and time.monotonic() - aud["last_rx"] < 1.0

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
                tr.speech_source = "clean" if clean_active() else "browser"
                snap = tr.update(dets, speaking=None if clean_active() else msg.get("speech"))
                tm["tracker"] = (time.perf_counter() - t1) * 1000
                aud["vis_t"] = time.monotonic()
                aud["vis_tc"] = msg.get("t")
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

    def asr_event(ev: dict, sess_t0: float, session: int):
        # session-relative turn times: stream start (monotonic) - session start + word offset
        t0 = ev.pop("t0", None)
        for k in ("start_ms", "end_ms"):
            if t0 is not None and ev.get(k) is not None:
                ev[k.replace("_ms", "_s")] = round(t0 - sess_t0 + ev[k] / 1000, 2)
        asyncio.create_task(sock.send_json({"type": "asr", "session": session, **ev}))

    async def asr_close():
        if aud["asr"] is not None:
            s_, aud["asr"] = aud["asr"], None
            await s_.close()

    async def asr_step(tr, raw: bytes, clean: bytes, speech: bool, now: float):
        """Cloud ASR gate: only during an active session, only if selected and a key is set."""
        c = tr.cfg
        if tr.session != aud["sess"]:
            aud["sess"], aud["sess_t0"] = tr.session, now
        want = (
            c.asr_provider == "assemblyai"
            and asrmod.key_status()["set"]
            and st["running"]
            and tr.state.value != "ENDED"
        )
        if aud["asr"] is not None and (not want or aud["asr_sess"] != tr.session):
            await asr_close()
        if not want:
            return
        if aud["asr"] is None:
            sess, t0 = tr.session, aud["sess_t0"]
            aud["asr"] = asrmod.AssemblyAIStream(c, lambda ev: asr_event(ev, t0, sess), url=asrmod.URL_OVERRIDE)
            aud["asr_sess"] = sess
            aud["asr"].start()
        src = clean if c.asr_source == "clean" else raw
        if c.asr_feed == "speech" and not speech:
            src = bytes(len(src))  # keep real-time pacing; no content leaves the device
        aud["asr"].feed(src)

    async def audio_worker():  # one chunk at a time, in order; isolation runs in a thread
        while True:
            t_client, pcm = await aud["q"].get()
            tr = st["tr"]
            if tr.cfg.audio_pipeline != "puresound":
                aud["stream"] = None
                continue
            if aud["stream"] is None:
                ok, why = puresound_available()
                if not ok:
                    if aud["err"] != why:
                        aud["err"] = why
                        logging.warning("audio_pipeline=puresound unavailable: %s", why)
                        await sock.send_json({"type": "audio", "available": False, "error": why})
                    continue
                aud["stream"] = await asyncio.to_thread(AudioStream, tr.cfg)
                aud["err"] = None
            a = aud["stream"]
            a.cfg = tr.cfg
            now = time.monotonic()
            snap = tr.snapshot() if st["running"] else None  # None -> acoustic_only (no vision)
            if snap is not None:
                await asyncio.to_thread(a.on_vision, snap, now)
            t0 = time.perf_counter()
            a.calibrated = aud["calibrated"]
            out, speech = await asyncio.to_thread(a.process, pcm, snap, now, now - aud["vis_t"])
            proc_ms = (time.perf_counter() - t0) * 1000
            aud["last_rx"] = time.monotonic()
            if out:
                await sock.send_bytes(b"\x02" + struct.pack("<d", t_client) + out)
            await asr_step(tr, pcm, out, speech, now)
            async with lock:
                tr = st["tr"]
                tr.speech_source = "clean"
                if st["running"] and (speech != tr.speaking):
                    await send(tr.set_speech(speech))
            if now - aud["meta_t"] >= 0.1:
                aud["meta_t"] = now
                await sock.send_json(
                    {
                        "type": "audio",
                        "available": True,
                        **a.meta(),
                        "p_group": tr.snapshot().get("p_group"),
                        "state": tr.state.value,
                        "t_server": time.time(),
                        "t_client": t_client,
                        "proc_ms": round(proc_ms, 2),
                        "t_vision_client": aud["vis_tc"],
                        "asr": {
                            "provider": tr.cfg.asr_provider,
                            "key_set": asrmod.key_status()["set"],
                            "connected": bool(aud["asr"] and aud["asr"].connected),
                            "sent_s": round(aud["asr"].sent_ms / 1000, 1) if aud["asr"] else 0,
                        },
                        "dropped": aud["dropped"],
                        "queue": aud["q"].qsize(),
                    }
                )

    tasks = [asyncio.create_task(ticker()), asyncio.create_task(worker()), asyncio.create_task(audio_worker())]
    try:
        while True:
            raw = await sock.receive()
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(raw.get("code", 1000))
            if raw.get("bytes") is not None:
                # PRIVACY: PCM audio lives in memory only (queue -> isolation -> socket), never logged/written.
                b = raw["bytes"]
                if len(b) > 9 and b[0] == 1 and st["tr"].cfg.audio_pipeline == "puresound":
                    if aud["q"].full():
                        aud["q"].get_nowait()
                        aud["dropped"] += 1
                    aud["q"].put_nowait((struct.unpack("<d", b[1:9])[0], b[9:]))
                continue
            msg = json.loads(raw.get("text") or "{}")
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
                    if st["running"] and not clean_active():  # puresound: the clean VAD drives speech
                        await send(tr.set_speech(bool(msg.get("speech"))))
                elif kind == "config":
                    warns = tr.cfg.update(msg.get("config", {}))
                    await send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns})
                elif kind == "calibrate":
                    warns = calibrate(tr, msg.get("meters"))
                    aud["calibrated"] = aud["calibrated"] or any(w.startswith("calibrated") for w in warns)
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
        with contextlib.suppress(Exception):
            await asr_close()


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
