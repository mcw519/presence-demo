"""FastAPI server: serves the demo page and runs one PresenceTracker per WebSocket."""

from __future__ import annotations

import asyncio
import atexit
import base64
import collections
import concurrent.futures
import contextlib
import faulthandler
import json
import logging
import os
import secrets
import struct
import threading
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import asr as asrmod
from . import diag
from . import reid as reidmod
from .audio import AudioStream, puresound_available
from .body import calibration_factor
from .optimizer import FrameProcessor, recommend_fps
from .stats import ConnStats, build_stats
from .tracker import RANGES, Config, PresenceTracker
from .vision import VisionPipeline, decode_jpeg, load_pose_detector

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Presence demo")
_pipeline: VisionPipeline | None = None
TOKEN = os.environ.get("PRESENCE_TOKEN", "")  # empty = no auth (local use)


@app.middleware("http")
async def require_token(request: Request, call_next):
    # vendored design-system assets (public CSS + fonts, no data) load without the token
    public = request.url.path.startswith("/static/vendor/") or request.url.path == "/static/app.css"
    if TOKEN and not public and not secrets.compare_digest(request.query_params.get("token", ""), TOKEN):
        return PlainTextResponse("forbidden", status_code=403)
    return await call_next(request)


_pose_cache: dict = {}
_PIPE_LOCK = threading.Lock()


def pose_detector():
    """Load the pose detector once per process (downloads MoveNet MultiPose Lightning on first use)."""
    if "det" not in _pose_cache:
        t = time.perf_counter()
        try:
            _pose_cache["det"] = load_pose_detector()
        except Exception as e:  # noqa: BLE001
            _pose_cache["det"] = None
            diag.event("pose_load_failed", "error", error=f"{type(e).__name__}: {e}")
        det = _pose_cache["det"]
        diag.event(
            "module_init",
            module="pose",
            name=det.name if det else "unavailable",
            ms=round((time.perf_counter() - t) * 1000, 1),
        )
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


# One shared VisionPipeline (OpenCV DNN models) serves every WebSocket connection. cv2.FaceDetectorYN keeps its
# input size as mutable state and is not thread-safe: two connections detecting at once corrupted results (and
# could crash the process). All vision inference is therefore serialized through this lock.
VISION_LOCK = threading.Lock()
# Audio isolation gets its own single worker thread so it never waits behind vision jobs in the default pool.
AUDIO_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio")


class VisionBusy(RuntimeError):
    """The shared vision lock could not be acquired in time (another inference is hung or very slow)."""


def _vision_job(proc: FrameProcessor, data: bytes, cfg, stable: bool, conn: str | None = None):
    img, dec_ms = decode_jpeg(data)  # in memory only
    if img is None:
        return None
    t0 = time.perf_counter()
    if not VISION_LOCK.acquire(timeout=VISION_LOCK_TIMEOUT):
        diag.event("vision_lock_timeout", "error", conn, waited_s=VISION_LOCK_TIMEOUT)
        diag.log.error("stack dump (vision_lock_timeout)\n%s", diag.dump_stacks())
        raise VisionBusy(f"vision lock not acquired within {VISION_LOCK_TIMEOUT} s")
    try:
        wait_ms = (time.perf_counter() - t0) * 1000
        dets, tm = proc.process(img, cfg, stable)
    finally:
        VISION_LOCK.release()
    tm["lock_wait"] = wait_ms
    tm.update(decode=dec_ms, w=img.shape[1], h=img.shape[0])
    return dets, tm


def pipeline() -> VisionPipeline:
    global _pipeline
    with _PIPE_LOCK:  # two connections opening at once must not build two pipelines
        if _pipeline is None:
            t = time.perf_counter()
            _pipeline = VisionPipeline()
            diag.event(
                "module_init", module="vision", name=_pipeline.name, ms=round((time.perf_counter() - t) * 1000, 1)
            )
    return _pipeline


app.mount("/static/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")


@app.get("/static/app.css")
def app_css():
    return FileResponse(STATIC / "app.css", media_type="text/css", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__VERSION__", _version())
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("presence-demo")
    except Exception:  # noqa: BLE001
        return "dev"


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


@app.get("/api/reid_status")
def reid_status():
    return reidmod.key_status()  # never the key itself


@app.post("/api/reid_key")
async def reid_key(request: Request):
    """Set/clear the re-ID backend key in server memory (token-protected). Not logged/stored."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    return reidmod.set_key(str(body.get("key") or "")[:512])


@app.post("/api/reid_reset")
def reid_reset():
    """Wipe every connection's in-memory reference faces."""
    n = reidmod.wipe_all()
    diag.event("reid_store", action="wipe", reason="api", count=n)
    return {"wiped": n}


@app.get("/debug/events")
def debug_events(n: int = 200, conn: str | None = None):
    """Recent structured events (token-protected by the middleware; no media, keys or transcript text)."""
    return {"events": diag.recent(max(1, min(n, 2000)), conn)}


VISION_LOCK_TIMEOUT = float(os.environ.get("PRESENCE_VISION_LOCK_TIMEOUT", "10"))
WARN_FRAME_MS = float(os.environ.get("PRESENCE_WARN_FRAME_MS", "500"))
HEARTBEAT_S = float(os.environ.get("PRESENCE_HEARTBEAT_S", "10"))
STALL_S = float(os.environ.get("PRESENCE_STALL_S", "5"))
SEND_AUDIO_MAX = 50  # queued outbound audio messages (~1 s); more -> new audio is dropped (slow client)
SEND_AUDIO_STALE = 0.3  # an audio message waiting longer than this is skipped if newer audio is queued
SEND_JSON_MAX = 60  # queued droppable JSON messages; more -> drop (state/meta are superseded by the next one)


class Outbox:
    """Non-blocking outbound queue for one WebSocket: processing never awaits the network (a slow client used to
    block the vision/audio workers on ``send``). Audio and droppable JSON are dropped when the backlog grows."""

    def __init__(self, sock: WebSocket):
        self.sock, self.q = sock, collections.deque()
        self.ev = asyncio.Event()
        self.n_audio = self.n_json = 0
        self.sent = self.dropped_audio = self.dropped_json = 0
        self.send_ms_max = 0.0
        self.last_send = time.monotonic()

    def post(self, item, *, audio: bool = False, droppable: bool = False) -> None:
        if audio and self.n_audio >= SEND_AUDIO_MAX:
            self.dropped_audio += 1
            return
        if droppable and self.n_json >= SEND_JSON_MAX:
            self.dropped_json += 1
            return
        self.n_audio += audio
        self.n_json += droppable
        self.q.append((item, audio, droppable, time.monotonic()))
        self.ev.set()

    def backlog(self) -> int:
        return len(self.q)

    async def run(self) -> None:
        while True:
            if not self.q:
                self.ev.clear()
                await self.ev.wait()
                continue
            item, audio, droppable, t_post = self.q.popleft()
            self.n_audio -= audio
            self.n_json -= droppable
            if audio and self.n_audio and time.monotonic() - t_post > SEND_AUDIO_STALE:
                self.dropped_audio += 1  # catch up: playback latency matters more than completeness
                continue
            t0 = time.perf_counter()
            if isinstance(item, bytes):
                await self.sock.send_bytes(item)
            else:
                await self.sock.send_text(json.dumps(item, separators=(",", ":")))
            ms = (time.perf_counter() - t0) * 1000
            self.send_ms_max = max(self.send_ms_max, ms)
            self.sent += 1
            self.last_send = time.monotonic()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    if TOKEN and not secrets.compare_digest(sock.query_params.get("token", ""), TOKEN):
        await sock.close(code=1008)
        return
    await sock.accept()
    cid = secrets.token_hex(3)
    peer = sock.client.host if sock.client else "?"
    diag.event("conn_open", conn=cid, peer=peer)
    t_open = time.monotonic()
    t = time.perf_counter()
    pipe = await asyncio.to_thread(pipeline)
    st = {"tr": PresenceTracker(Config()), "running": False}
    stats = ConnStats()
    proc = FrameProcessor(pipe, pose_loader=pose_detector)
    stats.counts = proc.counts
    fpsstate = {"sig": (), "changed_at": 0.0, "rec": 0.0}
    lock = asyncio.Lock()
    slot: dict = {"frame": None}
    wake = asyncio.Event()
    out = Outbox(sock)
    # audio (audio_pipeline=puresound): binary PCM16 chunks in, clean PCM16 + "audio" meta out; memory only
    aud: dict = {
        "stream": None,
        "q": asyncio.Queue(maxsize=25),  # <= 500 ms backlog; older chunks are dropped (latency over completeness)
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
        "rec": False,  # browser-side recording requested: also stream aligned vi + raw (server stores nothing)
        "recv": 0,
        "processed": 0,
        "rtf": None,
    }
    mon = {
        "frame_rx_t": 0.0,  # last frame received (monotonic)
        "frame_done_t": time.monotonic(),
        "frames_done": 0,
        "audio_rx_t": 0.0,
        "audio_done_t": time.monotonic(),
        "loop_lag_ms": 0.0,
        "loop_lag_max": 0.0,
        "state": None,
        "session": None,
        "warn_t": {},
    }
    rd = reidmod.CrossSessionReid(
        st["tr"].cfg,
        emit=lambda d: out.post(d),
        event=lambda name, level="info", **kv: diag.event(name, level, cid, **kv),
    )
    diag.event("conn_ready", conn=cid, vision_ms=round((time.perf_counter() - t) * 1000, 1), pipeline=pipe.name)

    def warn(name: str, every: float = 30.0, level: str = "warning", **kv) -> None:  # rate-limited warnings
        now = time.monotonic()
        if now - mon["warn_t"].get(name, -1e9) >= every:
            mon["warn_t"][name] = now
            diag.event(name, level, cid, **kv)

    def notify_ui(rec: dict) -> None:  # this connection's events (and global ones) -> UI 事件紀錄
        if rec.get("conn") in (cid, None):
            out.post({"type": "event", **rec}, droppable=rec.get("level") == "info")

    unsub = diag.subscribe(notify_ui)

    def on_crash(task: str, err: str) -> None:
        out.post({"type": "error", "where": task, "error": err})

    def clean_active() -> bool:
        return aud["stream"] is not None and time.monotonic() - aud["last_rx"] < 1.0

    def track_state(snap: dict) -> None:
        s, sess = snap.get("state"), snap.get("session")
        if s != mon["state"] or sess != mon["session"]:
            diag.event(
                "state",
                conn=cid,
                frm=mon["state"],
                to=s,
                session=sess,
                participants=sum(1 for x in snap.get("tracks", []) if x.get("participant")),
                p_group=snap.get("p_group"),
            )
            mon["state"], mon["session"] = s, sess
        for e in snap.get("events") or []:
            name = e.get("event") if isinstance(e, dict) else str(e)
            if name in ("NEW_GROUP", "ENDED", "SESSION_START", "JOIN", "LEAVE"):
                diag.event("session_event", conn=cid, kind=name, session=sess)

    def send(snap, extra=None, droppable=True):
        track_state(snap)
        snap.update({"type": "state", "running": st["running"], "pipeline": pipe.name})
        if extra:
            snap.update(extra)
            if extra.get("config_ack"):
                snap["config"] = asdict(st["tr"].cfg)
        out.post(snap, droppable=droppable)

    async def ticker():  # keep timers moving; stats ~1/s while running; loop-lag probe
        n = 0
        while True:
            t0 = time.monotonic()
            await asyncio.sleep(0.5)
            lag = (time.monotonic() - t0 - 0.5) * 1000
            mon["loop_lag_ms"] = lag
            mon["loop_lag_max"] = max(mon["loop_lag_max"], lag)
            if lag > 250:
                warn("loop_lag", 10, lag_ms=round(lag))
            n += 1
            async with lock:
                tr = st["tr"]
                snap_t = tr.tick() if st["running"] else tr.snapshot()
                send(snap_t)
                if st["running"]:
                    rd.observe(snap_t)  # timeouts / TTL only; crops come from the frame worker
                if n % 2 == 0 and (rd.on() or rd.calls):
                    out.post({"type": "reid", **rd.status()}, droppable=True)
                if st["running"] and n % 2 == 0:
                    out.post(
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
                        ),
                        droppable=True,
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
            r = await asyncio.to_thread(_vision_job, proc, data, st["tr"].cfg, stable, cid)
            if r is None:
                del data
                continue
            dets, tm = r
            reqs: list = []
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
                mon["frame_done_t"] = now
                mon["frames_done"] += 1
                if tm["total"] > WARN_FRAME_MS:
                    warn(
                        "slow_frame",
                        10,
                        total_ms=round(tm["total"]),
                        detect_ms=round(tm.get("detect", 0)),
                        pose_ms=round(tm.get("pose", 0)),
                        embed_ms=round(tm.get("embed", 0)),
                    )
                send(
                    snap,
                    {
                        "detections": [{"bbox": d.bbox, "conf": d.conf} for d in dets],
                        "skipped": tm.get("skipped", False),
                    },
                )
                reqs = rd.observe(snap, now)
            # cross-session re-ID: crop faces in memory only when the driver asks (never stored to disk)
            for kind, bbox, q in reqs:
                crop = await asyncio.to_thread(reidmod.face_crop, data, bbox)
                rd.on_crop(kind, crop, quality=q)
            del data

    def asr_event(ev: dict, sess_t0: float, session: int):
        # session-relative turn times: stream start (monotonic) - session start + word offset
        t0 = ev.pop("t0", None)
        for k in ("start_ms", "end_ms"):
            if t0 is not None and ev.get(k) is not None:
                ev[k.replace("_ms", "_s")] = round(t0 - sess_t0 + ev[k] / 1000, 2)
        if ev.get("kind") in ("status", "error"):  # never the transcript text
            diag.event(
                "asr",
                "warning" if ev.get("kind") == "error" else "info",
                cid,
                status=ev.get("status"),
                error=ev.get("error"),
                session=session,
            )
        out.post({"type": "asr", "session": session, **ev})

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
        loop = asyncio.get_running_loop()
        while True:
            batch = [await aud["q"].get()]
            while not aud["q"].empty() and len(batch) < 25:  # catch up after a stall: one thread hop for the backlog
                batch.append(aud["q"].get_nowait())
            aud["backlog_max"] = max(aud.get("backlog_max", 0), len(batch))
            if len(batch) > 10:
                warn("audio_backlog", 10, backlog_ms=len(batch) * 20)
            tr = st["tr"]
            if tr.cfg.audio_pipeline != "puresound":
                aud["stream"] = None
                continue
            if aud["stream"] is None:
                ok, why = puresound_available()
                if not ok:
                    if aud["err"] != why:
                        aud["err"] = why
                        diag.event("audio_unavailable", "warning", cid, reason=why)
                        out.post({"type": "audio", "available": False, "error": why})
                    continue
                t_init = time.perf_counter()
                aud["stream"] = await asyncio.to_thread(AudioStream, tr.cfg)
                diag.event("audio_init", conn=cid, ms=round((time.perf_counter() - t_init) * 1000, 1))
                aud["err"] = None
            a = aud["stream"]
            a.cfg = tr.cfg
            now = time.monotonic()
            snap = tr.snapshot() if st["running"] else None  # None -> acoustic_only (no vision)
            a.calibrated = aud["calibrated"]
            vis_age = now - aud["vis_t"]

            rec = aud["rec"]

            def job(a=a, snap=snap, now=now, vis_age=vis_age, batch=batch, rec=rec):
                if snap is not None:
                    a.on_vision(snap, now)
                res = []
                for _, pcm in batch:
                    o, sp = a.process(pcm, snap, now, vis_age)
                    res.append((o, sp, a.recording_tracks() if rec else None))
                return res

            t0 = time.perf_counter()
            res = await loop.run_in_executor(AUDIO_POOL, job)
            proc_ms = (time.perf_counter() - t0) * 1000 / len(batch)
            aud["last_rx"] = time.monotonic()
            aud["processed"] += len(batch)
            mon["audio_done_t"] = time.monotonic()
            for (t_client, pcm), (o, speech, tracks) in zip(batch, res, strict=True):
                if o:
                    out.post(b"\x02" + struct.pack("<d", t_client) + o, audio=True)
                if o and tracks is not None:  # 0x03: t, n, out, vi, raw (same n samples each, aligned)
                    vi, rw = tracks
                    out.post(b"\x03" + struct.pack("<dI", t_client, len(o) // 2) + o + vi + rw)
                await asr_step(tr, pcm, o, speech, now)
            async with lock:
                tr = st["tr"]
                tr.speech_source = "clean"
                if st["running"] and (speech != tr.speaking):
                    send(tr.set_speech(speech), droppable=False)
            if now - aud["meta_t"] >= 0.1:
                aud["meta_t"] = now
                meta = a.meta()
                aud["rtf"] = meta.get("rtf")
                aud["counts"] = meta.get("counts")
                out.post(
                    {
                        "type": "audio",
                        "available": True,
                        **meta,
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
                        "backlog_max": aud.get("backlog_max", 0),
                        "send_dropped": out.dropped_audio,
                    },
                    droppable=True,
                )

    tasks: dict[str, asyncio.Task] = {}
    factories = {"ticker": ticker, "vision": worker, "audio": audio_worker, "sender": out.run}

    def spawn(name: str) -> None:
        tasks[name] = asyncio.create_task(diag.supervise(name, factories[name], cid, on_crash))

    async def watchdog():  # heartbeat + stall detection (+ recovery) every HEARTBEAT_S / 1 s
        last_hb = time.monotonic()
        last_frames = last_audio = 0
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            run = st["running"]
            ps = aud["stream"] is not None and st["tr"].cfg.audio_pipeline == "puresound"
            # vision: frames arrive but none finished for STALL_S
            if run and mon["frame_rx_t"] > mon["frame_done_t"] + STALL_S:
                warn(
                    "vision_stalled",
                    30,
                    "error",
                    since_s=round(now - mon["frame_done_t"], 1),
                    lock_held=VISION_LOCK.locked(),
                )
                diag.log.error("stack dump (vision_stalled conn=%s)\n%s", cid, diag.dump_stacks())
                out.post({"type": "error", "where": "vision", "error": "影像處理停滯，正在重新啟動"})
                tasks["vision"].cancel()
                proc.reset()
                spawn("vision")
                mon["frame_done_t"] = now
                wake.set()
            if run and ps and mon["audio_rx_t"] > mon["audio_done_t"] + STALL_S:
                warn("audio_stalled", 30, "error", since_s=round(now - mon["audio_done_t"], 1), queue=aud["q"].qsize())
                diag.log.error("stack dump (audio_stalled conn=%s)\n%s", cid, diag.dump_stacks())
                out.post({"type": "error", "where": "audio", "error": "音訊處理停滯，正在重新啟動"})
                tasks["audio"].cancel()
                aud["stream"] = None
                spawn("audio")
                mon["audio_done_t"] = now
            if run and now - mon["frame_rx_t"] > 3 and now - t_open > 5:
                warn("no_frames", 30, since_s=round(now - mon["frame_rx_t"], 1) if mon["frame_rx_t"] else None)
            if run and st["tr"].cfg.audio_pipeline == "puresound" and now - mon["audio_rx_t"] > 1 and now - t_open > 5:
                warn("no_audio", 30, since_s=round(now - mon["audio_rx_t"], 1) if mon["audio_rx_t"] else None)
            if out.backlog() > 50 or now - out.last_send > 5 and out.backlog():
                warn(
                    "send_backlog",
                    10,
                    backlog=out.backlog(),
                    dropped_audio=out.dropped_audio,
                    dropped_json=out.dropped_json,
                    last_send_s=round(now - out.last_send, 1),
                )
            if now - last_hb >= HEARTBEAT_S:
                fr = {"received": stats.received, "processed": stats.processed, "dropped": stats.dropped}
                c = proc.counts
                snap = st["tr"].snapshot()
                diag.event(
                    "heartbeat",
                    conn=cid,
                    running=run,
                    frames_rx=fr.get("received"),
                    frames_done=fr.get("processed"),
                    frames_dropped=fr.get("dropped"),
                    fps=round((mon["frames_done"] - last_frames) / (now - last_hb), 2),
                    detect_calls=c.get("detect_calls"),
                    detect_hits=c.get("detect_hits"),
                    tracks=len(snap.get("tracks", [])),
                    state=snap.get("state"),
                    p_group=snap.get("p_group"),
                    audio_rx=aud["recv"],
                    audio_done=aud["processed"],
                    audio_dropped=aud["dropped"],
                    audio_rate=round((aud["processed"] - last_audio) / (now - last_hb), 1),
                    audio_queue=aud["q"].qsize(),
                    rtf=aud["rtf"],
                    xcheck=aud.get("counts"),
                    send_backlog=out.backlog(),
                    send_dropped=out.dropped_audio + out.dropped_json,
                    send_ms_max=round(out.send_ms_max, 1),
                    loop_lag_ms=round(mon["loop_lag_max"], 1),
                    vision_lock_held=VISION_LOCK.locked(),
                    reid_calls=rd.calls,
                    reid_tokens=rd.tokens,
                    reid_cost_est=round(rd.calls * st["tr"].cfg.reid_cost_per_call, 4),
                    reid_refs=len(rd.store.refs),
                )
                mon["loop_lag_max"] = 0.0
                out.send_ms_max = 0.0
                last_hb, last_frames, last_audio = now, mon["frames_done"], aud["processed"]

    for name in factories:
        spawn(name)
    tasks["watchdog"] = asyncio.create_task(diag.supervise("watchdog", watchdog, cid))
    reason = "client"
    try:
        while True:
            raw = await sock.receive()
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(raw.get("code", 1000))
            if raw.get("bytes") is not None:
                # PRIVACY: PCM audio lives in memory only (queue -> isolation -> socket), never logged/written.
                b = raw["bytes"]
                if len(b) > 9 and b[0] == 1 and st["tr"].cfg.audio_pipeline == "puresound":
                    aud["recv"] += 1
                    mon["audio_rx_t"] = time.monotonic()
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
                mon["frame_rx_t"] = time.monotonic()
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
                    out.post({"type": "pong", "t": msg.get("t")})
                elif kind == "vad":
                    if st["running"] and not clean_active():  # puresound: the clean VAD drives speech
                        send(tr.set_speech(bool(msg.get("speech"))), droppable=False)
                elif kind == "config":
                    new = msg.get("config", {})
                    before = asdict(tr.cfg)
                    warns = tr.cfg.update(new)
                    after = asdict(tr.cfg)
                    changed = {k: after[k] for k in after if after[k] != before.get(k)}
                    if changed:
                        diag.event("config", conn=cid, **changed)
                    if warns:
                        diag.event("config_warnings", "warning", cid, warnings="; ".join(warns)[:300])
                    send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns}, False)
                elif kind == "calibrate":
                    warns = calibrate(tr, msg.get("meters"))
                    aud["calibrated"] = aud["calibrated"] or any(w.startswith("calibrated") for w in warns)
                    diag.event("calibrate", conn=cid, result="; ".join(warns)[:200])
                    send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns}, False)
                elif kind == "reid_wipe":
                    rd.wipe("ui")
                    out.post({"type": "reid", **rd.status()})
                elif kind == "record":  # client-side recording on/off (PRIVACY: the server never writes audio)
                    aud["rec"] = bool(msg.get("on"))
                    diag.event("record_stream", conn=cid, on=aud["rec"])
                elif kind == "start":
                    st["running"] = True
                    diag.event("start", conn=cid)
                    send(tr.snapshot(), droppable=False)
                elif kind in ("stop", "reset"):
                    proc.reset()
                    stats.counts = proc.counts
                    st["running"] = kind == "reset" and st["running"]
                    if kind == "stop":
                        aud["rec"] = False
                    diag.event(kind, conn=cid)
                    st["tr"] = PresenceTracker(tr.cfg)  # clears tracks + embeddings
                    rd.session = 0  # the new tracker counts sessions from 1 again
                    if kind == "reset":
                        rd.wipe("reset")
                    slot["frame"] = None
                    send(st["tr"].snapshot(), {"detections": []}, False)
    except WebSocketDisconnect as e:
        reason = f"disconnect code={e.code}"
    except Exception as e:  # noqa: BLE001
        reason = f"error {type(e).__name__}: {e}"
        diag.log.exception("receive loop failed conn=%s", cid)
    finally:
        unsub()
        rd.close()  # wipes this connection's reference faces
        for t in tasks.values():
            t.cancel()
        with contextlib.suppress(Exception):
            await asr_close()
        diag.event(
            "conn_close",
            conn=cid,
            reason=reason,
            duration_s=round(time.monotonic() - t_open, 1),
            frames=mon["frames_done"],
            audio_chunks=aud["processed"],
            send_dropped=out.dropped_audio + out.dropped_json,
        )


atexit.register(reidmod.wipe_all)  # shutdown: wipe in-memory reference faces


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    diag.setup_logging()
    faulthandler.enable()  # hard crashes (segfault) still print all thread stacks to stderr
    # access_log=False: request lines would contain the ?token= query string. Frames travel over the
    # WebSocket and are never logged. log_config=None: uvicorn logs go through our handler (rotation, redaction).
    diag.event("server_start", host=a.host, port=a.port, log_level=logging.getLevelName(logging.getLogger().level))
    uvicorn.run(
        app,
        host=a.host,
        port=a.port,
        access_log=False,
        log_config=None,
        ws_max_size=8 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()

_RedactToken = diag.Redact
