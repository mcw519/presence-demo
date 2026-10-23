"""One WebSocket connection = one session: receives frames/audio/commands, runs the vision and audio workers,
the watchdog and re-ID driver, and streams state back through the :class:`Outbox`."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import struct
import time
from dataclasses import asdict

from fastapi import WebSocket, WebSocketDisconnect

from .. import asr as asrmod
from .. import diag
from .. import reid as reidmod
from ..audio import SR, AudioStream, puresound_available
from ..optimizer import FrameProcessor, recommend_fps
from ..stats import AudioStats, ConnStats, build_stats
from ..tracker import Config, PresenceTracker
from .continuity import PAGE_TOKEN, PAGES, ObservationClock
from .outbox import Outbox
from .protocol import VIDEO_TAG, binary_frame, legacy_frame
from .runtime import AUDIO_POOL, TOKEN, VISION_LOCK, _pose_cache, _vision_job, calibrate, pipeline, pose_detector

WARN_FRAME_MS = float(os.environ.get("PRESENCE_WARN_FRAME_MS", "500"))
HEARTBEAT_S = float(os.environ.get("PRESENCE_HEARTBEAT_S", "10"))
STALL_S = float(os.environ.get("PRESENCE_STALL_S", "5"))


async def ws(sock: WebSocket):
    if TOKEN and not secrets.compare_digest(sock.query_params.get("token", ""), TOKEN):
        await sock.close(code=1008)
        return
    protocols = [p.strip() for p in sock.headers.get("sec-websocket-protocol", "").split(",")]
    capabilities = [p[5:] for p in protocols if p.startswith("page-")]
    page_token = capabilities[0] if len(capabilities) == 1 else None
    if capabilities and (len(capabilities) != 1 or not PAGE_TOKEN.fullmatch(page_token)):
        await sock.close(code=1008)
        return
    if "presence-v1" in protocols:
        await sock.accept(subprotocol="presence-v1")
    else:
        await sock.accept()
    page, page_generation = None, None
    if page_token:
        try:
            page, page_generation = PAGES.acquire(page_token)
        except ValueError:
            await sock.close(code=1008)
            return
    cid = secrets.token_hex(3)
    peer = sock.client.host if sock.client else "?"
    diag.event("conn_open", conn=cid, peer=peer)
    t_open = time.monotonic()
    t = time.perf_counter()
    try:
        pipe = await asyncio.to_thread(pipeline)
    except BaseException:
        if page:
            PAGES.release(page, page_generation)
        raise
    if page and page.generation != page_generation:
        await sock.close(code=1000)
        return
    observation_clock = ObservationClock()
    st = {"tr": PresenceTracker(Config(), clock=observation_clock), "running": False}
    if page:
        st["tr"].session = page.session
    stats = ConnStats()
    audio_stats = AudioStats()
    proc = FrameProcessor(pipe, pose_loader=pose_detector)
    stats.counts = proc.counts
    fpsstate = {"sig": (), "changed_at": 0.0, "rec": 0.0}
    lock = asyncio.Lock()
    slot: dict = {"frame": None}
    wake = asyncio.Event()
    out = Outbox(sock, cid)
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
        "tracks": {},  # id -> {"visible": bool, "since": t}  (track lifecycle events)
        "fm": {"frames": 0, "detected": 0, "confs": [], "widths": [], "out_roi": 0, "coast": 0, "t0": time.monotonic()},
        "wall0": time.time(),
        "mono0": time.monotonic(),
    }
    rd = reidmod.CrossSessionReid(
        st["tr"].cfg,
        emit=lambda d: out.post(d),
        event=lambda name, level="info", **kv: diag.event(name, level, cid, **kv),
        store=page.store if page else None,
    )
    if page:
        page.interrupt = rd.interrupt
        page.disconnect = lambda: asyncio.create_task(sock.close(code=1000))
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
        tracks = snap.get("tracks", [])
        if s != mon["state"] or sess != mon["session"]:
            tr_ev = [e for e in snap.get("log") or [] if e.get("event") == "transition" and e.get("to") == s]
            diag.event(
                "state",
                conn=cid,
                frm=mon["state"],
                to=s,
                session=sess,
                reason=tr_ev[-1].get("reason") if tr_ev else None,
                participants=sum(1 for x in tracks if x.get("participant")),
                faces=sum(1 for x in tracks if x.get("visible") and x.get("has_face") is not False),
                speech=snap.get("speaking"),
                p_group=snap.get("p_group"),
            )
            if s == "ACTIVE" and mon["state"] == "ENDED":
                diag.event(
                    "session_open",
                    conn=cid,
                    session=sess,
                    reason=tr_ev[-1].get("reason") if tr_ev else None,
                    participants=sum(1 for x in tracks if x.get("participant")),
                    faces=sum(1 for x in tracks if x.get("visible") and x.get("has_face") is not False),
                    tracks=len(tracks),
                    speech=snap.get("speaking"),
                )
            mon["state"], mon["session"] = s, sess
        # track lifecycle: new / lost (missing) / recovered / dropped
        now_t, seen = time.monotonic(), set()
        for t in tracks:
            tid, vis = t.get("id"), bool(t.get("visible")) and t.get("visible_kind") != "predicted"
            seen.add(tid)
            prev = mon["tracks"].get(tid)
            if prev is None:
                mon["tracks"][tid] = {"visible": vis, "since": now_t}
                diag.event(
                    "track_new",
                    conn=cid,
                    track=tid,
                    participant=t.get("participant"),
                    has_face=t.get("has_face"),
                    session=sess,
                )
            elif prev["visible"] != vis:
                if vis:
                    diag.event(
                        "track_recovered",
                        conn=cid,
                        track=tid,
                        missing_s=round(now_t - prev["since"], 1),
                        participant=t.get("participant"),
                    )
                else:
                    diag.event(
                        "track_lost",
                        conn=cid,
                        track=tid,
                        kind=t.get("visible_kind") or "missing",
                        visible_s=round(now_t - prev["since"], 1),
                        participant=t.get("participant"),
                    )
                prev.update(visible=vis, since=now_t)
        for tid in [k for k in mon["tracks"] if k not in seen]:
            prev = mon["tracks"].pop(tid)
            diag.event(
                "track_dropped", conn=cid, track=tid, was_visible=prev["visible"], for_s=round(now_t - prev["since"], 1)
            )
        for e in snap.get("events") or []:
            name = e.get("event") if isinstance(e, dict) else str(e)
            if name in ("NEW_GROUP", "ENDED", "SESSION_START", "JOIN", "LEAVE"):
                diag.event("session_event", conn=cid, kind=name, session=sess)

    def send(snap, extra=None, droppable=True):
        if page:
            if page.generation != page_generation:
                return
            page.session = max(page.session, int(snap.get("session") or 0))
        track_state(snap)
        now = time.monotonic()
        sig = tuple(sorted((t["id"], t["participant"], t["visible"]) for t in snap["tracks"]))
        if sig != fpsstate["sig"]:
            fpsstate["sig"], fpsstate["changed_at"] = sig, now
        fpsstate["rec"] = recommend_fps(snap, st["tr"].cfg, now - fpsstate["changed_at"] < 3.0)
        stats.recommended_fps = fpsstate["rec"]
        snap["recommended_fps"] = fpsstate["rec"]
        snap.update(
            {
                "type": "state",
                "running": st["running"],
                "pipeline": pipe.name,
                "sensor_status": "interrupted" if observation_clock.paused_at is not None else "available",
            }
        )
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
                if page and page.generation != page_generation:
                    return
                tr = st["tr"]
                if st["running"] and time.monotonic() - mon["frame_done_t"] > 3:
                    observation_clock.pause(at=mon["frame_done_t"] + 3)
                PAGES.prune()
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
            data = msg["data"]
            stable = st["tr"].state.value not in ("MAYBE_AWAY", "AWAY")
            submitted = time.monotonic()

            def vision_job(data=data, cfg=st["tr"].cfg, stable=stable, submitted=submitted):
                started = time.monotonic()
                result = _vision_job(proc, data, cfg, stable, cid)
                if result is not None:
                    result[1]["executor_wait"] = (started - submitted) * 1000
                return result

            r = await asyncio.to_thread(vision_job)
            if r is None:
                del data
                continue
            dets, tm = r
            tm["queue_wait"] = max(0.0, (t0 - msg["_received_perf"]) * 1000)
            reqs: list = []
            async with lock:
                if not st["running"]:
                    continue
                t1 = time.perf_counter()
                tr = st["tr"]
                if page and page.generation != page_generation:
                    continue
                observation_clock.observed()
                tr.speech_source = "clean" if clean_active() else "browser"
                snap = tr.update(dets, speaking=None if clean_active() else msg.get("speech"))
                if not tm.get("skipped"):
                    fm = mon["fm"]
                    fm["frames"] += 1
                    faces = [d for d in dets if d.face_bbox is not None]
                    if faces:
                        fm["detected"] += 1
                        r_ = tr.cfg.face_roi
                        for d in faces:
                            fm["confs"].append(float(d.conf))
                            fm["widths"].append(float(d.face_bbox[2] - d.face_bbox[0]))
                            cx, cy = (d.face_bbox[0] + d.face_bbox[2]) / 2, (d.face_bbox[1] + d.face_bbox[3]) / 2
                            fm["out_roi"] += not (r_[0] <= cx <= r_[2] and r_[1] <= cy <= r_[3])
                    fm["coast"] += any(
                        t.get("participant") and t.get("visible_kind") == "predicted" for t in snap["tracks"]
                    )
                tm["tracker"] = (time.perf_counter() - t1) * 1000
                aud["vis_t"] = time.monotonic()
                aud["vis_tc"] = msg.get("t")
                tm["total"] = (time.perf_counter() - t0) * 1000
                now = time.monotonic()
                stats.on_processed(tm)
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
        if (
            c.asr_provider == "assemblyai"
            and not asrmod.key_status()["set"]
            and st["running"]
            and tr.state.value != "ENDED"
        ):
            warn("asr_skip", 60, reason="no_key")
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
                for _, pcm, received in batch:
                    started = time.monotonic()
                    o, sp = a.process(pcm, snap, now, vis_age)
                    done = time.monotonic()
                    res.append((o, sp, a.recording_tracks() if rec else None, done, done - started, started - received))
                return res

            t0 = time.perf_counter()
            res = await loop.run_in_executor(AUDIO_POOL, job)
            proc_ms = (time.perf_counter() - t0) * 1000 / len(batch)
            aud["last_rx"] = time.monotonic()
            aud["processed"] += len(batch)
            mon["audio_done_t"] = time.monotonic()
            for (t_client, pcm, _), (o, speech, tracks, done, elapsed, wait) in zip(batch, res, strict=True):
                audio_stats.record(done, elapsed, len(pcm) / (2 * SR), wait)
                if o:
                    out.post(b"\x02" + struct.pack("<d", t_client) + o, audio=True)
                if o and tracks is not None:  # 0x03: t, n, out, vi, raw (same n samples each, aligned)
                    vi, rw = tracks
                    out.post(b"\x03" + struct.pack("<dI", t_client, len(o) // 2) + o + vi + rw)
                await asr_step(tr, pcm, o, speech, now)
            async with lock:
                tr = st["tr"]
                tr.speech_source = "clean"
                if st["running"] and observation_clock.paused_at is None and (speech != tr.speaking):
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
                        **audio_stats.snapshot(),
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
            wall_d, mono_d = time.time() - mon["wall0"], now - mon["mono0"]
            if wall_d - mono_d > 2.0:  # CLOCK_MONOTONIC does not advance while the host is suspended
                diag.event("host_suspend", "warning", cid, gap_s=round(wall_d - mono_d, 1))
            mon["wall0"], mon["mono0"] = time.time(), now
            fm = mon["fm"]
            if run and now - fm["t0"] >= 30 and fm["frames"]:
                rate = fm["detected"] / fm["frames"]
                cs, ws_ = fm["confs"], fm["widths"]
                diag.event(
                    "face_miss_summary",
                    "warning" if rate < 0.6 else "info",
                    cid,
                    window_s=round(now - fm["t0"]),
                    frames=fm["frames"],
                    face_frames=fm["detected"],
                    hit_rate=round(rate, 2),
                    conf_avg=round(sum(cs) / len(cs), 2) if cs else None,
                    conf_min=round(min(cs), 2) if cs else None,
                    face_w_avg=round(sum(ws_) / len(ws_), 3) if ws_ else None,
                    out_of_roi=fm["out_roi"],
                    coast_frames=fm["coast"],
                    min_conf=st["tr"].cfg.min_conf,
                )
                mon["fm"] = {"frames": 0, "detected": 0, "confs": [], "widths": [], "out_roi": 0, "coast": 0, "t0": now}
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
            if page and page.generation != page_generation:
                break
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(raw.get("code", 1000))
            frame = None
            if raw.get("bytes") is not None:
                # PRIVACY: audio/video payloads stay in memory, never logged/written.
                b = raw["bytes"]
                if b and b[0] == VIDEO_TAG:
                    if not st["running"]:
                        continue
                    try:
                        frame = binary_frame(b)
                    except ValueError:
                        continue
                    wire_bytes = len(b)
                else:
                    if len(b) > 9 and b[0] == 1 and st["tr"].cfg.audio_pipeline == "puresound":
                        aud["recv"] += 1
                        mon["audio_rx_t"] = time.monotonic()
                        if aud["q"].full():
                            aud["q"].get_nowait()
                            aud["dropped"] += 1
                        aud["q"].put_nowait((struct.unpack("<d", b[1:9])[0], b[9:], mon["audio_rx_t"]))
                    continue
                msg = {}
                kind = "frame"
            else:
                text = raw.get("text") or "{}"
                msg = json.loads(text)
                kind = msg.get("type")
                if kind == "frame":
                    if not st["running"]:
                        continue
                    try:
                        frame = legacy_frame(msg)
                    except ValueError:
                        continue
                    wire_bytes = len(text.encode("utf-8"))
            if frame is not None:
                # Application wire bytes: complete JSON UTF-8 or binary header + JPEG.
                stats.on_received(wire_bytes)
                mon["frame_rx_t"] = time.monotonic()
                if slot["frame"] is not None:
                    stats.dropped += 1
                frame["_received_perf"] = time.perf_counter()
                slot["frame"] = frame
                wake.set()
                continue
            async with lock:
                tr = st["tr"]
                if kind == "ping":
                    if msg.get("rtt") is not None:
                        stats.rtt.append(float(msg["rtt"]))
                    out.post({"type": "pong", "t": msg.get("t")})
                elif kind == "vad":
                    if st["running"] and observation_clock.paused_at is None and not clean_active():
                        send(tr.set_speech(bool(msg.get("speech"))), droppable=False)
                elif kind == "config":
                    new = msg.get("config", {})
                    before = asdict(tr.cfg)
                    warns = tr.cfg.update(new)
                    after = asdict(tr.cfg)
                    changed = {k: after[k] for k in after if after[k] != before.get(k)}
                    if not tr.cfg.reid_enabled or not tr.cfg.reid_consent:
                        rd.wipe("disabled")
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
                elif kind == "client":  # browser-side report: visibility / auto-pause / media ended / reconnect
                    what = str(msg.get("what") or "")[:32]
                    info = {
                        k: (v if isinstance(v, (int, float, bool)) else str(v)[:64])
                        for k, v in msg.items()
                        if k not in ("type", "what", "level", "conn", "_event")
                    }
                    diag.event(
                        "client",
                        "warning" if what in ("media_ended", "reconnect") else "info",
                        cid,
                        what=what,
                        **dict(list(info.items())[:8]),
                    )
                elif kind == "reid_wipe":
                    rd.wipe("ui")
                    out.post({"type": "reid", **rd.status()})
                elif kind == "record":  # client-side recording on/off (PRIVACY: the server never writes audio)
                    aud["rec"] = bool(msg.get("on"))
                    diag.event("record_stream", conn=cid, on=aud["rec"])
                elif kind == "start":
                    st["running"] = True
                    mon["frame_done_t"] = time.monotonic()
                    observation_clock.observed()
                    diag.event("start", conn=cid)
                    send(tr.snapshot(), droppable=False)
                elif kind in ("stop", "reset"):
                    proc.reset()
                    stats.counts = proc.counts
                    st["running"] = kind == "reset" and st["running"]
                    if kind == "stop":
                        aud["rec"] = False
                    diag.event(kind, conn=cid)
                    rd.interrupt()
                    previous_session = tr.session
                    observation_clock.resume()
                    if st["running"]:
                        observation_clock.observed()
                    st["tr"] = PresenceTracker(tr.cfg, clock=observation_clock)
                    st["tr"].session = previous_session
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
        if page:
            PAGES.release(page, page_generation)
        else:
            rd.close()
        if st["running"] and st["tr"].state.value != "ENDED":
            diag.event("session_interrupted", conn=cid, session=st["tr"].session, reason="transport_closed")
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
