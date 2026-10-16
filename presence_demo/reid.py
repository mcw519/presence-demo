"""Optional cross-session re-identification: "is this a new user?" (``reid_enabled``).

When the presence tracker opens a NEW session, the first good face of that session is compared against the
reference face(s) of the previous session(s) by a pluggable cloud backend. Same person -> the new session
*resumes* the previous ``conversation_id``; different / uncertain / any failure -> a new conversation. There
are NO calls inside a session, and none at all when no unexpired reference exists.

PRIVACY (documented exception to "the server never stores images", see docs/REID.md):
* Off by default; needs ``reid_enabled`` AND ``reid_consent``.
* Reference crops (one face crop per past user, 320x320 JPEG) live in this process's memory only, per
  connection, for ``reid_ttl`` seconds (default 600) and at most ``reid_max_refs`` users. They are wiped on
  expiry, disable, reset, the UI "clear" button, ``POST /api/reid_reset``, connection close and shutdown.
  Never written to disk, never logged, never sent back to the browser.
* With ``reid_show_crops`` (default on) a 112 px thumbnail of the probe and compared reference is sent to this
  connection's browser only for display (``reid_view``); never written to disk, logs or diagnostic events.
* Only face crops (never whole frames) are sent to the backend. The API key is memory-only (env
  ``TYPESAFE_API_KEY`` or set from the UI) and never logged or echoed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger("presence.reid")

JEV_URL = os.environ.get("REID_JEV_URL") or "https://api.typesafe.ai/v1/systemone"  # override: tests / proxies
QUESTION = (
    "Are the person in `reference` and the person in `query` the same individual? "
    "Judge identity from facial features, not clothing, pose, or background."
)
CROP_MAX = 320
BREAKER_ERRORS, BREAKER_PAUSE = 3, 60.0

_key: dict = {"value": os.environ.get("TYPESAFE_API_KEY", "") or "", "source": "env"}
if not _key["value"]:
    _key["source"] = None


def set_key(value: str | None) -> dict:
    v = (value or "").strip()
    _key["value"], _key["source"] = v, ("ui" if v else None)
    return key_status()


def key_status() -> dict:
    return {"set": bool(_key["value"]), "source": _key["source"]}


# ---------------------------------------------------------------------------------------------- crops
def face_crop(jpeg: bytes, bbox, margin: float = 0.15) -> bytes | None:
    """Decode (in memory), crop a normalized bbox with margin, pad to square, scale to CROP_MAX, re-encode JPEG."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox[:4]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    a, b = int(max(0, (x0 - mx) * w)), int(max(0, (y0 - my) * h))
    c, d = int(min(w, (x1 + mx) * w)), int(min(h, (y1 + my) * h))
    if c - a < 8 or d - b < 8:
        return None
    crop = img[b:d, a:c]
    # square crop scaled to exactly CROP_MAX (the API rejects small images: "Invalid image dimensions")
    side = max(crop.shape[:2])
    pad = np.zeros((side, side, 3), np.uint8)
    pad[(side - crop.shape[0]) // 2 :][: crop.shape[0], (side - crop.shape[1]) // 2 :][:, : crop.shape[1]] = crop
    interp = cv2.INTER_AREA if side > CROP_MAX else cv2.INTER_CUBIC
    crop = cv2.resize(pad, (CROP_MAX, CROP_MAX), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------------------------- backend
class ReidError(RuntimeError):
    def __init__(self, kind: str, msg: str = ""):
        super().__init__(msg or kind)
        self.kind = kind


class CloudReid:
    """Backend interface: ``compare(reference_jpeg, query_jpeg) -> (score 0..1, meta)``. Raises ReidError."""

    name = "base"

    async def compare(self, ref: bytes, query: bytes, *, model: str, timeout: float) -> tuple[float, dict]:
        raise NotImplementedError


class JevClient(CloudReid):
    """Typesafe Jev images API (``noul`` question over two image assets). Call shape follows the LFW benchmark."""

    name = "jev"

    def __init__(self, url: str | None = None, key_fn: Callable[[], str] = lambda: _key["value"]):
        self.url = url or JEV_URL
        self.key_fn = key_fn

    def body(self, ref: bytes, query: bytes, model: str) -> dict:
        r1, r2 = str(uuid.uuid4()), str(uuid.uuid4())
        enc = lambda b: base64.b64encode(b).decode()  # noqa: E731
        return {
            "model": model,
            "state": {"reference": r1, "query": r2},
            "assets": {
                r1: {"type": "image", "format": "jpg", "data_base64": enc(ref)},
                r2: {"type": "image", "format": "jpg", "data_base64": enc(query)},
            },
            "questions": {"same": {"type": "noul", "instructions": QUESTION}},
        }

    def _post(self, data: bytes, key: str, timeout: float):
        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "TypeSafe-Danger-Zone-Images": "true",  # opt-in for image assets (meaning unverified; docs/REID.md)
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https URL / test override)
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise ReidError(f"http_{e.code}") from None
        except (TimeoutError, OSError) as e:
            raise ReidError(
                "timeout" if "timed out" in str(e).lower() or isinstance(e, TimeoutError) else "network"
            ) from None
        except ValueError:
            raise ReidError("bad_response") from None

    async def compare(self, ref: bytes, query: bytes, *, model: str, timeout: float) -> tuple[float, dict]:
        key = self.key_fn()
        if not key:
            raise ReidError("no_key")
        data = json.dumps(self.body(ref, query, model)).encode()
        j = await asyncio.to_thread(self._post, data, key, timeout)
        try:
            score = float(((j.get("answers") or {}).get("same") or {})["noul"])
        except (KeyError, TypeError, ValueError):
            raise ReidError("bad_response") from None
        return max(0.0, min(1.0, score)), {
            "model": j.get("model"),
            "in_tok": (j.get("usage") or {}).get("input_tokens"),
        }


THUMB_PX = 112


def thumb(jpeg: bytes | None, px: int = THUMB_PX) -> str | None:
    """Small display thumbnail (base64 JPEG) for this connection's browser only (``reid_show_crops``).
    Never written to disk, logs or diagnostic events."""
    if not jpeg:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# ---------------------------------------------------------------------------------------------- store
@dataclass
class Ref:
    session: int
    conversation_id: str
    jpeg: bytes = field(repr=False)
    t: float = 0.0


class ReferenceStore:
    """Past users' face crops, in process memory only, with TTL and a size cap."""

    def __init__(self):
        self.refs: list[Ref] = []

    def add(self, ref: Ref, max_refs: int) -> None:
        self.refs = [r for r in self.refs if r.conversation_id != ref.conversation_id] + [ref]
        del self.refs[: max(0, len(self.refs) - max_refs)]

    def expire(self, now: float, ttl: float) -> int:
        n = len(self.refs)
        self.refs = [r for r in self.refs if now - r.t < ttl]
        return n - len(self.refs)

    def valid(self, now: float, ttl: float) -> list[Ref]:
        return [r for r in reversed(self.refs) if now - r.t < ttl]  # newest first

    def wipe(self) -> int:
        n = len(self.refs)
        for r in self.refs:
            r.jpeg = b""
        self.refs.clear()
        return n


_STORES: weakref.WeakSet[ReferenceStore] = weakref.WeakSet()


def wipe_all() -> int:
    """Wipe every connection's references (``POST /api/reid_reset``, shutdown)."""
    return sum(s.wipe() for s in list(_STORES))


# ---------------------------------------------------------------------------------------------- decision
def face_quality(t: dict) -> float:
    """Quality of a track's face for re-ID (0 = unusable). Uses the tracker snapshot only (no pixels)."""
    if not t.get("visible") or t.get("has_face") is False or t.get("visible_kind") == "predicted":
        return 0.0
    b = t.get("bbox") or [0, 0, 0, 0]
    w = b[2] - b[0]
    q = w * float(t.get("conf") or 0)
    if t.get("facing") == "away":
        return 0.0
    if t.get("facing") == "side":
        q *= 0.5
    return q


class CrossSessionReid:
    """Per-connection driver. ``observe(snap, now)`` returns crop requests ``[(kind, bbox)]`` for the caller to
    fulfil with :func:`face_crop` and hand back via ``on_crop(kind, jpeg, now)``. Decisions are emitted with
    ``emit(dict)`` (``type: handoff``) and log events with ``event(name, **kv)``."""

    def __init__(self, cfg, client: CloudReid | None = None, emit=None, event=None, clock=time.monotonic):
        self.cfg = cfg
        self.client = client or JevClient()
        self.emit = emit or (lambda d: None)
        self.event = event or (lambda name, **kv: None)
        self.clock = clock
        self.store = ReferenceStore()
        _STORES.add(self.store)
        self.session = 0
        self.conversation_id: str | None = None
        self.best: tuple[float, bytes] | None = None  # current session's best face (memory only)
        self.best_t = -1e9
        self.phase = "idle"  # idle | await_query | calling | await_retry
        self.waited = 0.0  # seconds waited for a usable face (ACTIVE/IDLE only; paused in MAYBE_AWAY/AWAY)
        self.first_part_t: float | None = None  # waited-time when a participant first appeared
        self.wait_t = 0.0
        self.wait_why: dict = {}
        self.retry_ref: Ref | None = None
        self.retried = False
        self.task: asyncio.Task | None = None
        self.calls = 0
        self.tokens = 0
        self.hour: deque = deque()
        self.errors = 0
        self.paused_until = 0.0
        self.last_decision_t = -1e9
        self.last: dict | None = None
        self.committed = True
        self.was_on = False
        self.view_probe: bytes | None = None  # last query crop and best-scoring compared ref (display only)
        self.view_ref: tuple[float, Ref] | None = None

    # -- helpers
    def on(self) -> bool:
        return bool(self.cfg.reid_enabled and self.cfg.reid_consent)

    def status(self) -> dict:
        now = self.clock()
        refs = self.store.valid(now, self.cfg.reid_ttl)
        return {
            "enabled": self.on(),
            "key_set": key_status()["set"],
            "phase": self.phase,
            "refs": len(refs),
            "ttl_left": round(min((self.cfg.reid_ttl - (now - r.t) for r in refs), default=0), 0),
            "calls": self.calls,
            "tokens": self.tokens,
            "cost_est": round(self.calls * self.cfg.reid_cost_per_call, 4),
            "conversation_id": self.conversation_id,
            "last": self.last,
        }

    def wipe(self, reason: str) -> None:
        n = self.store.wipe()
        self.best = None
        self.view_probe = self.view_ref = None
        self.emit({"type": "reid_view_clear"})
        if n:
            self.event("reid_store", action="wipe", reason=reason, count=n)

    def _commit_best(self, now: float) -> None:
        if not self.committed and self.best is not None and self.conversation_id:
            self.store.add(Ref(self.session, self.conversation_id, self.best[1], now), int(self.cfg.reid_max_refs))
            self.event("reid_store", action="add", session=self.session, count=len(self.store.refs))
        self.committed = True
        self.best = None

    def _decide(self, decision: str, reason: str, now: float, ref: Ref | None = None, score=None) -> None:
        self.phase = "idle"
        self.last_decision_t = now
        if decision == "resume" and ref is not None:
            self.conversation_id = ref.conversation_id
        else:
            self.conversation_id = uuid.uuid4().hex
        d = {
            "type": "handoff",
            "session": self.session,
            "conversation_id": self.conversation_id,
            "decision": decision,
            "resumed_from_session": ref.session if decision == "resume" and ref else None,
            "score": None if score is None else round(score, 3),
            "reason": reason,
            "calls": self.calls,
        }
        self.last = {k: d[k] for k in ("session", "decision", "resumed_from_session", "score", "reason")}
        self.event("reid_decision", **{k: v for k, v in d.items() if k != "type"})
        self.emit(d)
        probe, vr = self.view_probe, self.view_ref
        self.view_probe = self.view_ref = None
        if self.cfg.reid_show_crops:  # display-only thumbnails, to this connection's websocket; never logged
            r = ref if ref is not None else (vr[1] if vr else None)
            self.emit(
                {
                    "type": "reid_view",
                    "session": self.session,
                    "decision": decision,
                    "reason": reason,
                    "score": d["score"] if d["score"] is not None else (round(vr[0], 3) if vr else None),
                    "threshold": self.cfg.reid_threshold,
                    "resumed_from_session": d["resumed_from_session"],
                    "ref_session": r.session if r else None,
                    "conversation_id": self.conversation_id,
                    "t": time.time(),
                    "probe": thumb(probe),
                    "ref": thumb(r.jpeg if r else None),
                }
            )

    def _budget(self, now: float) -> str | None:
        while self.hour and now - self.hour[0] > 3600:
            self.hour.popleft()
        if now < self.paused_until:
            return "breaker"
        if len(self.hour) >= self.cfg.reid_cap_per_hour or self.calls >= self.cfg.reid_cap_total:
            return "cap"
        if not key_status()["set"] and isinstance(self.client, JevClient):
            return "no_key"
        return None

    # -- main entry (called after each tracker update; sync, never blocks)
    def observe(self, snap: dict, now: float | None = None) -> list[tuple]:
        now = self.clock() if now is None else now
        on = self.on()
        if self.was_on and not on:
            self.wipe("disabled")
        self.was_on = on
        if not on:
            return []
        n = self.store.expire(now, self.cfg.reid_ttl)
        if n:
            self.event("reid_store", action="expire", count=n)
        sess, state = int(snap.get("session") or 0), snap.get("state")
        if state == "ENDED":
            self._commit_best(now)
        if sess != self.session and sess >= 1 and state != "ENDED":
            self._commit_best(now)
            self.session, self.committed, self.best, self.best_t = sess, False, None, -1e9
            self.retried, self.retry_ref = False, None
            self.waited, self.first_part_t, self.wait_t, self.wait_why = 0.0, None, now, {}
            if not self.store.valid(now, self.cfg.reid_ttl):
                self._decide("new", "no_reference", now)  # no previous-round data: brand-new, no call
            elif now - self.last_decision_t < self.cfg.reid_cooldown:
                self._decide("new", "cooldown", now)
            else:
                self.phase = "await_query"
                self.event("reid_start", session=sess, refs=len(self.store.valid(now, self.cfg.reid_ttl)))
        reqs: list[tuple] = []
        if state == "ENDED":
            return reqs
        cands = sorted(
            ((face_quality(t), t) for t in snap.get("tracks", []) if t.get("participant")),
            key=lambda x: -x[0],
        )
        good = [(q, t) for q, t in cands if q > 0 and (t["bbox"][2] - t["bbox"][0]) >= self.cfg.reid_min_face]
        if self.phase in ("await_query", "await_retry"):
            dt, self.wait_t = max(0.0, now - self.wait_t), now
            paused = state in ("MAYBE_AWAY", "AWAY")  # nobody to look at: do not burn the face-wait budget
            if not paused:
                self.waited += dt
            parts = [t for t in snap.get("tracks", []) if t.get("participant") and t.get("visible")]
            if parts and self.first_part_t is None:
                self.first_part_t = self.waited
            if good:
                reqs.append(("query", good[0][1]["bbox"], good[0][0]))
            elif not paused:
                if parts:  # remember why the visible participant's face is not usable (for reid_wait)
                    t = parts[0]
                    b = t.get("bbox") or [0, 0, 0, 0]
                    self.wait_why = {
                        "why": "facing_away"
                        if t.get("facing") == "away"
                        else (
                            "no_face"
                            if t.get("has_face") is False
                            else (
                                "predicted"
                                if t.get("visible_kind") == "predicted"
                                else ("face_too_small" if b[2] - b[0] < self.cfg.reid_min_face else "low_quality")
                            )
                        ),
                        "face_w": round(b[2] - b[0], 3),
                    }
                face_wait = self.first_part_t is not None and self.waited - self.first_part_t > self.cfg.reid_face_wait
                if face_wait or self.waited > self.cfg.reid_decision_timeout:
                    reason = "no_face_timeout" if face_wait else "decision_timeout"
                    self.event(
                        "reid_wait",
                        session=self.session,
                        result=reason,
                        waited_s=round(self.waited, 1),
                        participant_seen=self.first_part_t is not None,
                        **(self.wait_why or {"why": "no_participant"}),
                    )
                    self._decide("new", reason, now)
        if good and not self.committed and now - self.best_t >= 1.0:
            q = good[0][0]
            if self.best is None or q > self.best[0] * 1.1:
                reqs.append(("best", good[0][1]["bbox"], q))
        return reqs

    def on_crop(self, kind: str, jpeg: bytes | None, now: float | None = None, quality: float = 0.0) -> None:
        now = self.clock() if now is None else now
        if not jpeg:
            return
        if kind == "best":
            self.best, self.best_t = (quality, jpeg), now
            return
        if kind == "query" and self.phase in ("await_query", "await_retry"):
            if self.best is None:  # the first query also seeds this session's reference
                self.best, self.best_t = (quality, jpeg), now
            refs = (
                [self.retry_ref]
                if self.phase == "await_retry" and self.retry_ref
                else self.store.valid(now, self.cfg.reid_ttl)
            )
            self.phase = "calling"
            self.task = asyncio.get_running_loop().create_task(self._run(jpeg, refs, self.session))

    async def _run(self, query: bytes, refs: list[Ref], session: int) -> None:
        best_unc: tuple[float, Ref] | None = None
        self.view_probe = query
        for ref in refs:
            now = self.clock()
            why = self._budget(now)
            if why:
                self.event("reid_skip", reason=why, session=session)
                return self._finish(session, "new", why, now)
            t0 = time.perf_counter()
            try:
                score, meta = await self.client.compare(
                    ref.jpeg, query, model=self.cfg.reid_model, timeout=self.cfg.reid_timeout
                )
            except ReidError as e:
                self.calls += 1 if e.kind not in ("no_key",) else 0
                self.hour.append(now)
                self.errors += 1
                self.event("reid_error", "warning", kind=e.kind, session=session, errors=self.errors)
                if self.errors >= BREAKER_ERRORS:
                    self.paused_until = now + BREAKER_PAUSE
                    self.event("reid_breaker", "warning", action="open", pause_s=BREAKER_PAUSE)
                return self._finish(session, "new", "error", self.clock())
            except Exception as e:  # noqa: BLE001
                self.event("reid_error", "warning", kind=type(e).__name__, session=session)
                return self._finish(session, "new", "error", self.clock())
            self.errors = 0
            self.calls += 1
            self.hour.append(now)
            self.tokens += int(meta.get("in_tok") or 0)
            if self.view_ref is None or score > self.view_ref[0]:
                self.view_ref = (score, ref)
            self.event(
                "reid_call",
                session=session,
                ref_session=ref.session,
                score=round(score, 3),
                latency_ms=round((time.perf_counter() - t0) * 1000),
                model=meta.get("model"),
                in_tok=meta.get("in_tok"),
                calls=self.calls,
                cost_est=round(self.calls * self.cfg.reid_cost_per_call, 4),
            )
            if score >= self.cfg.reid_threshold:
                return self._finish(session, "resume", "match", self.clock(), ref, score)
            if score >= self.cfg.reid_uncertain_low and (best_unc is None or score > best_unc[0]):
                best_unc = (score, ref)
        now = self.clock()
        if best_unc and not self.retried and session == self.session:
            self.retried, self.retry_ref, self.phase = True, best_unc[1], "await_retry"  # one more frame
            self.event("reid_retry", session=session, score=round(best_unc[0], 3), ref_session=best_unc[1].session)
            return None
        self._finish(
            session, "new", "uncertain" if best_unc else "below_threshold", now, score=best_unc[0] if best_unc else None
        )

    def _finish(self, session, decision, reason, now, ref=None, score=None):
        if session != self.session:  # a newer session started meanwhile: drop
            return None
        self._decide(decision, reason, now, ref, score)
        return None

    def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
        self.wipe("conn_close")
