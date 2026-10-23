"""Optional cross-session re-identification: "is this a new user?" (``reid_enabled``).

When the presence tracker opens a NEW session, the first good face of that session is compared against the
reference face(s) of the previous session(s) by a pluggable cloud backend. Same person -> the new session
*resumes* the previous ``conversation_id``; different / uncertain / any failure -> a new conversation. There
are NO calls inside a session, and none at all when no unexpired reference exists.

PRIVACY (documented exception to "the server never stores images", see docs/reid.md):
* Off by default; needs ``reid_enabled`` AND ``reid_consent``.
* Reference crops (up to three participant faces per conversation, 320x320 JPEG) are memory-only.
  ``reid_max_refs`` caps inactive conversations; ``reid_ttl`` starts at their last normal ENDED.
  The transport may retain this store across same-page reconnects. Expiry, consent withdrawal, reset,
  page release and shutdown wipe references; interrupt() preserves active references for reconnect.
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
                "TypeSafe-Danger-Zone-Images": "true",  # opt-in for image assets (meaning unverified; docs/reid.md)
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
    participant: str = ""


class ReferenceStore:
    """Memory-only references; the history cap counts conversations, not faces.

    ``active`` survives a same-page transport interruption separately from history.
    Its timestamps are not refreshed by reconnecting or unsuccessful comparisons.
    """

    def __init__(self):
        self.refs: list[Ref] = []
        self.active: list[Ref] = []
        self.generation = 0
        self.active_connected = False
        self.active_until = 0.0
        self.active_cid = None
        self.active_expiries = {}
        self.ended_at = {}  # Updated only by a normal session ENDED commit.
        self.drivers = weakref.WeakSet()

    @staticmethod
    def _clear(refs):
        for ref in refs:
            ref.jpeg = b""

    def add(self, ref: Ref, max_refs: int) -> None:
        members = [r for r in self.refs if r.conversation_id == ref.conversation_id]
        members = [r for r in members if r.participant != ref.participant] + [ref]
        self.commit(members, max_refs, ref.t)

    def commit(self, members: list[Ref], max_refs: int, now: float) -> None:
        if not members:
            return
        cid = members[0].conversation_id
        old = [r for r in self.refs + self.active if r.conversation_id == cid]
        merged = {r.participant: r for r in old}
        for ref in members:
            previous = merged.get(ref.participant)
            if previous is not None and previous is not ref:
                previous.jpeg = b""
            merged[ref.participant] = ref
        selected = list(merged.values())[-3:]
        self._clear([r for r in old if r not in selected])
        for ref in selected:
            ref.t = now
        self.refs = [r for r in self.refs if r.conversation_id != cid] + selected
        self.active = [r for r in self.active if r.conversation_id != cid]
        if self.active_cid == cid:
            self.active_cid = None
            self.active_connected = False
        self.active_expiries.pop(cid, None)
        self.ended_at[cid] = now
        self.trim(max_refs)

    def trim(self, max_refs):
        inactive = [
            r for r in self.refs + self.active if not (self.active_connected and r.conversation_id == self.active_cid)
        ]
        times = {}
        for ref in inactive:
            times[ref.conversation_id] = max(times.get(ref.conversation_id, -1e30), ref.t)
        oldest = sorted(times, key=times.get)
        evict = set(oldest[: max(0, len(oldest) - max_refs)])
        self._clear([r for r in self.refs + self.active if r.conversation_id in evict])
        self.refs = [r for r in self.refs if r.conversation_id not in evict]
        self.active = [r for r in self.active if r.conversation_id not in evict]
        for cid in evict:
            self.active_expiries.pop(cid, None)
            self.ended_at.pop(cid, None)

    def interrupt(self, conversation_id, now, ttl):
        # A reconnect must never renew an ended conversation's retention lease.
        # A never-ended conversation gets exactly one lease at first interruption.
        if conversation_id in self.ended_at:
            deadline = self.ended_at[conversation_id] + ttl
            self.active_expiries[conversation_id] = deadline
        else:
            deadline = self.active_expiries.setdefault(conversation_id, now + ttl)
        self.active_connected = False
        self.active_until = deadline

    def activate(self, conversation_id):
        self.active.extend(r for r in self.refs if r.conversation_id == conversation_id)
        self.refs = [r for r in self.refs if r.conversation_id != conversation_id]
        self.active_connected = True
        self.active_cid = conversation_id

    def expire(self, now: float, ttl: float) -> int:
        expired = [r for r in self.refs if now - r.t >= ttl]
        self.refs = [r for r in self.refs if now - r.t < ttl]
        expired_active = [
            r
            for r in self.active
            if not (self.active_connected and r.conversation_id == self.active_cid)
            and now >= self.active_expiries.get(r.conversation_id, r.t + ttl)
        ]
        expired.extend(expired_active)
        self.active = [r for r in self.active if r not in expired_active]
        self._clear(expired)
        retained = {r.conversation_id for r in self.refs + self.active}
        for cid in (set(self.ended_at) | set(self.active_expiries)) - retained:
            self.ended_at.pop(cid, None)
            self.active_expiries.pop(cid, None)
        return len(expired)

    def valid(self, now: float, ttl: float) -> list[Ref]:
        history = [r for r in self.refs if now - r.t < ttl]
        active = [
            r
            for r in self.active
            if (self.active_connected and r.conversation_id == self.active_cid)
            or now < self.active_expiries.get(r.conversation_id, r.t + ttl)
        ]
        return [r for r in reversed(history + active) if r.jpeg]

    def wipe(self) -> int:
        n = len(self.refs) + len(self.active)
        self._clear(self.refs + self.active)
        self.refs.clear()
        self.active.clear()
        self.active_connected = False
        self.active_until = 0.0
        self.active_cid = None
        self.active_expiries.clear()
        self.ended_at.clear()
        self.generation += 1
        for driver in list(self.drivers):
            driver._store_wiped()
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
    """Nonblocking, per-page re-ID driver with generation-safe crop and task delivery."""

    def __init__(self, cfg, client=None, emit=None, event=None, clock=time.monotonic, store=None):
        self.cfg = cfg
        self.client = client or JevClient()
        self.emit = emit or (lambda d: None)
        self.event = event or (lambda *args, **kv: None)
        self.clock = clock
        self.store = store if store is not None else ReferenceStore()
        self.store.drivers.add(self)
        _STORES.add(self.store)
        self.session = 0
        self.conversation_id = None
        self.members = {}  # track ID -> (quality, JPEG, last improvement time)
        self.member_ids = {}  # query track ID -> historical participant identity on resume
        self.phase = "idle"
        self.started = 0.0
        self.first_part_t = None
        self.wait_why = {}
        self.retried = False
        self.task = None
        self.deadline_task = None
        self.epoch = 0
        self.ended = True
        self.queries = {}
        self.expected = set()
        self.calls = self.tokens = self.errors = 0
        self.hour = deque()
        self.paused_until = 0.0
        self.last_decision_t = -1e9
        self.last = None
        self.committed = True
        self.was_on = False
        self.view_probe = self.view_ref = None

    def on(self):
        return bool(self.cfg.reid_enabled and self.cfg.reid_consent)

    def status(self):
        now = self.clock()
        refs = self.store.valid(now, self.cfg.reid_ttl)
        deadlines = [
            self.store.active_expiries.get(r.conversation_id, r.t + self.cfg.reid_ttl)
            for r in refs
            if not (self.store.active_connected and r.conversation_id == self.store.active_cid)
        ]
        return {
            "enabled": self.on(),
            "session": self.session,
            "key_set": key_status()["set"],
            "phase": self.phase,
            "refs": len(set(r.conversation_id for r in refs)),
            "ttl_left": round(max(0, min(deadlines) - now), 0) if deadlines else None,
            "calls": self.calls,
            "tokens": self.tokens,
            "cost_est": round(self.calls * self.cfg.reid_cost_per_call, 4),
            "conversation_id": self.conversation_id,
            "last": self.last,
        }

    def _cancel(self):
        self.epoch += 1
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in (self.task, self.deadline_task):
            if task and not task.done() and task is not current:
                task.cancel()
        self.task = self.deadline_task = None
        self.queries.clear()
        self.expected.clear()

    def _store_wiped(self):
        self._cancel()
        self.members.clear()
        self.member_ids.clear()
        self.view_probe = self.view_ref = None
        self.phase = "idle"
        self.committed = True

    def wipe(self, reason):
        n = self.store.wipe()
        self.emit({"type": "reid_view_clear"})
        if n:
            self.event("reid_store", action="wipe", reason=reason, count=n)

    def _refs_for_members(self, now):
        return [
            Ref(self.session, self.conversation_id, jpeg, now, self.member_ids.get(track, f"{self.session}:{track}"))
            for track, (_quality, jpeg, _t) in self.members.items()
        ]

    def _commit_best(self, now):
        if not self.committed and self.conversation_id:
            self.store.commit(self._refs_for_members(now), int(self.cfg.reid_max_refs), now)
            self.event("reid_store", action="add", session=self.session, count=len(self.store.refs))
        self.committed = True
        self.members.clear()

    def _decide(self, decision, reason, now, ref=None, score=None):
        if self.phase == "idle" and self.last and self.last["session"] == self.session:
            return
        previous = self.conversation_id
        if decision == "resume" and ref:
            self.conversation_id = ref.conversation_id
            self.store.activate(ref.conversation_id)
        self.phase = "idle"
        self.last_decision_t = now
        if self.deadline_task:
            self.deadline_task.cancel()
        d = {
            "type": "handoff",
            "session": self.session,
            "conversation_id": self.conversation_id,
            "previous_conversation_id": previous,
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
        if self.cfg.reid_show_crops:
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

    def _budget(self, now):
        while self.hour and now - self.hour[0] > 3600:
            self.hour.popleft()
        if now < self.paused_until:
            return "breaker"
        if len(self.hour) >= self.cfg.reid_cap_per_hour or self.calls >= self.cfg.reid_cap_total:
            return "cap"
        if not key_status()["set"] and isinstance(self.client, JevClient):
            return "no_key"
        return None

    def _alive(self, session, epoch, generation):
        return (
            session == self.session
            and epoch == self.epoch
            and generation == self.store.generation
            and not self.ended
            and self.on()
            and self.phase != "idle"
        )

    def _remaining(self):
        return self.cfg.reid_decision_timeout - (self.clock() - self.started)

    async def _deadline(self, session, epoch, generation):
        await asyncio.sleep(max(0, self._remaining()))
        if self._alive(session, epoch, generation):
            if self.task and not self.task.done():
                self.task.cancel()
            self._decide("new", "decision_timeout", self.clock())

    def observe(self, snap, now=None):
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
            if sess == self.session and not self.ended:
                self._cancel()
                self.phase = "idle"
                self.ended = True
                self._commit_best(now)
            return []
        if sess >= 1 and sess != self.session:
            self._cancel()
            # A tracker reset without ENDED is interruption, never a fabricated normal ending.
            if not self.committed:
                self.interrupt()
            self.members = {}
            self.member_ids = {}
            self.session, self.committed, self.ended = sess, False, False
            self.started, self.first_part_t, self.wait_why = now, None, {}
            self.retried = False
            self.conversation_id = uuid.uuid4().hex
            self.phase = "await_query"
            refs = self.store.valid(now, self.cfg.reid_ttl)
            if not refs:
                self._decide("new", "no_reference", now)
            elif now - self.last_decision_t < self.cfg.reid_cooldown:
                self._decide("new", "cooldown", now)
            else:
                self.event("reid_start", session=sess, refs=len(refs))
                self.deadline_task = asyncio.get_running_loop().create_task(
                    self._deadline(sess, self.epoch, self.store.generation)
                )
        if self.ended or sess < 1:
            return []
        parts = [t for t in snap.get("tracks", []) if t.get("participant")][:3]
        if parts and self.first_part_t is None:
            self.first_part_t = now
        good = [(face_quality(t), t) for t in parts]
        good = [(q, t) for q, t in good if q > 0 and t["bbox"][2] - t["bbox"][0] >= self.cfg.reid_min_face]
        if self.phase != "idle" and self._remaining() <= 0:
            if self.task and not self.task.done():
                self.task.cancel()
            self.event(
                "reid_wait",
                session=self.session,
                result="decision_timeout",
                waited_s=round(now - self.started, 1),
                participant_seen=self.first_part_t is not None,
                **(self.wait_why or {"why": "no_participant"}),
            )
            self._decide("new", "decision_timeout", now)
        participant_ids = {str(track["id"]) for track in parts}
        if self.phase == "calling" and participant_ids - self.expected:
            # Membership changed while remote comparisons were in flight. Invalidate
            # that result set and compare the expanded group under the original deadline.
            queries, expected = dict(self.queries), set(self.expected)
            self._cancel()
            self.queries, self.expected = queries, expected | participant_ids
            self.view_probe = self.view_ref = None
            self.phase = "await_query"
            self.deadline_task = asyncio.get_running_loop().create_task(
                self._deadline(self.session, self.epoch, self.store.generation)
            )
            self.event("reid_members_changed", session=self.session, participants=len(self.expected))
        waiting = self.phase in ("await_query", "await_retry")
        if waiting:
            self.expected.update(str(t["id"]) for t in parts)
            if parts and len(good) < len(parts):
                t = next((t for t in parts if t not in [g[1] for g in good]), parts[0])
                b = t.get("bbox") or [0, 0, 0, 0]
                why = (
                    "facing_away"
                    if t.get("facing") == "away"
                    else "no_face"
                    if t.get("has_face") is False
                    else "predicted"
                    if t.get("visible_kind") == "predicted"
                    else "face_too_small"
                    if b[2] - b[0] < self.cfg.reid_min_face
                    else "low_quality"
                )
                self.wait_why = {"why": why, "face_w": round(b[2] - b[0], 3)}
            face_wait = self.first_part_t is not None and now - self.first_part_t >= self.cfg.reid_face_wait
            if face_wait and not self.queries and not good:
                self.event(
                    "reid_wait",
                    session=self.session,
                    result="no_face_timeout",
                    waited_s=round(now - self.started, 1),
                    participant_seen=True,
                    **self.wait_why,
                )
                self._decide("new", "no_face_timeout", now)
                waiting = False
            elif face_wait and self.queries and not ({str(t["id"]) for _q, t in good} - self.queries.keys()):
                self._launch()
                waiting = False
        reqs = []
        for q, track in good:
            tid = str(track["id"])
            if waiting and tid not in self.queries:
                reqs.append((f"query:{self.session}:{self.epoch}:{tid}", track["bbox"], q))
            prev = self.members.get(tid)
            if not self.committed and (prev is None or (now - prev[2] >= 1 and q > prev[0] * 1.1)):
                reqs.append((f"best:{self.session}:{self.epoch}:{tid}", track["bbox"], q))
        return reqs

    def on_crop(self, kind, jpeg, now=None, quality=0.0):
        now = self.clock() if now is None else now
        if not jpeg or not self.on() or self.ended or self.committed:
            return
        bits = kind.split(":", 3)
        if len(bits) == 4:
            name, session, epoch, tid = bits
            if name not in ("best", "query"):
                return
            if session != str(self.session) or epoch != str(self.epoch):
                return
        elif kind in ("best", "query"):
            name, tid = kind, "1"
        else:
            return
        if tid not in self.members and len(self.members) >= 3:
            return
        previous = self.members.get(tid)
        if previous is None or quality >= previous[0]:
            self.members[tid] = (quality, jpeg, now)
        if name == "query" and self.phase in ("await_query", "await_retry"):
            if self._remaining() <= 0:
                self._decide("new", "decision_timeout", now)
                return
            self.queries[tid] = jpeg
            if self.expected <= self.queries.keys():
                self._launch()

    def _launch(self):
        self.phase = "calling"
        refs = self.store.valid(self.clock(), self.cfg.reid_ttl)
        self.task = asyncio.get_running_loop().create_task(
            self._run(dict(self.queries), refs, self.session, self.epoch, self.store.generation)
        )

    async def _run(self, queries, refs, session, epoch, generation):
        matched = {}
        identities = {}
        best_unc = None
        uncertain_conversations = set()
        for tid, query in queries.items():
            for ref in refs:
                if not self._alive(session, epoch, generation):
                    return
                now = self.clock()
                remaining = self._remaining()
                if remaining <= 0:
                    return self._finish(session, epoch, generation, "new", "decision_timeout", now)
                why = self._budget(now)
                if why:
                    self.event("reid_skip", reason=why, session=session)
                    return self._finish(session, epoch, generation, "new", why, now)
                t0 = time.perf_counter()
                timeout = min(self.cfg.reid_timeout, remaining)
                # Reserve the call before awaiting: cancellation cannot evade caps.
                self.calls += 1
                self.hour.append(now)
                try:
                    score, meta = await asyncio.wait_for(
                        self.client.compare(ref.jpeg, query, model=self.cfg.reid_model, timeout=timeout), timeout
                    )
                except asyncio.CancelledError:
                    return
                except Exception as error:
                    if not self._alive(session, epoch, generation):
                        return
                    kind = (
                        "timeout" if isinstance(error, TimeoutError) else getattr(error, "kind", type(error).__name__)
                    )
                    self.errors += 1
                    self.event("reid_error", "warning", kind=kind, session=session, errors=self.errors)
                    if self.errors >= BREAKER_ERRORS:
                        self.paused_until = now + BREAKER_PAUSE
                        self.event("reid_breaker", "warning", action="open", pause_s=BREAKER_PAUSE)
                    return self._finish(session, epoch, generation, "new", "error", self.clock())
                if not self._alive(session, epoch, generation):
                    return
                self.tokens += int(meta.get("in_tok") or 0)
                self.errors = 0
                if self._remaining() <= 0:
                    return self._finish(session, epoch, generation, "new", "decision_timeout", self.clock())
                if self.view_ref is None or score > self.view_ref[0]:
                    self.view_probe, self.view_ref = query, (score, ref)
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
                    old = matched.get(ref.conversation_id)
                    if old is None or score > old[0]:
                        matched[ref.conversation_id] = (score, ref)
                    identities.setdefault(ref.conversation_id, {})[tid] = ref.participant
                elif score >= self.cfg.reid_uncertain_low:
                    uncertain_conversations.add(ref.conversation_id)
                    if best_unc is None or score > best_unc:
                        best_unc = score
        now = self.clock()
        if len(matched) > 1:
            return self._finish(session, epoch, generation, "new", "ambiguous", now)
        if len(matched) == 1:
            cid, (score, ref) = next(iter(matched.items()))
            if uncertain_conversations - {cid}:
                return self._finish(session, epoch, generation, "new", "ambiguous", now)
            self.member_ids = identities[cid]
            return self._finish(session, epoch, generation, "resume", "match", now, ref, score)
        if best_unc is not None and not self.retried and self._alive(session, epoch, generation):
            self.retried, self.phase = True, "await_retry"
            self.queries.clear()
            self.event("reid_retry", session=session, score=round(best_unc, 3))
            return
        self._finish(
            session, epoch, generation, "new", "uncertain" if best_unc else "below_threshold", now, score=best_unc
        )

    def _finish(self, session, epoch, generation, decision, reason, now, ref=None, score=None):
        if self._alive(session, epoch, generation):
            if decision == "resume" and not any(
                candidate is ref for candidate in self.store.valid(now, self.cfg.reid_ttl)
            ):
                self.member_ids.clear()
                return self._decide("new", "reference_expired", now)
            self._decide(decision, reason, now, ref, score)

    def interrupt(self):
        """Preserve current participant references without inventing an ENDED timestamp."""
        self._cancel()
        if not self.committed and self.conversation_id:
            cid = self.conversation_id
            # Keep the original retention lease across repeated interruptions.
            existing = {r.participant: r for r in self.store.active if r.conversation_id == cid}
            for ref in self._refs_for_members(self.clock()):
                previous = existing.get(ref.participant)
                if previous is not None:
                    previous.jpeg = b""
                existing[ref.participant] = ref
            selected = list(existing.values())[-3:]
            self.store._clear([r for r in existing.values() if r not in selected])
            self.store.active = [r for r in self.store.active if r.conversation_id != cid] + selected
            self.store.interrupt(cid, self.clock(), self.cfg.reid_ttl)
            self.store.trim(int(self.cfg.reid_max_refs))
        self.members.clear()
        self.ended, self.committed, self.phase = True, True, "idle"
        self.view_probe = self.view_ref = None
        self.emit({"type": "reid_view_clear"})

    def close(self):
        self.wipe("conn_close")
