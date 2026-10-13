"""Pure-python PresenceTracker (no OpenCV dependency). Only numpy is required.

Coordinates are normalized [0,1] (x0, y0, x1, y1).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class State(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    MAYBE_AWAY = "MAYBE_AWAY"
    AWAY = "AWAY"
    ENDED = "ENDED"


@dataclass
class Config:
    """Tracker parameters. All coordinates are normalized; times in seconds. See docs/CONFIGURATION.md."""

    roi: tuple = (0.2, 0.05, 0.8, 0.95)  # central box, normalized
    min_conf: float = 0.6  # detections below are "low confidence"
    iou_match: float = 0.3
    cos_match: float = 0.40  # cosine >= this => same person (SFace ~0.363)
    cos_weak: float = 0.20  # with good IoU, this weaker cosine is accepted
    embed_ema: float = 0.15  # template update rate
    join_seconds: float = 2.0  # continuous time in ROI to become participant
    grace_seconds: float = 15.0  # keep missing tracks this long
    absent_frames: int = 3  # consecutive empty frames before "empty"
    present_frames: int = 2  # consecutive confident frames before "present"
    idle_after: float = 3.0  # silence before ACTIVE -> IDLE
    base_timeout: float = 20.0  # silence timeout when vision is unknown
    present_timeout: float = 15.0  # extended timeout when participants visible
    empty_timeout: float = 5.0  # shortened timeout when ROI is empty
    maybe_away_frac: float = 0.5  # fraction of timeout to enter MAYBE_AWAY
    end_after_away: float = 5.0  # AWAY -> ENDED
    new_group_seconds: float = 3.0  # old group missing + new group confirmed this long
    priority_mode: str = "vision"  # vision | balanced | audio
    present_max_silence: float = 0.0  # vision mode: cap on silence with people present (0 = off)

    def update(self, d: dict) -> list[str]:
        """Validate + clamp; unknown keys / bad values are skipped. Returns warnings."""
        warnings = []
        for k, v in (d or {}).items():
            if k not in RANGES:
                warnings.append(f"unknown key {k}")
                continue
            lo, hi, typ = RANGES[k]
            try:
                if k == "priority_mode":
                    if v not in lo:
                        raise ValueError
                    setattr(self, k, v)
                    continue
                if k == "roi":
                    vals = [min(hi, max(lo, float(x))) for x in v]
                    if len(vals) != 4:
                        raise ValueError
                    x0, y0, x1, y1 = vals
                    x0, x1 = sorted((x0, x1))
                    y0, y1 = sorted((y0, y1))
                    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                        warnings.append("roi too small; ignored")
                        continue
                    new = (x0, y0, x1, y1)
                else:
                    raw = float(v)
                    if raw != raw:  # NaN
                        raise ValueError
                    new = typ(round(min(hi, max(lo, raw))) if typ is int else min(hi, max(lo, raw)))
                    if new != (typ(raw) if typ is float else round(raw)):
                        warnings.append(f"{k} clamped to {new}")
            except (TypeError, ValueError):
                warnings.append(f"invalid value for {k}")
                continue
            setattr(self, k, new)
        if self.cos_weak > self.cos_match:
            self.cos_weak = self.cos_match
            warnings.append("cos_weak lowered to cos_match")
        return warnings


# (min, max, type) for every Config field
RANGES = {
    "roi": (0.0, 1.0, tuple),
    "min_conf": (0.0, 1.0, float),
    "iou_match": (0.0, 1.0, float),
    "cos_match": (0.0, 1.0, float),
    "cos_weak": (0.0, 1.0, float),
    "embed_ema": (0.0, 1.0, float),
    "join_seconds": (0.0, 30.0, float),
    "grace_seconds": (0.0, 120.0, float),
    "absent_frames": (1, 20, int),
    "present_frames": (1, 20, int),
    "idle_after": (0.0, 60.0, float),
    "base_timeout": (1.0, 300.0, float),
    "present_timeout": (1.0, 600.0, float),
    "empty_timeout": (1.0, 120.0, float),
    "maybe_away_frac": (0.05, 0.95, float),
    "end_after_away": (0.0, 120.0, float),
    "new_group_seconds": (0.0, 60.0, float),
    "present_max_silence": (0.0, 3600.0, float),
    "priority_mode": (("vision", "balanced", "audio"), None, str),
}


@dataclass
class Detection:
    bbox: tuple  # normalized x0,y0,x1,y1
    conf: float
    embedding: np.ndarray | None = None


@dataclass
class Track:
    id: int
    bbox: tuple
    embedding: np.ndarray | None
    conf: float
    first_seen: float
    last_seen: float
    in_roi_since: float | None
    participant: bool = False
    visible: bool = True
    similarity: float = 0.0  # similarity of last match


def _norm(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


class PresenceTracker:
    """Fuses per-frame face detections with VAD speech state into a presence state machine.

    Feed it with :meth:`update` (one call per processed frame) and :meth:`set_speech`
    (on VAD changes); call :meth:`tick` periodically. The clock is injectable for tests.
    """

    def __init__(self, config: Config | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = config or Config()
        self.clock = clock
        self.log: list[dict] = []
        self._next_id = 1
        self._reset_session(self.clock(), initial=True)

    # ------------------------------------------------------------------ session
    def _reset_session(self, now, initial=False):
        self.tracks: dict[int, Track] = {}  # embeddings live only here (memory)
        self.state = State.ENDED if initial else State.ACTIVE
        self.state_since = now
        self.last_speech = now
        self.speaking = False
        self.present = False
        self.empty = True
        self._empty_streak = 0
        self._present_streak = 0
        self.empty_since: float | None = now
        self._newgroup_since: float | None = None
        self.timeout = self.cfg.base_timeout
        self.session = getattr(self, "session", 0) + (0 if initial else 1)

    def _event(self, now, kind, reason, **extra):
        e = {"t": round(now, 3), "event": kind, "reason": reason, **extra}
        self.log.append(e)
        self.log = self.log[-200:]

    def _set_state(self, now, new, reason):
        if new != self.state:
            self._event(now, "transition", reason, frm=self.state.value, to=new.value)
            self.state = new
            self.state_since = now

    def _end(self, now, reason):
        self._set_state(now, State.ENDED, reason)
        self.tracks.clear()  # drop embeddings
        self.present, self.empty = False, True

    def _start(self, now, reason):
        speaking = self.speaking
        self._reset_session(now)
        self.speaking = speaking
        self.state = State.ENDED
        self._set_state(now, State.ACTIVE, reason)

    # ------------------------------------------------------------------ inputs
    def in_roi(self, bbox) -> bool:
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        r = self.cfg.roi
        return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]

    def set_speech(self, speaking: bool, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        self.speaking = bool(speaking)
        if speaking:
            self.last_speech = now
            if self.state == State.ENDED and self.cfg.priority_mode != "vision":
                self._start(now, "speech detected")
        self._evaluate(now)
        return self.snapshot(now)

    def update(
        self, detections: list[Detection], speaking: bool | None = None, now: float | None = None
    ) -> dict:
        now = self.clock() if now is None else now
        if speaking is not None:
            self.speaking = bool(speaking)
            if speaking:
                self.last_speech = now
        dets = [d for d in detections if self.in_roi(d.bbox)]
        for d in dets:
            d.embedding = _norm(d.embedding)
        confident = [d for d in dets if d.conf >= self.cfg.min_conf]
        low = [d for d in dets if d.conf < self.cfg.min_conf]

        if self.state == State.ENDED and (
            confident or (self.speaking and self.cfg.priority_mode != "vision")
        ):
            self._start(now, "person in ROI" if confident else "speech detected")

        self._match(confident, now)
        self._update_presence(now, bool(confident), bool(low))
        if self.state != State.ENDED:
            self._check_new_group(now)
        self._evaluate(now)
        return self.snapshot(now)

    # ------------------------------------------------------------------ tracking
    def _match(self, dets, now):
        c = self.cfg
        for t in self.tracks.values():
            t.visible = False
        pairs = []
        for di, d in enumerate(dets):
            for t in self.tracks.values():
                cs = cosine(d.embedding, t.embedding)
                ov = iou(d.bbox, t.bbox)
                if (
                    cs >= c.cos_match
                    or (ov >= c.iou_match and cs >= c.cos_weak)
                    or (ov >= c.iou_match and (d.embedding is None or t.embedding is None))
                ):
                    pairs.append((cs + 0.5 * ov, di, t.id, cs))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for _, di, tid, cs in pairs:  # greedy assignment
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            d, t = dets[di], self.tracks[tid]
            if not t.participant and now - t.last_seen > 2.5:  # gap -> restart join timer
                t.in_roi_since = now
            t.bbox, t.conf, t.last_seen, t.visible, t.similarity = d.bbox, d.conf, now, True, cs
            if d.embedding is not None:
                t.embedding = (
                    d.embedding
                    if t.embedding is None
                    else _norm((1 - c.embed_ema) * t.embedding + c.embed_ema * d.embedding)
                )
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            t = Track(self._next_id, d.bbox, d.embedding, d.conf, now, now, now)
            self._next_id += 1
            self.tracks[t.id] = t
            self._event(now, "track_new", "new face in ROI", track=t.id)
        for t in list(self.tracks.values()):
            if (
                t.visible
                and not t.participant
                and (now - t.in_roi_since >= c.join_seconds or self.speaking)
                and self._newgroup_since is None
            ) and (not self._old_group_missing() or not self._participants()):
                t.participant = True
                self._event(
                    now,
                    "join",
                    "in ROI %.1fs" % (now - t.in_roi_since) if not self.speaking else "speaking while in ROI",
                    track=t.id,
                )
            if now - t.last_seen > c.grace_seconds:
                del self.tracks[t.id]
                self._event(
                    now,
                    "track_drop",
                    f"missing > grace {c.grace_seconds:.0f}s",
                    track=t.id,
                    participant=t.participant,
                )

    def _participants(self):
        return [t for t in self.tracks.values() if t.participant]

    def _old_group_missing(self):
        p = self._participants()
        return bool(p) and not any(t.visible for t in p)

    def _check_new_group(self, now):
        c = self.cfg
        newcomers = [
            t
            for t in self.tracks.values()
            if not t.participant and t.visible and now - t.in_roi_since >= c.join_seconds
        ]
        if self._old_group_missing() and newcomers:
            if self._newgroup_since is None:
                self._newgroup_since = now
            if now - self._newgroup_since >= c.new_group_seconds:
                ids = [t.id for t in newcomers]
                self._event(now, "NEW_GROUP", f"old participants gone, unmatched faces {ids} took over")
                keep = [Track(**{**asdict_shallow(t), "participant": True}) for t in newcomers]
                self._end(now, "new group replaced old group")
                self._start(now, "new group")
                for t in keep:
                    self.tracks[t.id] = t
                self.present, self.empty = True, False
        else:
            self._newgroup_since = None

    def _update_presence(self, now, confident: bool, low: bool):
        c = self.cfg
        if not (low and not confident):
            self._low_streak = 0
        if confident:
            self._present_streak += 1
            self._empty_streak = 0
        elif low:
            # low confidence: neither confirms presence nor absence (down-weighted)
            self._present_streak = 0
            self._low_streak = getattr(self, "_low_streak", 0) + 1
            if self._low_streak >= c.absent_frames and (self.present or self.empty):
                self._event(
                    now, "uncertain", "%d low-confidence frames -> audio-only timeout" % self._low_streak
                )
                self.present, self.empty, self.empty_since = False, False, None
        else:
            self._empty_streak += 1
            self._present_streak = 0
        if self._present_streak >= c.present_frames or (confident and self.present):
            if not self.present:
                self._event(now, "presence", "%d confident frames" % self._present_streak)
            self.present, self.empty, self.empty_since = True, False, None
        if self._empty_streak >= c.absent_frames:
            if not self.empty:
                self._event(now, "empty", "%d consecutive empty frames" % self._empty_streak)
                self.empty_since = now - (self._empty_streak - 1) * 1.0
            self.present, self.empty = False, True
        if low and not confident and not self.present:
            self.empty = False  # uncertain -> fall back to audio-only base timeout

    # ------------------------------------------------------------------ fusion
    def _evaluate(self, now):
        c = self.cfg
        if self.state == State.ENDED:
            self.elapsed, self.path = 0.0, "ended"
            return
        silence = 0.0 if self.speaking else now - self.last_speech
        mode = c.priority_mode
        empty_conf = self.empty and self.empty_since is not None
        self.path = "present" if self.present else "empty" if empty_conf else "uncertain"
        vision_led = mode == "vision" and self.path != "uncertain"
        if mode == "vision" and self.present:
            # confident person in ROI: silence alone never ends the conversation (optional cap)
            self.timeout, elapsed = c.present_max_silence, silence
            why = "silence %.0fs reached present_max_silence with people present"
        elif mode == "vision" and empty_conf:
            # confidently empty ROI: leave after empty_timeout regardless of audio
            self.timeout, elapsed = c.empty_timeout, now - self.empty_since
            why = "ROI empty %.0fs (vision priority, audio ignored)"
        elif mode == "audio":
            k = 1.5 if self.present else 0.5 if empty_conf else 1.0
            self.timeout, elapsed = max(1.0, c.base_timeout * k), silence
            why = "silence %.0fs (audio priority, vision-adjusted)"
        elif self.present:
            self.timeout, elapsed, why = c.present_timeout, silence, "silence %.0fs with people present"
        elif empty_conf:
            self.timeout = c.empty_timeout
            elapsed = min(silence, now - self.empty_since)
            why = "silence + empty ROI %.0fs"
        else:
            self.timeout, elapsed, why = c.base_timeout, silence, "silence %.0fs (vision uncertain)"
        self.elapsed = elapsed
        returned = self.present if vision_led else (self.speaking or self.present)
        if self.state == State.AWAY:
            if returned:
                self._set_state(now, State.ACTIVE, "user returned")
            elif now - self.state_since >= c.end_after_away:
                self._end(now, f"away for {c.end_after_away:.0f}s")
            return
        if self.timeout > 0 and elapsed >= self.timeout:
            self._set_state(now, State.AWAY, why % elapsed)
        elif (
            self.timeout > 0
            and elapsed >= c.maybe_away_frac * self.timeout
            and not (vision_led and self.path == "present" and silence < c.idle_after)
        ):
            self._set_state(now, State.MAYBE_AWAY, why % elapsed)
        elif silence < c.idle_after:
            self._set_state(now, State.ACTIVE, "speech")
        else:
            self._set_state(now, State.IDLE, f"silence {silence:.0f}s")

    def tick(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        self._evaluate(now)
        return self.snapshot(now)

    def snapshot(self, now=None) -> dict:
        now = self.clock() if now is None else now
        return {
            "state": self.state.value,
            "session": self.session,
            "speaking": self.speaking,
            "silence": 0.0
            if self.speaking or self.state == State.ENDED
            else round(now - self.last_speech, 1),
            "path": getattr(self, "path", "ended"),
            "priority_mode": self.cfg.priority_mode,
            "present": self.present,
            "empty": self.empty,
            "timeout": self.timeout,
            "elapsed": round(getattr(self, "elapsed", 0.0), 1),
            "in_state": 0.0 if self.state == State.ENDED else round(now - self.state_since, 1),
            "roi": list(self.cfg.roi),
            "tracks": [
                {
                    "id": t.id,
                    "bbox": [round(x, 4) for x in t.bbox],
                    "participant": t.participant,
                    "visible": t.visible,
                    "similarity": round(t.similarity, 3),
                    "conf": round(t.conf, 2),
                    "missing": round(now - t.last_seen, 1),
                }
                for t in self.tracks.values()
            ],
            "log": self.log[-30:],
        }


def asdict_shallow(t: Track) -> dict:
    return {f: getattr(t, f) for f in t.__dataclass_fields__}
