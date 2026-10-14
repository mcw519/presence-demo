"""Pure-python PresenceTracker (no OpenCV dependency). Only numpy is required.

Coordinates are normalized [0,1] (x0, y0, x1, y1).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import presence_model as pm
from .body import Body, estimate_distance, orientation, shoulder_width
from .kalman import KalmanBox, appearance_similarity, hungarian


class State(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    MAYBE_AWAY = "MAYBE_AWAY"
    AWAY = "AWAY"
    ENDED = "ENDED"


@dataclass
class Config:
    """Tracker parameters. All coordinates are normalized; times in seconds. See docs/CONFIGURATION.md."""

    face_roi: tuple = (0.2, 0.05, 0.8, 0.95)  # face ROI (formerly `roi`), normalized
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
    present_timeout: float = 15.0  # silence timeout while participants are visible (balanced mode)
    empty_timeout: float = 5.0  # shortened timeout when ROI is empty
    maybe_away_frac: float = 0.5  # fraction of timeout to enter MAYBE_AWAY
    end_after_away: float = 5.0  # AWAY -> ENDED
    new_group_seconds: float = 3.0  # old group missing + new group confirmed this long
    priority_mode: str = "vision"  # vision | balanced | audio
    # --- performance / power (see presence_demo/optimizer.py) ---
    motion_gate: bool = True  # skip detection when the (downscaled) frame barely changed
    motion_threshold: float = 0.015  # mean abs gray diff (0..1) below which a frame is "static"
    force_detect_seconds: float = 3.0  # full detection at least this often even if static
    embed_throttle: bool = True  # reuse a face embedding when the face barely moved
    embed_interval: float = 3.0  # recompute each face embedding at least this often
    embed_move_thresh: float = 0.15  # center shift / size change (relative to box) forcing recompute
    adaptive_fps: bool = True  # server recommends client fps from state
    fps_low: float = 0.5  # recommended fps when stable
    fps_high: float = 2.0  # recommended fps when things change / MAYBE_AWAY
    roi_crop: bool = True  # crop ROI (+margin) and downscale before detection
    crop_margin: float = 0.1  # margin around ROI, fraction of frame
    max_width: int = 320  # max width of the image fed to the detector
    # --- pose / body (optional) ---
    pose_enabled: bool = False  # run the pose detector (MoveNet MultiPose Lightning)
    pose_conf: float = 0.4  # min pose/person confidence
    body_min_conf: float = 0.4  # confidence needed for a body-only (no face) detection to count as confident
    pose_roi: tuple = (0.0, 0.0, 1.0, 1.0)  # body ROI (center of body box), normalized
    roi_rule: str = "either"  # either | face | body : which ROI makes a person count
    turned_away_early: bool = True  # turned away + moving away -> early MAYBE_AWAY
    turned_away_seconds: float = 2.0  # how long the cue must persist
    recede_ratio: float = 0.1  # relative size shrink over ~2 s that counts as "moving away"
    # --- distance (optional) ---
    distance_enabled: bool = False
    camera_hfov: float = 60.0  # horizontal field of view, degrees
    face_width_m: float = 0.15  # average face width
    shoulder_width_m: float = 0.38  # average shoulder width
    distance_calib: float = 1.0  # calibration factor (see calibrate)
    distance_ema: float = 0.3  # weight of the newest estimate
    zone_min: float = 0.4  # conversation zone, meters
    zone_max: float = 2.0
    leave_distance: float = 2.5  # beyond this a visible person counts as left
    # --- tracking (Layer 4) ---
    tracker_mode: str = "legacy"  # legacy (greedy IoU+cosine) | kalman (Kalman + Hungarian + appearance)
    coast_seconds: float = 2.0  # kalman: unmatched tracks coast on the prediction this long
    coast_present_conf: float = 0.4  # kalman: a coasting track counts as present while vis_conf >= this
    gate_mahalanobis: float = 9.21  # kalman: chi-square gate on the predicted center (2 dof, 99%)
    appearance_min_sim: float = 0.6  # kalman: torso-histogram similarity below this => different person (no face)
    appearance_ema: float = 0.2  # kalman: appearance template update rate
    # --- presence model (Layer 2) ---
    presence_model: str = "rules"  # rules (counters + timeouts) | hmm (per-track Bayes filter)
    leave_halflife: float = 0.6  # hmm: with nobody seen, odds of "present" halve every this many seconds
    detect_gain: float = 4.0  # hmm: evidence rate per second for a detection (x confidence)
    lowconf_factor: float = 0.15  # hmm: low-confidence frames decay this fraction of the normal rate
    p_active: float = 0.8  # hmm: group P >= this -> ACTIVE/IDLE (from MAYBE_AWAY/AWAY)
    p_maybe: float = 0.6  # hmm: group P < this -> MAYBE_AWAY
    p_away: float = 0.3  # hmm: group P < this for away_hold_seconds -> AWAY
    away_hold_seconds: float = 1.0
    speech_join: str = "auto"  # auto (vision: off, else single_candidate) | off | single_candidate | all
    present_max_silence: float = 0.0  # vision mode: cap on silence with people present (0 = off)

    @property
    def roi(self) -> tuple:
        """Backward-compatible alias for :attr:`face_roi`."""
        return self.face_roi

    @roi.setter
    def roi(self, v: tuple) -> None:
        self.face_roi = tuple(v)

    def update(self, d: dict) -> list[str]:
        """Validate + clamp; unknown keys / bad values are skipped. Returns warnings."""
        warnings = []
        d = dict(d or {})
        if "roi" in d:  # migrate legacy key
            v = d.pop("roi")
            d.setdefault("face_roi", v)
        for k, v in d.items():
            if k not in RANGES:
                warnings.append(f"unknown key {k}")
                continue
            lo, hi, typ = RANGES[k]
            try:
                if typ is str:
                    if v not in lo:
                        raise ValueError
                    setattr(self, k, v)
                    continue
                if typ is bool:
                    if isinstance(v, str):
                        if v.lower() not in ("true", "false", "1", "0", "on", "off"):
                            raise ValueError
                        v = v.lower() in ("true", "1", "on")
                    setattr(self, k, bool(v))
                    continue
                if typ is tuple:
                    vals = [min(hi, max(lo, float(x))) for x in v]
                    if len(vals) != 4:
                        raise ValueError
                    x0, y0, x1, y1 = vals
                    x0, x1 = sorted((x0, x1))
                    y0, y1 = sorted((y0, y1))
                    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                        warnings.append(f"{k} too small; ignored")
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
        if self.zone_min > self.zone_max:
            self.zone_min = self.zone_max
            warnings.append("zone_min lowered to zone_max")
        if self.fps_low > self.fps_high:
            self.fps_low = self.fps_high
            warnings.append("fps_low lowered to fps_high")
        if not (self.p_away <= self.p_maybe <= self.p_active):
            self.p_away = min(self.p_away, self.p_maybe, self.p_active)
            self.p_active = max(self.p_active, self.p_maybe)
            warnings.append("presence thresholds reordered: p_away <= p_maybe <= p_active")
        if self.cos_weak > self.cos_match:
            self.cos_weak = self.cos_match
            warnings.append("cos_weak lowered to cos_match")
        return warnings


# (min, max, type) for every Config field
RANGES = {
    "face_roi": (0.0, 1.0, tuple),
    "pose_roi": (0.0, 1.0, tuple),
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
    "roi_rule": (("either", "face", "body"), None, str),
    "pose_enabled": (False, True, bool),
    "pose_conf": (0.05, 1.0, float),
    "body_min_conf": (0.05, 1.0, float),
    "tracker_mode": (("legacy", "kalman"), None, str),
    "coast_seconds": (0.0, 10.0, float),
    "coast_present_conf": (0.0, 1.0, float),
    "gate_mahalanobis": (1.0, 100.0, float),
    "appearance_min_sim": (0.0, 1.0, float),
    "appearance_ema": (0.0, 1.0, float),
    "presence_model": (("rules", "hmm"), None, str),
    "leave_halflife": (0.1, 30.0, float),
    "detect_gain": (0.1, 20.0, float),
    "lowconf_factor": (0.0, 1.0, float),
    "p_active": (0.05, 0.99, float),
    "p_maybe": (0.02, 0.98, float),
    "p_away": (0.01, 0.95, float),
    "away_hold_seconds": (0.0, 30.0, float),
    "speech_join": (("auto", "off", "single_candidate", "all"), None, str),
    "turned_away_early": (False, True, bool),
    "turned_away_seconds": (0.0, 30.0, float),
    "recede_ratio": (0.01, 0.9, float),
    "distance_enabled": (False, True, bool),
    "camera_hfov": (20.0, 170.0, float),
    "face_width_m": (0.08, 0.3, float),
    "shoulder_width_m": (0.2, 0.6, float),
    "distance_calib": (0.2, 5.0, float),
    "distance_ema": (0.01, 1.0, float),
    "zone_min": (0.0, 10.0, float),
    "zone_max": (0.1, 20.0, float),
    "leave_distance": (0.2, 30.0, float),
    "motion_gate": (False, True, bool),
    "motion_threshold": (0.0, 0.2, float),
    "force_detect_seconds": (0.5, 30.0, float),
    "embed_throttle": (False, True, bool),
    "embed_interval": (0.0, 30.0, float),
    "embed_move_thresh": (0.0, 1.0, float),
    "adaptive_fps": (False, True, bool),
    "fps_low": (0.2, 5.0, float),
    "fps_high": (0.2, 5.0, float),
    "roi_crop": (False, True, bool),
    "crop_margin": (0.0, 0.5, float),
    "max_width": (96, 1920, int),
}


@dataclass
class Detection:
    """One person in a frame. ``bbox`` is the track box (face box, or head box estimated from the body)."""

    bbox: tuple  # normalized x0,y0,x1,y1
    conf: float
    embedding: np.ndarray | None = None
    face_bbox: tuple | None = None
    body: Body | None = None
    appearance: np.ndarray | None = None  # torso HSV histogram (kalman mode)

    def __post_init__(self):
        if self.face_bbox is None and self.body is None:
            self.face_bbox = self.bbox  # plain face detection (legacy constructor)


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
    has_face: bool = True
    body: Body | None = None
    facing: str | None = None  # facing | side | away (pose only)
    facing_score: float = 0.0
    distance: float | None = None  # meters (EMA, calibrated)
    dist_raw: float | None = None  # meters with calib = 1 (for calibration)
    trend: str = "steady"  # approaching | receding | steady
    far: bool = False  # beyond leave_distance
    size_hist: deque = field(default_factory=lambda: deque(maxlen=12))
    visible_kind: str = "detected"  # detected | predicted | none
    vis_conf: float = 1.0  # 1 when detected; decays while coasting (kalman)
    appearance: np.ndarray | None = None
    kf: KalmanBox | None = None
    logodds: float = 0.0  # hmm: log P(present)/P(left)

    @property
    def p(self) -> float:
        return pm.sigmoid(self.logodds)


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
        self._low_streak = 0
        self._first_empty_at: float | None = None
        self._frame_confident = False
        self.empty_since: float | None = now
        self._newgroup_since: float | None = None
        self._away_cue_since: float | None = None
        self._far_logged = False
        self._hmm_t = now
        self._below_away_since: float | None = None
        self.p_group: float | None = None
        self.p_tracks: float | None = None
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
    @staticmethod
    def _center_in(bbox, r) -> bool:
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]

    def in_roi(self, bbox) -> bool:
        """Face-ROI test for a bare bbox (legacy helper)."""
        return self._center_in(bbox, self.cfg.face_roi)

    def det_in_roi(self, d: Detection) -> bool:
        """Apply ``roi_rule``: face center in face_roi and/or body center in pose_roi."""
        c = self.cfg
        face_ok = d.face_bbox is not None and self._center_in(d.face_bbox, c.face_roi)
        body_ok = d.body is not None and self._center_in(d.body.bbox, c.pose_roi)
        if c.roi_rule == "face" or not c.pose_enabled:
            return face_ok
        if c.roi_rule == "body":
            return body_ok
        return face_ok or body_ok

    def set_speech(self, speaking: bool, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        self.speaking = bool(speaking)
        if speaking:
            self.last_speech = now
            if self.state == State.ENDED and self.cfg.priority_mode != "vision":
                self._start(now, "speech detected")
        self._evaluate(now)
        return self.snapshot(now)

    def update(self, detections: list[Detection], speaking: bool | None = None, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        if speaking is not None:
            self.speaking = bool(speaking)
            if speaking:
                self.last_speech = now
        dets = [d for d in detections if self.det_in_roi(d)]
        for d in dets:
            d.embedding = _norm(d.embedding)
        confident = [d for d in dets if d.conf >= self._min_conf(d)]
        low = [d for d in dets if d.conf < self._min_conf(d)]
        starters = [d for d in confident if not self._det_far(d)]

        if self.state == State.ENDED and (starters or (self.speaking and self.cfg.priority_mode != "vision")):
            self._start(now, "person in ROI" if starters else "speech detected")

        if self.cfg.tracker_mode == "kalman":
            self._match_kalman(confident, now)
        else:
            self._match(confident, now)
        near = any(t.visible and not t.far and t.visible_kind == "detected" for t in self.tracks.values())
        coasting = any(t.visible and not t.far and t.visible_kind == "predicted" for t in self.tracks.values())
        if confident and not near and not self._far_logged:
            self._event(now, "far", f"visible people beyond leave_distance {self.cfg.leave_distance:.1f} m")
        self._far_logged = bool(confident) and not near
        self._frame_confident = bool(confident)
        self._update_away_cue(now)
        if self.cfg.presence_model == "hmm":
            self._hmm_update(now, bool(low) and not confident)
        else:
            self._update_presence(now, near, bool(low), coasting)
        if self.state != State.ENDED:
            self._check_new_group(now)
        self._evaluate(now)
        return self.snapshot(now)

    def _min_conf(self, d: Detection) -> float:
        """Body-only detections (no face) use ``body_min_conf``; anything with a face uses ``min_conf``."""
        return self.cfg.body_min_conf if d.face_bbox is None and d.body is not None else self.cfg.min_conf

    def _det_far(self, d: Detection) -> bool:
        """Single-frame distance check for a detection (used before it has a track)."""
        c = self.cfg
        if not c.distance_enabled:
            return False
        fw = (d.face_bbox[2] - d.face_bbox[0]) if d.face_bbox is not None else None
        sw = shoulder_width(d.body) if d.body is not None else None
        facing = orientation(d.body)[1] if d.body is not None else ("facing" if fw else None)
        raw = estimate_distance(fw, sw, facing, c.camera_hfov, c.face_width_m, c.shoulder_width_m, 1.0)
        return raw is not None and raw * c.distance_calib > c.leave_distance

    def _speech_join_mode(self) -> str:
        m = self.cfg.speech_join
        if m == "auto":
            return "off" if self.cfg.priority_mode == "vision" else "single_candidate"
        return m

    def _speech_joinable(self, t: Track, cands: list[Track]) -> bool:
        """Whether speech lets track ``t`` join instantly (without waiting ``join_seconds``)."""
        if not self.speaking:
            return False
        mode = self._speech_join_mode()
        if mode == "all":
            return True
        if mode != "single_candidate" or len(cands) != 1:
            return False
        c = self.cfg
        face_ok = t.has_face and self._center_in(t.bbox, c.face_roi)
        near_ok = not c.distance_enabled or t.distance is None or t.distance <= c.zone_max
        return face_ok and t.facing == "facing" and near_ok

    # ------------------------------------------------------------------ tracking
    def _match(self, dets, now):
        c = self.cfg
        for t in self.tracks.values():
            t.visible, t.visible_kind, t.vis_conf = False, "none", 0.0
        pairs = []
        for di, d in enumerate(dets):
            for t in self.tracks.values():
                cs = cosine(d.embedding, t.embedding)
                ov = iou(d.bbox, t.bbox)
                if d.body is not None and t.body is not None and (d.embedding is None or t.embedding is None):
                    ov = max(ov, iou(d.body.bbox, t.body.bbox))  # identity continuity via body when no face
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
            self._apply_match(self.tracks[tid], dets[di], now, cs)
        self._new_tracks([d for di, d in enumerate(dets) if di not in used_d], now)
        self._join_and_drop(now)

    def _apply_match(self, t: Track, d: Detection, now: float, cs: float) -> None:
        c = self.cfg
        if not t.participant and now - t.last_seen > 2.5:  # gap -> restart join timer
            t.in_roi_since = now
        t.bbox, t.conf, t.last_seen, t.visible, t.similarity = d.bbox, d.conf, now, True, cs
        t.visible_kind, t.vis_conf = "detected", 1.0
        self._update_body(t, d, now)
        if d.embedding is not None:
            t.embedding = (
                d.embedding
                if t.embedding is None
                else _norm((1 - c.embed_ema) * t.embedding + c.embed_ema * d.embedding)
            )
        if d.appearance is not None:
            if t.appearance is None:
                t.appearance = d.appearance
            else:
                a = (1 - c.appearance_ema) * t.appearance + c.appearance_ema * d.appearance
                t.appearance = a / max(float(a.sum()), 1e-9)

    def _new_tracks(self, dets, now) -> None:
        for d in dets:
            t = Track(self._next_id, d.bbox, d.embedding, d.conf, now, now, now, appearance=d.appearance)
            if self.cfg.tracker_mode == "kalman":
                t.kf = KalmanBox(d.bbox, now)
            self._update_body(t, d, now)
            self._next_id += 1
            self.tracks[t.id] = t
            self._event(now, "track_new", "new face in ROI", track=t.id)

    def _match_kalman(self, dets, now):
        """Kalman predict -> gated cost (IoU, motion, face cosine, torso appearance) -> Hungarian -> coast."""
        c = self.cfg
        tracks = list(self.tracks.values())
        for t in tracks:
            if t.kf is None:
                t.kf = KalmanBox(t.bbox, t.last_seen)
            t.kf.predict_to(now)
        pred = [t.kf.bbox() for t in tracks]
        BIG = 1e6
        cost = np.full((len(dets), len(tracks)), BIG)
        cos_of = {}
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                ov = iou(d.bbox, pred[j])
                if d.body is not None and t.body is not None:
                    ov = max(ov, iou(d.body.bbox, t.body.bbox))
                m2 = t.kf.mahalanobis(d.bbox)
                cs = cosine(d.embedding, t.embedding) if d.embedding is not None and t.embedding is not None else None
                ap = appearance_similarity(d.appearance, t.appearance)
                strong_face = cs is not None and cs >= c.cos_match
                if m2 > c.gate_mahalanobis and ov < c.iou_match and not strong_face:
                    continue  # motion gate (a confident face match may re-identify anywhere)
                if cs is not None and cs < c.cos_weak:
                    continue  # faces say: different people
                if cs is None and ap is not None and ap < c.appearance_min_sim:
                    continue  # no face: clothing says different people
                comps = [(1.0, ov), (1.0, float(np.exp(-m2 / c.gate_mahalanobis)))]
                if cs is not None:
                    comps.append((2.0, min(1.0, max(0.0, cs) / max(c.cos_match, 1e-6))))
                if ap is not None:
                    comps.append((1.5, min(1.0, max(0.0, (ap - 0.5) / 0.5))))
                score = sum(w * v for w, v in comps) / sum(w for w, _ in comps)
                if score >= 0.3:
                    cost[i, j] = 1.0 - score
                    cos_of[i, j] = 0.0 if cs is None else cs
        used_d, used_t = set(), set()
        for i, j in hungarian(cost) if len(dets) and len(tracks) else []:
            if cost[i, j] >= BIG:
                continue
            used_d.add(i)
            used_t.add(j)
            t = tracks[j]
            t.kf.update(dets[i].bbox)
            self._apply_match(t, dets[i], now, cos_of[i, j])
        for j, t in enumerate(tracks):
            if j in used_t:
                continue
            missing = now - t.last_seen
            if c.coast_seconds > 0 and missing <= c.coast_seconds:
                t.bbox = pred[j]
                t.visible_kind = "predicted"
                t.vis_conf = max(0.0, 1.0 - missing / c.coast_seconds)
                t.visible = t.vis_conf >= c.coast_present_conf
            else:
                t.visible, t.visible_kind, t.vis_conf = False, "none", 0.0
        self._new_tracks([d for i, d in enumerate(dets) if i not in used_d], now)
        self._join_and_drop(now)

    def _join_and_drop(self, now):
        c = self.cfg
        cands = [
            t
            for t in self.tracks.values()
            if t.visible and t.visible_kind == "detected" and not t.participant and not t.far
        ]
        for t in list(self.tracks.values()):
            by_speech = t in cands and self._speech_joinable(t, cands)
            if (
                t.visible
                and t.visible_kind == "detected"
                and not t.participant
                and not t.far
                and (now - t.in_roi_since >= c.join_seconds or by_speech)
                and self._newgroup_since is None
            ) and (not self._old_group_missing_rules() or not self._participants()):  # join rule unchanged
                t.participant = True
                self._event(
                    now,
                    "join",
                    "in ROI %.1fs" % (now - t.in_roi_since) if not by_speech else "speaking while in ROI",
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

    def _update_body(self, t: Track, d: Detection, now: float) -> None:
        """Pose, orientation, size trend and distance for a matched/new track."""
        c = self.cfg
        t.has_face = d.face_bbox is not None
        t.body = d.body
        if d.body is not None:
            t.facing_score, t.facing = orientation(d.body)
        else:
            t.facing_score, t.facing = (1.0, "facing") if t.has_face else (0.0, None)
        sw = shoulder_width(d.body) if d.body is not None else None
        fw = (d.face_bbox[2] - d.face_bbox[0]) if d.face_bbox is not None else None
        # size for the trend: body height if available, else face width
        size = (d.body.bbox[3] - d.body.bbox[1]) if d.body is not None else fw
        if size:
            t.size_hist.append((now, size))
            old = [sz for ts, sz in t.size_hist if now - ts >= 1.5]
            if old:
                ratio = size / old[-1]
                t.trend = (
                    "receding"
                    if ratio < 1 - c.recede_ratio
                    else "approaching"
                    if ratio > 1 + c.recede_ratio
                    else "steady"
                )
        raw = estimate_distance(fw, sw, t.facing, c.camera_hfov, c.face_width_m, c.shoulder_width_m, 1.0)
        if raw is not None:
            t.dist_raw = raw
            dist = raw * c.distance_calib
            t.distance = dist if t.distance is None else (1 - c.distance_ema) * t.distance + c.distance_ema * dist
        t.far = bool(c.distance_enabled and t.distance is not None and t.distance > c.leave_distance)

    def _update_away_cue(self, now: float) -> None:
        """All visible participants turned away (side/away) and moving away -> start the early cue timer."""
        vis = [t for t in self._participants() if t.visible and not t.far]
        cue = (
            self.cfg.turned_away_early
            and bool(vis)
            and all(t.facing in ("side", "away") and t.trend == "receding" for t in vis)
        )
        if cue and self._away_cue_since is None:
            self._away_cue_since = now
            self._event(now, "turned_away", "participants turned away and moving away")
        elif not cue:
            self._away_cue_since = None

    def zone(self, t: Track) -> str | None:
        """Conversation-zone label for a track (distance mode only)."""
        c = self.cfg
        if not c.distance_enabled or t.distance is None:
            return None
        if t.distance > c.leave_distance:
            return "left"
        if t.distance < c.zone_min:
            return "near"
        return "in" if t.distance <= c.zone_max else "far"

    def _participants(self):
        return [t for t in self.tracks.values() if t.participant]

    def _old_group_missing(self):
        if self.cfg.presence_model == "hmm":
            p = self._participants()
            return bool(p) and all(t.p < self.cfg.p_away for t in p)
        return self._old_group_missing_rules()

    def _old_group_missing_rules(self):
        """All participants confidently gone: none visible (and near), and this frame was not merely
        low-confidence (a confident detection or a confirmed-empty ROI is required)."""
        p = self._participants()
        evidence = self._frame_confident or (self.empty and self.empty_since is not None)
        return bool(p) and evidence and not any(t.visible and not t.far for t in p)

    def _check_new_group(self, now):
        c = self.cfg
        newcomers = [
            t
            for t in self.tracks.values()
            if not t.participant
            and t.visible
            and t.visible_kind == "detected"
            and not t.far
            and now - t.in_roi_since >= c.join_seconds
            and (c.presence_model != "hmm" or t.p >= c.p_active)
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

    def _update_presence(self, now, confident: bool, low: bool, coasting: bool = False):
        """``coasting``: a predicted (kalman) track still above ``coast_present_conf``. It keeps presence
        and holds off the 'empty' decision; the empty clock starts when coasting ends."""
        c = self.cfg
        if not (low and not confident):
            self._low_streak = 0
        if confident or low:
            self._first_empty_at = None
        if confident:
            self._present_streak += 1
            self._empty_streak = 0
        elif low:
            # low confidence: neither confirms presence nor absence (down-weighted)
            self._present_streak = 0
            self._low_streak += 1
            if self._low_streak >= c.absent_frames and (self.present or self.empty):
                self._event(now, "uncertain", "%d low-confidence frames -> audio-only timeout" % self._low_streak)
                self.present, self.empty, self.empty_since = False, False, None
        else:
            if self._first_empty_at is None and not coasting:
                self._first_empty_at = now
            self._empty_streak += 1
            self._present_streak = 0
        if self._present_streak >= c.present_frames or (confident and self.present):
            if not self.present:
                self._event(now, "presence", "%d confident frames" % self._present_streak)
            self.present, self.empty, self.empty_since = True, False, None
        if self._empty_streak >= c.absent_frames and not coasting:
            if not self.empty:
                self._event(now, "empty", "%d consecutive empty frames" % self._empty_streak)
                self.empty_since = self._first_empty_at if self._first_empty_at is not None else now
            self.present, self.empty = False, True
        if low and not confident and not self.present:
            self.empty = False  # uncertain -> fall back to audio-only base timeout

    # ------------------------------------------------------------------ presence model (hmm)
    def _hmm_update(self, now: float, low_frame: bool) -> None:
        """Integrate this frame's evidence into every track's log-odds; derive present/empty flags."""
        c = self.cfg
        dt = now - self._hmm_t
        self._hmm_t = now
        for t in self.tracks.values():
            turned = c.turned_away_early and t.facing in ("side", "away") and t.trend == "receding"
            rate = pm.track_rate(
                t.visible_kind,
                t.conf,
                t.vis_conf,
                halflife=c.leave_halflife,
                detect_gain=c.detect_gain,
                lowconf_factor=c.lowconf_factor,
                low_frame=low_frame,
                turned_away=turned,
                far=t.far,
            )
            if t.first_seen == now:  # new track: start from the neutral prior + this frame's evidence
                t.logodds = pm.step(0.0, rate, 0.5)
            else:
                t.logodds = pm.step(t.logodds, rate, dt)
        p = self._track_group_p()
        self.p_tracks = p
        was_empty = self.empty
        self.present = p >= c.p_active or (self.present and p >= c.p_maybe)
        self.empty = p < c.p_away
        if self.empty and not was_empty:
            self.empty_since = now
        elif not self.empty:
            self.empty_since = None

    def _track_group_p(self) -> float:
        """Group presence from tracks: max P over participants (else over all tracks). With no track at all
        (e.g. a session started by speech) it is unknown (0.5), except in vision mode (nobody seen)."""
        c = self.cfg
        ts = self._participants() or list(self.tracks.values())
        unknown = pm.P_MIN if c.priority_mode == "vision" else 0.5
        p = max((t.p for t in ts), default=unknown)
        if self._participants() and p < c.p_away:
            # old group gone but a stable new candidate is here: a NEW_GROUP is forming, don't flap to AWAY
            now = self._hmm_t
            forming = [
                t.p
                for t in self.tracks.values()
                if not t.participant
                and t.visible_kind == "detected"
                and not t.far
                and t.p >= c.p_active
                and now - t.in_roi_since >= c.join_seconds
            ]
            p = max([p, *forming])
        return p

    def _evaluate_hmm(self, now, silence):
        c = self.cfg
        p_tr = self.p_tracks if self.p_tracks is not None else self._track_group_p()
        a = pm.audio_offset(c.priority_mode, self.speaking, silence, c.base_timeout)
        pg = pm.sigmoid(pm.logit(p_tr) + a)
        self.p_group = pg
        self.path = "present" if self.present else "empty" if self.empty else "uncertain"
        self.timeout, self.elapsed = 0.0, 0.0
        if pg < c.p_away:
            if self._below_away_since is None:
                self._below_away_since = now
        else:
            self._below_away_since = None
        # vision: only the cameras can bring the conversation back (audio never does)
        p_ret = p_tr if c.priority_mode == "vision" else pg
        if self.state == State.AWAY:
            if p_ret >= c.p_active:
                self._set_state(now, State.ACTIVE, "user returned (P=%.2f)" % p_ret)
            elif now - self.state_since >= c.end_after_away:
                self._end(now, f"away for {c.end_after_away:.0f}s")
            return
        capped = c.priority_mode == "vision" and c.present_max_silence > 0 and silence >= c.present_max_silence
        if capped:
            self._set_state(now, State.AWAY, "silence %.0fs reached present_max_silence" % silence)
        elif self._below_away_since is not None and now - self._below_away_since >= c.away_hold_seconds:
            self._set_state(now, State.AWAY, "P=%.2f < %.2f for %.1fs" % (pg, c.p_away, now - self._below_away_since))
        elif pg < c.p_maybe or (self.state == State.MAYBE_AWAY and pg < c.p_active):
            self._set_state(now, State.MAYBE_AWAY, "P=%.2f" % pg)
        elif silence < c.idle_after:
            self._set_state(now, State.ACTIVE, "speech")
        else:
            self._set_state(now, State.IDLE, f"silence {silence:.0f}s")

    # ------------------------------------------------------------------ fusion
    def _evaluate(self, now):
        c = self.cfg
        if self.state == State.ENDED:
            self.elapsed, self.path = 0.0, "ended"
            return
        if c.presence_model == "hmm":
            self._evaluate_hmm(now, 0.0 if self.speaking else now - self.last_speech)
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
        # vision mode: only a confident person brings the conversation back (speech alone never does)
        returned = self.present if mode == "vision" else (self.speaking or self.present)
        if self.state == State.AWAY:
            if returned:
                self._set_state(now, State.ACTIVE, "user returned")
            elif now - self.state_since >= c.end_after_away:
                self._end(now, f"away for {c.end_after_away:.0f}s")
            return
        early = (
            self._away_cue_since is not None
            and not self.speaking
            and now - self._away_cue_since >= c.turned_away_seconds
        )
        if self.timeout > 0 and elapsed >= self.timeout:
            self._set_state(now, State.AWAY, why % elapsed)
        elif early:
            self._set_state(now, State.MAYBE_AWAY, "turned away + moving away %.1fs" % (now - self._away_cue_since))
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
            "silence": 0.0 if self.speaking or self.state == State.ENDED else round(now - self.last_speech, 1),
            "path": getattr(self, "path", "ended"),
            "priority_mode": self.cfg.priority_mode,
            "present": self.present,
            "empty": self.empty,
            "timeout": self.timeout,
            "elapsed": round(getattr(self, "elapsed", 0.0), 1),
            "in_state": 0.0 if self.state == State.ENDED else round(now - self.state_since, 1),
            "roi": list(self.cfg.roi),  # legacy alias of face_roi
            "face_roi": list(self.cfg.face_roi),
            "pose_roi": list(self.cfg.pose_roi),
            "tracks": [
                {
                    "id": t.id,
                    "bbox": [round(x, 4) for x in t.bbox],
                    "participant": t.participant,
                    "visible": t.visible,
                    "similarity": round(t.similarity, 3),
                    "visible_kind": t.visible_kind,
                    "vis_conf": round(t.vis_conf, 2),
                    "p": round(t.p, 3) if self.cfg.presence_model == "hmm" else None,
                    "dwell_seconds": round(now - t.first_seen, 1),
                    "velocity": [round(v, 3) for v in t.kf.velocity] if t.kf is not None else None,
                    "conf": round(t.conf, 2),
                    "missing": round(now - t.last_seen, 1),
                    "has_face": t.has_face,
                    "body": [round(x, 4) for x in t.body.bbox] if t.body is not None else None,
                    "keypoints": {
                        k: [round(v[0], 4), round(v[1], 4), round(v[2], 2)] for k, v in t.body.keypoints.items()
                    }
                    if t.body is not None
                    else None,
                    "facing": t.facing,
                    "facing_score": t.facing_score,
                    "distance": round(t.distance, 2) if t.distance is not None and self.cfg.distance_enabled else None,
                    "trend": t.trend,
                    "far": t.far,
                    "zone": self.zone(t),
                }
                for t in self.tracks.values()
            ],
            "away_cue": self._away_cue_since is not None,
            "presence_model": self.cfg.presence_model,
            "p_group": None if self.p_group is None or self.state == State.ENDED else round(self.p_group, 3),
            "p_tracks": None if self.p_tracks is None or self.state == State.ENDED else round(self.p_tracks, 3),
            "log": self.log[-30:],
        }


def asdict_shallow(t: Track) -> dict:
    return {f: getattr(t, f) for f in t.__dataclass_fields__}
