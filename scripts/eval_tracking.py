"""Compare tracker_mode=legacy vs kalman on synthetic tracking scenarios.

PRIVACY: no images at all. Scenarios are generated in memory as detector outputs (boxes, noisy face
embeddings, torso colour histograms, body keypoints) with ground-truth person ids.

    uv run python scripts/eval_tracking.py            # markdown table
    uv run python scripts/eval_tracking.py --seeds 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presence_demo.body import Body  # noqa: E402
from presence_demo.tracker import Config, Detection, PresenceTracker  # noqa: E402

FACE_W, FACE_H = 0.08, 0.11


@dataclass
class Person:
    pid: int
    path: callable  # t -> (cx, cy) of the face, or None when out of the scene
    occluded: callable = lambda t: False  # in the scene but not detected
    no_face: callable = lambda t: False  # side/back view: body only
    emb: np.ndarray = None
    app: np.ndarray = None


@dataclass
class Scenario:
    name: str
    persons: list
    duration: float
    dt: callable = lambda t: 1.0  # frame interval at time t
    expect_new_people: bool = False  # NEW_GROUP / new session is legitimate
    leave_at: float | None = None  # all people leave at this time (for time-to-detect-leave)


def unit(v):
    return v / np.linalg.norm(v)


def make_identity(rng, p: Person):
    p.emb = unit(rng.normal(size=128))
    p.app = rng.dirichlet(np.full(48, 0.3))


def body_for(cx, cy, face):
    w = FACE_W * 3.2
    top = cy - FACE_H * 0.7
    kp = {
        "l_shoulder": (cx + w * 0.35, cy + FACE_H * 1.1, 0.9),
        "r_shoulder": (cx - w * 0.35, cy + FACE_H * 1.1, 0.9),
        "l_hip": (cx + w * 0.2, cy + FACE_H * 4.0, 0.8),
        "r_hip": (cx - w * 0.2, cy + FACE_H * 4.0, 0.8),
    }
    if face:
        kp.update(nose=(cx, cy, 0.9), l_eye=(cx + 0.015, cy - 0.01, 0.9), r_eye=(cx - 0.015, cy - 0.01, 0.9))
    else:  # side view: shoulders collapse
        kp["l_shoulder"], kp["r_shoulder"] = (cx - 0.01, cy + FACE_H * 1.1, 0.9), (cx + 0.01, cy + FACE_H * 1.1, 0.9)
    return Body((cx - w / 2, top, cx + w / 2, min(1.0, top + 0.75)), 0.8, kp)


def detections(scn: Scenario, t: float, rng) -> tuple[list[Detection], list[int], set[int]]:
    dets, gt, in_scene = [], [], set()
    for p in scn.persons:
        pos = p.path(t)
        if pos is None:
            continue
        in_scene.add(p.pid)
        if p.occluded(t):
            continue
        cx, cy = pos[0] + rng.normal(0, 0.004), pos[1] + rng.normal(0, 0.004)
        face = not p.no_face(t)
        body = body_for(cx, cy, face)
        app = 0.85 * p.app + 0.15 * rng.dirichlet(np.full(48, 0.3))
        if face:
            fb = (cx - FACE_W / 2, cy - FACE_H / 2, cx + FACE_W / 2, cy + FACE_H / 2)
            emb = unit(p.emb + rng.normal(size=128) * 0.06)
            dets.append(Detection(fb, 0.9, emb, face_bbox=fb, body=body, appearance=app))
        else:
            hb = (cx - FACE_W / 2, cy - FACE_H / 2, cx + FACE_W / 2, cy + FACE_H / 2)
            dets.append(Detection(hb, 0.75, None, face_bbox=None, body=body, appearance=app))
        gt.append(p.pid)
    return dets, gt, in_scene


def lerp(a, b, t0, t1):
    def f(t):
        if t < t0:
            return a
        if t > t1:
            return b
        k = (t - t0) / (t1 - t0)
        return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)

    return f


def window(*spans):
    return lambda t: any(a <= t < b for a, b in spans)


def gone_at(t_leave, xy=(0.5, 0.4)):
    return lambda t: xy if t < t_leave else None


OCC = window((6, 7), (13, 15), (21, 24))  # 1 s, 2 s, 3 s occlusions


def scenarios() -> list[Scenario]:
    still = lambda xy: lambda t: xy  # noqa: E731
    return [
        Scenario(
            "遮擋 1–3 s（1 fps）", [Person(1, still((0.5, 0.4)), occluded=window((6, 7), (13, 15), (21, 24)))], 30
        ),
        Scenario(
            "遮擋 1–3 s（2 fps）",
            [Person(1, still((0.5, 0.4)), occluded=OCC)],
            30,
            dt=lambda t: 0.5,
        ),
        Scenario(
            "兩人交錯走過",
            [
                Person(1, lerp((0.3, 0.4), (0.7, 0.4), 6, 14), occluded=window((9.5, 10.5))),
                Person(2, lerp((0.7, 0.42), (0.3, 0.42), 6, 14)),
            ],
            24,
        ),
        Scenario(
            "側身看不到臉",
            [
                Person(1, lerp((0.4, 0.4), (0.46, 0.4), 6, 14), no_face=window((6, 14))),
                Person(2, still((0.65, 0.4))),
            ],
            22,
        ),
        Scenario(
            "fps 變動 0.5–2",
            [Person(1, lambda t: (0.4 + 0.15 * np.sin(t / 4), 0.4))],
            30,
            dt=lambda t: 2.0 if int(t / 6) % 2 else 0.5,
        ),
        Scenario(
            "兩人側身互換位置",
            [
                Person(1, lerp((0.4, 0.4), (0.62, 0.4), 8, 11), no_face=window((7, 13))),
                Person(2, lerp((0.62, 0.4), (0.4, 0.4), 8, 11), no_face=window((7, 13))),
            ],
            20,
            dt=lambda t: 0.5,
        ),
        Scenario(
            "離開後新的人站到同一位置",
            [Person(1, lambda t: (0.5, 0.4) if t < 10 else None), Person(2, lambda t: (0.5, 0.4) if t >= 12 else None)],
            25,
            expect_new_people=True,
        ),
        Scenario(
            "離開（1 fps，量測判定時間）", [Person(1, lambda t: (0.5, 0.4) if t < 10 else None)], 25, leave_at=10.0
        ),
        Scenario(
            "離開（2 fps，量測判定時間）",
            [Person(1, gone_at(10))],
            25,
            dt=lambda t: 0.5,
            leave_at=10.0,
        ),
    ]


@dataclass
class Result:
    id_switch: int = 0
    frag: int = 0
    false_maybe: int = 0
    false_away: int = 0
    false_new_group: int = 0
    leave_s: list = field(default_factory=list)


def run(scn: Scenario, mode: str, seed: int) -> Result:
    rng = np.random.default_rng(seed)
    for p in scn.persons:
        make_identity(rng, p)
    clk = [0.0]
    tr = PresenceTracker(Config(tracker_mode=mode, pose_enabled=True), clock=lambda: clk[0])
    res = Result()
    owner: dict[int, int] = {}  # track id -> first gt that used it
    last: dict[int, int] = {}  # gt -> last track id
    t = 0.0
    while t < scn.duration:
        dets, gt, in_scene = detections(scn, t, rng)
        clk[0] = t
        n_log = len(tr.log)
        tr.update(dets)
        for d, g in zip(dets, gt, strict=True):
            tid = next(
                (k.id for k in tr.tracks.values() if k.visible_kind == "detected" and tuple(k.bbox) == tuple(d.bbox)),
                None,
            )
            if tid is None:
                continue
            prev = last.get(g)
            if prev is not None and prev != tid:
                if owner.get(tid, g) != g:
                    res.id_switch += 1
                else:
                    res.frag += 1
            elif prev is None and owner.get(tid, g) != g:
                res.id_switch += 1  # a new person inherited someone else's id
            owner.setdefault(tid, g)
            last[g] = tid
        # ticker between frames (0.5 s)
        step = scn.dt(t)
        k = 0.5
        while k < step - 1e-9:
            clk[0] = t + k
            tr.tick()
            k += 0.5
        for e in tr.log[n_log:]:
            if e["event"] == "transition" and in_scene_at(scn, e["t"]):
                res.false_maybe += e["to"] == "MAYBE_AWAY"
                res.false_away += e["to"] == "AWAY"
            if e["event"] == "NEW_GROUP" and not scn.expect_new_people:
                res.false_new_group += 1
            leave = scn.leave_at is not None and e["t"] >= scn.leave_at
            if leave and e["event"] == "transition" and e["to"] == "AWAY" and not res.leave_s:
                res.leave_s.append(e["t"] - scn.leave_at)
        t += step
    return res


def in_scene_at(scn: Scenario, t: float) -> bool:
    return any(p.path(t) is not None for p in scn.persons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    keys = ("id_switch", "frag", "false_maybe", "false_away", "false_new_group")
    print(f"每個情境跑 {args.seeds} 個亂數種子，數字是總和（離開判定時間取平均）。\n")
    print("| 情境 | 模式 | ID 互換 | 斷裂 | 誤判 MAYBE_AWAY | 誤判 AWAY | 誤判 NEW_GROUP | 離開判定 (s) |")
    print("|---|---|---|---|---|---|---|---|")
    tot = {m: dict.fromkeys(keys, 0) for m in ("legacy", "kalman")}
    for scn in scenarios():
        for mode in ("legacy", "kalman"):
            agg = dict.fromkeys(keys, 0)
            leave = []
            for s in range(args.seeds):
                r = run(scn, mode, s)
                for k in keys:
                    agg[k] += getattr(r, k)
                leave += r.leave_s
            for k in keys:
                tot[mode][k] += agg[k]
            lv = f"{np.mean(leave):.1f}" if leave else "—"
            print(f"| {scn.name} | {mode} | " + " | ".join(str(agg[k]) for k in keys) + f" | {lv} |")
    for mode in ("legacy", "kalman"):
        print(f"| **合計** | {mode} | " + " | ".join(f"**{tot[mode][k]}**" for k in keys) + " | |")


if __name__ == "__main__":
    main()
