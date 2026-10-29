"""Step 6: who is who.

Each of the four players is followed with torso colour (dark / light / skin), leg colour,
the side of the net and the last position. Position alone swaps names when partners
switch sides, so colour and the net side carry most of the weight.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court

NEAR, FAR = 0, 1


def _region_lab(lab, box, fx, fy):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    a = lab[int(y0 + fy[0] * h):int(y0 + fy[1] * h), int(x0 + fx[0] * w):int(x0 + fx[1] * w)]
    if a.size == 0:
        return np.array([np.nan] * 3)
    return np.median(a.reshape(-1, 3), axis=0).astype(float)


def appearance(frame_lab, box):
    torso = _region_lab(frame_lab, box, (0.3, 0.7), (0.18, 0.42))
    legs = _region_lab(frame_lab, box, (0.3, 0.7), (0.58, 0.82))
    return np.r_[torso, legs]


def shirt_class(torso_lab):
    L, a, b = torso_lab
    # skin: warm and mid-bright
    if 80 < L < 210 and a > 135 and b > 138:
        return "skin"
    return "dark" if L < 110 else "light"


def detections(people, Hs, frames):
    """Per frame: list of dicts with box, court position and appearance."""
    out = []
    h, w = frames[0].shape[:2]
    for t, rows in enumerate(people):
        lab = cv2.cvtColor(frames[t], cv2.COLOR_BGR2LAB)
        Hi = np.linalg.inv(Hs[t])
        dets = []
        for r in rows:
            tid, x0, y0, x1, y1, conf = r
            bh = y1 - y0
            if bh < 30:
                continue
            # feet cut off by the bottom of the picture: no position on the floor
            if y1 > h - 4:
                continue
            foot = court.apply_h(Hi, [((x0 + x1) / 2, y1)])[0]
            if not np.isfinite(foot).all():
                continue
            if not (-1.2 < foot[0] < court.COURT_W + 1.2 and -2.0 < foot[1] < court.COURT_L + 2.0):
                continue
            # feet hidden behind someone nearer the camera (their box covers our feet)
            occluded = any(min(o[3], x1) - max(o[1], x0) > 0.3 * (x1 - x0)
                           and o[2] < y1 - 2 and o[4] > y1 + 10 for o in rows)
            dets.append({"tid": int(tid), "box": (x0, y0, x1, y1), "conf": float(conf),
                         "pos": foot, "app": appearance(lab, (x0, y0, x1, y1)),
                         "occluded": occluded, "Hi": Hi})
        out.append(dets)
    return out


def _app_dist(a, b):
    d = a - b
    if not np.isfinite(d).all():
        return 1.0
    # torso counts double; lightness matters more than hue for dark/light shirts
    wts = np.array([1.0, 0.7, 0.7, 0.5, 0.35, 0.35])
    return float(np.sqrt(np.sum((d * wts) ** 2)) / 40.0)


def _init_slots(dets, n_look=45):
    """Pick two players per side from the first frames that have them."""
    for t in range(min(n_look, len(dets))):
        near = [d for d in dets[t] if 0.0 < d["pos"][1] < court.NET_Y and -0.5 < d["pos"][0] < court.COURT_W + 0.5]
        far = [d for d in dets[t] if court.NET_Y < d["pos"][1] < court.COURT_L and -0.5 < d["pos"][0] < court.COURT_W + 0.5]
        if len(near) >= 2 and len(far) >= 2:
            near = sorted(near, key=lambda d: -d["conf"])[:2]
            far = sorted(far, key=lambda d: -d["conf"])[:2]
            # darker torso first (Player 1 / Player 3)
            near.sort(key=lambda d: d["app"][0])
            far.sort(key=lambda d: d["app"][0])
            return t, near + far
    raise RuntimeError("could not find two players on each side of the net")


def assign(dets, n_frames):
    """Follow the four players. Returns dict with per-slot boxes, positions and info."""
    t0, init = _init_slots(dets)
    team = [NEAR, NEAR, FAR, FAR]
    model = [d["app"].copy() for d in init]
    pos = [d["pos"].copy() for d in init]
    vel = [np.zeros(2) for _ in range(4)]
    last_t = [t0] * 4
    tids = [d["tid"] for d in init]
    heights = [d["box"][3] - d["box"][1] for d in init]
    boxes = [[None] * n_frames for _ in range(4)]
    P = np.full((4, n_frames, 2), np.nan)
    conf_log = np.zeros((4, n_frames))

    last_box = [d["box"] for d in init]
    last_any = [t0] * 4

    def step(t):
        D = dets[t]
        if not D:
            return
        cost = np.full((4, len(D)), 50.0)
        for i in range(4):
            dt = t - last_t[i]
            pred = pos[i] + vel[i] * min(abs(dt), 10)
            gate = 1.5 + 0.12 * abs(t - last_any[i])
            for j, d in enumerate(D):
                if d["occluded"]:
                    # feet hidden: compare where the head is, in units of body height
                    lb = last_box[i]
                    a = np.array([(d["box"][0] + d["box"][2]) / 2, d["box"][1]])
                    b = np.array([(lb[0] + lb[2]) / 2, lb[1]])
                    dp = 1.7 * np.linalg.norm(a - b) / max(1.0, heights[i])
                    ratio = 1.0
                else:
                    y = d["pos"][1]
                    side_ok = (y < court.NET_Y + 0.6) if team[i] == NEAR else (y > court.NET_Y - 0.6)
                    if not side_ok:
                        continue
                    dp = np.linalg.norm(d["pos"] - pred)
                    # someone further away or nearer has a different size in the picture
                    ratio = (d["box"][3] - d["box"][1]) / max(1.0, heights[i])
                    if not 0.65 < ratio < 1.5:
                        continue
                if dp > gate:
                    continue
                c = 0.8 * dp + 1.5 * _app_dist(d["app"], model[i]) + 1.0 * abs(np.log(ratio))
                if d["tid"] == tids[i]:
                    c -= 0.4
                cost[i, j] = c
        # a slot may also stay empty this frame (dummy columns), so nobody is forced
        # onto someone else's detection just to fill every slot
        full = np.hstack([cost, np.full((4, 4), 2.5)])
        r, c = linear_sum_assignment(full)
        for i, j in zip(r, c):
            if j >= len(D) or cost[i, j] >= 50:
                continue
            d = D[j]
            tids[i] = d["tid"]
            last_any[i] = t
            conf_log[i, t] = 1
            if np.isfinite(d["app"]).all():
                model[i] = 0.95 * model[i] + 0.05 * d["app"]
            if d["occluded"]:
                # feet hidden: estimate them one usual body height below the head
                x0, y0, x1, _ = d["box"]
                boxes[i][t] = (x0, y0, x1, y0 + heights[i])
                last_box[i] = boxes[i][t]
                P[i, t] = court.apply_h(d["Hi"], [((x0 + x1) / 2, y0 + heights[i])])[0]
                continue
            dt = t - last_t[i]
            if dt != 0:
                v_new = (d["pos"] - pos[i]) / dt
                vel[i] = 0.6 * vel[i] + 0.4 * v_new
                sp = np.linalg.norm(vel[i])
                if sp > 0.2:  # 6 m/s
                    vel[i] *= 0.2 / sp
            pos[i] = d["pos"].copy()
            last_t[i] = t
            boxes[i][t] = d["box"]
            last_box[i] = d["box"]
            heights[i] = 0.85 * heights[i] + 0.15 * (d["box"][3] - d["box"][1])
            P[i, t] = d["pos"]

    # forward from the start frame, then backward to the beginning
    for t in range(t0, n_frames):
        step(t)
    pos = [d["pos"].copy() for d in init]
    vel = [np.zeros(2) for _ in range(4)]
    last_t = [t0] * 4
    tids = [d["tid"] for d in init]
    heights[:] = [d["box"][3] - d["box"][1] for d in init]
    last_box[:] = [d["box"] for d in init]
    last_any[:] = [t0] * 4
    for t in range(t0 - 1, -1, -1):
        step(t)
    return {"boxes": boxes, "pos": P, "seen": conf_log,
            "shirts": [shirt_class(m[:3]) for m in model], "team": team, "start": t0}


def fill_and_smooth(P, boxes, max_gap=150, win=7):
    """Interpolate short gaps and smooth the court positions; interpolate boxes too."""
    n = P.shape[1]
    Ps = P.copy()
    B = [[b for b in bl] for bl in boxes]
    for i in range(P.shape[0]):
        seen = np.flatnonzero(np.isfinite(P[i, :, 0]))
        if len(seen) == 0:
            continue
        for a, b in zip(seen[:-1], seen[1:]):
            if 1 < b - a <= max_gap:
                for t in range(a + 1, b):
                    f = (t - a) / (b - a)
                    Ps[i, t] = (1 - f) * P[i, a] + f * P[i, b]
                    if boxes[i][t] is None:
                        ba, bb = np.array(boxes[i][a]), np.array(boxes[i][b])
                        B[i][t] = tuple((1 - f) * ba + f * bb)
        # hold the ends
        Ps[i, :seen[0]] = P[i, seen[0]]
        Ps[i, seen[-1] + 1:] = P[i, seen[-1]]
        for t in range(0, seen[0]):
            B[i][t] = boxes[i][t] or boxes[i][seen[0]]
        for t in range(seen[-1] + 1, n):
            B[i][t] = boxes[i][t] or boxes[i][seen[-1]]
        # nobody runs faster than ~4.5 m/s: a jump lifts the feet in the picture, which
        # reads as a sudden step backwards
        vmax = 0.15
        for t in range(1, n):
            if not (np.isfinite(Ps[i, t]).all() and np.isfinite(Ps[i, t - 1]).all()):
                continue
            step = Ps[i, t] - Ps[i, t - 1]
            d = np.linalg.norm(step)
            if np.isfinite(d) and d > vmax:
                Ps[i, t] = Ps[i, t - 1] + step * (vmax / d)
        # then a median and a moving average
        from scipy.ndimage import median_filter
        for c in range(2):
            Ps[i, :, c] = median_filter(Ps[i, :, c], size=15, mode="nearest")
        k = np.ones(win) / win
        for c in range(2):
            x = Ps[i, :, c]
            ok = np.isfinite(x)
            if ok.sum() > win:
                xp = np.pad(x, win // 2, mode="edge")
                Ps[i, :, c] = np.convolve(xp, k, mode="valid")
    return Ps, B


def stamina_and_distance(Ps, fps):
    """Distance run (m) and a game-style stamina value (100 -> lower with running)."""
    n = Ps.shape[1]
    dist = np.zeros((4, n))
    stam = np.full((4, n), 100.0)
    for i in range(4):
        for t in range(1, n):
            d = np.linalg.norm(Ps[i, t] - Ps[i, t - 1])
            if not np.isfinite(d) or d < 0.01:
                d = 0.0
            dist[i, t] = dist[i, t - 1] + d
            speed = d * fps
            s = stam[i, t - 1] - 1.6 * d - (2.5 / fps if speed > 3.0 else 0)
            if speed < 0.5:
                s += 1.5 / fps
            stam[i, t] = np.clip(s, 0, 100)
    return dist, stam
