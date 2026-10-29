"""Step 7: find the shuttle.

Candidates are small bright blobs that are brighter than the frames +-3 and +-10
(each neighbour is aligned twice: by the background motion and by the floor plane).
Candidates are chained into tracklets, and tracklets are joined into one path that
may only change direction at the racket of one of the four players.
"""
import cv2
import numpy as np

from . import camera, court

OFFSETS = (-10, -3, 3, 10)


def _gray(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]


def body_mask(shape, boxes):
    """Torso and legs of every person (a shuttle is never seen there; arms and rackets are)."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = b[:4]
        bw, bh = x1 - x0, y1 - y0
        if bh > 0.45 * h:
            # someone right in front of the camera: ignore all of them
            m[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 1
            continue
        m[max(0, int(y0 + 0.22 * bh)):int(y1), max(0, int(x0 + 0.12 * bw)):int(x1 - 0.12 * bw)] = 1
    return m


def candidates(frames, C, Hs, boxes_per_frame=None, thr=18, local_thr=22, progress=None):
    """Per frame: array of [x, y, area, contrast, elongation]."""
    g = _gray(frames)
    n = len(g)
    h, w = g[0].shape
    out = []
    k3 = np.ones((3, 3), np.uint8)
    for t in range(n):
        cur = g[t]
        warped = []
        for d in OFFSETS:
            s = t + d
            if not 0 <= s < n:
                continue
            Hb = camera.warp_between(C, s, t)
            Hf = Hs[t] @ np.linalg.inv(Hs[s])
            for Hm in (Hb, Hf):
                warped.append(cv2.warpPerspective(g[s], Hm, (w, h), flags=cv2.INTER_LINEAR,
                                                  borderValue=255))
        if len(warped) < 4:
            out.append(np.zeros((0, 5)))
            continue
        # a tiny tolerance for sub-pixel misalignment: dilate the neighbours
        mx = cv2.dilate(np.max(warped, axis=0), k3)
        diff = cv2.subtract(cur, mx)
        # local contrast: brighter than its own 15x15 surroundings in this frame
        bg = cv2.medianBlur(cur, 15)
        local = cv2.subtract(cur, bg)
        m = ((diff > thr) & (local > local_thr)).astype(np.uint8)
        if boxes_per_frame is not None:
            m[body_mask(cur.shape, boxes_per_frame[t]) > 0] = 0
        m = cv2.dilate(m, k3)
        nl, lab, st, cen = cv2.connectedComponentsWithStats(m)
        rows = []
        for i in range(1, nl):
            area = st[i, cv2.CC_STAT_AREA]
            bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if area < 4 or area > 400 or max(bw, bh) > 60:
                continue
            x0, y0 = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            patch = diff[y0:y0 + bh, x0:x0 + bw][lab[y0:y0 + bh, x0:x0 + bw] == i]
            elong = max(bw, bh) / max(1, min(bw, bh))
            rows.append([cen[i][0], cen[i][1], area, float(patch.max()), elong])
        out.append(np.array(rows) if rows else np.zeros((0, 5)))
        if progress:
            progress(t)
    return out


def build_tracklets(cands, max_gap=2, max_jump=110, min_len=5, min_travel=25):
    """Greedy constant-velocity linking of candidates into tracklets."""
    active = []   # dicts: t (list), p (list of xy)
    done = []
    for t, c in enumerate(cands):
        pts = c[:, :2] if len(c) else np.zeros((0, 2))
        used = np.zeros(len(pts), bool)
        # predict each active tracklet
        preds = []
        for tr in active:
            if len(tr["p"]) >= 2:
                dt = tr["t"][-1] - tr["t"][-2]
                v = (np.array(tr["p"][-1]) - np.array(tr["p"][-2])) / dt
            else:
                v = np.zeros(2)
            gap = t - tr["t"][-1]
            preds.append(np.array(tr["p"][-1]) + v * gap)
        # assign by increasing distance
        pairs = []
        for i, pr in enumerate(preds):
            if len(pts):
                d = np.linalg.norm(pts - pr, axis=1)
                tr = active[i]
                speed = 0 if len(tr["p"]) < 2 else np.linalg.norm(
                    np.array(tr["p"][-1]) - np.array(tr["p"][-2]))
                lim = (25 if len(tr["p"]) >= 2 else max_jump) + 0.3 * speed
                for j in np.flatnonzero(d < lim):
                    pairs.append((d[j], i, j))
        pairs.sort()
        taken = set()
        for d, i, j in pairs:
            if i in taken or used[j]:
                continue
            taken.add(i)
            used[j] = True
            active[i]["t"].append(t)
            active[i]["p"].append(tuple(pts[j]))
            active[i]["a"].append(float(c[j, 3]))
        keep = []
        for i, tr in enumerate(active):
            if t - tr["t"][-1] > max_gap:
                done.append(tr)
            else:
                keep.append(tr)
        active = keep
        for j in np.flatnonzero(~used):
            active.append({"t": [t], "p": [tuple(pts[j])], "a": [float(c[j, 3])]})
    done += active
    pieces = []
    for tr in done:
        t = np.array(tr["t"], float)
        p = np.array(tr["p"], float)
        a = np.array(tr["a"], float)
        pieces += _split_smooth(t, p, a, min_len)
    out = []
    for t, p, a in pieces:
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        # the shuttle moves; static flicker and jitter do not
        if np.median(step / np.diff(t)) < 2.0 or np.linalg.norm(p[-1] - p[0]) < min_travel:
            continue
        # smooth flight: quadratic fit residual
        res = _quad_residual(t, p)
        if res > 3.0:
            continue
        out.append({"t": t, "p": p, "res": res, "contrast": float(np.mean(a))})
    out.sort(key=lambda tr: tr["t"][0])
    return out


def _split_smooth(t, p, a, min_len, thr=3.0):
    """Cut a linked track where it turns sharply (a hit) until every piece is smooth."""
    if len(t) < min_len:
        return []
    if _quad_residual(t, p) <= thr:
        return [(t, p, a)]
    v = np.diff(p, axis=0)
    sp = np.linalg.norm(v, axis=1) + 1e-6
    cos = (v[:-1] * v[1:]).sum(1) / (sp[:-1] * sp[1:])
    # cut at the sharpest turn (ties: the middle)
    k = int(np.argmin(cos + 1e-3 * np.abs(np.arange(len(cos)) - len(cos) / 2))) + 1
    return (_split_smooth(t[:k + 1], p[:k + 1], a[:k + 1], min_len, thr)
            + _split_smooth(t[k + 1:], p[k + 1:], a[k + 1:], min_len, thr))


def _quad_residual(t, p):
    """Smoothness of a flight: residual of a low-order polynomial fit (pixels).

    Air drag makes shuttle flights far from parabolic, so longer tracklets get a cubic.
    """
    if len(t) < 4:
        return 0.0
    t = t - t[0]
    A = np.c_[t ** 3, t ** 2, t, np.ones_like(t)] if len(t) >= 7 else np.c_[t ** 2, t, np.ones_like(t)]
    r = []
    for k in range(2):
        coef, *_ = np.linalg.lstsq(A, p[:, k], rcond=None)
        r.append(p[:, k] - A @ coef)
    return float(np.sqrt(np.mean(np.square(r))))


def _extrapolate(tr, t_new, end=True):
    """Quadratic extrapolation of a tracklet to time t_new from its end (or start)."""
    t, p = tr["t"], tr["p"]
    sl = slice(-6, None) if end else slice(0, 6)
    tt, pp = t[sl], p[sl]
    deg = 2 if len(tt) >= 5 else 1
    out = []
    for k in range(2):
        coef = np.polyfit(tt - tt[0], pp[:, k], deg)
        out.append(np.polyval(coef, t_new - tt[0]))
    return np.array(out)


def _velocity(tr, end=True):
    t, p = tr["t"], tr["p"]
    if end:
        return (p[-1] - p[-3 if len(p) >= 3 else 0]) / max(1e-6, t[-1] - t[-3 if len(t) >= 3 else 0])
    return (p[min(2, len(p) - 1)] - p[0]) / max(1e-6, t[min(2, len(t) - 1)] - t[0])


def racket_zone(box):
    """Where a racket can meet the shuttle, given the player's box (pixels).

    Right after a hit the shuttle travels more than a metre per frame, so the zone is
    generous and scales with the player's height.
    """
    x0, y0, x1, y1 = box
    bh = y1 - y0
    return (x0 - 0.7 * bh, y0 - 0.8 * bh, x1 + 0.7 * bh, y1 - 0.3 * bh)


def _player_box(players_boxes, pid, f):
    boxes = players_boxes[pid]
    return boxes[min(max(int(round(f)), 0), len(boxes) - 1)]


def _in_zone(pt, box):
    if box is None or not np.isfinite(pt).all():
        return False
    z = racket_zone(box)
    return z[0] <= pt[0] <= z[2] and z[1] <= pt[1] <= z[3]


def _zone_dist(pt, box):
    x0, y0, x1, y1 = box
    c = np.array([(x0 + x1) / 2, y0])
    return np.linalg.norm(pt - c) / max(1.0, y1 - y0)


def _contacts(tr, players_boxes, start=True, reach=10):
    """Players who could have hit (start=True) or received (start=False) this tracklet.

    The tracklet is extrapolated up to `reach` frames back (or forward); every player
    whose racket zone it enters is a candidate. Returns {player: (score, frame, xy)},
    lower score = better (closer to the head, fewer frames extrapolated).
    """
    t0 = tr["t"][0] if start else tr["t"][-1]
    # image speed near the contact; a shuttle at a player's distance moving at >= 3 m/s
    # covers at least ~6 % of that player's height per frame (distant shuttles crawl)
    speed = np.linalg.norm(_velocity(tr, end=not start))
    out = {}
    for k in range(0, reach + 1):
        f = t0 - k if start else t0 + k
        p = (tr["p"][0] if start else tr["p"][-1]) if k == 0 else _extrapolate(tr, f, end=not start)
        for pid in players_boxes:
            box = _player_box(players_boxes, pid, f)
            if _in_zone(p, box) and speed >= 0.05 * (box[3] - box[1]):
                d = _zone_dist(p, box) + 0.05 * k
                if pid not in out or d < out[pid][0]:
                    out[pid] = (d, float(f), p)
    return out


def _racket_swing(tr, players_boxes, max_len=15):
    """True for a short tracklet that stays close to one player's body (arm or racket)."""
    if len(tr["t"]) > max_len:
        return False
    for pid in players_boxes:
        ok = True
        for t, p in zip(tr["t"], tr["p"]):
            b = _player_box(players_boxes, pid, t)
            if b is None:
                ok = False
                break
            x0, y0, x1, y1 = b
            h = y1 - y0
            if not (x0 - 0.35 * h <= p[0] <= x1 + 0.35 * h and y0 - 0.4 * h <= p[1] <= y1 - 0.2 * h):
                ok = False
                break
        if ok:
            return True
    return False


def best_path(tracklets, players_boxes, team_of, max_gap=20, continue_px=35, fps=30,
              min_gap_s=0.3, max_gap_s=2.5):
    """Choose tracklets forming one rally path.

    Consecutive tracklets are joined in one of three ways:
      cont   - the same flight continues (the end of one extrapolates onto the next)
      hit    - the first flight arrives at a player's racket and the next leaves it
      bridge - the first arrives at player X, the next leaves player Y of the other
               team, and the flight X -> Y in between was not seen
    The direction may only change at a racket of one of our players, and hits must
    alternate between the two teams.
    Returns (list of tracklet indices, list of hits (frame, player, xy)).
    """
    keep = [i for i, tr in enumerate(tracklets) if not _racket_swing(tr, players_boxes)]
    trs = [tracklets[i] for i in keep]
    n = len(trs)
    if n == 0:
        return [], []
    hitter = [_contacts(tr, players_boxes, start=True) for tr in trs]
    recv = [_contacts(tr, players_boxes, start=False) for tr in trs]
    edges = {i: [] for i in range(n)}
    for i, a in enumerate(trs):
        for j in range(i + 1, n):
            b = trs[j]
            gap = b["t"][0] - a["t"][-1]
            if gap < 1 or gap > max_gap_s * fps:
                continue
            if gap <= 10:
                pe = _extrapolate(a, b["t"][0], end=True)
                if np.linalg.norm(pe - b["p"][0]) < continue_px + 4 * gap:
                    edges[i].append((j, "cont", [], 0.0))
                    continue
            va, vb = _velocity(a, True), _velocity(b, False)
            cosang = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            for X, (dx, fx, px) in recv[i].items():
                for Y, (dy, fy, py) in hitter[j].items():
                    if X == Y and abs(fy - fx) <= 4 and gap <= max_gap and cosang < 0.8:
                        f = (fx + fy) / 2
                        edges[i].append((j, "hit", [(f, X, (px + py) / 2)], dx + dy))
                    elif team_of[X] != team_of[Y] and min_gap_s * fps <= fy - fx <= max_gap_s * fps:
                        edges[i].append((j, "bridge", [(fx, X, px), (fy, Y, py)], dx + dy))
    # long tracklets that travel far are worth more than short wiggles
    L = np.array([len(tr["t"]) * min(1.0, np.linalg.norm(tr["p"][-1] - tr["p"][0]) / 80.0)
                  for tr in trs], float)
    best = {}  # (i, team of last hit) -> (score, previous state, hits on the link)
    for i in range(n):
        best[(i, None)] = (L[i], None, [])
        for Y, (dy, fy, py) in hitter[i].items():
            k = (i, team_of[Y])
            sc = L[i] - 0.5 * dy + 3
            if k not in best or best[k][0] < sc:
                best[k] = (sc, None, [(fy, Y, py)])
    for i in range(n):
        for team in (None, 0, 1):
            if (i, team) not in best:
                continue
            sc = best[(i, team)][0]
            for j, kind, hits, dz in edges[i]:
                gap = trs[j]["t"][0] - trs[i]["t"][-1]
                ok = True
                tm = team
                for _, pid, _ in hits:
                    if tm is not None and team_of[pid] == tm:
                        ok = False
                    tm = team_of[pid]
                if not ok:
                    continue
                add = L[j] - 0.2 * gap - 0.5 * dz + {"cont": 0, "hit": 3, "bridge": -2}[kind]
                nk = (j, tm)
                if nk not in best or best[nk][0] < sc + add:
                    best[nk] = (sc + add, (i, team), hits)
    end = max(best, key=lambda k: best[k][0])
    path, hits = [], []
    k = end
    while k is not None:
        path.append(keep[k[0]])
        sc, prev, link_hits = best[k]
        hits.extend(reversed(link_hits))
        k = prev
    hits = [(f, pid, tuple(np.asarray(p, float))) for f, pid, p in reversed(hits)]
    return path[::-1], hits


def on_floor(tr, Hs, max_speed=5.0, fps=30):
    """True when a tracklet moves like a shuttle lying or sliding on the floor.

    Projected onto the floor plane, an airborne shuttle jumps around fast and far;
    one on the floor stays put or slides slowly.
    """
    q = []
    for t, p in zip(tr["t"].astype(int), tr["p"]):
        H = Hs[min(t, len(Hs) - 1)]
        q.append(court.apply_h(np.linalg.inv(H), [p])[0])
    q = np.array(q)
    if not np.isfinite(q).all():
        return False
    if not ((q[:, 0] > -2).all() and (q[:, 0] < court.COURT_W + 2).all()
            and (q[:, 1] > -2).all() and (q[:, 1] < court.COURT_L + 2).all()):
        return False
    v = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.diff(tr["t"]) * fps
    return float(np.median(v)) < max_speed


def landing(tracklets, path, hits, Hs, fps=30):
    """Time (s) the shuttle came down after the last real shot, and hits to keep."""
    first_hit = hits[0][0] if hits else 0
    for i in path:
        tr = tracklets[i]
        if tr["t"][0] <= first_hit or not on_floor(tr, Hs, fps=fps):
            continue
        t_land = tr["t"][0] / fps
        keep = [h for h in hits if h[0] / fps < t_land - 0.1]
        return t_land, keep
    return None, hits


def path_positions(tracklets, path, n_frames):
    """Per-frame shuttle pixel position (NaN where unseen) along the chosen path."""
    pos = np.full((n_frames, 2), np.nan)
    for i in path:
        tr = tracklets[i]
        for t, p in zip(tr["t"].astype(int), tr["p"]):
            pos[t] = p
    return pos
