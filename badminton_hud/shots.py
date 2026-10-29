"""Step 8: shots, hitters and launch speeds."""
import numpy as np

K_DRAG = 9.81 / 6.8 ** 2  # 1/m, shuttle terminal velocity 6.8 m/s


def launch_speed_kmh(D, T):
    """Launch speed for a flight of D metres in T seconds with quadratic drag."""
    if D is None or T is None or T <= 0 or D <= 0:
        return None
    v = (np.exp(K_DRAG * D) - 1) / (K_DRAG * T)
    return float(v * 3.6)


def _closest_member(team_members, frame, shuttle_px, player_boxes, player_pos, ref_court=None):
    """Team member closer to the shuttle (pixels); fall back to court distance to ref_court."""
    best, bd = None, np.inf
    if shuttle_px is not None and np.isfinite(shuttle_px).all():
        for pid in team_members:
            b = player_boxes[pid][frame]
            if b is None:
                continue
            # distance in units of the player's height ~ metres at that player's depth
            # (pixels alone favour whoever is nearer the camera)
            c = np.array([(b[0] + b[2]) / 2, b[1] + 0.15 * (b[3] - b[1])])
            d = np.linalg.norm(c - shuttle_px) / max(1.0, b[3] - b[1])
            if d < bd:
                best, bd = pid, d
        if best is not None:
            return best
    if ref_court is not None:
        for pid in team_members:
            p = player_pos[pid, frame]
            if np.isfinite(p).all():
                d = np.linalg.norm(p - ref_court)
                if d < bd:
                    best, bd = pid, d
    return best if best is not None else team_members[0]


def _nearest_seen(shuttle_px, frame, window):
    n = len(shuttle_px)
    for dt in range(0, window + 1):
        for f in (frame - dt, frame + dt):
            if 0 <= f < n and np.isfinite(shuttle_px[f]).all():
                return shuttle_px[f]
    return None


def detect_shots(visual_hits, onsets, shuttle_px, player_boxes, player_pos, team, fps,
                 n_frames, fixes=None, min_gap=0.33, max_gap=2.4):
    """Combine visual hits and racket sounds into an alternating list of shots.

    visual_hits: list of (frame (float), player id, xy pixel)
    onsets: list of (time_s, strength)
    Returns a list of dicts sorted by time.
    """
    fixes = fixes or {}
    members = {0: [i for i in range(4) if team[i] == 0], 1: [i for i in range(4) if team[i] == 1]}
    removed = [float(x) for x in fixes.get("remove_shot", [])]

    def is_removed(t):
        return any(abs(t - r) < 0.12 for r in removed)

    def frame_of(t):
        return min(n_frames - 1, max(0, int(round(t * fps))))

    shots = []
    for f, pid, xy in visual_hits:
        t = f / fps
        # the contact time comes from extrapolated flights; a racket sound is sharper
        snd = [(abs(ts - t), ts) for ts, _ in onsets if abs(ts - t) < 0.17]
        if snd:
            t = min(snd)[1]
        fi = frame_of(t)
        hitter = _closest_member(members[team[pid]], fi, np.asarray(xy), player_boxes, player_pos)
        shots.append({"time": t, "frame": fi, "player": hitter, "team": team[hitter],
                      "source": "shuttle" + ("+sound" if snd else ""), "notes": []})
    shots = [s for s in shots if not is_removed(s["time"])]
    for a in fixes.get("add_shot", []):
        t = float(a["time"])
        pid = a.get("player")
        shots.append({"time": t, "frame": frame_of(t), "player": pid,
                      "team": team[pid] if pid is not None else None,
                      "source": "config", "notes": ["added in config"]})
    shots.sort(key=lambda s: s["time"])
    # two detections of the same shot
    dedup = []
    for sh in shots:
        if dedup and sh["time"] - dedup[-1]["time"] < 0.15 and sh["source"] != "config":
            continue
        dedup.append(sh)
    shots = dedup

    sounds = [(ts, st) for ts, st in onsets
              if not is_removed(ts) and not any(abs(ts - sh["time"]) < 0.2 for sh in shots)]

    def sound_in(t0, t1):
        c = [(st, ts) for ts, st in sounds if t0 + min_gap * 0.9 < ts < t1 - min_gap * 0.9
             and not any(abs(ts - sh["time"]) < 0.2 for sh in shots)]
        return max(c)[1] if c else None

    def new_shot(t, source, strength=None):
        note = f"from racket sound ({strength:.0f} sigma)" if strength else \
            "not seen: inferred because shots alternate sides"
        return {"time": t, "frame": frame_of(t), "player": None, "team": None,
                "source": source, "notes": [note]}

    # the serve: the last racket sound shortly before the first shot found (earlier
    # sounds are usually the shuttle being passed to the server)
    if shots:
        t1 = shots[0]["time"]
        c = [ts for ts, _ in sounds if t1 - max_gap < ts < t1 - 0.5]
        ts = max(c) if c else None
        if ts is not None and shots[0]["source"] not in ("config", "shuttle", "shuttle+sound"):
            st = dict(sounds)[ts]
            shots.insert(0, new_shot(ts, "sound", st))

    # repair: sides must alternate, and no gap longer than max_gap
    for _ in range(50):
        changed = False
        for i in range(len(shots) - 1):
            a, b = shots[i], shots[i + 1]
            same = a["team"] is not None and b["team"] is not None and a["team"] == b["team"]
            long_gap = b["time"] - a["time"] > max_gap
            if not (same or long_gap):
                continue
            ts = sound_in(a["time"], b["time"])
            if ts is not None:
                shots.insert(i + 1, new_shot(ts, "sound", dict(sounds)[ts]))
                changed = True
                break
            if same and b["time"] - a["time"] > 2 * min_gap:
                shots.insert(i + 1, new_shot((a["time"] + b["time"]) / 2, "inferred"))
                changed = True
                break
            if same:
                # too close for a shot in between: keep the better supported one
                rank = {"config": 3, "shuttle+sound": 2, "shuttle": 1}
                drop = i + 1 if rank.get(b["source"], 0) <= rank.get(a["source"], 0) else i
                keep = shots[i + 1 - (drop - i)]
                keep["notes"].append(f"dropped a second detection {abs(b['time'] - a['time']):.2f} s away "
                                     "on the same side")
                shots.pop(drop)
                changed = True
                break
        if not changed:
            break
        # give inserted shots a team right away so the next pass can check alternation
        for j, sh in enumerate(shots):
            if sh["team"] is None:
                nb = shots[j - 1] if j > 0 else (shots[j + 1] if j + 1 < len(shots) else None)
                if nb is not None and nb["team"] is not None:
                    sh["team"] = 1 - nb["team"]

    # hitters for the shots that have none yet
    for j, sh in enumerate(shots):
        if sh["team"] is None:
            sh["team"] = 0
        if sh["player"] is not None:
            continue
        fi = sh["frame"]
        spx = _nearest_seen(shuttle_px, fi, int(0.3 * fps))
        if j == 0 and spx is None:
            # the serve: the server stands nearer the net than the partner
            sh["player"] = min(members[sh["team"]], key=lambda p: abs(6.7 - player_pos[p, fi, 1])
                               if np.isfinite(player_pos[p, fi]).all() else 99)
        else:
            ref = player_pos[shots[j - 1]["player"], fi] if j > 0 and shots[j - 1]["player"] is not None else None
            sh["player"] = _closest_member(members[sh["team"]], fi, spx, player_boxes, player_pos,
                                           ref_court=ref)
    for i, sh in enumerate(shots):
        sh["index"] = i + 1
    return shots


def rally_end(shots, shuttle_px, fps, n_frames, fixes=None, max_flight=2.4, landing=None):
    fixes = fixes or {}
    if "end_time" in fixes.get("rally", {}):
        return float(fixes["rally"]["end_time"]), "config"
    if landing is not None and (not shots or landing > shots[-1]["time"]):
        return float(landing), "shuttle landed"
    if not shots:
        return n_frames / fps, "clip end"
    last = shots[-1]["frame"]
    seen = np.flatnonzero(np.isfinite(shuttle_px[:, 0]))
    after = seen[(seen > last) & (seen < last + max_flight * fps)]
    if len(after):
        return float(min(n_frames - 1, after[-1] + 2) / fps), "shuttle last seen"
    return float(min(n_frames - 1, last + 1.0 * fps) / fps), "last shot + 1 s"


def add_speeds(shots, player_pos, end_time, end_court, fps):
    """Launch speed of every shot from the distance to the next hitter and the flight time."""
    for i, sh in enumerate(shots):
        a = player_pos[sh["player"], sh["frame"]]
        if i + 1 < len(shots):
            nx = shots[i + 1]
            b = player_pos[nx["player"], nx["frame"]]
            T = nx["time"] - sh["time"]
        else:
            b = end_court
            T = end_time - sh["time"]
        D = float(np.linalg.norm(b - a)) if b is not None and np.isfinite(a).all() and np.isfinite(b).all() else None
        sh["D"] = D
        sh["T"] = T
        sh["speed"] = launch_speed_kmh(D, T)
    return shots


def warnings(shots, min_speed=25, max_speed=250, min_gap=0.33, max_gap=2.4):
    out = []
    for i, sh in enumerate(shots):
        v = sh.get("speed")
        if v is not None and v > max_speed:
            out.append((sh["index"], f"shot {sh['index']} at {sh['time']:.2f} s: {v:.0f} km/h is above {max_speed} km/h"))
        if v is not None and v < min_speed:
            out.append((sh["index"], f"shot {sh['index']} at {sh['time']:.2f} s: {v:.0f} km/h is below {min_speed} km/h"))
        if i + 1 < len(shots):
            g = shots[i + 1]["time"] - sh["time"]
            if g < min_gap:
                out.append((sh["index"], f"shots {sh['index']} and {sh['index'] + 1} are only {g:.2f} s apart"))
            if g > max_gap:
                out.append((sh["index"], f"shots {sh['index']} and {sh['index'] + 1} are {g:.2f} s apart (no shot found in between)"))
        if sh["source"] in ("sound", "inferred"):
            out.append((sh["index"], f"shot {sh['index']} at {sh['time']:.2f} s was not seen on the shuttle path ({sh['source']})"))
    return out
