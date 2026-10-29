"""hud <clip>: add a game-style HUD to a single-rally badminton clip."""
import argparse
import json
import pickle
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from . import camera, court, identity, people, prepare, render, report, shots, shuttle
from .video import read_frames

DEFAULT_NAMES = ["Player 1", "Player 2", "Player 3", "Player 4"]
STAGES = ["prepare", "people", "camera", "court", "identity", "shuttle"]

FIXES_TEMPLATE = """# Fixes for {clip}
# Edit and run `hud {clip}` again. Only the shot, speed, drawing and report steps are redone.

# Names: Player 1 and 2 are on the camera's side (1 = darker shirt), 3 and 4 on the far side.
names = {names}

# Times (s) of shots to delete (matches within 0.12 s), e.g. [5.17, 7.30]:
remove_shot = []

[rally]
# end_time = 10.40        # seconds; uncomment to set when the rally ends

# Shots to add; player (1-4) is optional. Copy the block once per shot:
# [[add_shot]]
# time = 6.50
# player = 2
"""


def log(msg):
    print(f"[hud] {msg}", flush=True)


def load_fixes(path, names, clip_name):
    if not path.exists():
        path.write_text(FIXES_TEMPLATE.format(clip=clip_name, names=json.dumps(names)))
    with open(path, "rb") as f:
        fx = tomllib.load(f)
    # tolerate keys written under [rally] by mistake
    for k in ("remove_shot", "add_shot"):
        if k in fx.get("rally", {}) and k not in fx:
            fx[k] = fx["rally"].pop(k)
    for a in fx.get("add_shot", []):
        if a.get("player") is not None:
            a["player"] = int(a["player"]) - 1
    return fx


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hud", description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write the results (default: next to the clip)")
    ap.add_argument("--redo", choices=STAGES, default=None, help="recompute from this step on")
    ap.add_argument("--model", default="yolo11m.pt")
    args = ap.parse_args(argv)

    clip = args.clip.resolve()
    if not clip.exists():
        sys.exit(f"no such file: {clip}")
    out_dir = (args.out_dir or clip.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = clip.stem
    work = out_dir / f"{name}_HUD_work"
    work.mkdir(exist_ok=True)
    out_video = out_dir / f"{name}_HUD.mp4"
    report_html = out_dir / f"{name}_HUD_report.html"
    report_dir = out_dir / f"{name}_HUD_report"
    fixes_path = out_dir / f"{name}_HUD_fixes.toml"
    fixes = load_fixes(fixes_path, DEFAULT_NAMES, clip.name)
    names = list(fixes.get("names", DEFAULT_NAMES))[:4]
    names += DEFAULT_NAMES[len(names):]

    stamp = {"clip": str(clip), "mtime": clip.stat().st_mtime, "size": clip.stat().st_size}
    stamp_path = work / "source.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        args.redo = "prepare"
    stamp_path.write_text(json.dumps(stamp))
    redo_from = STAGES.index(args.redo) if args.redo else len(STAGES)

    def cached(stage, path):
        return path.exists() and STAGES.index(stage) < redo_from

    T0 = time.time()
    # 1. prepare ------------------------------------------------------------
    meta_p = work / "meta.json"
    if not cached("prepare", meta_p):
        log("1/10 prepare: normalising video, extracting audio")
        meta = prepare.normalise(clip, work / "video.mp4", work / "audio.wav")
        meta["onsets"] = prepare.racket_onsets(work / "audio.wav") if meta["audio"] else []
        meta_p.write_text(json.dumps(meta))
    meta = json.loads(meta_p.read_text())
    fps = meta["fps"]
    frames = read_frames(work / "video.mp4")
    n = len(frames)
    log(f"     {meta['width']}x{meta['height']} @ {fps} fps, {n} frames, {len(meta['onsets'])} racket-sound onsets")

    # 3. people (run before the camera step so people can be masked out) ----
    ppl_p = work / "people.pkl"
    if not cached("people", ppl_p):
        log("3/10 people: YOLO11m + ByteTrack, imgsz 1280")
        t = time.time()
        ppl, dev = people.track_people(work / "video.mp4", model_name=args.model,
                                       progress=lambda i: i % 50 == 0 and log(f"     frame {i}/{n} ({time.time() - t:.0f} s)"))
        pickle.dump({"rows": ppl, "device": dev}, open(ppl_p, "wb"))
    ppl = pickle.load(open(ppl_p, "rb"))["rows"]
    ppl = (ppl + [np.zeros((0, 6))] * n)[:n]
    boxes_all = [[r[1:5] for r in rows] for rows in ppl]

    # 2. camera motion ---------------------------------------------------------
    cam_p = work / "camera.npz"
    if not cached("camera", cam_p):
        log("2/10 camera motion: background homographies (ceiling and walls)")
        C, ninl = camera.all_motion(frames, boxes_all)
        np.savez(cam_p, C=C, n=ninl)
    z = np.load(cam_p)
    C, ninl = z["C"], z["n"]

    # 4-5. court ---------------------------------------------------------------
    court_p = work / "court.npz"
    if not cached("court", court_p):
        log("4/10 court: first-frame line fit")
        feet = [((b[0] + b[2]) / 2, b[3]) for b in boxes_all[0]]
        H0, info, _ = court.fit_first_frame(frames[0], boxes_all[0], feet)
        log(f"     first frame: {info}")
        log("5/10 court: following the court through the clip")
        Hs, stats = court.track_court(frames, H0, C, boxes_all,
                                      log=lambda t, s: t % 50 == 0 and log(f"     frame {t}/{n} {s['source']} cov={s['cov']:.3f}"))
        np.savez(court_p, Hs=Hs)
        (work / "court_stats.json").write_text(json.dumps({"first": info, "frames": stats}, default=float))
    Hs = np.load(court_p)["Hs"]
    cstats = json.loads((work / "court_stats.json").read_text())

    # 6. identity ----------------------------------------------------------------
    id_p = work / "identity.pkl"
    if not cached("identity", id_p):
        log("6/10 who is who: torso colour + side of the net + last position")
        dets = identity.detections(ppl, Hs, frames)
        ids = identity.assign(dets, n)
        pickle.dump(ids, open(id_p, "wb"))
    ids = pickle.load(open(id_p, "rb"))
    team = ids["team"]
    Ps, B = identity.fill_and_smooth(ids["pos"], ids["boxes"])
    dist, stam = identity.stamina_and_distance(Ps, fps)
    log(f"     shirts: {ids['shirts']}")

    # 7. shuttle -----------------------------------------------------------------
    sh_p = work / "shuttle.pkl"
    if not cached("shuttle", sh_p):
        log("7/10 shuttle: bright-blob candidates, tracklets, one rally path")
        cands = shuttle.candidates(frames, C, Hs, boxes_all,
                                   progress=lambda i: i % 100 == 0 and log(f"     frame {i}/{n}"))
        trs = shuttle.build_tracklets(cands)
        pb = {i: B[i] for i in range(4)}
        path, hits = shuttle.best_path(trs, pb, team, fps=fps)
        pickle.dump({"cands": cands, "tracklets": trs, "path": path, "hits": hits}, open(sh_p, "wb"))
    S = pickle.load(open(sh_p, "rb"))
    trs, path, hits = S["tracklets"], S["path"], S["hits"]
    t_land, hits = shuttle.landing(trs, path, hits, Hs, fps)
    spx = shuttle.path_positions(trs, path, n)
    log(f"     {sum(len(c) for c in S['cands'])} candidates, {len(trs)} tracklets, "
        f"path of {len(path)} tracklets covering {int(np.isfinite(spx[:, 0]).sum())} frames, {len(hits)} visual hits")

    # 8. shots ---------------------------------------------------------------------
    log("8/10 shots and speeds")
    shot_list = shots.detect_shots(hits, meta["onsets"], spx, B, Ps, team, fps, n, fixes)
    end_t, end_src = shots.rally_end(shot_list, spx, fps, n, fixes, landing=t_land)
    end_f = min(n - 1, int(round(end_t * fps)))
    end_court = None
    seen = np.flatnonzero(np.isfinite(spx[:, 0]))
    if shot_list:
        last_seen = seen[(seen > shot_list[-1]["frame"]) & (seen <= end_f + 2)]
        if len(last_seen):
            f = last_seen[-1]
            q = court.apply_h(np.linalg.inv(Hs[f]), [spx[f]])[0]
            if -1 < q[0] < court.COURT_W + 1 and -1 < q[1] < court.COURT_L + 1:
                end_court = q
    shots.add_speeds(shot_list, Ps, end_t, end_court, fps)
    warns = shots.warnings(shot_list)
    start_t = shot_list[0]["time"] if shot_list else 0.0
    for sh in shot_list:
        v = f"{sh['speed']:.0f} km/h" if sh["speed"] else "-"
        log(f"     #{sh['index']:2d} {sh['time']:5.2f}s {names[sh['player']]:<10} {sh['source']:<13} {v}")
    log(f"     rally {start_t:.2f}s -> {end_t:.2f}s ({end_src}); {len(warns)} warnings")

    # 9. draw ----------------------------------------------------------------------
    log("9/10 drawing the HUD")
    R = render.HudRenderer(meta["width"], meta["height"], names, fps)

    def states():
        k = -1
        for t in range(n):
            ts = t / fps
            while k + 1 < len(shot_list) and shot_list[k + 1]["frame"] <= t:
                k += 1
            in_rally = start_t <= ts <= end_t if shot_list else False
            st = {"t": t, "H": Hs[t],
                  "pos": [Ps[i, t] for i in range(4)],
                  "box": [B[i][t] for i in range(4)],
                  "stamina": stam[:, t], "dist": dist[:, t],
                  "count": k + 1 if shot_list else 0,
                  "rally_time": (min(ts, end_t) - start_t) if shot_list and ts >= start_t else None,
                  "speed": shot_list[k]["speed"] if k >= 0 else None}
            if in_rally and k >= 0:
                cur = shot_list[k]
                hitter = names[cur["player"]]
                sp = f"{cur['speed']:.0f} km/h" if cur["speed"] else ""
                st["lock_label"] = f"{hitter.upper()}  {sp}".strip()
                p = spx[t]
                if not np.isfinite(p).all():
                    # bridge short gaps in the path
                    prev = seen[seen < t]
                    nxt = seen[seen > t]
                    if len(prev) and len(nxt) and nxt[0] - prev[-1] <= 8:
                        a, b = prev[-1], nxt[0]
                        f = (t - a) / (b - a)
                        p = (1 - f) * spx[a] + f * spx[b]
                        st["shuttle_lost"] = True
                if np.isfinite(p).all():
                    st["shuttle"] = p
                a = Ps[cur["player"], cur["frame"]]
                if k + 1 < len(shot_list):
                    nx = shot_list[k + 1]
                    b = Ps[nx["player"], nx["frame"]]
                    T = nx["time"] - cur["time"]
                else:
                    b, T = end_court, end_t - cur["time"]
                if b is not None and np.isfinite(a).all() and np.isfinite(b).all() and T > 0:
                    st["shot_line"] = (a, b)
                    fr = float(np.clip((ts - cur["time"]) / T, 0, 1))
                    st["shuttle_court"] = a + fr * (b - a)
                age = ts - cur["time"]
                if age < 0.7 and cur["speed"]:
                    bx = B[cur["player"]][t]
                    if bx is not None:
                        st["popup"] = (((bx[0] + bx[2]) / 2, bx[1]), f"{cur['speed']:.0f} KM/H", age / 0.7)
            yield R.draw(frames[t], st)

    render.encode(states(), (meta["width"], meta["height"]), fps, out_video,
                  work / "audio.wav" if meta["audio"] else None)
    log(f"     wrote {out_video}")

    # 10. report ---------------------------------------------------------------------
    log("10/10 error report")
    for sub in ("shots", "gaps"):
        for old in (report_dir / sub).glob("*.jpg") if (report_dir / sub).exists() else []:
            old.unlink()
    cimgs = report.court_check(frames, Hs, report_dir / "court")
    simgs = report.shot_pictures(frames, shot_list, names, B, spx, fps, report_dir / "shots")
    gaps = report.gap_pictures(frames, shot_list, names, B, spx, fps, start_t, end_t, report_dir / "gaps")
    covs = [s["cov"] for s in cstats["frames"] if s.get("cov") is not None]
    summary = {
        "clip": clip.name, "frames": f"{n} @ {fps} fps ({meta['width']}x{meta['height']})",
        "court fit (first frame)": json.dumps(cstats["first"], default=float),
        "court line coverage (near half)": f"median {np.median(covs):.3f}, min {np.min(covs):.3f}",
        "camera features (median inliers)": int(np.median(ninl[1:])) if n > 1 else 0,
        "shirts": ", ".join(f"{names[i]}: {s}" for i, s in enumerate(ids["shirts"])),
        "rally": f"{start_t:.2f} s -> {end_t:.2f} s ({end_src})",
        "shots": len(shot_list),
        "shuttle seen on path": f"{int(np.isfinite(spx[:, 0]).sum())} of {n} frames",
    }
    report.write_html(report_html, f"{name} — HUD error check", summary, cimgs, shot_list, simgs,
                      gaps, warns, names, fixes_path)
    log(f"     wrote {report_html}")
    for w in warns:
        log(f"     WARNING {w[1]}")
    log(f"done in {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
