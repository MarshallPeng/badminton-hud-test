"""Step 10: error-check report (HTML + pictures)."""
import html
from pathlib import Path

import cv2
import numpy as np

from . import court

COLORS_BGR = [(255, 229, 0), (79, 255, 124), (38, 167, 255), (160, 64, 255)]


def _label(img, text, org, scale=1.0, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def court_check(frames, Hs, out_dir, n=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    paths = []
    for t in idx:
        img = court.draw_court(frames[t].copy(), Hs[t], color=(0, 0, 255), thickness=2)
        _label(img, f"frame {t}", (20, 50), 1.4)
        p = out_dir / f"court_{t:04d}.jpg"
        cv2.imwrite(str(p), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def _mark_frame(frame, t, names, boxes, shuttle_px, hitter=None, text=None):
    img = frame.copy()
    for i in range(len(names)):
        b = boxes[i][t]
        if b is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in b]
        th = 5 if i == hitter else 2
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS_BGR[i], th)
        _label(img, names[i], (x0, max(30, y0 - 10)), 0.9, COLORS_BGR[i])
    p = shuttle_px[t] if 0 <= t < len(shuttle_px) else None
    if p is not None and np.isfinite(p).all():
        cv2.circle(img, (int(p[0]), int(p[1])), 22, (0, 0, 255), 3)
    if text:
        _label(img, text, (20, 50), 1.3)
    return img


def shot_pictures(frames, shots, names, boxes, shuttle_px, fps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n = len(frames)
    for sh in shots:
        t0 = min(n - 1, sh["frame"])
        t1 = min(n - 1, t0 + int(round(0.2 * fps)))
        a = _mark_frame(frames[t0], t0, names, boxes, shuttle_px, sh["player"],
                        f"shot {sh['index']}  {sh['time']:.2f}s  {names[sh['player']]}  (hit)")
        b = _mark_frame(frames[t1], t1, names, boxes, shuttle_px, sh["player"],
                        f"+0.2s  ({sh['source']})")
        img = np.hstack([a, b])
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        p = out_dir / f"shot_{sh['index']:02d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def gap_pictures(frames, shots, names, boxes, shuttle_px, fps, start, end, out_dir, max_gap=2.4):
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [start] + [s["time"] for s in shots] + [end]
    out = []
    for a, b in zip(times[:-1], times[1:]):
        if b - a <= max_gap:
            continue
        ts = np.linspace(a, b, 5)[1:-1]
        imgs = []
        for t in ts:
            f = min(len(frames) - 1, int(round(t * fps)))
            imgs.append(_mark_frame(frames[f], f, names, boxes, shuttle_px, None, f"{t:.2f}s"))
        img = np.hstack(imgs)
        img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        p = out_dir / f"gap_{a:05.2f}-{b:05.2f}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append((a, b, p))
    return out


def write_html(path, title, summary, court_imgs, shots, shot_imgs, gaps, warns, names, fixes_path):
    rel = lambda p: html.escape(str(Path(p).relative_to(path.parent)))
    rows = []
    for sh, img in zip(shots, shot_imgs):
        v = f"{sh['speed']:.0f} km/h" if sh.get("speed") else "–"
        D = f"{sh['D']:.1f} m" if sh.get("D") else "–"
        warn = any(w[0] == sh["index"] for w in warns)
        rows.append(f"""<tr class="{'warn' if warn else ''}"><td>{sh['index']}</td><td>{sh['time']:.2f}</td>
<td>{html.escape(names[sh['player']])}</td><td>{html.escape(sh['source'])}</td><td>{D}</td>
<td>{sh['T']:.2f} s</td><td>{v}</td><td>{html.escape('; '.join(sh['notes']))}</td></tr>""")
    shots_html = "\n".join(
        f'<figure><img src="{rel(p)}" loading="lazy"><figcaption>Shot {sh["index"]} at {sh["time"]:.2f} s'
        f' — {html.escape(names[sh["player"]])}</figcaption></figure>' for sh, p in zip(shots, shot_imgs))
    court_html = "\n".join(f'<figure><img src="{rel(p)}" loading="lazy"></figure>' for p in court_imgs)
    gaps_html = "\n".join(
        f'<figure><img src="{rel(p)}" loading="lazy"><figcaption>No shot between {a:.2f} s and {b:.2f} s'
        f'</figcaption></figure>' for a, b, p in gaps) or "<p>None.</p>"
    warn_html = "\n".join(f"<li>{html.escape(w[1])}</li>" for w in warns) or "<li>No warnings.</li>"
    summ = "\n".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:system-ui,sans-serif;margin:0 auto;max-width:1300px;padding:16px;background:#0e141c;color:#e6eef6}}
h1,h2{{color:#00e5ff}} table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #2a3644;padding:4px 8px;text-align:left}}
tr.warn td{{background:#3a2412}} figure{{margin:0 0 16px}} img{{max-width:100%;border:1px solid #2a3644}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}
code,pre{{background:#18222e;padding:2px 4px}} pre{{padding:10px;overflow:auto}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<table>{summ}</table>
<h2>Warnings</h2><ul>{warn_html}</ul>
<h2>Fixes</h2>
<p>Edit <code>{html.escape(str(fixes_path))}</code> and run the same <code>hud</code> command again
(only the shot, speed and drawing steps are redone). Examples:</p>
<pre>[rally]
end_time = 10.4          # seconds; when the rally ends

remove_shot = [5.17]     # times (s) of shots to delete, +-0.12 s

[[add_shot]]
time = 6.50              # seconds
player = 2               # optional: 1-4
</pre>
<h2>Shots</h2>
<table><tr><th>#</th><th>time (s)</th><th>hitter</th><th>found by</th><th>D</th><th>T</th><th>speed</th><th>notes</th></tr>
{''.join(rows)}</table>
<h2>Court lines (8 frames)</h2><div class="grid">{court_html}</div>
<h2>Every shot (at the hit and 0.2 s later)</h2><div class="grid">{shots_html}</div>
<h2>Long stretches without a shot</h2><div class="grid">{gaps_html}</div>
</body></html>"""
    path.write_text(doc)
