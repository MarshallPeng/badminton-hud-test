"""Step 9: draw the HUD with Pillow and put the original audio back."""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import court

FONT_DIR = Path(__file__).parent / "fonts"
COLORS = [(0, 229, 255), (124, 255, 79), (255, 167, 38), (255, 64, 160)]
ACCENT = (0, 229, 255)
PANEL = (8, 14, 24, 170)
WHITE = (240, 248, 255)


def font(size, bold=True):
    name = "Rajdhani-Bold.ttf" if bold else "Rajdhani-SemiBold.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", int(size * 0.8))


class HudRenderer:
    def __init__(self, width, height, names, fps):
        self.w, self.h = width, height
        self.s = min(width, height) / 1080.0
        self.names = names
        self.fps = fps
        S = self.s
        self.f_big = font(int(44 * S))
        self.f_mid = font(int(30 * S))
        self.f_small = font(int(22 * S))
        self.f_tiny = font(int(18 * S), bold=False)

    # ---------------------------------------------------------------- pieces
    def _panel(self, d, box, accent=ACCENT, cut=14):
        x0, y0, x1, y1 = box
        c = cut * self.s
        poly = [(x0 + c, y0), (x1, y0), (x1, y1 - c), (x1 - c, y1), (x0, y1), (x0, y0 + c)]
        d.polygon(poly, fill=PANEL)
        d.line(poly + [poly[0]], fill=accent + (220,), width=max(1, int(2 * self.s)))

    def foot_ring(self, d, H, pos, color, t):
        if pos is None or not np.isfinite(pos).all():
            return
        ang = np.linspace(0, 2 * np.pi, 64)
        for r, a_fill, width in ((0.55, 50, 3), (0.38, 0, 2)):
            pts = np.c_[pos[0] + r * np.cos(ang), pos[1] + r * np.sin(ang)]
            if not court.in_front(H, pts).all():
                return
            p = court.apply_h(H, pts)
            poly = [tuple(q) for q in p]
            if a_fill:
                d.polygon(poly, fill=color + (a_fill,))
            d.line(poly + [poly[0]], fill=color + (235,), width=max(1, int(width * self.s)))
        # four rotating ticks lying on the floor
        rot = t * 0.08
        for k in range(4):
            a = rot + k * np.pi / 2
            seg = np.array([[pos[0] + 0.58 * np.cos(a), pos[1] + 0.58 * np.sin(a)],
                            [pos[0] + 0.75 * np.cos(a), pos[1] + 0.75 * np.sin(a)]])
            p = court.apply_h(H, seg)
            d.line([tuple(p[0]), tuple(p[1])], fill=color + (255,), width=max(2, int(4 * self.s)))

    def name_tag(self, d, box, name, color, lift=0.0):
        if box is None:
            return
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        S = self.s
        tw = d.textlength(name, font=self.f_small)
        ty = y0 - 34 * S - lift
        bx = (cx - tw / 2 - 10 * S, ty - 14 * S, cx + tw / 2 + 10 * S, ty + 14 * S)
        d.rectangle(bx, fill=(8, 14, 24, 190))
        d.rectangle((bx[0], bx[1], bx[0] + 4 * S, bx[3]), fill=color + (255,))
        d.text((cx, ty), name, font=self.f_small, fill=WHITE, anchor="mm")
        d.polygon([(cx - 7 * S, bx[3]), (cx + 7 * S, bx[3]), (cx, bx[3] + 9 * S)], fill=color + (230,))

    def reticle(self, d, xy, label, t, lost=False):
        x, y = xy
        S = self.s
        r = (30 + 4 * np.sin(t * 0.5)) * S
        col = (255, 60, 60) if not lost else (255, 160, 60)
        L = 12 * S
        wdt = max(2, int(3 * S))
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = x + sx * r, y + sy * r
                d.line([(cx, cy), (cx - sx * L, cy)], fill=col + (255,), width=wdt)
                d.line([(cx, cy), (cx, cy - sy * L)], fill=col + (255,), width=wdt)
        d.ellipse((x - 4 * S, y - 4 * S, x + 4 * S, y + 4 * S), outline=col + (255,), width=max(1, int(2 * S)))
        d.line([(x - r - 10 * S, y), (x - r + 4 * S, y)], fill=col + (200,), width=1)
        d.line([(x + r - 4 * S, y), (x + r + 10 * S, y)], fill=col + (200,), width=1)
        d.text((x, y - r - 14 * S), "LOCK ON", font=self.f_small, fill=col + (255,), anchor="mm")
        if label:
            d.text((x, y + r + 16 * S), label, font=self.f_small, fill=WHITE, anchor="mm",
                   stroke_width=2, stroke_fill=(0, 0, 0))

    def top_bar(self, d, rally_time, count, speed):
        S = self.s
        items = [("RALLY TIME", f"{rally_time:05.2f}" if rally_time is not None else "--.--"),
                 ("RALLY COUNT", f"{count}"),
                 ("SHOT SPEED", f"{speed:.0f} km/h" if speed else "---")]
        bw, bh = 250 * S, 78 * S
        total = bw * 3 + 16 * S * 2
        x = (self.w - total) / 2
        y = 18 * S
        for lab, val in items:
            self._panel(d, (x, y, x + bw, y + bh))
            d.text((x + bw / 2, y + 20 * S), lab, font=self.f_small, fill=ACCENT + (255,), anchor="mm")
            d.text((x + bw / 2, y + 52 * S), val, font=self.f_big, fill=WHITE, anchor="mm")
            x += bw + 16 * S

    def stats_panel(self, d, stam, dist):
        S = self.s
        rows = len(self.names)
        bw, rh = 380 * S, 46 * S
        x0 = 24 * S
        y0 = self.h - 24 * S - rh * rows - 40 * S
        self._panel(d, (x0, y0, x0 + bw, self.h - 24 * S))
        d.text((x0 + 16 * S, y0 + 20 * S), "STAMINA", font=self.f_small, fill=ACCENT + (255,), anchor="lm")
        d.text((x0 + bw - 16 * S, y0 + 20 * S), "DISTANCE", font=self.f_small, fill=ACCENT + (255,), anchor="rm")
        for i, name in enumerate(self.names):
            y = y0 + 40 * S + i * rh + rh / 2
            col = COLORS[i]
            d.rectangle((x0 + 14 * S, y - 12 * S, x0 + 20 * S, y + 12 * S), fill=col + (255,))
            d.text((x0 + 28 * S, y), name.upper(), font=self.f_small, fill=WHITE, anchor="lm")
            bx0, bx1 = x0 + 140 * S, x0 + 270 * S
            d.rectangle((bx0, y - 8 * S, bx1, y + 8 * S), outline=(255, 255, 255, 120), width=1)
            frac = float(np.clip(stam[i] / 100.0, 0, 1))
            g = int(60 + 195 * frac)
            r = int(255 - 155 * frac)
            segs = 10
            for k in range(int(round(frac * segs))):
                sx = bx0 + 2 * S + k * (bx1 - bx0 - 4 * S) / segs
                d.rectangle((sx, y - 6 * S, sx + (bx1 - bx0 - 4 * S) / segs - 2 * S, y + 6 * S),
                            fill=(r, g, 70, 255))
            d.text((x0 + bw - 16 * S, y), f"{dist[i]:5.1f} m", font=self.f_mid, fill=WHITE, anchor="rm")

    def court_map(self, d, players, shot_line, shuttle_court):
        S = self.s
        # court drawn with its length running left -> right (near baseline on the left),
        # so the camera's left-hand sideline is at the top
        mw, mh = 330 * S, 150 * S
        pad = 18 * S
        x1 = self.w - 24 * S
        y1 = self.h - 24 * S
        x0, y0 = x1 - mw - 2 * pad, y1 - mh - 2 * pad - 26 * S
        self._panel(d, (x0, y0, x1, y1))
        d.text((x0 + 16 * S, y0 + 16 * S), "COURT MAP", font=self.f_small, fill=ACCENT + (255,), anchor="lm")
        cx0, cy0 = x0 + pad, y0 + pad + 26 * S

        def m2p(p):
            return (cx0 + p[1] / court.COURT_L * mw, cy0 + p[0] / court.COURT_W * mh)

        d.rectangle((cx0, cy0, cx0 + mw, cy0 + mh), fill=(20, 110, 80, 150))
        for (a, b) in court.model_segments():
            d.line([m2p(a), m2p(b)], fill=(255, 255, 255, 200), width=1)
        d.line([m2p((-0.3, court.NET_Y)), m2p((court.COURT_W + 0.3, court.NET_Y))],
               fill=(255, 230, 0, 255), width=max(2, int(3 * S)))
        if shot_line is not None:
            a, b = shot_line
            d.line([m2p(a), m2p(b)], fill=(255, 60, 60, 220), width=max(2, int(2 * S)))
        if shuttle_court is not None:
            q = m2p(shuttle_court)
            d.ellipse((q[0] - 4 * S, q[1] - 4 * S, q[0] + 4 * S, q[1] + 4 * S), fill=(255, 255, 255, 255))
        for i, p in enumerate(players):
            if p is None or not np.isfinite(p).all():
                continue
            q = m2p(p)
            r = 7 * S
            d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=COLORS[i] + (255,),
                      outline=(0, 0, 0, 255), width=1)
            d.text((q[0], q[1] - r - 8 * S), f"P{i + 1}", font=self.f_tiny, fill=WHITE, anchor="mm")

    def shot_popup(self, d, xy, text, age):
        if xy is None:
            return
        S = self.s
        a = int(255 * max(0.0, 1 - age))
        d.text((xy[0], xy[1] - 70 * S - 30 * S * age), text, font=self.f_mid,
               fill=(255, 230, 0, a), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0, a))

    # ---------------------------------------------------------------- frame
    def draw(self, frame_bgr, st):
        img = Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        t = st["t"]
        for i in range(len(self.names)):
            self.foot_ring(d, st["H"], st["pos"][i], COLORS[i], t)
        img = Image.alpha_composite(img, ov)
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # tags that would overlap are stacked upwards (nearest player keeps the lowest)
        placed = []
        order = sorted((i for i in range(len(self.names)) if st["box"][i] is not None),
                       key=lambda i: -st["box"][i][3])
        for i in order:
            x0, y0, x1, _ = st["box"][i]
            tw = d.textlength(self.names[i].upper(), font=self.f_small) + 20 * self.s
            cx = (x0 + x1) / 2
            lift = 0.0
            for _ in range(len(placed) + 1):
                ty = y0 - 34 * self.s - lift
                hit = [p for p in placed if abs(p[0] - cx) < (p[1] + tw) / 2 and abs(p[2] - ty) < 30 * self.s]
                if not hit:
                    break
                lift += (ty - min(p[2] for p in hit)) + 32 * self.s
            placed.append((cx, tw, y0 - 34 * self.s - lift))
            self.name_tag(d, st["box"][i], self.names[i].upper(), COLORS[i], lift)
        if st.get("shuttle") is not None:
            self.reticle(d, st["shuttle"], st.get("lock_label"), t, lost=st.get("shuttle_lost", False))
        if st.get("popup"):
            self.shot_popup(d, *st["popup"])
        self.top_bar(d, st["rally_time"], st["count"], st["speed"])
        self.stats_panel(d, st["stamina"], st["dist"])
        self.court_map(d, st["pos"], st.get("shot_line"), st.get("shuttle_court"))
        img = Image.alpha_composite(img, ov)
        return np.asarray(img.convert("RGB"))


def encode(frames_iter, size, fps, out_path, audio_wav=None):
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", str(audio_wav), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k",
                "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_iter:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the HUD video")
