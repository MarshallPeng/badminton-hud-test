"""Steps 4 and 5: find the court in the first frame and follow it through the clip.

Court coordinates are metres on the floor: x runs across the court from the left
doubles sideline (as seen from the camera) to the right one (0 .. 6.10), y runs
along the court from the near baseline (0) to the far baseline (13.40).
"""
import itertools

import cv2
import numpy as np
from scipy.optimize import least_squares

COURT_W = 6.10
COURT_L = 13.40
NET_Y = 6.70
# Lines running away from the camera (constant x) and parallel to the net (constant y)
X_LINES = [0.0, 0.46, 3.05, 5.64, 6.10]
Y_LINES = [0.0, 0.76, 4.72, 8.68, 12.64, 13.40]


def model_segments(near_only=False):
    """Painted lines as ((x0, y0), (x1, y1)) in metres."""
    ymax = NET_Y if near_only else COURT_L
    segs = []
    for y in Y_LINES:
        if y <= ymax:
            segs.append(((0.0, y), (COURT_W, y)))
    for x in (0.0, 0.46, 5.64, 6.10):
        segs.append(((x, 0.0), (x, ymax)))
    segs.append(((3.05, 0.0), (3.05, 4.72)))
    if not near_only:
        segs.append(((3.05, 8.68), (3.05, COURT_L)))
    return segs


def model_lines(near_only=False):
    """Model lines as (kind, value, lo, hi): kind 'x' means x=value for y in [lo, hi]."""
    out = []
    for (x0, y0), (x1, y1) in model_segments(near_only):
        if y0 == y1:
            out.append(("y", y0, x0, x1))
        else:
            out.append(("x", x0, y0, y1))
    return out


def apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def in_front(H, pts):
    """True where the model points project in front of the camera (positive w)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return (np.c_[pts, np.ones(len(pts))] @ H.T)[:, 2] > 0


def sample_model(near_only=True, step=0.10):
    """Points along the painted model lines plus, for each, the line it belongs to."""
    pts, ids = [], []
    for i, ((x0, y0), (x1, y1)) in enumerate(model_segments(near_only)):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0)])
        ids.append(np.full(n, i))
    return np.vstack(pts), np.concatenate(ids)


# ----------------------------------------------------------------------------- white paint

def line_response(frame_bgr):
    """Ridge strength of thin white floor lines (float32, 0..255)."""
    mn = frame_bgr.min(axis=2)
    mx = frame_bgr.max(axis=2).astype(np.int16)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, k).astype(np.float32)
    # white = all channels high, so penalise saturated pixels
    sat = (mx - mn.astype(np.int16)).astype(np.float32)
    resp = th - np.clip(sat - 40, 0, None)
    return np.clip(resp, 0, 255)


def floor_mask(frame_bgr):
    """The green court floor, with the white lines on it filled in."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (35, 20, 60), (100, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return (g > 0).astype(np.uint8)


def white_mask(frame_bgr, person_boxes=(), thr=28):
    resp = line_response(frame_bgr) * floor_mask(frame_bgr)
    m = (resp > thr).astype(np.uint8)
    for b in person_boxes:
        x0, y0, x1, y1 = [int(v) for v in b[:4]]
        pad = int(0.08 * (x1 - x0))
        m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
    return m, resp


def coverage(H, mask, valid=None, near_only=True):
    """Fraction of projected model line points (inside the picture) that land on white paint."""
    pts, _ = sample_model(near_only, step=0.05)
    ok = in_front(H, pts)
    p = apply_h(H, pts)
    h, w = mask.shape
    ok &= np.isfinite(p).all(1)
    ok &= (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
    if ok.sum() < 20:
        return 0.0, 0
    xi = p[ok, 0].astype(int)
    yi = p[ok, 1].astype(int)
    if valid is not None:
        keep = valid[yi, xi] > 0
        xi, yi = xi[keep], yi[keep]
        if len(xi) < 20:
            return 0.0, 0
    md = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return float(md[yi, xi].mean()), int(len(xi))


# ----------------------------------------------------------------------------- Hough lines

def hough_segments(mask, min_len=40):
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 720, threshold=40,
                           minLineLength=min_len, maxLineGap=10)
    if segs is None:
        return np.zeros((0, 4))
    return np.asarray(segs).reshape(-1, 4).astype(np.float64)


def _seg_dirs(segs):
    d = segs[:, 2:4] - segs[:, 0:2]
    L = np.linalg.norm(d, axis=1)
    return d / L[:, None], L, (segs[:, 0:2] + segs[:, 2:4]) / 2


def _vp_inliers(v, segs, tol_deg):
    d, L, m = _seg_dirs(segs)
    if abs(v[2]) < 1e-9:
        to = np.tile(v[:2], (len(segs), 1))
    else:
        to = v[:2] / v[2] - m
    to = to / (np.linalg.norm(to, axis=1)[:, None] + 1e-12)
    cosang = np.abs((d * to).sum(1))
    return cosang > np.cos(np.radians(tol_deg))


def find_vp(segs, iters=3000, tol_deg=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(segs) < 2:
        return None, np.zeros(len(segs), bool)
    _, L, _ = _seg_dirs(segs)
    lines = np.cross(np.c_[segs[:, :2], np.ones(len(segs))], np.c_[segs[:, 2:], np.ones(len(segs))])
    p = L / L.sum()
    best, best_score, best_in = None, -1, None
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False, p=p)
        v = np.cross(lines[i], lines[j])
        n = np.linalg.norm(v)
        if n < 1e-12:
            continue
        v = v / n
        inl = _vp_inliers(v, segs, tol_deg)
        s = L[inl].sum()
        if s > best_score:
            best, best_score, best_in = v, s, inl
    # refine: least squares intersection of inlier lines
    li = lines[best_in]
    li = li / np.linalg.norm(li[:, :2], axis=1)[:, None]
    w = L[best_in]
    _, _, vt = np.linalg.svd(li * w[:, None])
    v = vt[-1]
    v = v / np.linalg.norm(v)
    return v, _vp_inliers(v, segs, tol_deg)


def _line_through(v, pt):
    l = np.cross(v, np.r_[pt, 1.0])
    return l / np.linalg.norm(l[:2])


def cluster_family(v, segs, img_shape, tol_px=None):
    """Group segments of one family into distinct lines through the vanishing point v."""
    h, w = img_shape[:2]
    cands = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]], float)
    if abs(v[2]) > 1e-9:
        vf = v[:2] / v[2]
        R = cands[np.argmax(np.linalg.norm(cands - vf, axis=1))]
    else:
        R = np.array([w / 2, h / 2])
    _, L, m = _seg_dirs(segs)
    ls = np.array([_line_through(v, mi) for mi in m])
    par = ls @ np.r_[R, 1.0]  # signed distance of R from each line
    order = np.argsort(par)
    tol = tol_px or 5.0
    clusters, cur = [], [order[0]] if len(order) else []
    for a, b in zip(order[:-1], order[1:]):
        if par[b] - par[a] < tol:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    if cur:
        clusters.append(cur)
    out = []
    for c in clusters:
        c = np.array(c)
        tot = L[c].sum()
        # a painted line gives long straight segments; lettering only gives short strokes
        if tot < 150 or L[c].max() < 100:
            continue
        pc = float((par[c] * L[c]).sum() / tot)
        # line through v at weighted mean parameter: refit from endpoints, keeping the VP
        pts = np.r_[segs[c, :2], segs[c, 2:]]
        wts = np.r_[L[c], L[c]]
        dense = np.vstack([np.linspace(sg[:2], sg[2:], max(2, int(np.hypot(*(sg[2:] - sg[:2])) / 10)))
                           for sg in segs[c]])
        if abs(v[2]) > 1e-9:
            vf = v[:2] / v[2]
            dvec = pts - vf
            ang = np.arctan2(dvec[:, 1], dvec[:, 0])
            a0 = ang[0]
            ang = (ang - a0 + np.pi) % (2 * np.pi) - np.pi + a0
            # directions may point either way from the VP; fold to one half-plane
            ref = np.arctan2(*(m[c[0]] - vf)[::-1])
            ang = np.where(np.abs(((ang - ref + np.pi) % (2 * np.pi)) - np.pi) > np.pi / 2,
                           ang + np.pi, ang)
            ang = (ang - ref + np.pi) % (2 * np.pi) - np.pi + ref
            a = float((ang * wts).sum() / wts.sum())
            p2 = vf + 100 * np.array([np.cos(a), np.sin(a)])
            l = np.cross(np.r_[vf, 1], np.r_[p2, 1])
        else:
            mid = (pts * wts[:, None]).sum(0) / wts.sum()
            l = _line_through(v, mid)
        l = l / np.linalg.norm(l[:2])
        out.append({"line": l, "param": pc, "length": float(tot), "pts": dense})
    return _merge_clusters(out)


def _merge_clusters(cls, tol=4.0):
    """Merge clusters whose lines coincide where they have evidence."""
    cls = sorted(cls, key=lambda c: c["param"])
    merged = True
    while merged:
        merged = False
        for i in range(len(cls) - 1):
            a, b = cls[i], cls[i + 1]
            da = np.median(np.abs(b["pts"] @ a["line"][:2] + a["line"][2]))
            db = np.median(np.abs(a["pts"] @ b["line"][:2] + b["line"][2]))
            if max(da, db) < tol:
                keep = a if a["length"] >= b["length"] else b
                keep = dict(keep)
                keep["pts"] = np.r_[a["pts"], b["pts"]]
                keep["length"] = a["length"] + b["length"]
                cls[i:i + 2] = [keep]
                merged = True
                break
    return cls


def _is_thin_line(cl, mask, inner=3, band=(6, 16)):
    """A painted line is a narrow ridge: little white just beside it (lettering is wide)."""
    l = cl["line"]
    pts = cl["pts"]
    d = np.array([-l[1], l[0]])
    proj = pts @ d
    t = np.linspace(proj.min(), proj.max(), 40)
    base = -l[2] * l[:2]  # closest point of the line to the origin
    cen = base[None, :] + t[:, None] * d[None, :]
    h, w = mask.shape
    offs = np.arange(-band[1], band[1] + 1)
    q = cen[:, None, :] + offs[None, :, None] * l[None, None, :2]
    qi = np.round(q).astype(int)
    ok = (qi[..., 0] >= 0) & (qi[..., 0] < w) & (qi[..., 1] >= 0) & (qi[..., 1] < h)
    qc = np.clip(qi, 0, [w - 1, h - 1])
    v = mask[qc[..., 1], qc[..., 0]] * ok
    centre = v[:, np.abs(offs) <= inner].max(1)
    side = v[:, (np.abs(offs) >= band[0])].mean(1)
    on = centre > 0
    if on.sum() < 5:
        return False
    return float(side[on].mean()) < 0.18


def _refine_vp(v, segs, inl):
    if inl.sum() < 2:
        return v, inl
    _, L, _ = _seg_dirs(segs)
    lines = np.cross(np.c_[segs[:, :2], np.ones(len(segs))], np.c_[segs[:, 2:], np.ones(len(segs))])
    li = lines[inl] / np.linalg.norm(lines[inl, :2], axis=1)[:, None]
    _, _, vt = np.linalg.svd(li * L[inl][:, None])
    v2 = vt[-1] / np.linalg.norm(vt[-1])
    return v2, _vp_inliers(v2, segs, 1.0)


def detect_line_families(frame, person_boxes=(), H_prior=None):
    """White floor lines grouped into the two families of the court.

    With H_prior the vanishing points come from the predicted court instead of RANSAC.
    """
    mask, resp = white_mask(frame, person_boxes)
    segs = hough_segments(mask)
    if len(segs) < 6:
        return None
    h, w = frame.shape[:2]
    c = np.array([w / 2, h / 2])
    if H_prior is not None:
        vs = H_prior @ np.array([0.0, 1.0, 0.0])
        vn = H_prior @ np.array([1.0, 0.0, 0.0])
        vs, vn = vs / np.linalg.norm(vs), vn / np.linalg.norm(vn)
        vs, ins = _refine_vp(vs, segs, _vp_inliers(vs, segs, 2.0))
        vn, inn = _refine_vp(vn, segs, _vp_inliers(vn, segs, 2.0) & ~ins)
        inn &= ~ins
        S = [c_ for c_ in cluster_family(vs, segs[ins], frame.shape) if _is_thin_line(c_, mask)] if ins.any() else []
        N = [c_ for c_ in cluster_family(vn, segs[inn], frame.shape) if _is_thin_line(c_, mask)] if inn.any() else []
        return {"vs": vs, "vn": vn, "S": S, "N": N, "mask": mask, "resp": resp, "segs": segs,
                "ins": ins, "inn": inn}
    rng = np.random.default_rng(1)
    v1, in1 = find_vp(segs, rng=rng)
    v2, in2r = find_vp(segs[~in1], rng=rng)
    if v1 is None or v2 is None:
        return None
    in2 = np.zeros(len(segs), bool)
    in2[np.flatnonzero(~in1)[in2r]] = True

    def dist(v):
        return np.inf if abs(v[2]) < 1e-9 else np.linalg.norm(v[:2] / v[2] - c)

    # the family running away from the camera converges close to the picture
    if dist(v1) <= dist(v2):
        vs, vn, ins, inn = v1, v2, in1, in2
    else:
        vs, vn, ins, inn = v2, v1, in2, in1
    S = [c for c in cluster_family(vs, segs[ins], frame.shape) if _is_thin_line(c, mask)]
    N = [c for c in cluster_family(vn, segs[inn], frame.shape) if _is_thin_line(c, mask)]
    return {"vs": vs, "vn": vn, "S": S, "N": N, "mask": mask, "resp": resp, "segs": segs,
            "ins": ins, "inn": inn}


# ----------------------------------------------------------------------------- fitting

def _intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def _fit_1d(xs, us):
    """Fit u = (a x + b) / (c x + 1); return params and RMS residual."""
    A = np.c_[xs, np.ones(len(xs)), -xs * us]
    sol, *_ = np.linalg.lstsq(A, us, rcond=None)
    a, b, c = sol
    pred = (a * xs + b) / (c * xs + 1)
    return sol, float(np.sqrt(np.mean((pred - us) ** 2)))


def h_from_lines(xmatch, ymatch):
    """Homography from matched lines: xmatch/ymatch are lists of (model value, image line)."""
    src, dst = [], []
    for xv, lx in xmatch:
        for yv, ly in ymatch:
            src.append((xv, yv))
            dst.append(_intersect(lx, ly))
    src, dst = np.float64(src), np.float64(dst)
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def refine_h(H, obs, iters=2, loss_scale=3.0):
    """Refine H so that observed image points lie on their model lines.

    obs: list of (kind, value, pts) with pts an (n, 2) array of image points.
    """
    if not obs:
        return H
    kinds = np.concatenate([[k == "x"] * len(p) for k, _, p in obs])
    vals = np.concatenate([[v] * len(p) for _, v, p in obs])
    pts = np.vstack([p for _, _, p in obs])
    h0 = H / H[2, 2]

    def res(params):
        Hm = np.r_[params, 1.0].reshape(3, 3)
        # model line in homogeneous coords: x = v -> (1, 0, -v); y = v -> (0, 1, -v)
        lm = np.where(kinds[:, None], np.c_[np.ones(len(vals)), np.zeros(len(vals)), -vals],
                      np.c_[np.zeros(len(vals)), np.ones(len(vals)), -vals])
        Hi = np.linalg.inv(Hm)
        li = lm @ Hi  # image lines (rows)  l_img = H^-T l_model
        li = li / np.linalg.norm(li[:, :2], axis=1)[:, None]
        return (li[:, 0] * pts[:, 0] + li[:, 1] * pts[:, 1] + li[:, 2])

    r = least_squares(res, h0.ravel()[:8], loss="huber", f_scale=loss_scale, max_nfev=200)
    return np.r_[r.x, 1.0].reshape(3, 3)


def obs_from_clusters(H, fam, near_only=False, max_px=12.0):
    """Match detected cluster evidence points to the nearest projected model line."""
    obs = []
    mls = model_lines(near_only)
    proj = []
    for kind, v, lo, hi in mls:
        a = (v, lo) if kind == "x" else (lo, v)
        b = (v, hi) if kind == "x" else (hi, v)
        pa, pb = apply_h(H, [a, b])
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            proj.append(None)
            continue
        l = np.cross(np.r_[pa, 1], np.r_[pb, 1])
        proj.append((l / np.linalg.norm(l[:2]), pa, pb))
    for cl in fam["S"] + fam["N"]:
        pts = cl["pts"]
        best, bd = None, max_px
        for (kind, v, lo, hi), pr in zip(mls, proj):
            if pr is None:
                continue
            l, pa, pb = pr
            # only the evidence within the model segment's extent counts
            t = ((pts - pa) @ (pb - pa)) / max(1e-9, (pb - pa) @ (pb - pa))
            inside = (t > -0.03) & (t < 1.03)
            if inside.sum() < max(3, 0.3 * len(pts)):
                continue
            d = np.abs(pts[inside] @ l[:2] + l[2])
            md = float(np.median(d))
            if md < bd:
                best, bd, keep = (kind, v), md, inside
        if best:
            obs.append((best[0], best[1], pts[keep]))
    return obs


def plausible(H, shape, min_area=0.02):
    """The near half must project in front of the camera as a convex, sizeable quad."""
    q = np.array([[0, 0], [COURT_W, 0], [COURT_W, NET_Y], [0, NET_Y]], float)
    if not in_front(H, q).all():
        return False
    p = apply_h(H, q)
    if not np.isfinite(p).all():
        return False
    cross = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        u, v = b - a, c - b
        cross.append(u[0] * v[1] - u[1] * v[0])
    if not (all(x > 0 for x in cross) or all(x < 0 for x in cross)):
        return False
    area = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
    return area > min_area * shape[0] * shape[1]


def fit_first_frame(frame, person_boxes=(), feet=()):
    """Detect the court in a still frame. Returns (H, info) with H: court metres -> pixels.

    People are not masked for the line search (that cuts the long lines into pieces);
    they are only left out when scoring how much white paint a fit covers.
    """
    fam = detect_line_families(frame)
    if fam is None:
        raise RuntimeError("not enough white lines found")
    S, N = fam["S"], fam["N"]
    vs = fam["vs"]
    h, w = frame.shape[:2]
    valid = np.ones((h, w), np.uint8)
    for b in person_boxes:
        x0, y0, x1, y1 = [int(v) for v in b[:4]]
        valid[max(0, y0):y1, max(0, x0):x1] = 0

    # order the net-parallel lines by closeness to the camera (distance from VP_S
    # along the S line through the bottom centre of the picture)
    vsf = vs[:2] / vs[2] if abs(vs[2]) > 1e-9 else None
    centre_line = _line_through(vs, np.array([w / 2, h]))
    for n in N:
        p = _intersect(n["line"], centre_line)
        n["depth"] = np.linalg.norm(p - vsf) if vsf is not None else p[1]
    N_sorted = sorted(N, key=lambda n: -n["depth"])
    # keep the strongest running-away lines; tiny fragments only slow the search down
    S = sorted(S, key=lambda c: -c["length"])[:10]
    cands = []
    for bi in range(min(5, len(N_sorted) - 1)):
        base, ls = N_sorted[bi], N_sorted[bi + 1]
        bdir = np.array([-base["line"][1], base["line"][0]])
        if bdir[0] < 0:
            bdir = -bdir
        # order S lines left -> right along the baseline candidate
        u = np.array([_intersect(c["line"], base["line"]) for c in S]) @ bdir
        order = np.argsort(u)
        u = u[order]
        Ssorted = [S[i] for i in order]
        m = len(Ssorted)
        for k in ((5, 4) if m >= 4 else (3,)):
            if m < k:
                continue
            for sub in itertools.combinations(range(m), k):
                for msub in itertools.combinations(range(5), k):
                    xs = np.array([X_LINES[j] for j in msub])
                    rms = _fit_1d(xs, u[list(sub)])[1] if k >= 4 else 4.0
                    # cross-ratio check (relative to the width of the matched lines)
                    if rms > 0.01 * abs(u[sub[-1]] - u[sub[0]]) + 2.0:
                        continue
                    cands.append((rms, bi, [(X_LINES[j], Ssorted[i]["line"]) for i, j in zip(sub, msub)],
                                  [(0.0, base["line"]), (0.76, ls["line"])]))
    if not cands:
        raise RuntimeError("could not match the court lines")
    feet = np.asarray(feet, float).reshape(-1, 2)

    def people_inside(H):
        if not len(feet):
            return 0
        q = apply_h(np.linalg.inv(H), feet)
        return int(((q[:, 0] > -0.5) & (q[:, 0] < COURT_W + 0.5) &
                    (q[:, 1] > -1.0) & (q[:, 1] < COURT_L + 1.0)).sum())

    # quick score for every candidate, full refinement for the best few
    quick = []
    for rms, bi, xm, ym in cands:
        H = h_from_lines(xm, ym)
        if H is None or not np.isfinite(H).all() or not plausible(H, frame.shape):
            continue
        cov, nn = coverage(H, fam["mask"], valid, near_only=True)
        if nn < 50:
            continue
        quick.append((cov + 0.05 * len(xm) + 0.05 * min(people_inside(H), 4), rms, bi, H, len(xm)))
    quick.sort(key=lambda q: -q[0])
    best = None
    for _, rms, bi, H, nx in quick[:15]:
        try:
            for _ in range(2):
                obs = obs_from_clusters(H, fam, near_only=False)
                H = refine_h(H, obs)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not plausible(H, frame.shape):
            continue
        cov_near, nn = coverage(H, fam["mask"], valid, near_only=True)
        cov_all, na = coverage(H, fam["mask"], valid, near_only=False)
        if nn < 50:
            continue
        inside = people_inside(H)
        score = cov_near + 0.5 * cov_all + 0.05 * min(inside, 4) - 0.01 * rms - 0.03 * bi
        if best is None or score > best[0]:
            best = (score, H, dict(cov_near=cov_near, cov_all=cov_all, inside=inside, rms=rms,
                                   baseline_rank=bi, n_x=nx))
    if best is None:
        raise RuntimeError("no court fit covered enough white paint")
    return best[1], best[2], fam


# ----------------------------------------------------------------------------- tracking

def snap(H, resp, valid, near_only=True, search=14, step=0.15, thr=20.0):
    """Move H so the projected near-half lines sit on the ridge of white paint."""
    h, w = resp.shape
    obs = []
    for kind, v, lo, hi in model_lines(near_only):
        n = max(3, int((hi - lo) / step))
        t = np.linspace(lo, hi, n)
        mp = np.c_[np.full(n, v), t] if kind == "x" else np.c_[t, np.full(n, v)]
        if not in_front(H, mp).all():
            continue
        p = apply_h(H, mp)
        d = np.gradient(p, axis=0)
        nrm = np.c_[-d[:, 1], d[:, 0]]
        nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-9
        offs = np.arange(-search, search + 1)
        q = p[:, None, :] + offs[None, :, None] * nrm[:, None, :]
        qi = np.round(q).astype(int)
        inside = (qi[..., 0] >= 0) & (qi[..., 0] < w) & (qi[..., 1] >= 0) & (qi[..., 1] < h)
        qc = np.clip(qi, 0, [w - 1, h - 1])
        r = resp[qc[..., 1], qc[..., 0]] * inside * (valid[qc[..., 1], qc[..., 0]] > 0)
        k = r.argmax(1)
        peak = r[np.arange(n), k]
        good = (peak > thr) & (k > 0) & (k < len(offs) - 1)
        if good.sum() < 3:
            continue
        # sub-pixel peak
        kk = k[good]
        rr = r[good]
        a, b, c = rr[np.arange(len(kk)), kk - 1], rr[np.arange(len(kk)), kk], rr[np.arange(len(kk)), kk + 1]
        den = a - 2 * b + c
        sub = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den, 0)
        pts = p[good] + (offs[kk] + sub)[:, None] * nrm[good]
        obs.append((kind, v, pts))
    if len(obs) < 3:
        return H, obs
    return refine_h(H, obs, loss_scale=2.0), obs


def redetect(frame, H_pred, person_boxes, valid):
    """Re-read the lines from the picture and fit them, using H_pred to label them."""
    fam = detect_line_families(frame, H_prior=H_pred)
    if fam is None:
        return None
    obs = obs_from_clusters(H_pred, fam, near_only=True, max_px=25.0)
    kinds = {(k, v) for k, v, _ in obs}
    if sum(k == "x" for k, _ in kinds) < 2 or sum(k == "y" for k, _ in kinds) < 2:
        return None
    try:
        H = refine_h(H_pred, obs)
        H = refine_h(H, obs_from_clusters(H, fam, near_only=True, max_px=8.0))
    except (np.linalg.LinAlgError, ValueError):
        return None
    return H


def draw_court(img, H, color=(0, 0, 255), thickness=2, near_only=False):
    out = img
    for (x0, y0), (x1, y1) in model_segments(near_only):
        t = np.linspace(0, 1, 40)
        mp = np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0)]
        ok = in_front(H, mp)
        p = apply_h(H, mp)
        for a, b, oa, ob in zip(p[:-1], p[1:], ok[:-1], ok[1:]):
            if oa and ob and np.isfinite(a).all() and np.isfinite(b).all() \
                    and np.abs(a).max() < 1e5 and np.abs(b).max() < 1e5:
                cv2.line(out, tuple(np.int32(a)), tuple(np.int32(b)), color, thickness, cv2.LINE_AA)
    # net
    pa, pb = apply_h(H, [(0, NET_Y), (COURT_W, NET_Y)])
    if np.isfinite(pa).all() and np.isfinite(pb).all():
        cv2.line(out, tuple(np.int32(pa)), tuple(np.int32(pb)), (0, 255, 255), thickness, cv2.LINE_AA)
    return out


def person_valid_mask(shape, person_boxes):
    valid = np.ones(shape[:2], np.uint8)
    for b in person_boxes:
        x0, y0, x1, y1 = [int(v) for v in b[:4]]
        pad = int(0.08 * (x1 - x0))
        valid[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
    return valid


def track_court(frames, H0, C, boxes_per_frame, redetect_every=2, log=None):
    """Court homography for every frame.

    Each frame: predict with the camera motion, snap onto the white lines (near half),
    and every `redetect_every` frames also re-read the lines from the picture; keep the
    fit that covers more white paint.
    """
    Hs = [H0]
    stats = [{"source": "first", "cov": None}]
    for t in range(1, len(frames)):
        f = frames[t]
        boxes = boxes_per_frame[t]
        mask, resp = white_mask(f, boxes)
        valid = person_valid_mask(f.shape, boxes)
        resp = resp * valid
        H_pred = C[t] @ Hs[-1]
        cands = [("predicted", H_pred)]
        try:
            H_snap, _ = snap(H_pred, resp, valid)
            H_snap, _ = snap(H_snap, resp, valid, search=6)
            cands.append(("snapped", H_snap))
        except (np.linalg.LinAlgError, ValueError):
            pass
        if redetect_every and t % redetect_every == 0:
            try:
                H_re = redetect(f, H_pred, boxes, valid)
                if H_re is not None:
                    H_re, _ = snap(H_re, resp, valid, search=6)
                    cands.append(("re-read", H_re))
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                pass
        scored = []
        for name, H in cands:
            if not np.isfinite(H).all() or not plausible(H, f.shape):
                continue
            cov, n = coverage(H, mask, valid, near_only=True)
            # guard against a fit that jumps far from the prediction
            jump = _mean_shift(H, H_pred)
            scored.append((cov - (0.5 if jump > 40 else 0), cov, name, H, jump))
        scored.sort(key=lambda s: -s[0])
        _, cov, name, H, jump = scored[0]
        Hs.append(H)
        stats.append({"source": name, "cov": cov, "jump": jump,
                      "covs": {s[2]: round(s[1], 3) for s in scored}})
        if log:
            log(t, stats[-1])
    return np.array(Hs), stats


def _mean_shift(Ha, Hb):
    pts = np.array([[0, 0], [COURT_W, 0], [0, NET_Y], [COURT_W, NET_Y]], float)
    a, b = apply_h(Ha, pts), apply_h(Hb, pts)
    return float(np.nanmean(np.linalg.norm(a - b, axis=1)))
