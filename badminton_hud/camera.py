"""Step 2: frame-to-frame camera motion from features on the ceiling and walls."""
import cv2
import numpy as np


def background_mask(frame, person_boxes=()):
    """Everything that is not floor (green) and not a person: ceiling, walls, lights."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    floor = cv2.inRange(hsv, (35, 20, 60), (100, 255, 255))
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = (floor == 0).astype(np.uint8)
    for b in person_boxes:
        x0, y0, x1, y1 = [int(v) for v in b[:4]]
        pw, ph = int(0.15 * (x1 - x0)), int(0.05 * (y1 - y0))
        m[max(0, y0 - ph):y1 + ph, max(0, x0 - pw):x1 + pw] = 0
    # stay away from the picture border (rolling-shutter / lens edge) and from the floor edge
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    return m


def frame_motion(prev, cur, mask_prev):
    """Homography mapping pixels of prev onto cur, estimated on background features."""
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=1500, qualityLevel=0.005, minDistance=12,
                                  mask=mask_prev, blockSize=7)
    if pts is None or len(pts) < 12:
        return np.eye(3), 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=4)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, nxt, None, winSize=(21, 21), maxLevel=4)
    fb = np.linalg.norm(back - pts, axis=2)[:, 0]
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 0.7)
    if ok.sum() < 12:
        return np.eye(3), int(ok.sum())
    H, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 1.5)
    if H is None:
        return np.eye(3), 0
    return H, int(inl.sum())


def all_motion(frames, boxes_per_frame):
    """C[t] maps frame t-1 pixels to frame t (C[0] = I). Also returns inlier counts."""
    C = [np.eye(3)]
    n_in = [0]
    for t in range(1, len(frames)):
        m = background_mask(frames[t - 1], boxes_per_frame[t - 1])
        H, k = frame_motion(frames[t - 1], frames[t], m)
        C.append(H)
        n_in.append(k)
    return np.array(C), np.array(n_in)


def warp_between(C, a, b):
    """Homography mapping frame a pixels to frame b pixels using the chain C."""
    H = np.eye(3)
    if b > a:
        for t in range(a + 1, b + 1):
            H = C[t] @ H
    else:
        for t in range(a, b, -1):
            H = np.linalg.inv(C[t]) @ H
    return H
