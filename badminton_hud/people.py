"""Step 3: people detection and tracking (YOLO11m + ByteTrack, imgsz 1280)."""
import numpy as np


def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def track_people(video_path, model_name="yolo11m.pt", imgsz=1280, progress=None):
    """Returns a list (one per frame) of arrays [track_id, x0, y0, x1, y1, conf]."""
    from ultralytics import YOLO
    model = YOLO(model_name)
    device = pick_device()
    out = []
    for i, r in enumerate(model.track(source=str(video_path), stream=True, persist=True,
                                      tracker="bytetrack.yaml", classes=[0], imgsz=imgsz,
                                      conf=0.15, iou=0.6, device=device, verbose=False)):
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            rows = np.zeros((0, 6))
        else:
            rows = np.c_[b.id.cpu().numpy(), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()]
        out.append(rows)
        if progress:
            progress(i)
    return out, str(device)
