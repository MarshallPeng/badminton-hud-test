"""Step 1: normalise the video, extract the audio and find racket sounds."""
import json
import subprocess
import wave

import numpy as np
from scipy import signal


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    w, h = s["width"], s["height"]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(s.get("tags", {}).get("rotate", rot))
    if rot % 180:
        w, h = h, w
    num, den = s["r_frame_rate"].split("/")
    return w, h, float(num) / float(den)


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def normalise(src, dst_video, dst_wav):
    """Scale to 1080x1920 (portrait) or 1920x1080 (landscape input), 8-bit SDR, constant fps."""
    w, h, fps = probe(src)
    size = (1080, 1920) if h >= w else (1920, 1080)
    fps = round(fps) if abs(fps - round(fps)) < 0.05 else fps
    vf = (f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
          f"crop={size[0]}:{size[1]},fps={fps},format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf", vf,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                    str(dst_video)], check=True)
    audio = has_audio(src)
    if audio:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
                        "-ar", "44100", "-c:a", "pcm_s16le", str(dst_wav)], check=True)
    return {"width": size[0], "height": size[1], "fps": fps, "audio": audio}


def read_wav(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x, sr


def racket_onsets(wav_path, lo=2000, hi=8000, hop_s=0.005, min_gap_s=0.25):
    """Onsets of short broadband clicks in the 2-8 kHz band.

    Returns a list of (time_s, strength) where strength is in units of robust sigma
    above the median onset envelope.
    """
    x, sr = read_wav(wav_path)
    sos = signal.butter(4, [lo, min(hi, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x)
    hop = int(sr * hop_s)
    n = len(y) // hop
    e = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loge = 20 * np.log10(e)
    # onset strength: rise of the band energy over the previous 30 ms floor
    k = int(0.03 / hop_s)
    floor = np.array([loge[max(0, i - k):max(1, i - 1)].min() for i in range(n)])
    flux = np.clip(loge - floor, 0, None)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) * 1.4826 + 1e-6
    z = (flux - med) / mad
    peaks, props = signal.find_peaks(z, height=6.0, distance=int(min_gap_s / hop_s))
    return [(float(p * hop_s), float(h)) for p, h in zip(peaks, props["peak_heights"])]
