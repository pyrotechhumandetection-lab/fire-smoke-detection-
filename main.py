#!/usr/bin/env python3
"""
Fire & smoke detection on an RTSP camera stream - CPU only.

    python main.py                        # camera from config.yaml, login from .env
    python main.py --source video.mp4     # test with a video file (or 0 = webcam)
    python main.py --benchmark            # measure model speed on this CPU and exit

Three threads keep the display smooth on a CPU-only laptop:
    FrameGrabber    reads the stream non-stop and keeps only the LATEST frame
    DetectorWorker  runs YOLO on every Nth frame (default 3) and raises alerts
    main thread     shows the newest frame with the latest boxes and an FPS counter

Quit with 'q' (or Esc) in the video window, or Ctrl+C in the terminal.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import statistics
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

# Ultralytics: no usage analytics / online checks from a CCTV box (set YOLO_OFFLINE=0 to allow).
os.environ.setdefault("YOLO_OFFLINE", "1")

APP_DIR = Path(__file__).resolve().parent
LABELS = ("fire", "smoke")
COLORS = {"fire": (0, 0, 255), "smoke": (255, 144, 30)}  # BGR: red, blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
NETWORK_SCHEMES = ("rtsp", "rtsps", "rtmp", "http", "https")

DEFAULT_CONFIG: dict = {
    "stream": {
        "url": "",
        "ffmpeg_options": "rtsp_transport;tcp",
        "buffer_size": 1,
        "open_timeout_s": 10,
        "read_timeout_s": 10,
        "reconnect_delay_s": 2,
        "reconnect_max_delay_s": 30,
    },
    "model": {
        "variant": "auto",
        "min_fps": 5.0,
        "imgsz": 416,
        "cpu_threads": 0,
        "models_dir": "models",
        "small": {"repo_id": "leeyunjai/yolo11-firedetect", "filename": "firedetect-11s.pt"},
        "nano": {"repo_id": "elvinguseinov/wildfire-detection-yolo11-yolo26-nano", "filename": "yolo11n.pt"},
    },
    "detection": {"every_n_frames": 3, "conf_threshold": 0.50, "class_conf": {}, "iou": 0.45},
    "alert": {"consecutive_frames": 3, "cooldown_s": 10, "snapshot_dir": "snapshots", "jpeg_quality": 90},
    "display": {"enabled": True, "window_name": "Fire & Smoke Detection", "max_width": 1280},
    "stats_interval_s": 30,
}


def log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def display_path(path: Path) -> str:
    """Path relative to the current folder with forward slashes, e.g. snapshots/fire_x.jpg."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


# -----------------------------------------------------------------------------
# Configuration and credentials
# -----------------------------------------------------------------------------
def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    if not path.is_file():
        log("CONFIG", f"{path} not found - using built-in defaults")
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[CONFIG] {path} must contain YAML key: value settings")
    return deep_merge(DEFAULT_CONFIG, data)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def add_credentials(url: str, username: str, password: str) -> str:
    """Insert URL-encoded credentials into a stream URL (so '@', ':', '#'... are safe)."""
    if not url or not username:
        return url
    parts = urlsplit(url)
    if parts.scheme.lower() not in NETWORK_SCHEMES or "@" in parts.netloc:
        return url  # not a network URL, or it already carries credentials
    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    return urlunsplit((parts.scheme, f"{userinfo}@{parts.netloc}", parts.path, parts.query, parts.fragment))


def mask_url(url: str) -> str:
    """Hide the password before a URL is printed."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in NETWORK_SCHEMES or "@" not in parts.netloc:
        return url
    userinfo, host = parts.netloc.rsplit("@", 1)
    user = userinfo.split(":", 1)[0]
    return urlunsplit((parts.scheme, f"{user}:****@{host}", parts.path, parts.query, parts.fragment))


def resolve_source(cfg: dict, cli_source: str | None) -> str | int:
    """Stream to open: --source, else RTSP_URL from .env, else stream.url from config.yaml."""
    if cli_source is not None:
        source = cli_source.strip()
        if source.isdigit():
            return int(source)  # webcam index
    else:
        source = os.getenv("RTSP_URL", "").strip() or str(cfg["stream"].get("url") or "").strip()
    if source and "://" not in source:  # local video file
        return str(Path(source).resolve())
    username = os.getenv("RTSP_USERNAME", "").strip()
    if source and not username and "@" not in urlsplit(source).netloc:
        log("CONFIG", "No RTSP_USERNAME / RTSP_PASSWORD found (.env missing?) - connecting without a login")
    return add_credentials(source, username, os.getenv("RTSP_PASSWORD", ""))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def canonical_label(name: object) -> str | None:
    """Map a model class name ('Fire', 'smoke', ...) to 'fire' / 'smoke' (None = ignore)."""
    text = str(name).strip().lower()
    if "smoke" in text:
        return "smoke"
    if "fire" in text or "flame" in text:
        return "fire"
    return None


def ensure_weights(repo_id: str, filename: str, models_dir: Path) -> Path:
    """Local weights file; downloaded from Hugging Face into models_dir on first use."""
    target = models_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    from huggingface_hub import hf_hub_download

    log("MODEL", f"Downloading {filename} from https://huggingface.co/{repo_id} ...")
    models_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=models_dir))
    log("MODEL", f"Saved {display_path(path)} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def set_torch_threads(count: int) -> None:
    if count > 0:
        import torch

        torch.set_num_threads(count)


@dataclass(frozen=True)
class Detection:
    label: str                       # "fire" or "smoke"
    conf: float
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in frame pixels


class Detector:
    """Ultralytics YOLO model running on the CPU."""

    def __init__(self, weights: Path, imgsz: int, det_cfg: dict):
        from ultralytics import YOLO  # heavy import (PyTorch): only when a model is needed

        self.weights = weights
        self.imgsz = imgsz
        self.iou = float(det_cfg["iou"])
        self.model = YOLO(str(weights), task="detect")
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.label_of = {k: canonical_label(v) for k, v in self.names.items()}
        if not any(self.label_of.values()):
            raise SystemExit(f"[MODEL] {weights.name} has no fire/smoke classes: {self.names}")
        default = float(det_cfg["conf_threshold"])
        self.thresholds = dict.fromkeys(LABELS, default)
        for name, value in (det_cfg.get("class_conf") or {}).items():
            if canonical_label(name):
                self.thresholds[canonical_label(name)] = float(value)
        self.min_conf = min(self.thresholds.values())
        self.bench_fps = 0.0

    @property
    def label(self) -> str:
        return f"{self.weights.stem}@{self.imgsz}"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.min_conf, iou=self.iou,
                                    device="cpu", verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        found = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            label = self.label_of.get(int(cls))
            if label and conf >= self.thresholds[label]:
                found.append(Detection(label, float(conf), (int(x1), int(y1), int(x2), int(y2))))
        return found

    def benchmark(self, runs: int = 8, warmup: int = 2) -> float:
        """Median seconds per frame (pre-processing + inference + NMS) on this CPU."""
        frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        for _ in range(warmup):
            self.detect(frame)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            self.detect(frame)
            times.append(time.perf_counter() - start)
        return statistics.median(times)


def build_detector(cfg: dict, variant: str, models_dir: Path, benchmark: bool = True) -> Detector:
    spec = cfg["model"][variant]
    weights = ensure_weights(spec["repo_id"], spec["filename"], models_dir)
    detector = Detector(weights, int(cfg["model"]["imgsz"]), cfg["detection"])
    if benchmark:
        seconds = detector.benchmark()
        detector.bench_fps = 1.0 / seconds
        log("MODEL", f"{variant} = {weights.name}, classes {detector.names}: {seconds * 1000:.0f} ms/frame "
                     f"at imgsz {detector.imgsz} -> {detector.bench_fps:.1f} FPS on this CPU")
    else:
        detector.detect(np.zeros((480, 640, 3), np.uint8))  # warm-up
        log("MODEL", f"{variant} = {weights.name} loaded, classes {detector.names}")
    return detector


def load_detector(cfg: dict, variant: str, models_dir: Path) -> tuple[Detector, Callable[[], Detector] | None]:
    """Model to start with, plus (auto mode only) a factory for the nano model that the
    worker switches to if the small model later runs below min_fps on live video."""
    import torch
    import ultralytics

    log("MODEL", f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
                 f"CPU only, {torch.get_num_threads()} threads")
    if variant == "nano":
        return build_detector(cfg, "nano", models_dir), None
    detector = build_detector(cfg, "small", models_dir)
    if variant == "small":
        return detector, None
    min_fps = float(cfg["model"]["min_fps"])
    try:  # fetch nano now, so a later switch works without internet
        nano = cfg["model"]["nano"]
        ensure_weights(nano["repo_id"], nano["filename"], models_dir)
    except Exception as exc:
        log("MODEL", f"WARNING: nano model unavailable ({exc}) - staying on the small model")
        return detector, None
    if detector.bench_fps >= min_fps:
        log("MODEL", f"Using the small model (switches to nano if it drops below {min_fps:g} FPS on live video)")
        return detector, lambda: build_detector(cfg, "nano", models_dir, benchmark=False)
    log("MODEL", f"small model is below {min_fps:g} FPS on this CPU -> switching to the nano model")
    del detector
    gc.collect()
    return build_detector(cfg, "nano", models_dir), None


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw_detections(img: np.ndarray, detections, sx: float = 1.0, sy: float = 1.0) -> None:
    h, w = img.shape[:2]
    thickness = max(2, round(min(h, w) / 320))
    scale = max(0.45, min(h, w) / 1000)
    text_thick = max(1, round(scale * 1.6))
    for det in detections:
        x1, y1, x2, y2 = (int(det.box[0] * sx), int(det.box[1] * sy), int(det.box[2] * sx), int(det.box[3] * sy))
        color = COLORS.get(det.label, (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.label} {det.conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, FONT, scale, text_thick)
        top = y1 - th - base - 4
        if top < 0:  # no room above the box: put the label inside it
            top = y1
        cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(img, text, (x1 + 3, top + th + 2), FONT, scale, (255, 255, 255), text_thick, cv2.LINE_AA)


def draw_panel(img: np.ndarray, lines: list[str], scale: float = 0.55) -> None:
    """White text on a darkened box in the top-left corner."""
    thick = 1 if scale < 0.8 else 2
    line_h = int(40 * scale)
    width = max(cv2.getTextSize(line, FONT, scale, thick)[0][0] for line in lines) + 16
    height = line_h * len(lines) + int(18 * scale)
    roi = img[:height, :width]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, line_h * (i + 1)), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_alert_banner(img: np.ndarray, labels) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
    text = " + ".join(label.upper() for label in labels) + " DETECTED"
    (tw, th), base = cv2.getTextSize(text, FONT, 0.9, 2)
    x, y = (w - tw) // 2, h - 20
    cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + base + 6), (0, 0, 255), -1)
    cv2.putText(img, text, (x, y), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def fit_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Copy of the frame, scaled down if wider than max_width (display only)."""
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        return cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_LINEAR)
    return frame.copy()


def annotate_snapshot(frame: np.ndarray, detections, when: datetime) -> np.ndarray:
    img = frame.copy()
    draw_detections(img, detections)
    best: dict[str, float] = {}
    for det in detections:
        best[det.label] = max(best.get(det.label, 0.0), det.conf)
    summary = "  ".join(f"{label.upper()} {conf:.2f}" for label, conf in best.items())
    draw_panel(img, [f"{when:%Y-%m-%d %H:%M:%S}  {summary}"], scale=max(0.55, min(img.shape[:2]) / 900))
    return img


# -----------------------------------------------------------------------------
# Alerts
# -----------------------------------------------------------------------------
class AlertManager:
    """Fire/smoke must be seen in N consecutive detection frames; then one alert
    (terminal line + annotated JPG) per class, at most once per cooldown."""

    def __init__(self, alert_cfg: dict, snapshot_dir: Path):
        self.needed = max(1, int(alert_cfg["consecutive_frames"]))
        self.cooldown = max(0.0, float(alert_cfg["cooldown_s"]))
        self.jpeg_quality = int(alert_cfg["jpeg_quality"])
        self.snapshot_dir = snapshot_dir
        self.streak = dict.fromkeys(LABELS, 0)
        self.last_alert: dict[str, float | None] = dict.fromkeys(LABELS, None)
        self.counts = dict.fromkeys(LABELS, 0)
        self.active: frozenset[str] = frozenset()

    def reset_streaks(self) -> None:
        self.streak = dict.fromkeys(LABELS, 0)
        self.active = frozenset()

    def update(self, detections, frame: np.ndarray, now: float | None = None,
               when: datetime | None = None) -> list[tuple[str, float, Path | None]]:
        now = time.monotonic() if now is None else now
        when = datetime.now() if when is None else when
        best: dict[str, float] = {}
        for det in detections:
            best[det.label] = max(best.get(det.label, 0.0), det.conf)

        due = []
        for label in LABELS:
            self.streak[label] = self.streak[label] + 1 if label in best else 0
            last = self.last_alert[label]
            if self.streak[label] >= self.needed and (last is None or now - last >= self.cooldown):
                self.last_alert[label] = now
                due.append(label)
        self.active = frozenset(label for label in LABELS if self.streak[label] >= self.needed)

        fired = []
        if due:
            snapshot = annotate_snapshot(frame, detections, when)
            for label in due:
                self.counts[label] += 1
                try:
                    path = self._save(snapshot, label, when)
                    outcome = f"saved {display_path(path)}"
                except Exception as exc:  # disk full, permissions... still report the alert
                    path, outcome = None, f"snapshot NOT saved ({exc})"
                print(f"[ALERT] {label.upper()} DETECTED at {when:%Y-%m-%d %H:%M:%S} "
                      f"(conf {best[label]:.2f}) -> {outcome}", flush=True)
                fired.append((label, best[label], path))
        return fired

    def _save(self, image: np.ndarray, label: str, when: datetime) -> Path:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{label}_{when:%Y%m%d_%H%M%S}"
        path, n = self.snapshot_dir / f"{stem}.jpg", 1
        while path.exists():
            path, n = self.snapshot_dir / f"{stem}_{n}.jpg", n + 1
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        path.write_bytes(buf.tobytes())  # works with non-ASCII paths, unlike cv2.imwrite
        return path


# -----------------------------------------------------------------------------
# Threads
# -----------------------------------------------------------------------------
class RateMeter:
    """Events per second over a sliding window (thread-safe)."""

    def __init__(self, window_s: float = 2.0):
        self.window = window_s
        self._times: deque[float] = deque()
        self._lock = threading.Lock()
        self._start = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._times.append(now)
            self._trim(now)

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            span = min(self.window, now - self._start)
            return len(self._times) / span if span > 0.5 else 0.0

    def _trim(self, now: float) -> None:
        while self._times and now - self._times[0] > self.window:
            self._times.popleft()


class FrameGrabber(threading.Thread):
    """Reads the stream as fast as it arrives and keeps only the newest frame,
    so the display never falls behind. Reconnects automatically."""

    def __init__(self, source: str | int, stream_cfg: dict):
        super().__init__(name="FrameGrabber", daemon=True)
        self.source = source
        self.cfg = stream_cfg
        self.title = mask_url(source) if isinstance(source, str) else f"webcam {source}"
        self.is_file = isinstance(source, str) and "://" not in source
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._stop_evt = threading.Event()
        self.generation = 0          # +1 on every (re)connection
        self.frames_total = 0
        self.connected = False
        self.status = "Connecting..."
        self.meter = RateMeter()

    def stop(self) -> None:
        self._stop_evt.set()
        with self._cond:
            self._cond.notify_all()

    def latest(self, last_id: int, timeout: float) -> tuple[np.ndarray | None, int, int]:
        """Newest frame; waits up to `timeout` s if nothing newer than last_id has arrived."""
        with self._cond:
            if self._frame_id == last_id and not self._stop_evt.is_set():
                self._cond.wait(timeout)
            return self._frame, self._frame_id, self.generation

    def _open(self) -> cv2.VideoCapture | None:
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
        else:
            # Read by OpenCV's FFmpeg backend when the capture opens: RTSP over TCP.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = str(self.cfg["ffmpeg_options"])
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(float(self.cfg["open_timeout_s"]) * 1000),
                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(float(self.cfg["read_timeout_s"]) * 1000)]
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg["buffer_size"]))
        return cap

    def run(self) -> None:
        base_delay = float(self.cfg["reconnect_delay_s"])
        max_delay = float(self.cfg["reconnect_max_delay_s"])
        failures = 0
        while not self._stop_evt.is_set():
            started = time.monotonic()
            try:
                cap = self._open()
            except Exception as exc:  # never let the grabber thread die: treat as a failed attempt
                log("STREAM", f"Error opening the stream: {exc!r}")
                cap = None
            if self._stop_evt.is_set():  # asked to quit while connecting
                if cap is not None:
                    cap.release()
                break
            if cap is None:
                failures += 1
                wait = min(base_delay * 2 ** (failures - 1), max_delay)
                log("STREAM", f"{clock()} Cannot open {self.title} (attempt {failures}, gave up after "
                              f"{time.monotonic() - started:.0f} s) - retrying in {wait:.0f} s")
                if failures == 1 and not self.frames_total and isinstance(self.source, str) and not self.is_file:
                    log("STREAM", "Check the camera address (stream.url in config.yaml or RTSP_URL in .env), "
                                  "RTSP_USERNAME / RTSP_PASSWORD in .env, and that this PC can reach the camera "
                                  "(test the same URL in VLC)")
                self.status = f"Cannot connect - retrying in {wait:.0f} s (attempt {failures})"
                self._stop_evt.wait(wait)
                continue
            failures = 0
            with self._cond:
                self.generation += 1
            self.connected, self.status = True, "Connected"
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_text = f" @ {fps:.0f} fps" if 0 < fps < 121 else ""
            if not (self.is_file and self.frames_total):  # not again for every replay of a video file
                log("STREAM", f"{clock()} Connected to {self.title} ({w}x{h}{fps_text})")

            try:
                frames = self._read_frames(cap, fps)
            except Exception as exc:
                log("STREAM", f"Error reading the stream: {exc!r}")
                frames = 0
            cap.release()
            self.connected = False
            if self._stop_evt.is_set():
                break
            if self.is_file and frames:
                log("STREAM", "End of video file - playing it again")
                continue
            log("STREAM", f"{clock()} Stream lost after {frames} frames - reconnecting in {base_delay:.0f} s")
            self.status = "Stream lost - reconnecting..."
            self._stop_evt.wait(base_delay)

    def _read_frames(self, cap: cv2.VideoCapture, fps: float) -> int:
        frames = 0
        pace = 1.0 / fps if self.is_file and 0 < fps < 121 else 0.0  # play files in real time
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            ok, frame = cap.read()  # returns False on error / EOF / read timeout
            if not ok or frame is None:
                break
            frames += 1
            self.frames_total += 1
            with self._cond:
                self._frame = frame  # older frames are simply dropped
                self._frame_id += 1
                self._cond.notify_all()
            self.meter.tick()
            if pace:
                next_t += pace
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop_evt.wait(delay)
                else:
                    next_t = time.monotonic()
        return frames


@dataclass(frozen=True)
class DetectionResult:
    detections: list
    shape: tuple[int, int]   # (h, w) of the analysed frame
    generation: int          # stream connection it came from


class DetectorWorker(threading.Thread):
    """Runs YOLO on submitted frames in the background and feeds the AlertManager.
    submit() never blocks: if the model is still busy, the frame is skipped.
    With a fallback (model.variant=auto) it switches to the nano model once the
    median speed of the last 10 detections is below min_fps (a few seconds of video)."""

    def __init__(self, detector: Detector, alerts: AlertManager, cpu_threads: int,
                 fallback: Callable[[], Detector] | None = None, min_fps: float = 0.0):
        super().__init__(name="DetectorWorker", daemon=True)
        self.detector = detector
        self.alerts = alerts
        self.cpu_threads = cpu_threads
        self.fallback = fallback
        self.min_fps = min_fps
        # 10 samples: a slow model is replaced within seconds, because every slow detection
        # delays alerts (below 5 FPS, fire -> alert takes more than about 1 s).
        self._recent: deque[float] = deque(maxlen=10)
        self._cond = threading.Condition()
        self._job: tuple | None = None
        self._busy = False
        self._stop_evt = threading.Event()
        self._generation: int | None = None
        self.result: DetectionResult | None = None
        self.meter = RateMeter(window_s=5.0)
        self.latency = 0.0   # smoothed seconds per detection
        self.runs = 0
        self.errors = 0

    def submit(self, frame: np.ndarray, generation: int) -> bool:
        with self._cond:
            if self._busy or self._job is not None:
                return False
            self._job = (frame, generation, datetime.now())
            self._cond.notify()
            return True

    def stop(self) -> None:
        self._stop_evt.set()
        with self._cond:
            self._cond.notify_all()

    def run(self) -> None:
        set_torch_threads(self.cpu_threads)
        last_error_log = 0.0
        while True:
            with self._cond:
                while self._job is None and not self._stop_evt.is_set():
                    self._cond.wait(0.5)
                if self._stop_evt.is_set():
                    return
                (frame, generation, when), self._job = self._job, None
                self._busy = True
            try:
                if generation != self._generation:  # new connection: count streaks from zero
                    self._generation = generation
                    self.alerts.reset_streaks()
                start = time.perf_counter()
                detections = self.detector.detect(frame)
                elapsed = time.perf_counter() - start
                self.latency = elapsed if self.runs == 0 else 0.8 * self.latency + 0.2 * elapsed
                self.runs += 1
                self.meter.tick()
                self.result = DetectionResult(detections, frame.shape[:2], generation)
                self.alerts.update(detections, frame, when=when)
                self._recent.append(elapsed)
                if self.fallback and len(self._recent) == self._recent.maxlen:
                    self._check_speed()
            except Exception as exc:  # keep running; report at most every 10 s
                self.errors += 1
                if time.monotonic() - last_error_log > 10:
                    last_error_log = time.monotonic()
                    log("ERROR", f"Detection failed: {exc!r}")
            finally:
                with self._cond:
                    self._busy = False

    def _check_speed(self) -> None:
        live_fps = 1.0 / statistics.median(self._recent)
        if live_fps >= self.min_fps:
            return
        log("MODEL", f"{clock()} {self.detector.label} runs at {live_fps:.1f} FPS on live video "
                     f"(< {self.min_fps:g}) -> switching to the nano model")
        factory, self.fallback = self.fallback, None
        try:
            new_detector = factory()
        except Exception as exc:
            log("MODEL", f"WARNING: could not load the nano model ({exc}) - keeping {self.detector.label}")
            return
        self.detector = new_detector
        self._recent.clear()
        gc.collect()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def process_ram_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def screen_size() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        return None


def render_status(last_frame: np.ndarray | None, text: str, max_width: int) -> np.ndarray:
    """Dimmed last frame (or black) with a status message, shown while disconnected."""
    if last_frame is None:
        canvas = np.zeros((360, 640, 3), np.uint8)
    else:
        canvas = fit_width(last_frame, max_width)
        canvas[:] = (canvas * 0.3).astype(np.uint8)
    h, w = canvas.shape[:2]
    for i, line in enumerate(("NO VIDEO", text)):
        scale = 1.0 if i == 0 else 0.6
        (tw, th), _ = cv2.getTextSize(line, FONT, scale, 2)
        cv2.putText(canvas, line, ((w - tw) // 2, h // 2 + i * 40), FONT, scale, (0, 200, 255), 2, cv2.LINE_AA)
    return canvas


def run(cfg: dict, args: argparse.Namespace, models_dir: Path, snapshot_dir: Path) -> int:
    source = resolve_source(cfg, args.source)
    if source == "":
        log("CONFIG", "No stream configured: set stream.url in config.yaml (or RTSP_URL in .env), or use --source")
        return 2

    set_torch_threads(int(cfg["model"]["cpu_threads"]))
    detector, fallback = load_detector(cfg, str(args.model or cfg["model"]["variant"]).lower(), models_dir)
    alerts = AlertManager(cfg["alert"], snapshot_dir)
    grabber = FrameGrabber(source, cfg["stream"])
    worker = DetectorWorker(detector, alerts, int(cfg["model"]["cpu_threads"]),
                            fallback, float(cfg["model"]["min_fps"]))
    every_n = max(1, int(cfg["detection"]["every_n_frames"]))

    show = bool(cfg["display"]["enabled"]) and not args.no_display
    win = str(cfg["display"]["window_name"])
    max_width = int(cfg["display"]["max_width"] or 0)
    if show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        except cv2.error:
            log("DISPLAY", "OpenCV was built without GUI support (opencv-python-headless?) - no window")
            show = False

    thresholds = ", ".join(f"{k} >= {v:.2f}" for k, v in detector.thresholds.items())
    log("INFO", f"Source: {grabber.title}")
    log("INFO", f"Model {detector.label} on every {every_n} frames | {thresholds} | alert after "
                f"{alerts.needed} consecutive detections, cooldown {alerts.cooldown:g} s")
    log("INFO", f"Snapshots -> {display_path(snapshot_dir)}/")
    log("INFO", "Press 'q' in the video window to quit" if show else "No window - press Ctrl+C to quit")

    grabber.start()
    worker.start()
    display_meter = RateMeter()
    last_id, since_submit, shown = 0, every_n, 0     # since_submit=every_n: analyse the 1st frame
    last_frame: np.ndarray | None = None
    window_shown, window_sized, last_status_draw = False, False, 0.0
    stats_every = float(cfg.get("stats_interval_s") or 0)
    next_stats = time.monotonic() + stats_every if stats_every > 0 else float("inf")
    deadline = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    started, reason = time.monotonic(), "finished"

    def show_image(image: np.ndarray, video: bool = True) -> None:
        nonlocal window_shown, window_sized
        cv2.imshow(win, image)
        window_shown = True
        if video and not window_sized:  # size the window once, from the first video frame
            h, w = image.shape[:2]
            screen = screen_size()
            scale = min(1.0, 0.9 * screen[0] / w, 0.85 * screen[1] / h) if screen else 1.0
            cv2.resizeWindow(win, int(w * scale), int(h * scale))
            window_sized = True

    try:
        while True:
            frame, frame_id, generation = grabber.latest(last_id, 0.03 if show else 0.25)
            now = time.monotonic()
            if frame is not None and frame_id != last_id:
                last_id, last_frame = frame_id, frame
                shown += 1
                display_meter.tick()
                since_submit += 1
                if since_submit >= every_n and worker.submit(frame, generation):
                    since_submit = 0
                if show:
                    canvas = fit_width(frame, max_width)
                    draw_panel(canvas, [
                        f"FPS {display_meter.rate():4.1f}   stream {grabber.meter.rate():4.1f}",
                        f"Detect {worker.meter.rate():3.1f}/s  {worker.latency * 1000:3.0f} ms  "
                        f"{worker.detector.label}",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ])
                    result = worker.result
                    fresh = result is not None and result.generation == generation
                    if fresh and result.detections:  # last known boxes, scaled to this frame
                        draw_detections(canvas, result.detections,
                                        canvas.shape[1] / result.shape[1], canvas.shape[0] / result.shape[0])
                    if fresh and alerts.active:
                        draw_alert_banner(canvas, sorted(alerts.active))
                    show_image(canvas)
            elif show and not grabber.connected and now - last_status_draw >= 0.5:
                last_status_draw = now
                show_image(render_status(last_frame, grabber.status, max_width), video=False)

            if show:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    reason = "'q' pressed"
                    break
                if window_shown and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    reason = "window closed"
                    break
            if now >= next_stats:
                next_stats = now + stats_every
                ram = process_ram_mb()
                log("STATS", f"{clock()} display {display_meter.rate():.1f} fps | stream {grabber.meter.rate():.1f} fps"
                             f" | detect {worker.meter.rate():.1f}/s ({worker.latency * 1000:.0f} ms)"
                             f" | {worker.detector.label}" + (f" | RAM {ram:.0f} MB" if ram else "")
                             + f" | alerts fire={alerts.counts['fire']} smoke={alerts.counts['smoke']}")
            if now >= deadline:
                reason = f"--duration {args.duration:g} s reached"
                break
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        log("INFO", f"Stopping ({reason}) ...")
        worker.stop()
        grabber.stop()
        worker.join(timeout=10)  # finishes the detection in progress
        # A running stream stops within one frame. If the grabber is stuck in a connection attempt
        # to an unreachable camera (up to open_timeout_s), don't make the user wait for it: it is
        # a daemon thread and ends with the process.
        grabber.join(timeout=2)
        if show:
            cv2.destroyAllWindows()
    log("INFO", f"Stopped after {time.monotonic() - started:.0f} s | frames shown {shown} | detections {worker.runs}"
                f" | alerts fire={alerts.counts['fire']} smoke={alerts.counts['smoke']}"
                + (f" | detection errors {worker.errors}" if worker.errors else ""))
    return 0


def run_benchmark(cfg: dict, models_dir: Path) -> int:
    """Speed of the small and nano models on this CPU at imgsz 416 and 480."""
    import torch
    import ultralytics

    set_torch_threads(int(cfg["model"]["cpu_threads"]))
    log("BENCH", f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
                 f"{torch.get_num_threads()} CPU threads")
    min_fps = float(cfg["model"]["min_fps"])
    for variant in ("small", "nano"):
        spec = cfg["model"][variant]
        detector = Detector(ensure_weights(spec["repo_id"], spec["filename"], models_dir), 416, cfg["detection"])
        for imgsz in (416, 480):
            detector.imgsz = imgsz
            seconds = detector.benchmark(runs=10)
            verdict = "OK" if 1 / seconds >= min_fps else f"below {min_fps:g} FPS"
            log("BENCH", f"{variant:5s} {detector.weights.name:18s} imgsz {imgsz}: {seconds * 1000:5.0f} ms/frame"
                         f" = {1 / seconds:5.1f} FPS  ({verdict})")
        del detector
        gc.collect()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fire & smoke detection on an RTSP stream (CPU only)")
    parser.add_argument("--config", default=str(APP_DIR / "config.yaml"), help="settings file (default: config.yaml)")
    parser.add_argument("--env", help="credentials file (default: .env next to the config file)")
    parser.add_argument("--source", help="override the stream: RTSP URL, video file, or webcam index such as 0")
    parser.add_argument("--model", choices=("auto", "small", "nano"), help="override model.variant")
    parser.add_argument("--imgsz", type=int, help="override model.imgsz, e.g. 416 or 480")
    parser.add_argument("--no-display", action="store_true", help="run without a video window")
    parser.add_argument("--duration", type=float, default=0, help="stop after N seconds (for testing)")
    parser.add_argument("--benchmark", action="store_true", help="measure model speed on this CPU and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    env_path = Path(args.env).resolve() if args.env else base_dir / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)  # real environment variables win
    elif args.env:
        raise SystemExit(f"[CONFIG] {env_path} not found")

    cfg = load_config(config_path)
    if args.imgsz:
        cfg["model"]["imgsz"] = args.imgsz
    imgsz = int(cfg["model"]["imgsz"])
    if imgsz % 32:
        cfg["model"]["imgsz"] = imgsz = max(32, round(imgsz / 32) * 32)
        log("CONFIG", f"model.imgsz must be a multiple of 32 - using {imgsz}")

    models_dir = resolve_path(cfg["model"]["models_dir"], base_dir)
    snapshot_dir = resolve_path(cfg["alert"]["snapshot_dir"], base_dir)
    log("INFO", f"Fire & smoke detector | Python {sys.version.split()[0]} | OpenCV {cv2.__version__}")
    if args.benchmark:
        return run_benchmark(cfg, models_dir)
    return run(cfg, args, models_dir, snapshot_dir)


if __name__ == "__main__":
    sys.exit(main())
