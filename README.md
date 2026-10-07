# Fire & Smoke Detection on an RTSP Camera (CPU only)

Watches an IP camera's RTSP stream, shows it live with fire/smoke boxes, and when fire or
smoke is confirmed it prints an alert and saves an annotated snapshot:

```
[ALERT] FIRE DETECTED at 2026-09-19 11:16:43 (conf 0.70) -> saved snapshots/fire_20260919_111643.jpg
```

Built for a Windows laptop with **8 GB RAM and no GPU**. The whole app uses about 450-560 MB of RAM.

| Requirement | How it is done |
|---|---|
| Model | `leeyunjai/yolo11-firedetect` → `firedetect-11s.pt` (classes fire, smoke), downloaded with `hf_hub_download` into `./models` on first run |
| Slow CPU fallback | `model.variant: auto` switches to the nano model (`elvinguseinov/wildfire-detection-yolo11-yolo26-nano`, `yolo11n.pt`) when the small model runs below `min_fps` (5). This is checked at startup **and** on the live video (within about 3 s). You can also force `small` or `nano` |
| Speed | fire in the stream → alert + snapshot in **about 0.8 s** on this laptop (box on screen after about 0.3 s), see "How fast is an alert?" |
| No hard-coded login | RTSP URL in `config.yaml` (or `.env`), username/password only in `.env` |
| Live window | boxes, label + confidence, FPS counter, detection speed, red "FIRE DETECTED" banner |
| Alert | confidence ≥ 0.50, seen in **3 consecutive** detection frames, then terminal alert + JPG in `./snapshots`, **10 s cooldown** per class |
| Low latency | `cv2.CAP_FFMPEG` + `rtsp_transport;tcp` + buffer size 1; a reader thread keeps only the **latest** frame |
| Low CPU | detection on every **3rd** frame at `imgsz` 416 (or 480) in a background thread; boxes are reused in between |
| Robust | auto-reconnect with back-off (camera off, network drop, frozen stream); `q` to quit cleanly |

---

## 1. Setup (once)

Needs Python 3.10 or newer (tested with Python 3.14.7 on Windows 10). Open **PowerShell**:

```powershell
cd "D:\fire&smoke"
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch from the **CPU-only** index (about 125 MB instead of about 2.5 GB for a CUDA build).
Tested versions: torch 2.14.0+cpu, ultralytics 8.4.155, opencv-python 5.0.0, huggingface_hub 1.32.

> The folder name contains `&`, so always keep the path in quotes (`cd "D:\fire&smoke"`).
> The commands call `.venv\Scripts\python.exe` directly, so you don't need to activate the
> venv. PowerShell's execution policy often blocks `Activate.ps1`.

## 2. Configure

Create your local configuration from the sanitized example:
```powershell
Copy-Item config.example.yaml config.yaml
```
`config.yaml` is intentionally ignored by Git so your camera address stays local.

1. **Credentials**: copy the template and put in the camera login:
   ```powershell
   copy .env.example .env
   notepad .env
   ```
   ```ini
   RTSP_USERNAME=admin
   RTSP_PASSWORD='your password'
   ```
   Special characters (`@ : # / %`) are URL-encoded by the app automatically. Wrap the password
   in **single quotes** if it contains spaces, `#` or `$`. `.env` is listed in `.gitignore`.

2. **Camera address**: edit `stream.url` in your local `config.yaml`, **without** username/password:
   ```yaml
   url: "rtsp://192.168.1.64:554/Streaming/Channels/102"
   ```
   Use the camera's **sub-stream** (e.g. Hikvision `.../102`, Dahua `subtype=1`). A 640x360 or
   704x576 stream costs far less CPU to decode than 1080p/4K and is plenty for detection.
   Alternatively set `RTSP_URL=...` in `.env`, which overrides `config.yaml`.

All other settings (thresholds, cooldown, model, image size...) are commented in `config.yaml`.

## 3. Run

```powershell
.venv\Scripts\python.exe main.py
```

- The first run downloads the models (about 19 MB + 5.5 MB) into `models\`. Later runs work offline.
- Quit: click the video window and press **q** (or Esc), close the window, or press Ctrl+C in the terminal.
- Snapshots go to `snapshots\fire_YYYYMMDD_HHMMSS.jpg` / `smoke_...jpg` (the folder is created automatically).

Useful options:

| Command | What it does |
|---|---|
| `main.py --benchmark` | measure both models on this CPU at imgsz 416 and 480, then exit |
| `main.py --model nano` | force the nano model (`small` / `auto` also possible) |
| `main.py --imgsz 480` | slightly better for small/far-away fires, about 15-25% slower |
| `main.py --no-display` | no window (e.g. running as a background service), Ctrl+C to stop |
| `main.py --source video.mp4` | test with a video file (it loops), or `--source 0` for the webcam |
| `main.py --duration 60` | stop automatically after 60 s |

Example terminal output (excerpt from the test run in section 6):

```
[INFO] Fire & smoke detector | Python 3.14.7 | OpenCV 5.0.0
[MODEL] ultralytics 8.4.155 | torch 2.14.0+cpu | CPU only, 2 threads
[MODEL] small = firedetect-11s.pt, classes {0: 'Fire', 1: 'Smoke'}: 167 ms/frame at imgsz 416 -> 6.0 FPS on this CPU
[MODEL] Using the small model (switches to nano if it drops below 5 FPS on live video)
[INFO] Source: rtsp://admin:****@127.0.0.1:8554/live
[INFO] Model firedetect-11s@416 on every 3 frames | fire >= 0.50, smoke >= 0.50 | alert after 3 consecutive detections, cooldown 10 s
[STREAM] 11:16:42 Connected to rtsp://admin:****@127.0.0.1:8554/live (1280x720 @ 15 fps)
[ALERT] FIRE DETECTED at 2026-09-19 11:16:43 (conf 0.70) -> saved snapshots/fire_20260919_111643.jpg
[MODEL] 11:16:50 firedetect-11s@416 runs at 4.2 FPS on live video (< 5) -> switching to the nano model
[ALERT] FIRE DETECTED at 2026-09-19 11:16:53 (conf 0.68) -> saved snapshots/fire_20260919_111653.jpg
[ALERT] SMOKE DETECTED at 2026-09-19 11:17:04 (conf 0.84) -> saved snapshots/smoke_20260919_111704.jpg
[STREAM] 11:17:06 Stream lost after 390 frames - reconnecting in 2 s
...
[STREAM] 11:17:22 Connected to rtsp://admin:****@127.0.0.1:8554/live (1280x720 @ 15 fps)
[ALERT] FIRE DETECTED at 2026-09-19 11:17:27 (conf 0.68) -> saved snapshots/fire_20260919_111727.jpg
[INFO] Stopping ('q' pressed) ...
```

## 4. How it works (why it stays smooth on a CPU)

```
 camera --RTSP/TCP--> [FrameGrabber thread] -- newest frame only --> [main thread: window, FPS, keys]
                                                                             | every 3rd frame
                                                                             v (skipped if still busy)
                                                                  [DetectorWorker thread: YOLO]
                                                                             |
                                                  boxes for display <--------+--> AlertManager
                                                                                  (3 in a row, cooldown,
                                                                                   snapshot + [ALERT])
```

- **FrameGrabber** reads frames as fast as the camera sends them and overwrites a single
  "latest frame" slot, so old frames are dropped and the picture never lags behind. OpenCV's
  FFmpeg backend ignores `CAP_PROP_BUFFERSIZE` (it is still set, as requested). This thread is
  what actually prevents lag.
- **DetectorWorker** runs YOLO in the background, so a slow detection never freezes the video.
  In the tests the window stayed at the camera's 15 fps even while detection ran at only 2-5/s.
- **Alerts** count *consecutive detection frames* per class (fire and smoke separately). A
  reconnect restarts the count, and the snapshot is the exact frame that was analysed, with its boxes.
- **Reconnect**: a dropped connection is noticed at once. A frozen stream (connection open, no
  video) is noticed after `read_timeout_s` (10 s). Retries back off 2 → 4 → 8 → ... → 30 s. On
  Windows each attempt against a camera that is offline can take up to `open_timeout_s` (10 s).
- Ultralytics analytics/online checks are switched off (`YOLO_OFFLINE=1`). Set `YOLO_OFFLINE=0`
  in the environment to allow them.

## 5. Performance on this laptop

Measured on the target machine: Intel Core i5-6300U (2 cores / 4 threads, 2015), 8 GB, no GPU, PyTorch CPU.
Time per detection, including pre/post-processing. The ranges cover several runs with normal
background apps open; this laptop's speed varies a lot with load and heat:

| Model | imgsz 416 | imgsz 480 |
|---|---|---|
| small `firedetect-11s.pt` | 135-209 ms (4.8-7.4 FPS) | 207-263 ms (3.8-4.8 FPS) |
| nano `yolo11n.pt` (default nano) | 59-98 ms (10-17 FPS) | 115 ms (8.7 FPS) |
| nano `yolo26n.pt` | 62-89 ms (11-16 FPS) | 97 ms (10.3 FPS) |

- With another heavy app running (an IDE using about 70% CPU), the same models took 316 ms (small)
  and 160 ms (nano). That is exactly why `auto` also re-checks the speed on the live video.
- With a 1280x720 stream decoding at the same time, the small model dropped to 3-5 FPS on this
  laptop, so `auto` switched to nano within about 3 seconds. Expect the nano model here unless the
  PC is otherwise idle.
- 2 PyTorch threads (= physical cores, the default) were faster than 4. Leave `cpu_threads: 0`.
- **Accuracy trade-off**: on the test images the small model found smoke that nano missed. If
  your CPU allows it (e.g. a sub-stream and an otherwise idle PC), try `variant: small`.
- OpenVINO was also tried and was *slower* than PyTorch on this CPU (288 ms vs 192 ms), so it is not used.

Tips: use the camera sub-stream, keep `imgsz: 416`, raise `every_n_frames` to 4-5 if the
laptop gets hot, close heavy applications on a dedicated monitoring PC.

### How fast is an alert?

The video is live: frames are shown as they arrive, and old frames are dropped, never queued.
Measured from the moment fire appears in the stream to (a) the box appearing in the window
and (b) the `[ALERT]` line being printed with the snapshot already saved. These are 7-8 fire
events per row, measured on this laptop while other applications used about 80% of the CPU:

| Setup | Box on screen | Alert + snapshot saved |
|---|---|---|
| nano, 25 fps stream | 0.18-0.41 s (avg 0.31) | 0.54-1.00 s (avg 0.78) |
| nano, 15 fps stream | 0.15-0.48 s (avg 0.30) | 0.56-1.66 s (avg 0.84) |
| small, 25 fps stream | 0.39-0.82 s (avg 0.65) | 1.25-1.85 s (avg 1.60) |
| `auto` (default) | small → nano after 3 s | 0.55-0.85 s once on nano |

- The box appears after the **first** detection. The alert waits for **3 detections in a row**
  (about 0.2 s each) to filter out false alarms. Saving the snapshot adds about 20 ms.
- Add the camera's own delay (encoding + network, typically 0.2-0.5 s) for the time from a real flame.
- For a faster alert: fewer other programs running (detection then runs every 0.12 s instead of
  0.2 s), or `consecutive_frames: 2` (about 0.2 s faster, but more false alarms). The small model
  is too slow for alerts under 1 s on this laptop.
- A flame must be big enough in the picture for the model to see it. A tiny, distant flame is
  only detected once it grows, however fast the pipeline is.

## 6. Testing without a camera

**Unit tests** (alert rules, cooldown, file names, credential encoding), in under a second:
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Local fake camera**: `tools\rtsp_test_server.py` streams your own pictures as a real
RTSP/TCP camera (MJPEG), optionally with a Digest login. Use any street photo plus a fire photo:
```powershell
# terminal 1: a fake camera showing a street for 10 s, then fire for 20 s (loops)
.venv\Scripts\python.exe tools\rtsp_test_server.py --scene C:\pics\street.jpg:10 --scene C:\pics\fire.jpg:20

# terminal 2: the detector
.venv\Scripts\python.exe main.py --source rtsp://127.0.0.1:8554/live
```
Kill terminal 1 and start it again to watch the auto-reconnect. `--stall-after 15` freezes the
video once (connection kept open) to test the read timeout, and `--user/--password` require a login.

**Webcam**: `main.py --source 0`, then show a fire video on a phone to the webcam.

What was tested on this laptop (all passed): model download and class check, benchmarks, 11 unit
tests, and end-to-end runs against the fake camera with a Digest login and the password
`Fire@Test#1:x` from a `.env` file:
- fire and smoke alerts, 10 s cooldown, snapshots;
- the small → nano switch (both at startup and during live video);
- camera killed, then restarted: reconnected;
- frozen stream: detected after 10 s, reconnected;
- wrong password: clear 401 message, retry back-off;
- `q` in the window: clean exit, code 0;
- video-file source with looping.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open rtsp://... retrying` | Check the IP/port/path, and open the same URL in VLC (Media → Open Network Stream). Make sure the PC and the camera are on the same network. |
| `method DESCRIBE failed: 401 Unauthorized` | Wrong `RTSP_USERNAME` / `RTSP_PASSWORD` in `.env` (quote the password if it has spaces, `#` or `$`). |
| `[ WARN ...] Stream timeout triggered after 10000 ms` | Printed by OpenCV when the camera stops answering. Expected; the app reconnects by itself. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless, shown once while the models download. |
| Low FPS / laptop hot | Use the sub-stream, `imgsz: 416`, `variant: nano`, `every_n_frames: 4`. |
| Too many false alarms | Raise `conf_threshold` (e.g. 0.6) or `consecutive_frames` (e.g. 5), or set `class_conf: {smoke: 0.6}`. |
| No window appears | Something installed `opencv-python-headless`. Run `pip uninstall opencv-python-headless opencv-python`, then `pip install opencv-python`. |

## Files

```
main.py                    the application
config.yaml                settings (no passwords)
.env.example               template for .env (camera username/password)
requirements.txt           CPU-only dependencies
tools/rtsp_test_server.py  local fake RTSP camera for testing
tests/test_main.py         unit tests
models/                    downloaded weights (created on first run)
snapshots/                 alert images (created on first alert)
```
