# Phase 1: camera and live perception

## 1. Confirm the starting point

Hardware reported by the user: Raspberry Pi 4 Model B, 4 GB RAM, 64 GB microSD card, Camera Module 3 Wide, fan, and thermal paste. A live preview from `rpicam-hello` was confirmed on 2026-09-02.

The supplied screenshot shows the `imx708_wide` tuning file, registration of the camera, and configuration of video streams. The warnings concern missing static sensor properties and default sensor delays. The preview confirms basic capture works; it does not prove sustained stability or precise timing. No fatal error is visible in the screenshot. Do not modify sensor databases or install experimental firmware solely to suppress these warnings.

`rpicam-hello` normally exits after about five seconds. To leave the preview running, use the following and stop with Ctrl+C. [Official camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)

```bash
rpicam-hello --timeout 0
```

Close the preview before running Python so the camera is available to SAGE.

Before installing anything, run these commands **on the Pi**, and share the output:

```bash
cat /etc/os-release
uname -m
python3 --version
```

The OS release, userland architecture, and package availability have not yet been confirmed. The installation recipe below is conditional on a supported Raspberry Pi OS installation, with its system Python. If the output identifies another distribution, adapt the setup before proceeding. `uname -m` identifies the kernel architecture; `getconf LONG_BIT` can additionally check the userland bitness.

## 2. Put the project on the Pi

Copy this project's source folder to the Pi using a USB drive or an existing file-transfer setup. For a fresh Pi-side copy, `~/sage` is the intended location; an existing project folder can be used directly. This work has not created or published a GitHub repository, so there is no clone URL to use yet.

Copy `AGENTS.md`, `README.md`, `.gitignore`, `src/`, `scripts/`, `tests/`, and `docs/`. You can omit local `data/`, `models/`, caches, and `dist/`; the model downloader retrieves the required files. If using the supplied starter ZIP, extract its contents into `~/sage`.

On the Pi, open a terminal in the copied project root:

```bash
cd ~/sage
python3 scripts/diagnose_pi.py
```

The diagnostic tool prints information and does not open the camera or change settings.

## 3. Install the small runtime on Raspberry Pi OS

After confirming the OS, use the distribution packages so Picamera2 and libcamera remain compatible. Picamera2's maintainers recommend APT for that reason. OpenCV supplies the detector runtime; no separate TensorFlow or PyTorch install is required. [Picamera2 installation guidance](https://github.com/raspberrypi/picamera2#installation)

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy python3-psutil
python3 -c "import cv2, numpy, psutil; from picamera2 import Picamera2; print('Python dependencies OK; OpenCV', cv2.__version__)"
```

Use `python3` from the Pi's normal terminal, outside any existing isolated environment, for this first check. If you later create a virtual environment, it needs access to the system Picamera2/libcamera packages. The project runs directly from its folder and does not require `pip install` or root privileges to capture.

## 4. Check Python capture without detection

Run from the Pi desktop with its display attached:

```bash
python3 scripts/run_phase1.py --camera-only --preview --seconds 30
```

Expected behavior: a live image with a SAGE overlay, periodic processed-FPS/RAM output, and a final summary. Press `q`, Escape, or Ctrl+C to stop early. Verify image color and focus, and that it follows movement without an accumulating delay.

For an SSH terminal or any session without a local graphical desktop, omit `--preview`. Inspect `first-frame.jpg` and `sample.avi` afterward. OpenCV windows require a working graphical session; the script does not stream a browser preview.

The initial capture request is 640×480 at 15 FPS with four camera buffers, automatic exposure, and continuous autofocus where supported. These are experiment settings, not promised measured rates. The selected sensor mode and actual configuration are saved. Calibration is explicitly unset until Phase 3.

## 5. Add live object detection

Download the pinned pretrained model (approximately 23 MB of weights):

```bash
python3 scripts/download_model.py
python3 scripts/run_phase1.py --preview --seconds 60
```

Try a well-lit chair, bottle, person, or sofa. The confidence threshold defaults to 0.5. Keep scenes simple for the first functional check, then vary lighting, distance, and motion. False positives and missed objects are useful evidence to record.

This baseline uses the public `chuanqi305/MobileNet-SSD` model, revision `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`. The download checks SHA-256 hashes and retains the upstream license and a source manifest. Its label set is VOC, not an 80-class COCO detector. It has no backpack or door class. [Model and training context](https://github.com/chuanqi305/MobileNet-SSD)

Detection uses OpenCV DNN on the CPU, a 300×300 input, the model's BGR normalization, and the model's built-in detection output. The camera adapter uses Picamera2 `RGB888`, which provides BGR byte order suitable for OpenCV. [OpenCV DNN API](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html), [Picamera2 manual, image formats](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

## 6. Read the saved evidence

Each invocation prints its new folder under `data/runs/`:

| File | Contents |
| --- | --- |
| `run.json` | Arguments, OS/runtime versions, model hashes, source configuration, and Git state when available |
| `frames.jsonl` | One record per processed frame: timestamps, dimensions, metadata, detections, timings, and recording index |
| `health.jsonl` | Approximately one sample per second plus a final sample: CPU, RAM, temperature, and raw throttle flags |
| `summary.json` | Completion/failure status, frame counts, throughput, mean/max latency, sampled memory and temperature |
| `first-frame.jpg` | First processed frame with annotations |
| `sample.avi` | Raw processed frames for replay |
| `annotated.avi` | The corresponding frames with detection overlays |

By default, video is saved only during the first 10 seconds, capped at 300 frames. JPEG/AVI output adds overhead; the saved arguments identify runs with recording enabled. Set `--record-seconds 0` to disable videos. Logs stream to disk and timing summaries use constant memory. Each run is finite; the default duration is 60 seconds.

The AVI files contain processed frames only, encoded at nominal `--fps`. They may play faster than the original scene when detection is slower than capture. The frame timestamps and recording indices in `frames.jsonl` preserve the association; do not estimate real elapsed time from AVI playback. Replay is unpaced and does not restore original capture timing.

Metric definitions:

- `camera_delivered_fps`: cadence of camera requests delivered to Picamera2, counted by a lightweight callback. This is separate from frames actually processed by SAGE.
- `processed_fps`: processed frames divided by measured run time, including capture waits and output/monitoring overhead.
- `inference_fps`: detector invocations divided by the same run time; unset in camera-only mode. Every processed frame is inferred in detection mode.
- `inference_ms`: preprocessing, model forward pass, and decoding time for a processed frame. One initial model warmup is excluded.
- `receive_to_result_ms`: frame delivery to detection result. It excludes time already spent in the camera pipeline and excludes later drawing, display, logging, and recording.
- `sensor_to_result_ms`: sensor timestamp to result on the Pi, using the matching Linux `CLOCK_BOOTTIME` clock. It includes camera pipeline delay but is not glass-to-glass display latency. It is unavailable for file replay or missing sensor timestamps. [libcamera timestamp definition](https://docs.libcamera.org/master/internal-api/namespacelibcamera_1_1controls.html)
- `camera_requests_not_processed`: delivered requests minus processed frames during the measured interval; includes deliberate skips and boundary effects. This is not a count of sensor/driver frame loss. The baseline does not separately measure driver drops.
- Process CPU percentage uses psutil's convention: 100% represents one fully occupied CPU core and the process can exceed 100%. System CPU is aggregate utilization across the machine. Sampled maximum RSS can miss short peaks.
- Pi temperature and throttle data remain null if unavailable. Historical throttle flags can describe an event before the experiment; keep the raw flags for interpretation against official documentation.

## 7. Collect a sustained baseline

After the 30-second capture and 60-second detection checks pass, run a ten-minute experiment with recording disabled and the preview off:

```bash
python3 scripts/run_phase1.py --seconds 600 --record-seconds 0
```

Keep the first run's 640×480 capture, 15 requested FPS, two OpenCV threads, and 0.5 threshold. Record the scene, lighting, approximate distances, and any missed detections separately. Then change one setting at a time if there is a measured bottleneck. This model always uses 300×300 inference input, so reducing camera resolution alone does not shrink its neural network computation.

Inspect stability, mean/max latency, temperatures, available memory, and throttle flags. Compare camera-only and detection runs. Do not claim real-time performance until you have measured the useful rate and latency on the Pi and defined what the application requires.

Replay a saved clip for debugging, replacing the example directory with a real run:

```bash
python3 scripts/run_phase1.py --video data/runs/YOUR_RUN/sample.avi --seconds 60 --record-seconds 0
```

On Windows, use `python` instead of `python3` if that is the interpreter with OpenCV, NumPy, and psutil. Replay does not import Picamera2. Replaying an annotated clip can affect detections; use the raw `sample.avi` file.

## Troubleshooting and completion

- Camera busy: close `rpicam-hello` and other camera applications, then retry.
- Missing Picamera2/libcamera: verify Raspberry Pi OS and the system Python/package environment before installing anything else.
- GUI/plugin error: omit `--preview` and inspect saved output. A GUI backend can terminate before Python can write a failure summary, so headless mode is the default.
- Missing model: run `scripts/download_model.py`; it does not train a model.
- Incorrect/missing labels: check the 20-class baseline before assuming capture is broken.
- Poor frame rate: share `run.json`, `summary.json`, and `health.jsonl`; benchmark before switching models or reducing settings.
- Capture stall: this first version has no independent hardware watchdog. Stop it manually and collect the camera logs if frame capture stops returning. It is not a motion-control component.

Phase 1 is complete when Python capture and live detection work on the Pi, a replayable sample and detections are saved, and a sustained run has measured throughput, latency, CPU/RAM use, temperature, and available throttle information. Maintain actual results and remaining issues in `docs/status.md`.
