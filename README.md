# SAGE: Semantic Autonomous Ground Explorer

*Real-time monocular semantic mapping and object-goal navigation on resource-constrained hardware.*

SAGE is an indoor robotics project targeting a Raspberry Pi 4 (4 GB) and Camera Module 3 Wide. The subtitle describes the intended system; mapping and navigation are not implemented yet.

**Completed milestone: Phase 1 — camera and live perception.** Camera capture and live MobileNet-SSD detection have been verified on the Raspberry Pi. The fan-cooled sustained baseline processes approximately 2.40 inference frames per second, with 409 ms mean sensor-to-result age, about 249 MB process memory, a 64.27 C peak temperature, and no throttling during a five-minute run. Phase 2 object tracking is the next milestone.

Start with the [Phase 1 setup and run guide](docs/Phase1/guide.md). It covers checking the Pi installation, copying the project, running camera-only capture, adding detection, and saving a baseline. See [project status](docs/Phase1/status.md) for verification evidence and [AGENTS.md](AGENTS.md) for the full roadmap.

The initial detector is MobileNet-SSD with 20 VOC object classes, running on the CPU through OpenCV. It recognizes classes including people, chairs, bottles, dining tables, and sofas. It does **not** recognize backpacks or doors. This is a baseline for testing the pipeline; later model choices should follow measurements on the Pi. [Model source](https://github.com/chuanqi305/MobileNet-SSD)

Run commands from the project root on the Pi after following the setup guide:

```bash
python3 scripts/run_phase1.py --camera-only --preview --seconds 30
python3 scripts/download_model.py
python3 scripts/run_phase1.py --preview --seconds 60 --threads 4
python3 scripts/run_phase1.py --seconds 300 --record-seconds 0 --threads 4
```

Each run creates a new directory under `data/runs/` with its configuration, detections, health measurements, summary, first annotated frame, and bounded video samples. Model weights and recordings are excluded from version control.

Local verification, with OpenCV, NumPy, and psutil available:

```bash
python3 -m unittest discover -s tests -v
```

The detector integration test uses downloaded weights and is skipped if they are absent. Synthetic replay tests verify software behavior, not detection accuracy or Raspberry Pi performance.
