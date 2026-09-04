# Phase 2: object tracking

## 1. Question, scope, and starting point

**Question:** Is this the same object across nearby frames?

Phase 2 adds short-term image track IDs to Phase 1 MobileNet-SSD detections. It handles track creation, confirmation, short missed detections, reacquisition, and expiry. These IDs are scoped to one run and are not persistent semantic-map object IDs. A reappearing object receives a new ID after its earlier track expires.

Phase 1 is the verified prerequisite. Its fan-cooled Raspberry Pi baseline processes approximately 2.40 detector frames per second with 409 ms mean sensor-to-result age. Phase 2 therefore associates detector frames at their measured cadence; it does not produce or claim 15 FPS image tracking.

Phase 2 does not add geometric localization, visual odometry, semantic-map identity, custom model training, or rover control.

## 2. Tracker behavior

The initial tracker is deliberately small and dependency-free. It predicts each bounding box using its recent velocity, gates matches by detector class, and performs deterministic one-to-one association using intersection-over-union (IoU) and normalized center distance.

The lifecycle is:

```text
created/tentative -> confirmed -> lost -> confirmed (reacquired)
                                    \-> deleted (expired)
tentative -> deleted (if its next observation is missed)
```

Defaults:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--min-hits` | 2 | Consecutive observations needed to confirm a new track |
| `--max-missed-seconds` | 1.25 | Source-time gap allowed after a confirmed observation |
| `--min-iou` | 0.1 | Minimum overlap that can make a pair eligible |
| `--max-center-distance` | 1.5 | Alternative center-distance gate, normalized by object size |
| `--distance-weight` | 0.15 | Center-distance penalty in match ranking |
| `--velocity-smoothing` | 0.5 | Weight retained from the preceding velocity estimate |
| `--record-fps` | 2.4 | Approximate AVI playback rate, independent of requested camera FPS |

These are baseline experiment parameters, not established optima. Change one value at a time and record the reason and outcome.

Association is class-aware but not appearance-aware. It can still switch IDs when same-class objects cross, detections jump, or camera movement is too large. A `lost` bounding box is a prediction, not a current detection.

## 3. Software and local verification

Phase 2 uses the same Python, OpenCV, NumPy, psutil, Picamera2, and MobileNet-SSD setup as Phase 1. No SciPy, PyTorch, TensorFlow, or external tracking package is required.

From the project root, run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

On Windows, use `python` if that is the interpreter containing the project dependencies. Synthetic tests verify association and lifecycle rules; they do not establish tracking accuracy on real objects.

## 4. Replay an existing Phase 1 sample

Use the raw `sample.avi`, never the already annotated video. Supply its accompanying frame log so expiry and velocity use the original capture timestamps:

```bash
python3 scripts/run_phase2.py \
  --video data/runs/PHASE1_RUN/sample.avi \
  --source-frames data/runs/PHASE1_RUN/frames.jsonl \
  --seconds 120 \
  --record-seconds 0
```

Replay is unpaced, so `--seconds` is only a safety duration. `--source-frames` selects records with a non-null `recorded_video_frame_index` and restores their sensor or receive timestamps. The video and sidecar must come from the same Phase 1 run.

Without `--source-frames`, tracking uses the video's nominal timeline. That mode is acceptable for a software smoke test but not for evaluating time-based lifecycle behavior, because Phase 1 AVI files were written at the requested camera FPS rather than the measured detector cadence.

## 5. Run the live functional check on the Pi

Close other camera applications, confirm the fan is running, and execute:

```bash
python3 scripts/run_phase2.py --preview --seconds 60 --threads 4
```

Use well-lit objects from the VOC label set, such as people, chairs, bottles, dining tables, sofas, or potted plants. Start with one object, then introduce two objects, preferably two of the same class. Observe:

- a new orange `tentative` ID;
- confirmation after the next compatible observation;
- a stable ID during ordinary movement;
- a gray predicted box after a short miss;
- reacquisition with the same ID after a short occlusion;
- expiry and a new ID after a longer absence.

Press `q`, Escape, or Ctrl+C to stop early. If no graphical desktop is available, omit `--preview` and inspect `first-frame.jpg`, `annotated.avi`, and `frames.jsonl` afterward.

## 6. Record the evaluation sequence

Record a controlled 30-second run while keeping all processed frames from the interval. This should produce roughly 60–75 frames at the established Pi rate, keeping manual annotation manageable:

```bash
python3 scripts/run_phase2.py \
  --seconds 30 \
  --threads 4 \
  --record-seconds 30 \
  --record-max-frames 100 \
  --record-fps 2.4
```

Use a scene containing multiple objects and include ordinary motion, a short occlusion, an object leaving the frame, and a later re-entry. Avoid changing tracker settings while recording the initial baseline. Record scene, lighting, approximate object distances, object motions, and intentional occlusions in `docs/Phase2/status.md`.

Private camera footage and annotations belong under ignored `data/` paths unless deliberately prepared for publication.

## 7. Annotate and evaluate identities

Use the included frame-by-frame annotation tool on a computer with an OpenCV graphical window:

```bash
python scripts/annotate_phase2.py \
  --run data/pi-runs-phase2/PHASE2_RUN \
  --output data/annotations/phase2-eval.jsonl
```

The tool displays each raw frame and accepts commands in its terminal:

| Command | Action |
| --- | --- |
| `a OBJECT_ID CLASS_ID LABEL` | Draw and add an object, for example `a person-left 15 person` |
| `e OBJECT_ID` | Redraw that object's bounding box |
| `d OBJECT_ID` | Delete that object from the current frame |
| `c` | Copy the previous frame's annotations as a starting point |
| `n` / `p` | Move to the next or previous frame |
| `s` | Save progress |
| `q` | Save and quit; rerunning the command resumes the saved file |

Common VOC class IDs are bottle `5`, cat `8`, chair `9`, person `15`, and sofa `18`.

Create one ground-truth record for each evaluated Phase 2 `frame_id`. Use one physical-object ID only for the object's continuous visible trajectory. If an object fully leaves the scene beyond the intended short-term tracking interval, begin a new ground-truth trajectory when it returns; Phase 2 is not semantic re-identification.

Example:

```json
{"frame_id": 0, "objects": [{"object_id": "person-left", "class_id": 15, "label": "person", "bbox_xyxy": [42, 51, 188, 468]}]}
{"frame_id": 1, "objects": [{"object_id": "person-left", "class_id": 15, "label": "person", "bbox_xyxy": [49, 50, 196, 468]}]}
```

Bounding boxes use integer `xyxy` pixels with exclusive `x2/y2` and a top-left origin. Object IDs are annotation labels, not tracker IDs. Annotate the raw video while mapping its video index to `recorded_video_frame_index` in the run's `frames.jsonl`.

Evaluate the corresponding run:

```bash
python3 scripts/evaluate_phase2.py \
  --ground-truth data/annotations/phase2-eval.jsonl \
  --predictions data/runs-phase2/PHASE2_RUN/frames.jsonl \
  --iou-threshold 0.5 \
  --output data/runs-phase2/PHASE2_RUN/evaluation.json
```

The evaluator class-gates and IoU-matches observed track boxes to ground truth. Predicted-only `lost` boxes do not count as observations. It reports matched instances, misses, unmatched predictions, ID switches, fragmentation after a matched/unmatched/matched sequence, match fraction, and per-object results. These compact metrics are intended for the controlled Phase 2 dataset; they are not a full MOTChallenge implementation.

Use one sequence for initial tuning and a separate sequence for the reported check where possible. Document detector misses separately from incorrect tracker association when inspecting failures.

## 8. Read the saved evidence

Each invocation creates a directory under `data/runs-phase2/`:

| File | Contents |
| --- | --- |
| `run.json` | Environment, arguments, model hashes, tracker configuration, source, and Git state |
| `frames.jsonl` | Source timing, detections, tracks, lifecycle events, and per-frame timing |
| `health.jsonl` | CPU, RAM, temperature, and throttle samples |
| `summary.json` | Throughput, latency, tracker overhead, lifecycle counts, resource summaries, and run outcome |
| `first-frame.jpg` | First tracked annotation frame |
| `sample.avi` | Raw processed frames during the bounded recording interval |
| `annotated.avi` | The same frames with IDs and lifecycle states |

Phase 2 AVI files use `--record-fps` for approximate playback independently of the requested camera `--fps`. Because processing cadence varies, frame timestamps and recording indices in `frames.jsonl` remain authoritative; AVI duration is not an exact elapsed-time measurement.

Important fields:

- `source_timestamp_seconds` drives tracking and expiry.
- `source_timestamp_kind` records whether timing came from the live sensor, a Phase 1 sidecar, or the video timeline.
- `observed_this_frame=false` identifies a predicted lost track rather than a detector observation.
- `track_events` records `created`, `confirmed`, `lost`, `reacquired`, and `deleted` transitions.
- `tracking_ms` measures association and lifecycle processing separately from inference.

## 9. Sustained Pi comparison

After the functional and annotated checks, run a fan-cooled, headless five-minute comparison with recording disabled:

```bash
python3 scripts/run_phase2.py --seconds 300 --threads 4 --record-seconds 0
```

Compare processed FPS, inference time, sensor-to-result age, CPU, RSS, temperature, and throttle flags with the Phase 1 five-minute baseline. Report tracker update time separately. Do not attribute detector variation to tracking without comparing the measurements.

## Completion check

Phase 2 is complete only when a multi-object sequence demonstrates stable IDs and explicit lifecycle behavior, ID switches and fragmentation have been measured against manual annotations, failure cases are documented, and the detector-plus-tracker pipeline has a repeatable onboard Raspberry Pi run with resource measurements. Record actual evidence and remaining limitations in `docs/Phase2/status.md`.
