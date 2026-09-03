"""Phase 1 runner: capture, optional detection, annotated output, and measurements."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from sage.capture import PiCamera, VideoFile
from sage.detector import MobileNetSSD
from sage.metrics import Health, MeanMax, environment, sensor_age_ms


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--video", type=Path, help="Replay a file instead of opening the Pi camera")
    result.add_argument("--camera-only", action="store_true", help="Skip the detector")
    result.add_argument("--preview", action="store_true", help="Show a window on the local desktop")
    result.add_argument("--seconds", type=float, default=60, help="Run duration after warmup (default: 60)")
    result.add_argument("--warmup", type=float, default=2, help="Camera settling time, excluded from metrics")
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--fps", type=float, default=15, help="Requested camera FPS, not detector FPS")
    result.add_argument("--threads", type=int, default=2, help="OpenCV CPU threads")
    result.add_argument("--threshold", type=float, default=0.5)
    result.add_argument("--record-seconds", type=float, default=10, help="Save the first N seconds; 0 disables video")
    result.add_argument("--record-max-frames", type=int, default=300, help="Additional recording length cap")
    result.add_argument("--model-dir", type=Path, default=ROOT / "models" / "mobilenet-ssd")
    result.add_argument("--output", type=Path, default=ROOT / "data" / "runs", help="Parent for a new run folder")
    return result


def validate(args, argument_parser):
    for name in ("seconds", "fps", "width", "height", "threads", "record_max_frames"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("warmup", "record_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    if not 0 <= args.threshold <= 1:
        argument_parser.error("--threshold must be between 0 and 1")


def annotate(image, detections, frame_id, inference_ms):
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (80, 220, 80), 2)
        label = f"{detection['label']} {detection['confidence']:.2f}"
        cv2.putText(output, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 220, 80), 1, cv2.LINE_AA)
    status = f"SAGE | frame {frame_id} | "
    status += f"detector {inference_ms:.0f} ms" if inference_ms is not None else "camera only"
    cv2.putText(output, status, (8, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run(args):
    cv2.setNumThreads(args.threads)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving this run to {run_dir}", flush=True)
    manifest = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": vars(args), "environment": environment(), "model": None,
                "source": None, "bbox_convention": "xyxy pixels; exclusive x2/y2; origin top-left",
                "array_format": "BGR", "calibration": None,
                "recording": "Processed frames only. AVI uses nominal --fps; JSONL timestamps are authoritative."}
    write_json(run_dir / "run.json", manifest)
    source = None
    writers = []
    frames = recorded = 0
    start = None
    elapsed = 0
    status = "failed"
    failure = None
    inference = MeanMax()
    receive_to_result = MeanMax()
    sensor_to_result = MeanMax()
    temperatures = MeanMax()
    rss = MeanMax()
    source_stats = {}
    try:
        detector = None if args.camera_only else MobileNetSSD(args.model_dir, args.threshold)
        if detector:
            detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
            manifest["model"] = detector.info
        source = VideoFile(args.video) if args.video else PiCamera(args.width, args.height, args.fps)
        manifest["source"] = source.info
        write_json(run_dir / "run.json", manifest)
        if not args.video:
            time.sleep(args.warmup)
        if args.preview:
            cv2.namedWindow("SAGE Phase 1", cv2.WINDOW_NORMAL)
        health = Health()
        source.reset_stats()
        start = time.monotonic()
        last_health = start
        status = "duration-reached"
        with (run_dir / "frames.jsonl").open("w", encoding="utf-8", buffering=1) as frame_log, \
                (run_dir / "health.jsonl").open("w", encoding="utf-8", buffering=1) as health_log:
            while time.monotonic() - start < args.seconds:
                item = source.read()
                if item is None:
                    status = "end-of-video"
                    break
                image, timing = item
                detect_start = time.monotonic_ns()
                detections = detector.detect(image) if detector else []
                result_ns = time.monotonic_ns()
                inference_ms = (result_ns - detect_start) / 1e6 if detector else None
                receive_ms = (result_ns - timing["received_monotonic_ns"]) / 1e6
                sensor_ms = sensor_age_ms(timing["sensor_timestamp_boottime_ns"])
                inference.add(inference_ms)
                receive_to_result.add(receive_ms)
                sensor_to_result.add(sensor_ms)
                annotated = annotate(image, detections, frames, inference_ms)
                if frames == 0 and not cv2.imwrite(str(run_dir / "first-frame.jpg"), annotated):
                    raise RuntimeError("Could not save the first frame")
                video_index = None
                if time.monotonic() - start < args.record_seconds and recorded < args.record_max_frames:
                    if not writers:
                        height, width = image.shape[:2]
                        for name in ("sample.avi", "annotated.avi"):
                            writer = cv2.VideoWriter(str(run_dir / name), cv2.VideoWriter_fourcc(*"MJPG"),
                                                     args.fps, (width, height))
                            writers.append(writer)
                            if not writer.isOpened():
                                raise RuntimeError("MJPG recording is unavailable; try --record-seconds 0")
                    writers[0].write(image)
                    writers[1].write(annotated)
                    video_index = recorded
                    recorded += 1
                frame_log.write(json.dumps({"frame_id": frames, "width": image.shape[1],
                                            "height": image.shape[0], **timing,
                                            "result_monotonic_ns": result_ns,
                                            "inference_ms": inference_ms,
                                            "receive_to_result_ms": receive_ms,
                                            "sensor_to_result_ms": sensor_ms,
                                            "recorded_video_frame_index": video_index,
                                            "detections": detections}, default=str, allow_nan=False) + "\n")
                frames += 1
                if args.preview:
                    cv2.imshow("SAGE Phase 1", annotated)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        status = "user-stopped"
                        break
                    if cv2.getWindowProperty("SAGE Phase 1", cv2.WND_PROP_VISIBLE) < 1:
                        status = "user-stopped"
                        break
                if time.monotonic() - last_health >= 1:
                    sample = health.sample()
                    health_log.write(json.dumps(sample) + "\n")
                    temperatures.add(sample["temperature_c"])
                    rss.add(sample["process_rss_bytes"])
                    last_health = time.monotonic()
                    print(f"{frames} frames | {frames / (last_health - start):.1f} processed FPS | "
                          f"{sample['process_rss_bytes'] / 1024**2:.0f} MiB RAM", flush=True)
            sample = health.sample()
            health_log.write(json.dumps(sample) + "\n")
            temperatures.add(sample["temperature_c"])
            rss.add(sample["process_rss_bytes"])
        if frames == 0:
            raise RuntimeError("No frames were processed; check the video file or camera")
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "failed"
        failure = f"{type(error).__name__}: {error}"
    finally:
        elapsed = time.monotonic() - start if start is not None else 0
        if source is not None:
            try:
                source_stats = source.stats()
                source.close()
            except Exception as error:
                status = "failed"
                failure = failure or f"Camera cleanup failed: {error}"
        for writer in writers:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()
        camera_count = source_stats.get("camera_requests_received")
        summary = {"status": status, "error": failure, "frames_processed": frames,
                   "elapsed_seconds": elapsed, "processed_fps": frames / elapsed if elapsed else None,
                   "inference_fps": inference.count / elapsed if elapsed and inference.count else None,
                   **source_stats,
                   "camera_requests_not_processed": max(0, camera_count - frames) if camera_count is not None else None,
                   "recorded_frames": recorded, "inference_ms": inference.result(),
                   "receive_to_result_ms": receive_to_result.result(),
                   "sensor_to_result_ms": sensor_to_result.result(),
                   "sampled_rss_bytes": rss.result(), "sampled_temperature_c": temperatures.result()}
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    return 1 if status == "failed" else 130 if status == "interrupted" else 0


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    validate(args, argument_parser)
    return run(args)

