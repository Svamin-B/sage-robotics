"""Phase 2 runner: detection, short-term tracking, artifacts, and measurements."""

import argparse
from collections import Counter
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
from sage.tracker import MultiObjectTracker


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--video", type=Path, help="Replay a raw video instead of opening the Pi camera")
    result.add_argument("--source-frames", type=Path,
                        help="Phase 1 frames.jsonl supplying original timestamps for --video")
    result.add_argument("--preview", action="store_true", help="Show a window on the local desktop")
    result.add_argument("--seconds", type=float, default=60)
    result.add_argument("--warmup", type=float, default=2)
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--fps", type=float, default=15, help="Requested camera FPS, not detector FPS")
    result.add_argument("--threads", type=int, default=4, help="OpenCV CPU threads")
    result.add_argument("--threshold", type=float, default=0.5)
    result.add_argument("--min-hits", type=int, default=2)
    result.add_argument("--max-missed-seconds", type=float, default=1.25)
    result.add_argument("--min-iou", type=float, default=0.1)
    result.add_argument("--max-center-distance", type=float, default=1.5)
    result.add_argument("--distance-weight", type=float, default=0.15)
    result.add_argument("--velocity-smoothing", type=float, default=0.5)
    result.add_argument("--record-seconds", type=float, default=10)
    result.add_argument("--record-max-frames", type=int, default=300)
    result.add_argument("--record-fps", type=float, default=2.4,
                        help="Approximate playback FPS for saved AVI files (default: 2.4)")
    result.add_argument("--model-dir", type=Path, default=ROOT / "models" / "mobilenet-ssd")
    result.add_argument("--output", type=Path, default=ROOT / "data" / "runs-phase2")
    return result


def validate(args, argument_parser):
    for name in ("seconds", "fps", "width", "height", "threads", "record_max_frames", "record_fps",
                 "min_hits"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("warmup", "record_seconds", "max_missed_seconds", "max_center_distance",
                 "distance_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    for name in ("threshold", "min_iou", "velocity_smoothing"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            argument_parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.source_frames and not args.video:
        argument_parser.error("--source-frames requires --video")


def load_source_timestamps(path):
    timestamps = []
    if path is None:
        return timestamps
    if not path.is_file():
        raise FileNotFoundError(f"Source frame log not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recorded = [record for record in records if record.get("recorded_video_frame_index") is not None]
    recorded.sort(key=lambda record: record["recorded_video_frame_index"])
    expected = list(range(len(recorded)))
    actual = [record["recorded_video_frame_index"] for record in recorded]
    if actual != expected:
        raise ValueError("Source frame log has non-contiguous recorded video indices")
    for record in recorded:
        sensor_ns = record.get("sensor_timestamp_boottime_ns")
        received_ns = record.get("received_monotonic_ns")
        if sensor_ns is not None:
            timestamps.append(float(sensor_ns) / 1e9)
        elif received_ns is not None:
            timestamps.append(float(received_ns) / 1e9)
        else:
            raise ValueError("A recorded source frame has no usable timestamp")
    return timestamps


def source_timestamp(timing, sidecar_timestamps, frame_id):
    if sidecar_timestamps:
        if frame_id >= len(sidecar_timestamps):
            raise ValueError("Video contains more frames than the source frame log")
        return sidecar_timestamps[frame_id], "source-frames-jsonl"
    if timing["sensor_timestamp_boottime_ns"] is not None:
        return timing["sensor_timestamp_boottime_ns"] / 1e9, "sensor-boottime"
    if timing["video_time_ms"] is not None:
        return timing["video_time_ms"] / 1000, "video-timeline"
    return timing["received_monotonic_ns"] / 1e9, "receive-monotonic"


def annotate(image, tracks, frame_id, inference_ms, tracking_ms):
    output = image.copy()
    colors = {"confirmed": (80, 220, 80), "tentative": (0, 180, 255), "lost": (150, 150, 150)}
    for track in tracks:
        x1, y1, x2, y2 = track["bbox_xyxy"]
        color = colors[track["state"]]
        thickness = 1 if track["state"] == "lost" else 2
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
        suffix = " predicted" if not track["observed_this_frame"] else ""
        label = f"{track['label']} #{track['track_id']} {track['state']}{suffix}"
        cv2.putText(output, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    status = f"SAGE P2 | frame {frame_id} | detect {inference_ms:.0f} ms | track {tracking_ms:.2f} ms"
    cv2.putText(output, status, (8, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run(args):
    cv2.setNumThreads(args.threads)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving this run to {run_dir}", flush=True)
    tracker = MultiObjectTracker(args.min_hits, args.max_missed_seconds, args.min_iou,
                                 args.max_center_distance, args.distance_weight, args.velocity_smoothing)
    manifest = {
        "schema_version": 2,
        "phase": 2,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "environment": environment(),
        "model": None,
        "tracker": tracker.configuration,
        "source": None,
        "bbox_convention": "xyxy pixels; exclusive x2/y2; origin top-left",
        "track_id_scope": "short-term image track; unique only within this run; not a semantic-map ID",
        "array_format": "BGR",
        "calibration": None,
        "recording": "Processed frames only. AVI playback FPS is approximate; JSONL timestamps are authoritative.",
    }
    write_json(run_dir / "run.json", manifest)
    source = None
    writers = []
    frames = recorded = 0
    start = None
    status = "failed"
    failure = None
    inference = MeanMax()
    tracking = MeanMax()
    total_result = MeanMax()
    sensor_to_result = MeanMax()
    temperatures = MeanMax()
    rss = MeanMax()
    event_counts = Counter()
    maximum_active_tracks = 0
    source_stats = {}
    timing_kind = None
    try:
        sidecar_timestamps = load_source_timestamps(args.source_frames)
        detector = MobileNetSSD(args.model_dir, args.threshold)
        detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
        manifest["model"] = detector.info
        source = VideoFile(args.video) if args.video else PiCamera(args.width, args.height, args.fps)
        manifest["source"] = source.info
        manifest["source"]["timestamp_sidecar"] = str(args.source_frames.resolve()) if args.source_frames else None
        write_json(run_dir / "run.json", manifest)
        if not args.video:
            time.sleep(args.warmup)
        if args.preview:
            cv2.namedWindow("SAGE Phase 2", cv2.WINDOW_NORMAL)
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
                observation_timestamp, timing_kind = source_timestamp(timing, sidecar_timestamps, frames)
                detect_start = time.monotonic_ns()
                detections = detector.detect(image)
                detect_end = time.monotonic_ns()
                tracks, events = tracker.update(detections, observation_timestamp, frames)
                result_ns = time.monotonic_ns()
                inference_ms = (detect_end - detect_start) / 1e6
                tracking_ms = (result_ns - detect_end) / 1e6
                total_ms = (result_ns - timing["received_monotonic_ns"]) / 1e6
                sensor_ms = sensor_age_ms(timing["sensor_timestamp_boottime_ns"])
                inference.add(inference_ms)
                tracking.add(tracking_ms)
                total_result.add(total_ms)
                sensor_to_result.add(sensor_ms)
                event_counts.update(event["event"] for event in events)
                maximum_active_tracks = max(maximum_active_tracks, len(tracks))
                annotated = annotate(image, tracks, frames, inference_ms, tracking_ms)
                if frames == 0 and not cv2.imwrite(str(run_dir / "first-frame.jpg"), annotated):
                    raise RuntimeError("Could not save the first frame")
                video_index = None
                if time.monotonic() - start < args.record_seconds and recorded < args.record_max_frames:
                    if not writers:
                        height, width = image.shape[:2]
                        for name in ("sample.avi", "annotated.avi"):
                            writer = cv2.VideoWriter(str(run_dir / name), cv2.VideoWriter_fourcc(*"MJPG"),
                                                     args.record_fps, (width, height))
                            writers.append(writer)
                            if not writer.isOpened():
                                raise RuntimeError("MJPG recording is unavailable; try --record-seconds 0")
                    writers[0].write(image)
                    writers[1].write(annotated)
                    video_index = recorded
                    recorded += 1
                record = {
                    "frame_id": frames,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    **timing,
                    "source_timestamp_seconds": observation_timestamp,
                    "source_timestamp_kind": timing_kind,
                    "result_monotonic_ns": result_ns,
                    "inference_ms": inference_ms,
                    "tracking_ms": tracking_ms,
                    "receive_to_result_ms": total_ms,
                    "sensor_to_result_ms": sensor_ms,
                    "recorded_video_frame_index": video_index,
                    "detections": detections,
                    "tracks": tracks,
                    "track_events": events,
                }
                frame_log.write(json.dumps(record, default=str, allow_nan=False) + "\n")
                frames += 1
                if args.preview:
                    cv2.imshow("SAGE Phase 2", annotated)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        status = "user-stopped"
                        break
                    if cv2.getWindowProperty("SAGE Phase 2", cv2.WND_PROP_VISIBLE) < 1:
                        status = "user-stopped"
                        break
                if time.monotonic() - last_health >= 1:
                    sample = health.sample()
                    health_log.write(json.dumps(sample) + "\n")
                    temperatures.add(sample["temperature_c"])
                    rss.add(sample["process_rss_bytes"])
                    last_health = time.monotonic()
                    print(f"{frames} frames | {frames / (last_health - start):.1f} processed FPS | "
                          f"{len(tracks)} active tracks | {sample['process_rss_bytes'] / 1024**2:.0f} MiB RAM",
                          flush=True)
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
        summary = {
            "status": status,
            "error": failure,
            "frames_processed": frames,
            "elapsed_seconds": elapsed,
            "processed_fps": frames / elapsed if elapsed else None,
            "inference_fps": inference.count / elapsed if elapsed and inference.count else None,
            **source_stats,
            "camera_requests_not_processed": max(0, camera_count - frames) if camera_count is not None else None,
            "recorded_frames": recorded,
            "recording_fps": args.record_fps,
            "source_timestamp_kind": timing_kind,
            "inference_ms": inference.result(),
            "tracking_ms": tracking.result(),
            "receive_to_result_ms": total_result.result(),
            "sensor_to_result_ms": sensor_to_result.result(),
            "track_events": dict(sorted(event_counts.items())),
            "maximum_active_tracks": maximum_active_tracks,
            "active_tracks_at_end": len(tracker.tracks),
            "sampled_rss_bytes": rss.result(),
            "sampled_temperature_c": temperatures.result(),
        }
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    return 1 if status == "failed" else 130 if status == "interrupted" else 0


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    validate(args, argument_parser)
    return run(args)
