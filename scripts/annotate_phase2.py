"""Interactively annotate a Phase 2 raw sample for identity evaluation."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sage.annotation import load_recorded_frames, read_annotations, roi_to_xyxy, write_annotations


WINDOW = "SAGE Phase 2 ground truth"


def draw(frame, objects, video_index, frame_id):
    output = frame.copy()
    for item in objects:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (255, 180, 40), 2)
        text = f"{item['object_id']} | {item['label']}"
        cv2.putText(output, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 180, 40), 1, cv2.LINE_AA)
    status = f"video {video_index} | source frame {frame_id} | {len(objects)} objects"
    cv2.putText(output, status, (8, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def select_box(frame):
    print("Draw the box, then press Enter/Space. Press c to cancel.")
    roi = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
    return roi_to_xyxy(roi)


def print_help():
    print("Commands:")
    print("  a OBJECT_ID CLASS_ID LABEL  add an object and draw its box")
    print("  e OBJECT_ID                 redraw an existing object's box")
    print("  d OBJECT_ID                 delete an object from this frame")
    print("  c                           copy annotations from the previous frame")
    print("  n / p                       next / previous frame")
    print("  s                           save without quitting")
    print("  q                           save and quit")
    print("  h                           show this help")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Phase 2 run directory")
    parser.add_argument("--output", type=Path,
                        help="Ground-truth JSONL (default: data/annotations/<run-id>.jsonl)")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    video_path = run_dir / "sample.avi"
    frame_log_path = run_dir / "frames.jsonl"
    if not video_path.is_file() or not frame_log_path.is_file():
        parser.error("--run must contain sample.avi and frames.jsonl")
    output_path = (args.output.resolve() if args.output else
                   ROOT / "data" / "annotations" / f"{run_dir.name}.jsonl")
    recorded_frames = load_recorded_frames(frame_log_path)
    if not recorded_frames:
        parser.error("the run contains no recorded frames")
    annotations = read_annotations(output_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        parser.error(f"cannot decode {video_path}")
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_count and reported_count != len(recorded_frames):
        parser.error(f"video reports {reported_count} frames but frames.jsonl maps {len(recorded_frames)}")

    print(f"Annotating {len(recorded_frames)} frames; output: {output_path}")
    print("Use stable physical-object IDs within each continuous visible trajectory.")
    print_help()
    index = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode video frame {index}")
            frame_id = int(recorded_frames[index]["frame_id"])
            objects = annotations.setdefault(frame_id, [])
            cv2.imshow(WINDOW, draw(frame, objects, index, frame_id))
            cv2.waitKey(1)
            try:
                parts = shlex.split(input(f"[{index + 1}/{len(recorded_frames)}] command> ").strip())
            except EOFError:
                parts = ["q"]
            if not parts:
                continue
            command = parts[0].lower()
            if command == "h":
                print_help()
            elif command == "a":
                if len(parts) != 4:
                    print("Usage: a OBJECT_ID CLASS_ID LABEL")
                    continue
                try:
                    class_id = int(parts[2])
                except ValueError:
                    print("CLASS_ID must be an integer, such as 15 for person or 5 for bottle.")
                    continue
                if any(str(item["object_id"]) == parts[1] for item in objects):
                    print(f"Object {parts[1]!r} already exists on this frame; use e to redraw it.")
                    continue
                box = select_box(frame)
                if box:
                    objects.append({"object_id": parts[1], "class_id": class_id,
                                    "label": parts[3], "bbox_xyxy": box})
            elif command == "e":
                if len(parts) != 2:
                    print("Usage: e OBJECT_ID")
                    continue
                item = next((item for item in objects if str(item["object_id"]) == parts[1]), None)
                if item is None:
                    print(f"No object {parts[1]!r} on this frame.")
                    continue
                box = select_box(frame)
                if box:
                    item["bbox_xyxy"] = box
            elif command == "d":
                if len(parts) != 2:
                    print("Usage: d OBJECT_ID")
                    continue
                before = len(objects)
                annotations[frame_id] = [item for item in objects if str(item["object_id"]) != parts[1]]
                if len(annotations[frame_id]) == before:
                    print(f"No object {parts[1]!r} on this frame.")
            elif command == "c":
                if index == 0:
                    print("There is no previous frame to copy.")
                    continue
                previous_id = int(recorded_frames[index - 1]["frame_id"])
                annotations[frame_id] = json.loads(json.dumps(annotations.get(previous_id, [])))
            elif command == "n":
                if index + 1 < len(recorded_frames):
                    index += 1
                else:
                    print("Already at the final frame.")
            elif command == "p":
                index = max(0, index - 1)
            elif command == "s":
                write_annotations(output_path, recorded_frames, annotations)
                print(f"Saved {output_path}")
            elif command == "q":
                write_annotations(output_path, recorded_frames, annotations)
                print(f"Saved {output_path}")
                return 0
            else:
                print("Unknown command; enter h for help.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
