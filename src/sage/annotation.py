"""Data helpers for frame-by-frame Phase 2 ground-truth annotation."""

import json
from pathlib import Path


def load_recorded_frames(path):
    """Return frame-log records in raw-video order."""
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    recorded = [record for record in records if record.get("recorded_video_frame_index") is not None]
    recorded.sort(key=lambda record: record["recorded_video_frame_index"])
    indices = [record["recorded_video_frame_index"] for record in recorded]
    if indices != list(range(len(recorded))):
        raise ValueError("Frame log has non-contiguous recorded video indices")
    return recorded


def roi_to_xyxy(roi):
    x, y, width, height = map(int, roi)
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def annotation_records(recorded_frames, annotations):
    """Build evaluator JSONL records from annotations keyed by source frame ID."""
    result = []
    for frame in recorded_frames:
        frame_id = int(frame["frame_id"])
        objects = annotations.get(frame_id, [])
        result.append({"frame_id": frame_id, "objects": sorted(objects, key=lambda item: str(item["object_id"]))})
    return result


def read_annotations(path):
    path = Path(path)
    if not path.is_file():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(record["frame_id"]): record.get("objects", []) for record in records}


def write_annotations(path, recorded_frames, annotations):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(record, allow_nan=False) + "\n"
                      for record in annotation_records(recorded_frames, annotations))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
