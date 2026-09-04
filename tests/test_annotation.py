"""Tests for Phase 2 annotation data conversion and persistence."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sage.annotation import (annotation_records, load_recorded_frames, read_annotations,
                             roi_to_xyxy, write_annotations)


class AnnotationTests(unittest.TestCase):
    def test_recorded_frames_are_selected_and_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            records = [
                {"frame_id": 9, "recorded_video_frame_index": 1},
                {"frame_id": 7, "recorded_video_frame_index": None},
                {"frame_id": 8, "recorded_video_frame_index": 0},
            ]
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            self.assertEqual([item["frame_id"] for item in load_recorded_frames(path)], [8, 9])

    def test_roi_conversion_rejects_cancelled_selection(self):
        self.assertEqual(roi_to_xyxy((10, 20, 30, 40)), [10, 20, 40, 60])
        self.assertIsNone(roi_to_xyxy((0, 0, 0, 0)))

    def test_annotations_round_trip_and_include_empty_frames(self):
        frames = [{"frame_id": 2}, {"frame_id": 3}]
        objects = {2: [{"object_id": "person-a", "class_id": 15, "label": "person",
                        "bbox_xyxy": [1, 2, 10, 20]}]}
        records = annotation_records(frames, objects)
        self.assertEqual(records[1], {"frame_id": 3, "objects": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.jsonl"
            write_annotations(path, frames, objects)
            self.assertEqual(read_annotations(path), {2: objects[2], 3: []})


if __name__ == "__main__":
    unittest.main()
