"""Focused tests for Phase 2 association, lifecycle, and identity metrics."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sage.tracker import MultiObjectTracker, bbox_iou
from sage.tracking_evaluation import evaluate_tracking


def detection(label, class_id, box, confidence=0.9):
    return {"label": label, "class_id": class_id, "bbox_xyxy": box, "confidence": confidence}


class TrackerTests(unittest.TestCase):
    def test_track_confirms_survives_short_miss_and_is_reacquired(self):
        tracker = MultiObjectTracker(min_hits=2, max_missed_seconds=1.0)
        tracks, events = tracker.update([detection("person", 15, [10, 10, 30, 50])], 0.0, 0)
        self.assertEqual(tracks[0]["state"], "tentative")
        track_id = tracks[0]["track_id"]
        tracks, events = tracker.update([detection("person", 15, [12, 10, 32, 50])], 0.4, 1)
        self.assertEqual(tracks[0]["state"], "confirmed")
        self.assertIn("confirmed", [event["event"] for event in events])
        tracks, _ = tracker.update([], 0.8, 2)
        self.assertEqual(tracks[0]["state"], "lost")
        self.assertFalse(tracks[0]["observed_this_frame"])
        tracks, events = tracker.update([detection("person", 15, [16, 10, 36, 50])], 1.1, 3)
        self.assertEqual(tracks[0]["track_id"], track_id)
        self.assertEqual(tracks[0]["state"], "confirmed")
        self.assertIn("reacquired", [event["event"] for event in events])

    def test_expired_track_is_not_reused(self):
        tracker = MultiObjectTracker(min_hits=2, max_missed_seconds=0.5)
        tracks, _ = tracker.update([detection("chair", 9, [10, 10, 40, 50])], 0.0, 0)
        first_id = tracks[0]["track_id"]
        tracker.update([detection("chair", 9, [11, 10, 41, 50])], 0.2, 1)
        tracks, events = tracker.update([], 0.8, 2)
        self.assertEqual(tracks, [])
        self.assertIn("expired", [event.get("reason") for event in events])
        tracks, _ = tracker.update([detection("chair", 9, [11, 10, 41, 50])], 1.0, 3)
        self.assertNotEqual(tracks[0]["track_id"], first_id)

    def test_class_gate_keeps_overlapping_classes_separate(self):
        tracker = MultiObjectTracker(min_hits=1)
        tracks, _ = tracker.update([
            detection("person", 15, [0, 0, 20, 40]),
            detection("chair", 9, [0, 0, 20, 40]),
        ], 0.0, 0)
        ids = {track["label"]: track["track_id"] for track in tracks}
        tracks, _ = tracker.update([
            detection("chair", 9, [3, 0, 23, 40]),
            detection("person", 15, [3, 0, 23, 40]),
        ], 0.4, 1)
        self.assertEqual({track["label"]: track["track_id"] for track in tracks}, ids)

    def test_two_same_class_tracks_receive_one_detection_each(self):
        tracker = MultiObjectTracker(min_hits=1)
        tracks, _ = tracker.update([
            detection("person", 15, [0, 0, 20, 40]),
            detection("person", 15, [80, 0, 100, 40]),
        ], 0.0, 0)
        ids = [track["track_id"] for track in tracks]
        tracks, _ = tracker.update([
            detection("person", 15, [4, 0, 24, 40]),
            detection("person", 15, [76, 0, 96, 40]),
        ], 0.4, 1)
        self.assertEqual([track["track_id"] for track in tracks], ids)

    def test_nonmonotonic_timestamp_is_rejected(self):
        tracker = MultiObjectTracker()
        tracker.update([], 1.0, 0)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            tracker.update([], 0.9, 1)

    def test_iou_uses_exclusive_xyxy_convention(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)


class EvaluationTests(unittest.TestCase):
    def test_switch_fragmentation_miss_and_false_positive_counts(self):
        truth = [
            {"frame_id": 0, "objects": [{"object_id": "A", "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_id": 1, "objects": [{"object_id": "A", "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_id": 2, "objects": [{"object_id": "A", "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_id": 3, "objects": [{"object_id": "A", "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
        ]
        predictions = [
            {"frame_id": 0, "tracks": [{"track_id": 1, "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_id": 1, "tracks": [{"track_id": 2, "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_id": 2, "tracks": [{"track_id": 2, "label": "chair", "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_id": 3, "tracks": [{"track_id": 2, "label": "person", "bbox_xyxy": [0, 0, 10, 10]}]},
        ]
        result = evaluate_tracking(truth, predictions)
        self.assertEqual(result["matches"], 3)
        self.assertEqual(result["misses"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["id_switches"], 1)
        self.assertEqual(result["fragmentations"], 1)


if __name__ == "__main__":
    unittest.main()
