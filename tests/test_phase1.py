"""Verify decoding and real file-to-detector-to-artifact integration without a Pi."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sage.detector import decode_detections


class DetectionTests(unittest.TestCase):
    def test_clipping_and_class_mapping(self):
        output = np.array([[[[0, 15, 0.9, -0.1, 0.25, 1.2, 0.75]]]])
        detection, = decode_detections(output, 640, 480, 0.5)
        self.assertEqual(detection["label"], "person")
        self.assertEqual(detection["bbox_xyxy"], [0, 120, 640, 360])

    def test_invalid_and_low_confidence_detections_are_rejected(self):
        rows = [
            [0, 15, 0.2, 0, 0, 1, 1], [0, 0, 0.9, 0, 0, 1, 1],
            [0, 99, 0.9, 0, 0, 1, 1], [0, 15, 0.9, 0.8, 0.8, 0.2, 0.2],
            [0, 15, 0.9, 0, 0, float("nan"), 1], [0, 15, 1.1, 0, 0, 1, 1],
            [0, 15.5, 0.9, 0, 0, 1, 1],
        ]
        self.assertEqual(decode_detections(np.array(rows), 640, 480, 0.5), [])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "data" / "tests").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "data" / "tests")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.video = self.directory / "synthetic.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
        self.assertTrue(writer.isOpened())
        for index in range(6):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, 0] = 180
            cv2.rectangle(frame, (20 + index, 30), (60 + index, 80), (0, 255, 0), -1)
            writer.write(frame)
        writer.release()

    def launch(self, *extra):
        command = [sys.executable, str(ROOT / "scripts" / "run_phase1.py"),
                   "--video", str(self.video), "--seconds", "30", "--record-seconds", "30",
                   "--record-max-frames", "3", "--output", str(self.directory / "runs"), *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=ROOT)
        run_dir, = list((self.directory / "runs").iterdir())
        summary = json.loads((run_dir / "summary.json").read_text())
        return result, run_dir, summary

    def verify_artifacts(self, run_dir, summary):
        self.assertEqual(summary["frames_processed"], 6)
        self.assertEqual(summary["status"], "end-of-video")
        self.assertEqual(summary["recorded_frames"], 3)
        self.assertIsNone(summary["camera_delivered_fps"])
        frames = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text().splitlines()]
        self.assertEqual([frame["frame_id"] for frame in frames], list(range(6)))
        self.assertEqual([frame["recorded_video_frame_index"] for frame in frames], [0, 1, 2, None, None, None])
        self.assertTrue(all(frame["sensor_to_result_ms"] is None for frame in frames))
        self.assertTrue((run_dir / "first-frame.jpg").is_file())
        for name in ("sample.avi", "annotated.avi"):
            video = cv2.VideoCapture(str(run_dir / name))
            decoded = 0
            while video.read()[0]:
                decoded += 1
            video.release()
            self.assertEqual(decoded, 3)

    def test_capture_replay_and_recording(self):
        result, run_dir, summary = self.launch("--camera-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.verify_artifacts(run_dir, summary)
        self.assertIsNone(summary["inference_fps"])

    @unittest.skipUnless((ROOT / "models" / "mobilenet-ssd" / "mobilenet_iter_73000.caffemodel").is_file(),
                         "Download the model for the real inference integration test")
    def test_real_detector_replay_and_recording(self):
        result, run_dir, summary = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.verify_artifacts(run_dir, summary)
        self.assertEqual(summary["inference_ms"]["samples"], 6)
        self.assertGreater(summary["inference_fps"], 0)
        manifest = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(manifest["model"]["input_size"], [300, 300])

    def test_missing_video_records_failure(self):
        result, _, summary = self.launch("--camera-only", "--video", str(self.directory / "missing.avi"))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frames_processed"], 0)
        self.assertIn("Video file not found", summary["error"])


if __name__ == "__main__":
    unittest.main()
