"""Verify Phase 2 replay artifacts and authoritative sidecar timing."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "models" / "mobilenet-ssd" / "mobilenet_iter_73000.caffemodel").is_file(),
                     "Download the model for the Phase 2 integration test")
class Phase2ReplayTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "data" / "tests").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "data" / "tests")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.video = self.directory / "synthetic.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (160, 120))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()
        records = [
            {"frame_id": index, "recorded_video_frame_index": index,
             "sensor_timestamp_boottime_ns": 10_000_000_000 + index * 400_000_000}
            for index in range(4)
        ]
        self.sidecar = self.directory / "source-frames.jsonl"
        self.sidecar.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_replay_writes_tracks_events_and_original_timestamps(self):
        command = [
            sys.executable, str(ROOT / "scripts" / "run_phase2.py"),
            "--video", str(self.video), "--source-frames", str(self.sidecar),
            "--seconds", "30", "--record-seconds", "30", "--record-max-frames", "2",
            "--output", str(self.directory / "runs"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run_dir, = list((self.directory / "runs").iterdir())
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        frames = [json.loads(line) for line in
                  (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(manifest["phase"], 2)
        self.assertEqual(summary["status"], "end-of-video")
        self.assertEqual(summary["frames_processed"], 4)
        self.assertEqual(summary["recorded_frames"], 2)
        self.assertEqual(summary["recording_fps"], 2.4)
        self.assertEqual(summary["tracking_ms"]["samples"], 4)
        self.assertEqual(summary["source_timestamp_kind"], "source-frames-jsonl")
        self.assertEqual([frame["source_timestamp_seconds"] for frame in frames],
                         [10.0, 10.4, 10.8, 11.2])
        self.assertTrue(all("tracks" in frame and "track_events" in frame for frame in frames))


if __name__ == "__main__":
    unittest.main()
