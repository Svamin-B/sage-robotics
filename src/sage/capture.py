"""Live Picamera2 capture and offline video replay, both producing BGR images."""

import threading
import time
from pathlib import Path

import cv2


class PiCamera:
    def __init__(self, width, height, fps):
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError as error:
            raise RuntimeError("Live capture requires Picamera2 on Raspberry Pi OS. "
                               "See docs/Phase1/guide.md; use --video for desktop replay.") from error
        self.camera = Picamera2()
        self.lock = threading.Lock()
        self.received = 0
        self.first_sensor_ns = self.last_sensor_ns = None
        try:
            settings = {"FrameRate": fps}
            if "AfMode" in self.camera.camera_controls:
                settings["AfMode"] = controls.AfModeEnum.Continuous
            # Picamera2 RGB888 is a B,G,R byte array, as required by OpenCV.
            config = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls=settings, buffer_count=4, queue=False,
            )
            self.camera.configure(config)
            self.info = {"kind": "picamera2", "configuration": self.camera.camera_configuration(),
                         "properties": self.camera.camera_properties, "calibration": None}
            self.camera.post_callback = self._count_request
            self.camera.start()
        except BaseException:
            self.camera.close()
            raise

    def _count_request(self, request):
        sensor_ns = request.get_metadata().get("SensorTimestamp")
        with self.lock:
            self.received += 1
            if sensor_ns is not None:
                if self.first_sensor_ns is None:
                    self.first_sensor_ns = sensor_ns
                self.last_sensor_ns = sensor_ns

    def read(self):
        request = self.camera.capture_request()
        try:
            image = request.make_array("main").copy()
            metadata = request.get_metadata()
        finally:
            request.release()
        return image, {"received_monotonic_ns": time.monotonic_ns(),
                       "sensor_timestamp_boottime_ns": metadata.get("SensorTimestamp"),
                       "camera_metadata": metadata, "video_time_ms": None}

    def reset_stats(self):
        with self.lock:
            self.received = 0
            self.first_sensor_ns = self.last_sensor_ns = None

    def stats(self):
        with self.lock:
            span = ((self.last_sensor_ns - self.first_sensor_ns) / 1e9
                    if self.first_sensor_ns is not None and self.last_sensor_ns is not None else 0)
            return {"camera_requests_received": self.received,
                    "camera_delivered_fps": (self.received - 1) / span if span > 0 else None}

    def close(self):
        self.camera.stop()
        self.camera.close()


class VideoFile:
    def __init__(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        self.video = cv2.VideoCapture(str(path))
        if not self.video.isOpened():
            self.video.release()
            raise RuntimeError(f"Cannot decode video: {path}")
        self.info = {"kind": "offline-video", "path": str(Path(path).resolve()),
                     "nominal_fps": self.video.get(cv2.CAP_PROP_FPS), "calibration": None}

    def read(self):
        ok, image = self.video.read()
        if not ok:
            return None
        return image, {"received_monotonic_ns": time.monotonic_ns(),
                       "sensor_timestamp_boottime_ns": None, "camera_metadata": {},
                       "video_time_ms": self.video.get(cv2.CAP_PROP_POS_MSEC)}

    def reset_stats(self):
        pass

    def stats(self):
        return {"camera_requests_received": None, "camera_delivered_fps": None}

    def close(self):
        self.video.release()
