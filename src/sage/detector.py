"""MobileNet-SSD VOC baseline using OpenCV's CPU inference runtime."""

import hashlib
from pathlib import Path

import cv2
import numpy as np


LABELS = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_detections(output, width, height, threshold):
    """Return clipped xyxy pixel boxes; x2/y2 are exclusive image bounds."""
    detections = []
    for row in np.asarray(output).reshape(-1, 7):
        if not np.isfinite(row).all():
            continue
        _, class_value, confidence, left, top, right, bottom = map(float, row)
        class_id = int(class_value)
        if (class_value != class_id or not 0 < class_id < len(LABELS)
                or not threshold <= confidence <= 1):
            continue
        x1 = max(0, min(width, int(np.floor(left * width))))
        y1 = max(0, min(height, int(np.floor(top * height))))
        x2 = max(0, min(width, int(np.ceil(right * width))))
        y2 = max(0, min(height, int(np.ceil(bottom * height))))
        if x2 > x1 and y2 > y1:
            detections.append({"class_id": class_id, "label": LABELS[class_id],
                               "confidence": confidence, "bbox_xyxy": [x1, y1, x2, y2]})
    return detections


class MobileNetSSD:
    def __init__(self, model_dir, threshold=0.5):
        model_dir = Path(model_dir)
        config = model_dir / "deploy.prototxt"
        weights = model_dir / "mobilenet_iter_73000.caffemodel"
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError("Detector files missing. Run: python3 scripts/download_model.py")
        self.threshold = threshold
        self.network = cv2.dnn.readNetFromCaffe(str(config), str(weights))
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.info = {
            "name": "MobileNet-SSD VOC0712", "runtime": "opencv-dnn-cpu",
            "input_size": [300, 300], "labels": LABELS, "threshold": threshold,
            "preprocessing": "BGR; resize to 300x300; (pixel - 127.5) * 0.007843",
            "sha256": {p.name: file_sha256(p) for p in (config, weights)},
        }

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300),
                                     (127.5, 127.5, 127.5), swapRB=False, crop=False)
        self.network.setInput(blob)
        output = self.network.forward()
        height, width = image.shape[:2]
        return decode_detections(output, width, height, self.threshold)

