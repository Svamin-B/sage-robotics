"""Lightweight class-aware multi-object tracking for sparse detector frames."""

from dataclasses import dataclass
import math

import numpy as np


def bbox_iou(first, second):
    """Return intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(first, second):
    """Return center distance normalized by the larger box's area scale."""
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    first_scale = math.sqrt(max(1.0, (ax2 - ax1) * (ay2 - ay1)))
    second_scale = math.sqrt(max(1.0, (bx2 - bx1) * (by2 - by1)))
    return math.hypot(acx - bcx, acy - bcy) / max(first_scale, second_scale)


@dataclass
class _Track:
    track_id: int
    class_id: int
    label: str
    bbox: np.ndarray
    velocity: np.ndarray
    confidence: float
    state: str
    created_timestamp: float
    last_timestamp: float
    last_observed_timestamp: float
    hits: int = 1
    hit_streak: int = 1
    missed_updates: int = 0
    observed_this_frame: bool = True

    def predicted_bbox(self, timestamp):
        return self.bbox + self.velocity * max(0.0, timestamp - self.last_timestamp)

    def snapshot(self, frame_id):
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": [int(round(value)) for value in self.bbox],
            "state": self.state,
            "observed_this_frame": self.observed_this_frame,
            "age_seconds": self.last_timestamp - self.created_timestamp,
            "seconds_since_observation": self.last_timestamp - self.last_observed_timestamp,
            "hits": self.hits,
            "hit_streak": self.hit_streak,
            "missed_updates": self.missed_updates,
            "source_frame_id": frame_id,
            "source_timestamp_seconds": self.last_timestamp,
        }


class MultiObjectTracker:
    """Associate same-class detections using predicted boxes and explicit lifecycle rules."""

    def __init__(self, min_hits=2, max_missed_seconds=1.25, min_iou=0.1,
                 max_center_distance=1.5, distance_weight=0.15, velocity_smoothing=0.5):
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        if max_missed_seconds < 0:
            raise ValueError("max_missed_seconds must be nonnegative")
        if not 0 <= min_iou <= 1:
            raise ValueError("min_iou must be between 0 and 1")
        if max_center_distance < 0 or distance_weight < 0:
            raise ValueError("distance thresholds must be nonnegative")
        if not 0 <= velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must be between 0 and 1")
        self.min_hits = min_hits
        self.max_missed_seconds = max_missed_seconds
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.distance_weight = distance_weight
        self.velocity_smoothing = velocity_smoothing
        self.tracks = []
        self.next_track_id = 1
        self.last_timestamp = None

    @property
    def configuration(self):
        return {
            "kind": "class-aware-predicted-box-association",
            "min_hits": self.min_hits,
            "max_missed_seconds": self.max_missed_seconds,
            "min_iou": self.min_iou,
            "max_center_distance": self.max_center_distance,
            "distance_weight": self.distance_weight,
            "velocity_smoothing": self.velocity_smoothing,
            "assignment": "deterministic greedy by IoU minus normalized-center-distance penalty",
        }

    def update(self, detections, timestamp, frame_id):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("tracker timestamp must be finite")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("tracker timestamps must be monotonic")
        self.last_timestamp = timestamp

        predicted = [track.predicted_bbox(timestamp) for track in self.tracks]
        candidates = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if detection.get("class_id") != track.class_id:
                    continue
                box = detection["bbox_xyxy"]
                iou = bbox_iou(predicted[track_index], box)
                distance = normalized_center_distance(predicted[track_index], box)
                if iou >= self.min_iou or distance <= self.max_center_distance:
                    candidates.append((iou - self.distance_weight * distance, iou, -distance,
                                       -track.track_id, -detection_index,
                                       track_index, detection_index))
        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()
        matches = []
        for *_, track_index, detection_index in candidates:
            if track_index not in matched_tracks and detection_index not in matched_detections:
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)
                matches.append((track_index, detection_index))

        events = []
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            detection = detections[detection_index]
            previous_state = track.state
            new_bbox = np.asarray(detection["bbox_xyxy"], dtype=float)
            delta = timestamp - track.last_timestamp
            if delta > 0:
                measured_velocity = (new_bbox - track.bbox) / delta
                keep = self.velocity_smoothing
                track.velocity = keep * track.velocity + (1 - keep) * measured_velocity
            track.bbox = new_bbox
            track.confidence = float(detection["confidence"])
            track.last_timestamp = timestamp
            track.last_observed_timestamp = timestamp
            track.hits += 1
            track.hit_streak += 1
            track.missed_updates = 0
            track.observed_this_frame = True
            if previous_state == "lost":
                track.state = "confirmed"
                events.append({"event": "reacquired", "track_id": track.track_id})
            elif track.state != "confirmed" and track.hit_streak >= self.min_hits:
                track.state = "confirmed"
                events.append({"event": "confirmed", "track_id": track.track_id})

        survivors = []
        for track_index, track in enumerate(self.tracks):
            if track_index in matched_tracks:
                survivors.append(track)
                continue
            track.bbox = predicted[track_index]
            track.last_timestamp = timestamp
            track.hit_streak = 0
            track.missed_updates += 1
            track.observed_this_frame = False
            expired = timestamp - track.last_observed_timestamp > self.max_missed_seconds
            if track.state == "tentative" or expired:
                events.append({"event": "deleted", "track_id": track.track_id,
                               "reason": "unconfirmed" if track.state == "tentative" else "expired"})
            else:
                if track.state != "lost":
                    events.append({"event": "lost", "track_id": track.track_id})
                track.state = "lost"
                survivors.append(track)
        self.tracks = survivors

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            state = "confirmed" if self.min_hits == 1 else "tentative"
            track = _Track(
                track_id=self.next_track_id,
                class_id=int(detection["class_id"]),
                label=str(detection["label"]),
                bbox=np.asarray(detection["bbox_xyxy"], dtype=float),
                velocity=np.zeros(4, dtype=float),
                confidence=float(detection["confidence"]),
                state=state,
                created_timestamp=timestamp,
                last_timestamp=timestamp,
                last_observed_timestamp=timestamp,
            )
            self.next_track_id += 1
            self.tracks.append(track)
            events.append({"event": "created", "track_id": track.track_id})
            if state == "confirmed":
                events.append({"event": "confirmed", "track_id": track.track_id})

        snapshots = [track.snapshot(frame_id) for track in sorted(self.tracks, key=lambda item: item.track_id)]
        return snapshots, events
