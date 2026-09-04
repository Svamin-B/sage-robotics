"""Small, dependency-free identity evaluation for Phase 2 annotated sequences."""

from collections import defaultdict

from sage.tracker import bbox_iou


def evaluate_tracking(ground_truth_records, prediction_records, iou_threshold=0.5):
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    predictions_by_frame = {record["frame_id"]: record for record in prediction_records}
    identity_state = defaultdict(lambda: {"last_track_id": None, "gap": False})
    per_object = defaultdict(lambda: {"instances": 0, "matches": 0,
                                      "id_switches": 0, "fragmentations": 0})
    totals = {"ground_truth_instances": 0, "predicted_instances": 0, "matches": 0,
              "misses": 0, "false_positives": 0, "id_switches": 0, "fragmentations": 0}

    present_previous_frame = set()
    for truth_record in sorted(ground_truth_records, key=lambda item: item["frame_id"]):
        frame_id = truth_record["frame_id"]
        truth_objects = truth_record.get("objects", [])
        prediction = predictions_by_frame.get(frame_id, {})
        predicted_tracks = [track for track in prediction.get("tracks", [])
                            if track.get("observed_this_frame", True)]
        totals["ground_truth_instances"] += len(truth_objects)
        totals["predicted_instances"] += len(predicted_tracks)

        candidates = []
        for truth_index, truth in enumerate(truth_objects):
            for track_index, track in enumerate(predicted_tracks):
                same_class = (truth.get("class_id") == track.get("class_id")
                              if "class_id" in truth else truth.get("label") == track.get("label"))
                if same_class:
                    overlap = bbox_iou(truth["bbox_xyxy"], track["bbox_xyxy"])
                    if overlap >= iou_threshold:
                        candidates.append((overlap, -truth_index, -track_index, truth_index, track_index))
        candidates.sort(reverse=True)
        matched_truth = set()
        matched_tracks = set()
        matches = []
        for *_, truth_index, track_index in candidates:
            if truth_index not in matched_truth and track_index not in matched_tracks:
                matched_truth.add(truth_index)
                matched_tracks.add(track_index)
                matches.append((truth_index, track_index))

        current_ids = {str(item["object_id"]) for item in truth_objects}
        for identity in present_previous_frame - current_ids:
            identity_state[identity]["gap"] = False
        present_previous_frame = current_ids

        match_by_truth = {truth_index: track_index for truth_index, track_index in matches}
        for truth_index, truth in enumerate(truth_objects):
            identity = str(truth["object_id"])
            per_object[identity]["instances"] += 1
            state = identity_state[identity]
            if truth_index not in match_by_truth:
                totals["misses"] += 1
                if state["last_track_id"] is not None:
                    state["gap"] = True
                continue
            track = predicted_tracks[match_by_truth[truth_index]]
            track_id = track["track_id"]
            totals["matches"] += 1
            per_object[identity]["matches"] += 1
            if state["last_track_id"] is not None and state["last_track_id"] != track_id:
                totals["id_switches"] += 1
                per_object[identity]["id_switches"] += 1
            if state["gap"]:
                totals["fragmentations"] += 1
                per_object[identity]["fragmentations"] += 1
            state["last_track_id"] = track_id
            state["gap"] = False

        totals["false_positives"] += len(predicted_tracks) - len(matches)

    totals["match_fraction"] = (totals["matches"] / totals["ground_truth_instances"]
                                if totals["ground_truth_instances"] else None)
    return {**totals, "iou_threshold": iou_threshold,
            "per_object": {key: value for key, value in sorted(per_object.items())}}
