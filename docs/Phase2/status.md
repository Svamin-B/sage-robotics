# Phase 2 status

## 2026-09-04 — Phase 2 started

### Starting evidence and decision

- Phase 1 is complete on the Raspberry Pi 4. Its fan-cooled, four-thread, five-minute baseline achieved approximately 2.397 processed/inference FPS, 385.308 ms mean inference time, and 409.136 ms mean sensor-to-result age without throttling.
- Phase 2 therefore begins with association across sparse detector frames rather than claiming camera-rate tracking.
- Selected an in-project, class-aware predicted-box tracker to keep runtime dependencies and overhead small. It uses deterministic greedy one-to-one association over IoU and normalized center displacement. This is an initial baseline to be judged by annotated results, not a claim that it outperforms SORT or ByteTrack.
- Track expiry uses elapsed source time rather than processed-frame count. Short-term track IDs are unique only within a run and are explicitly separate from future semantic-map IDs.

### Implementation

- Added a tracker with tentative, confirmed, lost, reacquired, and deleted lifecycle behavior.
- Added configurable confirmation, expiry, IoU, center-distance, ranking, and velocity parameters.
- Added a separate Phase 2 live/video runner. The verified Phase 1 runner remains unchanged.
- Added per-frame track records and lifecycle events, tracked annotation video, tracker overhead measurements, and run-level lifecycle summaries.
- Added optional Phase 1 `frames.jsonl` timing restoration for raw sample replay so time-based behavior does not depend on nominal AVI playback speed.
- Added a dependency-free evaluator for matched instances, misses, unmatched predictions, ID switches, fragmentation, match fraction, and per-object counts.

### Local verification

Environment: Windows development workspace with Python 3.13.5, OpenCV 4.13.0, NumPy 2.1.3, and psutil 5.9.0.

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

Initial result: **13 tests passed, none skipped**, and compilation completed successfully. After adding the annotation workflow and independent recording playback rate, **16 tests passed, none skipped**. Checks cover confirmation, short misses, reacquisition, expiry and ID replacement, class gating, one-to-one same-class assignment, timestamp rejection, IoU convention, identity metric counting, annotation persistence, real MobileNet-SSD replay artifacts, and restoration of source timestamps from a Phase 1 frame log.

The Phase 2 replay integration uses blank synthetic software frames. It verifies the detector-to-tracker-to-artifact path but does not establish real-object tracking accuracy or ID stability.

## 2026-09-04 — First live detector-plus-tracker Pi run

- Run ID: `20260904T201757Z-574ee48f`.
- Configuration: Raspberry Pi 4 Model B with fan cooling; live Camera Module 3 Wide source; 640×480 capture requested at 15 FPS; four OpenCV threads; detector threshold 0.5; default tracker settings; local preview enabled; first 15 seconds recorded.
- Result: `duration-reached`, no error; 139 frames processed in 60.312 seconds.
- Camera delivery: 15.003 FPS; inference and processing: 2.305 FPS. Of 904 delivered requests, 765 were not processed because detector inference was slower than camera delivery.
- Inference: 382.430 ms mean and 471.925 ms maximum. Tracking: 0.350 ms mean and 1.156 ms maximum. Mean tracking time was about 0.09 percent of mean inference time.
- Sensor-to-result age: 408.274 ms mean and 501.577 ms maximum.
- Resource samples: 298,364,928 bytes maximum process RSS; 55.193 C mean and 60.374 C maximum temperature. The final health sample reported `throttled=0x0`.
- Lifecycle summary: 11 tracks created, 7 confirmed, 24 lost events, 18 reacquisitions, and 9 deletions; at most 3 tracks were active simultaneously.
- Detection observations included person, bottle, sofa, chair, and cat labels. Short-lived chair and cat detections did not confirm. Bottle tracks accounted for much of the repeated loss and reacquisition behavior.
- Both raw and annotated AVI files contain 31 decodable frames spanning 14.131 seconds of source time. They play for about 2.07 seconds because this run tagged them at the requested 15 FPS. No recorded frames were missing.
- User visually confirmed that live tracking works while noting imperfect stability. This establishes a functional live Pi demonstration, not measured identity accuracy.
- Compared with the Phase 1 fan-cooled headless/no-recording baseline, processed throughput was about 3.9 percent lower while mean sensor-to-result age was essentially unchanged. Because this run used preview and bounded recording, the difference must not be attributed solely to tracking.

### Implementation follow-up

- Added `--record-fps`, defaulting to 2.4 FPS, so future AVI playback approximates the established processed cadence without reducing the 15 FPS camera request. JSONL timestamps remain authoritative.
- Added `scripts/annotate_phase2.py`, a resumable frame-by-frame bounding-box and physical-identity annotation tool, plus tested mapping and persistence helpers.

### Current limitations

- No real multi-object sequence has yet been annotated or evaluated.
- The first Pi run was a functional check, not the controlled annotated evaluation or five-minute headless comparison.
- The tracker has no appearance descriptor. Same-class crossings, large camera motion, abrupt detector-box changes, and long detector gaps are expected failure risks.
- Default thresholds are engineering starting points and have not been tuned against real data.
- The evaluator is a focused controlled-sequence tool, not a complete implementation of an external multiple-object-tracking benchmark.

### Next prerequisite

Deploy the recording/annotation update, collect a controlled multi-object sequence, annotate it, evaluate ID switches and fragmentation, and then perform the five-minute headless Pi comparison. Phase 2 remains **in progress** until that evidence is recorded.
