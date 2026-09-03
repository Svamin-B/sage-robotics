# Project status

## 2026-09-02 — Phase 1 started

### Hardware evidence

- User reports the Raspberry Pi is set up with a 64 GB microSD card.
- User ran `rpicam-hello` and explicitly confirmed a live camera preview.
- Supplied log identifies `imx708_wide` and shows streams configured; it also contains static-property/default-delay warnings.
- Exact Pi OS release, architecture, and Python version are still awaiting user output.
- No remote Pi connection is established in this workspace.

### Implementation

- Added separate Picamera2 and video-file capture adapters.
- Added a MobileNet-SSD VOC baseline through OpenCV CPU inference and a pinned model downloader.
- Added camera-only and detector modes, optional local preview, bounded raw/annotated recording, per-frame logs, run manifests, and resource summaries.
- Kept model weights and camera data outside version control.
- Chose VOC MobileNet-SSD for an initial OpenCV baseline. It does not support backpack goals; a broader model remains a future measurement-driven decision.

### Verification

Local Windows environment: Python 3.13.5, OpenCV 4.13.0, NumPy 2.1.3, psutil 5.9.0. Public model files were downloaded at the pinned revision and a black-frame inference completed.

`python -m unittest discover -s tests -v`: **5 tests passed, none skipped**. Checks cover clipping/class decoding, invalid detections, camera-only replay, real-model inference on a synthetic clip, decodable raw/annotated recordings, frame-to-video indexing and recording caps, and failure reporting for a missing input. The test run used temporary fixtures under `data/tests/` and cleaned them up. `python -m compileall -q src scripts tests` also completed successfully.

Local test clips are synthetic software fixtures; they establish neither object-detection accuracy nor onboard performance. Pi-specific capture, autofocus, sensor timestamps, GUI behavior, resource readings, and sustained performance remain unverified.

### Next steps

1. Confirm Pi OS/architecture/Python output and copy the project to the Pi.
2. Follow `docs/Phase1/guide.md`: dependencies, camera-only check, detector check, sustained baseline.
3. Record actual run directories, configurations, scene notes, measurements, and failures here.
4. Begin Phase 2 only after Phase 1's on-Pi completion check is met.

## 2026-09-03 — Camera-only Pi baseline

- Environment: Raspberry Pi 4 Model B, 4 GB; Raspberry Pi OS Bookworm with `armhf` userland and `aarch64` kernel; Python 3.11.2.
- Known run configuration: camera-only mode for 30 seconds. The summary does not establish whether local preview was enabled.
- Result: `duration-reached`, no error; 445 frames processed in 30.012 s.
- Camera delivery: 15.002 FPS; processing: 14.827 FPS; 6 delivered requests not processed, consistent with the run boundaries and output overhead.
- Delivery-to-result time: 0.208 ms mean, 0.556 ms maximum. Sensor-to-result time: 24.578 ms mean, 42.758 ms maximum.
- Process RSS: 199,469,409 bytes mean and 200,065,024 bytes maximum (sampled).
- Temperature: 51.289 C mean and 53.069 C maximum (29 samples). The preceding diagnostic reported `throttled=0x0`.
- Recording: 144 processed frames saved during the bounded sample interval.
- Interpretation: the camera-only Phase 1 path meets the requested 15 FPS baseline without evidence of accumulating latency, excessive memory use, overheating, or throttling. Visual color, focus, and artifact inspection remain user-observed checks.



## 2026-09-03 — First live detector baseline

- Run configuration: MobileNet-SSD VOC detector with local preview, 60-second duration, default 15 FPS camera request, two OpenCV threads, and default bounded recording.
- Result: `duration-reached`, no error; 85 frames processed in 60.054 s.
- Camera delivery: 15.002 FPS; inference and processing: 1.415 FPS. Of 901 delivered requests, 816 were not processed because inference was slower than camera delivery.
- Inference time: 651.168 ms mean and 693.649 ms maximum. Sensor-to-result time: 676.940 ms mean and 731.497 ms maximum.
- Process RSS: 298,134,314 bytes mean and 298,217,472 bytes maximum (sampled).
- Temperature: 58.902 C mean and 63.296 C maximum (43 samples).
- Recording: 13 processed frames saved during the bounded sample interval.
- Interpretation: the detector path runs successfully, but this measured rate and observation age are insufficient for the intended moving-robot application. A headless, recording-disabled benchmark is required to isolate detector cost before changing the runtime or model.
- The summary does not establish detection correctness or throttle flags; inspect the visual output and health log separately.


## 2026-09-03 — Headless detector baseline

- Run configuration: MobileNet-SSD VOC detector, 60 seconds, default 15 FPS camera request, two OpenCV threads, preview disabled, recording disabled.
- Result: `duration-reached`, no error; 90 frames processed in 60.051 s.
- Camera delivery: 15.003 FPS; inference and processing: 1.499 FPS. Of 901 delivered requests, 811 were not processed because inference was slower than camera delivery.
- Inference time: 641.495 ms mean and 651.553 ms maximum. Sensor-to-result time: 664.518 ms mean and 675.133 ms maximum.
- Process RSS: 250,890,863 bytes mean and 250,892,288 bytes maximum (sampled).
- Temperature: 58.511 C mean and 62.809 C maximum (46 samples).
- Compared with the preview/recording run, processed throughput increased by about 5.9 percent and mean inference time decreased by about 1.5 percent. Detector computation remains the dominant bottleneck.


## 2026-09-03 — Four-thread detector comparison

- Run configuration: MobileNet-SSD VOC detector, 60 seconds, default 15 FPS camera request, four OpenCV threads, preview disabled, recording disabled.
- Result: `duration-reached`, no error; 151 frames processed in 60.401 s.
- Camera delivery: 15.003 FPS; inference and processing: 2.500 FPS. Of 906 delivered requests, 755 were not processed because inference was slower than camera delivery.
- Inference time: 376.009 ms mean and 387.822 ms maximum. Sensor-to-result time: 399.310 ms mean and 411.701 ms maximum.
- Process RSS: 251,360,196 bytes mean and 251,367,424 bytes maximum (sampled).
- Temperature: 66.590 C mean and 73.036 C maximum (51 samples).
- Compared with the otherwise matching two-thread run, throughput increased by about 66.8 percent, mean inference time decreased by about 41.4 percent, and mean result age decreased by about 39.9 percent. Peak temperature increased by about 10.2 C.
- Interpretation: four threads are materially faster for this OpenCV DNN baseline, but thermal/throttle status and sustained behavior must be checked before adopting the setting.


### Four-thread health confirmation

- Final in-run health sample: process CPU 323.0 percent (psutil convention, about 3.23 fully occupied cores), system CPU 81.8 percent, process RSS 251,367,424 bytes, available system RAM 3,356,622,848 bytes, temperature 72.062 C, and `throttled=0x0`.
- Immediate post-run checks: `vcgencmd get_throttled` returned `throttled=0x0`; `vcgencmd measure_temp` returned 49.6 C.
- Interpretation: no throttling or undervoltage was recorded, memory headroom was ample, and the cooling system reduced temperature substantially after the load ended. Sustained behavior remains to be measured.

- User visually confirmed that the live/recorded detector output displayed object labels. Per-class correctness and false detections have not yet been evaluated systematically.

## 2026-09-03 — Five-minute four-thread detector run

- Run configuration: MobileNet-SSD VOC detector, 300 seconds, default 15 FPS camera request, four OpenCV threads, preview disabled, recording disabled.
- Result: `duration-reached`, no error; 684 frames processed in 300.387 s.
- Camera delivery: 15.003 FPS; inference and processing: 2.277 FPS. Of 4,507 delivered requests, 3,823 were not processed because inference was slower than camera delivery.
- Inference time: 396.847 ms mean and 483.346 ms maximum. Sensor-to-result time: 422.437 ms mean and 510.178 ms maximum.
- Process RSS: 251,174,286 bytes mean and 251,203,584 bytes maximum (sampled).
- Temperature: 78.280 C mean and 84.237 C maximum (229 samples).
- Compared with the 60-second four-thread run, sustained throughput decreased by about 8.9 percent and mean inference time increased by about 5.5 percent. The temperature entered the documented 80-85 C progressive thermal-control range.
- Interpretation: four-thread performance is not thermally sustained with the tested cooling configuration. The per-run health throttle flag and physical fan operation must be checked before another long benchmark.


## 2026-09-03 — Five-minute thermal-status confirmation

After the 300-second, four-thread detector run, the final health sample reported 83.75 C and `throttled=0xe0008`. This indicates that the soft temperature limit was active at that sample and that ARM frequency capping, throttling, and the soft temperature limit had occurred during the run. A later idle check reported 55.5 C and `throttled=0xe0000`, indicating no active condition after cooling, while the historical thermal flags remained set. No undervoltage flag was present.

**Conclusion:** the current four-thread configuration is not thermally sustainable with the present cooling arrangement. Inspect fan operation and heatsink/thermal contact before another long four-thread run. Use a cooler configuration for the sustained baseline.

## 2026-09-03 — Fan-cooled sustained baseline and Phase 1 completion

- Cooling correction: the prior five-minute run was performed with the fan disconnected. The fan was connected, the Pi was rebooted, and the historical throttle flags cleared.
- Pre-run state: `throttled=0x0`; temperature 44.8 C.
- Run configuration: MobileNet-SSD VOC detector, 300 seconds, default 15 FPS camera request, four OpenCV threads, preview disabled, recording disabled.
- Result: `duration-reached`, no error; 719 frames processed in 300.011 s.
- Camera delivery: 15.003 FPS; inference and processing: 2.397 FPS. Of 4,501 delivered requests, 3,782 were not processed because inference was slower than camera delivery.
- Inference time: 385.308 ms mean and 447.858 ms maximum. Sensor-to-result time: 409.136 ms mean and 489.059 ms maximum.
- Process RSS: 249,410,539 bytes mean and 249,458,688 bytes maximum (sampled).
- Temperature: 60.853 C mean and 64.270 C maximum (241 samples).
- Final recorded health sample: 63.296 C and `throttled=0x0`. Post-run temperature was 44.8 C and `throttled=0x0`.
- Compared with the fan-disconnected five-minute run, throughput increased about 5.3 percent, mean inference time decreased about 2.9 percent, mean temperature decreased 17.4 C, and peak temperature decreased 20.0 C.

**Phase 1 status: complete.** Camera capture and live labeled detection have run continuously on the Raspberry Pi; the launch procedure, configuration, timestamped detections, replay sample, performance, memory use, CPU behavior, temperature, and throttle state have been recorded. The established fan-cooled four-thread baseline is approximately 2.40 inference FPS with 409 ms mean sensor-to-result age. Detection accuracy remains a later evaluation task, and the measured throughput is a constraint for tracking and eventual navigation.
