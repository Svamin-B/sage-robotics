# SAGE — Agent Context and Project Guide

## Project identity

- **Project:** SAGE — Semantic Autonomous Ground Explorer
- **Repository name:** `sage-robotics`
- **Intended project root folder:** `sage/`
- **Current workspace:** This directory is the project root, currently named `SAGE`. Do not create another nested `sage/` directory or rename the workspace merely to match the intended name.
- **Future README title:** `SAGE: Semantic Autonomous Ground Explorer`
- **Future README subtitle:** *Real-time monocular semantic mapping and object-goal navigation on resource-constrained hardware.*

This file applies throughout the project unless a more specific `AGENTS.md` provides local guidance. It records the project context and implementation roadmap; it is not evidence that any phase is implemented.

## Purpose and problem statement

Build a low-cost indoor mobile robot that detects, tracks, and spatially localizes objects from an onboard monocular RGB camera, maintains a semantic map, and navigates to a requested object under the compute constraints of a Raspberry Pi 4.

The central research question is: **How can a resource-constrained mobile robot turn monocular observations into a persistent spatial representation and use that representation for object-goal navigation?**

The system must eventually answer:

1. What objects are visible?
2. Have I observed this physical object before?
3. Where is it relative to the camera and robot?
4. Where am I, and where is the object in the room?
5. Which space can the robot traverse?
6. How can it reach a safe position near the requested object?
7. Should it change viewpoint to reduce uncertainty?

The intended result is an integrated robotics system with measured performance. Live detection is the first milestone, not the final project. Resume and research claims must be supported by actual experiments.

## Starting hardware and scope

The user already owns:

- Raspberry Pi 4 Model B with **4 GB RAM**.
- Raspberry Pi Camera Module 3 **Wide**, a single RGB camera.
- A fan and thermal paste for cooling.
- A **64 GB microSD card** used for the Pi setup (confirmed 2026-09-02).

Do not assume access to an ML accelerator, depth camera, LiDAR, IMU, wheel encoders, motors, chassis, motor driver, or rover battery. Do not assume the Pi is connected to this development workspace or that its camera setup has been validated.

Begin with the existing Pi and camera. Chassis and motor hardware belong to Phase 7; additional sensors are optional later extensions. Encoders may support motion control or localization, but an encoder-assisted result must be labeled separately from a vision-only result. Depth/stereo hardware changes the monocular sensing scope and should be documented as a separate configuration.

The initial operating domain is a controlled indoor environment. Flat-floor geometry is a conditional baseline, not a general depth solution. Exploration may initially be manual; autonomous exploration is not a prerequisite for the first object-goal demonstration.

## Working principles

- Work in the phase order below. Each phase should produce a runnable demonstration, saved evidence, and a documented completion check before dependent work proceeds.
- Treat phases as milestones rather than isolated silos: collect performance and accuracy measurements from Phase 1 onward, and revisit earlier components when integration reveals a problem.
- Start with lightweight pretrained detection. Training a custom model, learned depth, ROS, cloud inference, and an LLM interface are not initial requirements.
- Prefer Python for the first prototype, with Picamera2 for Pi capture and OpenCV for image handling and geometry. These are starting choices from the project discussion, not installed or verified dependencies.
- Choose detector, inference runtime, tracker, and localization library based on compatibility and measurements on the actual Pi. Candidate names below are options, not locked dependencies.
- Keep inference onboard for the baseline. Desktop replay and development are useful, but desktop benchmarks do not establish onboard performance.
- Keep camera capture, detection, tracking, geometry, localization, semantic mapping, obstacle mapping, planning, control, and evaluation separable. Avoid prematurely building a large framework.
- Use recorded inputs or simulated hardware where practical so development can continue without a connected Pi or rover. Label simulated results clearly.
- Before adding dependencies or giving hardware-specific setup instructions, verify relevant official documentation and the target OS, architecture, and package compatibility. Do not guess camera APIs, pin assignments, power requirements, or supported model formats.
- Do not invent completed milestones, measurements, validated commands, or installed hardware. Distinguish proposals, implementations, and hardware-verified behavior.

## Sequential roadmap

### Phase 1 — Camera and live perception

**Question:** What objects are currently visible?

- Establish reliable camera capture, then connect a lightweight pretrained detector.
- Consider MobileNet-SSD or a small quantized detector supported by the chosen runtime. Begin with its existing label set; verify available classes before choosing demo targets. Objects mentioned in the original brainstorm, such as doors, are not guaranteed model classes.
- Produce live annotated output and save timestamped detections plus a short replayable sample.
- Measure capture rate, inference throughput, end-to-end latency, CPU use, RAM use, and temperature/throttling over a sustained run.

**Completion check:** Demonstrate continuous capture and detection on the Pi with a repeatable launch procedure, recorded configuration, and measured resource usage. Establish a practical performance baseline rather than promising an arbitrary frame rate.

### Phase 2 — Object tracking

**Question:** Is this the same object across nearby frames?

- Add persistent track IDs with a lightweight association approach, such as SORT, ByteTrack, or centroid/Kalman tracking.
- Handle new tracks, missed detections, short occlusions, and objects leaving the frame.
- Measure ID switches and track fragmentation on a small annotated sequence.
- Keep short-term image track IDs distinct from persistent semantic-map object IDs. A reappearing object is not automatically the same physical object.

**Completion check:** Replay a sequence with multiple objects and demonstrate stable IDs, explicit track lifecycle behavior, and documented failure cases.

### Phase 3 — Camera calibration and geometry

**Question:** What viewing direction corresponds to an image point?

- Calibrate the wide-angle camera and evaluate a suitable distortion model using calibration residuals and visual checks.
- Save intrinsics, distortion coefficients, image resolution, and relevant camera/focus settings with the calibration record.
- Keep calibration consistent with image resizing, cropping, undistortion, and focus configuration. Validate or recalibrate after relevant changes.
- Convert image coordinates into rays or angles in an explicitly defined camera frame.

**Completion check:** Produce a reusable calibration artifact, report reprojection error, and verify ray directions against known image locations. Calibration alone does not provide object distance.

### Phase 4 — Local object position estimation

**Question:** How far away is an object, and where is it relative to the robot?

- Start with ray/ground-plane intersection using measured camera height, orientation, and a plausible floor contact point for a grounded object.
- Treat a bounding-box bottom point as a heuristic that may be invalid under occlusion or for elevated objects.
- Reject invalid intersections and rays near the horizon; carry uncertainty or validity information with each estimate.
- Compare estimated positions against manually measured distances and lateral offsets.
- Explore learned monocular depth only if the baseline exposes a concrete need and compute permits it. Do not assume relative depth is metric depth.

**Completion check:** Report local position error across several distances and viewpoints, with the assumptions and invalid cases documented.

### Phase 5 — Robot/camera localization

**Question:** Where is the camera or robot as it moves?

- Evaluate visual odometry or a lightweight SLAM approach using recorded sequences before attempting sustained onboard operation.
- For manual camera movement, estimate the necessary camera pose; reduce to planar robot position and heading only when motion and mounting assumptions support it.
- Explicitly address monocular scale ambiguity. Use a documented metric reference or constraint before combining a trajectory with metric object positions. Otherwise label the trajectory as unscaled.
- Measure drift and runtime; handle tracking loss and recovery explicitly.
- Use known or manually supplied poses as a labeled development fixture if needed for mapping work. They do not complete autonomous localization.

**Completion check:** Recover a repeatable trajectory with reported scale treatment, drift, resource usage, and tracking-loss behavior. Metric map integration requires compatible pose and object-position scales.

### Phase 6 — Semantic mapping

**Question:** What objects exist in the environment, and where are they?

- Transform local object estimates into a shared map frame using the pose at the observation timestamp.
- Associate repeated observations using compatible class, spatial, temporal, and uncertainty information. Do not merge objects solely because their class labels match.
- Maintain persistent object records with position, confidence/uncertainty, observation history, and last-seen time.
- Handle duplicate candidates, moved objects, and dynamic objects without silently treating everything as a permanent landmark.
- Save/load the map and provide a simple spatial visualization. Account for pose corrections if the localization system revises past poses.

**Completion check:** Map a small measured scene from multiple viewpoints; report object-position error, duplicates, incorrect merges, and missed objects. State whether localization was estimated or supplied.

### Phase 7 — Physical rover and motion control

**Question:** Can software reliably control physical movement?

- Specify a compatible chassis, motors, driver, power system, and optional encoders before implementing hardware-specific control.
- Document wiring, motor direction, camera mounting, and power arrangements against the selected hardware's specifications.
- Provide a motor abstraction, manual forward/reverse/turn/stop controls, command timeout, and an accessible emergency stop.
- Start with restrained or wheels-raised checks, then supervised low-speed floor tests. Motors must default to stopped at startup and after faults or stale commands.
- Measure the relationship between motion commands and actual displacement; distinguish open-loop commands from feedback-controlled movement.

**Completion check:** Demonstrate reliable manual movement and stop behavior, including timeout and fault handling, before enabling autonomous movement.

### Phase 8 — Obstacle detection and traversability

**Question:** Where can the robot safely drive?

- Build a local occupancy or cost map from a validated visual traversability/obstacle method.
- Represent free, occupied, and unknown space separately. Absence of an object detection is not evidence of free space.
- Include robot footprint, clearance, map freshness, and the limits of monocular sensing.
- Treat uncertain or unobserved space conservatively. Keep initial trials away from stairs, drop-offs, and other hazards that the current method cannot detect reliably.

**Completion check:** Evaluate free-space estimates and missed obstacles in a controlled course; demonstrate stop behavior when traversability information is missing, stale, or invalid.

### Phase 9 — Path planning

**Question:** Which route reaches the goal through traversable space?

- Begin with A* or Dijkstra on the occupancy/cost map.
- Plan for the robot's footprint and clearance, accounting for blocked and unknown regions.
- Choose a reachable approach position near the target object with a defined stand-off distance, rather than routing into the object's occupied position.
- Return waypoints and explicit outcomes for unreachable, invalid, or missing goals.

**Completion check:** Validate routes on recorded or synthetic maps, including narrow passages, blocked goals, and no-path cases, before connecting plans to motors.

### Phase 10 — Closed-loop object-goal navigation

**Question:** Can perception, localization, planning, and movement work together?

- Integrate the loop: observe, localize, update maps, plan/replan, command bounded movement, and observe again.
- Start with a simple structured command such as a target class or map-object ID. Natural-language interpretation is optional.
- Handle multiple matching objects, missing targets, localization loss, stale inputs, blocked paths, and motor/control failures explicitly.
- On arrival, stop at the stand-off position and visually verify the target. Report success, failure, or uncertainty using defined criteria.
- Keep initial trials supervised, slow, and in a bounded controlled area.

**Completion check:** Complete repeated object-goal trials with logged outcomes, time to goal, interventions, collisions/contacts, and failure causes. A single successful run is a demonstration, not a success-rate estimate.

### Phase 11 — Active perception

**Question:** Which safe viewpoint would reduce uncertainty?

- Use object identity or position uncertainty to decide whether another observation is useful.
- Begin with a small set of reachable candidate viewpoints and a simple information-benefit versus motion-cost rule.
- Bound movement and retries, preserve all navigation constraints, and support an explicit unresolved outcome.
- Do not equate higher detector confidence with improved accuracy without evaluating correctness.

**Completion check:** Compare a fixed-view or passive baseline against active viewpoint selection on the same scenes, reporting accuracy/localization changes, extra travel, time, and failures.

### Phase 12 — Evaluation and optimization

**Question:** How well does the complete system work, and what limits it?

- Consolidate measurements gathered throughout earlier phases into reproducible benchmarks.
- Report detection throughput and latency, tracking ID switches/fragmentation, local object error, pose drift, semantic-map accuracy, navigation success rate, collision/contact rate, interventions, and time to goal.
- Include CPU, peak RAM, temperature/throttling, and power measurements if instrumentation is available.
- Record hardware, OS, code revision when available, model/runtime versions, model input resolution, camera settings, configurations, scene/trial counts, and ground-truth method.
- Tune model size, quantization, image resolution, detection frequency, tracking, and scheduling against measured bottlenecks. Compare accuracy as well as speed.

**Completion check:** Provide a reproducible benchmark report and demo evidence with measured results, limitations, and comparisons. Use only these results for quantitative README or resume claims.

## Data and integration contracts

As components are introduced, keep their interfaces explicit:

- **Frames:** monotonic capture timestamp, frame ID, image dimensions, and calibration reference. Track wall-clock time separately when useful for logs.
- **Detections/tracks:** class label, score, bounding box, image coordinate convention, source timestamp, and optional track ID.
- **Local positions:** coordinate frame, units, estimate, validity, uncertainty, and source observation.
- **Poses:** timestamp, source/target frames, transform direction, scale status, and tracking quality.
- **Map objects:** persistent map ID, class evidence, location/uncertainty, timestamps, and supporting observations.
- **Occupancy/cost maps:** frame, origin, resolution, update time, and free/occupied/unknown semantics.
- **Motion commands:** documented units, bounded speed/duration, expiration time, and stop behavior.

Document coordinate conventions before integrating geometry. A sensible initial convention is meters and radians, robot `x` forward / `y` left / `z` up, and optical camera `x` right / `y` down / `z` forward. Keep camera-to-robot extrinsics explicit and never mix frames or units silently.

Bound queues and memory use. Prefer recent observations over accumulating latency, and log dropped frames. Measure capture FPS, inference FPS, control update rate, and observation age separately. Never continue physical motion based on indefinitely stale perception or pose data.

## Repository and handoff guidance

At creation of this file, the workspace had no implementation or verified setup. **Phase 1 is now in progress:** the user confirmed a working `rpicam-hello` preview, and a Python capture/detection prototype has been added. On-Pi Python capture and sustained detection measurements are still pending. See `docs/status.md` for current evidence and `docs/phase-1.md` for run instructions. Do not scaffold later phases merely because they appear in this guide.

As files become necessary, a reasonable organization is:

```text
sage/                       # project root; repository name: sage-robotics
  AGENTS.md
  README.md                 # setup, current capability, runnable demo
  src/sage/                 # implementation, separated by responsibility
  configs/                  # explicit runtime and experiment configuration
  tests/                    # meaningful algorithm and integration checks
  scripts/                  # small capture, replay, and benchmark entry points
  docs/                     # phase status, decisions, hardware, experiments
  data/                     # local recordings, calibration, and outputs
```

This layout is a proposal, not an existing tree. Keep large recordings, model weights, generated outputs, and private camera footage out of version control by default; document artifact locations and provenance instead.

For each implemented phase, record its status, exact run/replay command, configuration, artifact locations, verification environment, measured results, limitations, and next prerequisite. Use focused tests for geometry, association, planning, and control failure behavior where they protect real correctness risks. If hardware is unavailable, state what was verified locally and what remains to be checked on the Pi or rover.

Keep this file focused on durable context. Put evolving experiment results and detailed phase progress in `docs/` once implementation starts, and update this guide when the user changes scope or adopts a consequential architectural decision.
