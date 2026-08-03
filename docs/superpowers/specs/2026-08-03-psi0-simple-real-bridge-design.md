# PSI0 SIMPLE PC2 Bridge Design

Date: 2026-08-03

## Objective

Implement a simulation-first PC2 bridge that connects a SIMPLE-trained PSI0 HTTP policy server to the Python decoupled whole-body controller used by SIMPLE. The bridge must consume camera and 43-DoF robot state streams, build the PSI0 32-D observation, execute 36-D action chunks asynchronously, and publish whole-body goals at 50 Hz.

This work does not launch or modify a real Unitree G1 low-level control process. Real-hardware activation, dry-run/shadow mode, camera validation, supported-robot testing, and gain tuning remain later deployment gates.

## Scope

The implementation will add:

- `src/simple/deploy/psi0_simple_bridge.py`: transport-independent bridge core.
- `scripts/psi0_simple_real_bridge.py`: PC2 CLI with ROS2 WBC messaging, composed-camera ZMQ input, PSI0 HTTP input, and local keyboard control.
- `tests/test_psi0_simple_bridge.py`: deterministic unit and integration-style tests.
- A backward-compatible HTTP timeout option in `simple.baselines.client.HttpActionClient`.

The bridge will use the explicit 36-D command path. It will not use the 64-D SONIC token interface or the 78-D real SONIC PSI0 checkpoint contract.

## Architecture

```text
Composed camera ZMQ ── camera reader thread ─┐
                                             ├─ latest-input store
WBC state subscriber ── 50 Hz polling ──────┘          │
                                                       ▼
                                             inference worker thread
                                                       │
                                                       ▼
Local keyboard `p` ── state machine ── action queue ── 50 Hz goal publisher
                                                       │
                                                       ▼
                                      ControlPolicy/upper_body_pose
```

The bridge core receives four injected boundaries:

1. A state source returning the newest WBC state message.
2. A camera source returning the newest decoded camera message.
3. An action client implementing the existing PSI0 HTTP request contract.
4. A goal publisher accepting a Python dictionary.

This keeps joint mapping, validation, state transitions, inference scheduling, and command construction testable without ROS2, ZMQ, a GPU server, or a simulator.

## Operator State Machine

The bridge has four states:

- `PAUSED`: the startup state. It does not query PSI0. Once a valid robot state exists, it publishes a measured-pose, zero-navigation hold.
- `ACTIVE`: inference requests may be scheduled and validated actions may be executed.
- `FAULT`: a latched safe state entered on stale input, request failure, malformed output, absolute-bound violation, or action-queue underrun. It publishes a measured-pose, zero-navigation hold and ignores late inference results.
- `STOPPED`: terminal process-shutdown state.

Local keyboard `p` is the only activation control in this version:

- `PAUSED` + `p`: activate only if state and camera samples are valid and fresh.
- `ACTIVE` + `p`: pause immediately, invalidate in-flight results, clear queued actions, and capture a hold pose.
- `FAULT` + `p`: acknowledge and reactivate only if state and camera samples are valid and fresh.

Ctrl-C or normal process termination enters `STOPPED`. It does not stop the WBC control loop and must not claim that the robot remains balanced after the WBC command stream stops.

## Camera Contract

The PC2 CLI uses `ComposedCameraClientSensor` on a configurable host and port, defaulting to port 5555. The camera key is configurable because simulation and robot camera servers may publish different names.

The composed-camera decoder uses OpenCV and therefore returns BGR images. The bridge converts BGR to contiguous RGB `uint8` before sending:

```python
{"rgb_head_stereo_left": rgb_image}
```

The bridge records the local monotonic receive time for freshness checks. Server-provided timestamps are retained for diagnostics but are not compared directly to the local monotonic clock.

## Robot-State Contract

The WBC publishes `G1Env/env_state_act`. The bridge requires `q` to be a finite array with shape `(43,)`. It maps values by joint name using `instantiate_g1_robot_model(waist_location="lower_and_upper_body")`; raw slices from the WBC array are never assumed to match SIMPLE order.

The PSI0 state has shape `(1, 32)` and contains:

| PSI0 indices | Values |
|---|---|
| `0:3` | left thumb joints 0, 1, 2 |
| `3:5` | left middle joints 0, 1 |
| `5:7` | left index joints 0, 1 |
| `7:10` | right thumb joints 0, 1, 2 |
| `10:12` | right index joints 0, 1 |
| `12:14` | right middle joints 0, 1 |
| `14:21` | left arm joints |
| `21:28` | right arm joints |
| `28:31` | last executed waist roll, pitch, yaw command |
| `31` | last executed base-height command |

The final four values are command history, not measured waist configuration. They are initialized to `[0, 0, 0, 0.74]` and updated after each executed action. This matches the training converter and corrects the current `Psi0DecoupledWbcAgent` behavior, where the history value is never updated.

## Inference Contract

The worker sends the latest consistent camera/state snapshot through `HttpActionClient.query_action` with:

- instruction from the CLI;
- state `{"states": state_32}`;
- empty condition dictionary;
- dataset name `"simple"`;
- `history={"reset": True}` for the first request of each activation generation, then `{}`.

HTTP execution occurs only in the inference worker. The 50 Hz publisher must never wait on HTTP. Only one request may be in flight. A configurable low-water mark requests the next chunk before the current queue empties.

Each activation has a monotonically increasing generation number. Pausing, faulting, or stopping increments the generation. Results tagged with an older generation are discarded.

The client request timeout defaults to five seconds and is configurable. A returned result must also arrive within a configurable inference-latency budget, defaulting to 0.45 seconds, because a successful but stale response is unsafe. The prefetch low-water mark defaults to 20 actions, leaving 0.4 seconds at 50 Hz to replenish a 24-action execution chunk.

Immediately after activation, the bridge holds position while the first asynchronous request is in flight. Failure or expiry of that first request enters `FAULT`. After the first action has executed, reaching an empty action queue before a valid replacement chunk is available enters `FAULT` immediately. This prevents an old navigation command from being repeated while inference is unavailable.

## 36-D Action Mapping

Each action must be finite with shape `(36,)`. A server response must have shape `(T, 36)` with `T >= 1`.

The action layout is:

| Indices | Meaning |
|---|---|
| `0:7` | left hand: thumb 3, middle 2, index 2 |
| `7:14` | right hand: thumb 3, index 2, middle 2 |
| `14:21` | left arm |
| `21:28` | right arm |
| `28:31` | waist roll, pitch, yaw |
| `31` | base height |
| `32:36` | `vx`, `vy`, turning flag, target yaw |

The goal publisher constructs `target_upper_body_pose` in the 31-joint order returned by the WBC robot model:

1. waist yaw, roll, pitch;
2. left arm;
3. left index, middle, thumb;
4. right arm;
5. right index, middle, thumb.

Every published message contains:

- `target_upper_body_pose`: finite `(31,)` float32 array;
- `base_height_command`: finite `(1,)` float32 array;
- `navigate_cmd`: finite `(4,)` float32 array;
- `timestamp`: current monotonic time;
- `target_time`: current monotonic time plus one 50 Hz period.

The WBC loop adds interpolation garbage-collection timing itself.

## Safety Envelope

The bridge rejects an entire returned chunk if any action has the wrong shape, contains NaN or infinity, or violates an absolute bound. It does not partially execute an invalid chunk.

Absolute limits are:

- Named arm, hand, and waist targets must remain within the G1 robot model's position limits.
- Base height must be within `[0.20, 0.74]` metres.
- `vx` and `vy` must each be within `[-0.5, 0.5]` metres per second.
- Turning flag must be within `[-1, 1]`.
- Target yaw must be within `[-pi, pi]`.

After absolute validation, the command output is slew-limited relative to the last published command. The CLI-configurable simulation defaults are:

- arms: `1.0 rad/s`;
- hands: `2.0 rad/s`;
- waist: `0.5 rad/s`;
- base height: `0.1 m/s`;
- planar navigation: `0.5 m/s^2`;
- turning flag: `2.0/s`;
- target yaw: shortest-path change limited to `1.0 rad/s`.

These defaults must be reviewed during the later supported-robot phase. Slew limiting applies to valid commands; it never converts an absolute-bound violation into an accepted command.

State and camera freshness default to 0.5 seconds. Missing or stale data prevents activation and faults an active bridge.

## Hold Behavior

On startup, the bridge publishes nothing until the first valid 43-D state arrives. It never invents an upper-body initial pose.

When entering `PAUSED` or `FAULT`, it captures the upper-body joints from the newest measured state and uses that fixed snapshot as the hold target. It publishes:

- the captured upper-body pose;
- zero navigation `[0, 0, 0, 0]`;
- the last safe base height, or `0.74` before any action executes.

The hold command is republished at 50 Hz. Updating the hold target continuously from measured state is intentionally avoided because that would follow external motion instead of holding a fixed pose.

The bridge must be the sole publisher for `ControlPolicy/upper_body_pose` during a test. It does not arbitrate against a teleoperation publisher.

## Concurrency and Shutdown

Shared state is protected by a lock and exposed as immutable snapshots to inference. The camera reader and inference worker are daemon threads with explicit stop events. The main thread owns the operator keyboard state and 50 Hz publishing loop.

Shutdown order is:

1. enter `STOPPED` and invalidate the inference generation;
2. clear queued actions;
3. request worker shutdown;
4. close camera, messaging, and HTTP resources where supported;
5. join workers with bounded timeouts.

No thread receives permission to kill the PSI0 server, WBC process, simulator, or robot process.

## Testing

Tests use fake state, camera, inference, publisher, keyboard, and clock boundaries.

Required cases are:

1. Sentinel-based 43-D to 32-D mapping proves every hand and arm value is selected by name and command history occupies the last four values.
2. Sentinel-based 36-D mapping proves every named upper-body target, height, and navigation value reaches the correct goal field.
3. BGR to RGB conversion preserves shape and swaps only color channels.
4. Wrong state/action shapes and non-finite values are rejected.
5. Each absolute-bound category rejects the entire chunk.
6. Valid commands are slew-limited per 50 Hz tick.
7. The state machine starts paused and only activates on `p` with fresh inputs.
8. The 50 Hz tick continues publishing while a deliberately blocking inference fake runs.
9. State staleness, camera staleness, HTTP errors, and queue underrun latch `FAULT` and publish zero-navigation hold.
10. Late results from invalidated generations never enter the action queue.
11. Pause and shutdown clear queued actions and do not invoke process-control operations.

After unit tests pass, a bounded simulation smoke test will run with fake HTTP inference. It will verify WBC state subscription, camera decoding, 50 Hz goal publication, pause/activate behavior, and timeout-to-hold behavior without accessing Unitree DDS. A live PSI0 server integration test follows only after the fake-server smoke test passes.

## Deployment Gates After This Work

The bridge implementation does not authorize real-robot execution. The remaining sequence is:

1. Complete the bounded simulation test.
2. Add a true dry-run/shadow mode that never publishes control goals.
3. Validate real camera color, viewpoint, latency, and selected key.
4. Validate every observed and commanded joint with a supported G1 and low gains.
5. Review real-robot safety documentation, clear the safety zone, attach support, and staff the E-stop.
6. Only then launch the real DDS WBC loop and let the operator activate its balance policy with `]`.

The C++ SONIC controller and Python decoupled WBC controller must never publish real low-level commands concurrently.
