# PSI0 SIMPLE PC2 Bridge Design

Date: 2026-08-03
Status: revised after runtime-contract review

## Objective

Implement a fail-closed PC2 bridge between a SIMPLE PSI0 HTTP policy server and the Python decoupled whole-body controller (WBC). The bridge reads a composed camera stream and the WBC's 43-joint state, builds the corrected 32-D PSI0 observation, validates fixed-length 36-D action chunks, and publishes WBC goals at 50 Hz.

This milestone supports two modes only:

- `sim-control`: may publish goals only after proving that the connected WBC is the isolated MuJoCo configuration specified below.
- `shadow`: runs input, inference, validation, and scheduling but never creates a goal publisher. It may be connected to a real system for later observation-only validation.

There is deliberately no `real-control` mode in this milestone. No code in this work activates the WBC balance policy, publishes Unitree low-level commands, or stops an independently running WBC process.

## Verified blockers and decisions

### The current checkpoint is not certified for control

The inspected checkpoint records these facts:

- the stored observation has 32 meaningful values and four zero padding values;
- observation indices `28:31` vary like a three-axis torso command;
- observation index `31` is exactly `0.7400000095` throughout the recorded statistics;
- the model action horizon is 30;
- the served configuration currently returns 24 actions per request with RTC enabled.

However, `scripts/postprocess_psi0.py::build_vectors` currently constructs the three-axis history with:

```python
history_cmd[:to, 3:6][::-1]
```

That reverses episode rows. The separately implemented `build_proprio_obs` path uses the correct column reversal:

```python
history_cmd[:N, 3:6][:, ::-1]
```

The source command columns are yaw, pitch, roll, so reversing the columns produces the online order roll, pitch, yaw. Reversing rows instead is non-causal and cannot be reproduced by a live bridge. The processed training dataset needed to prove which path produced the checkpoint is not present locally.

The implementation therefore makes these decisions:

1. Fix `build_vectors` to reverse the three RPY columns, not time, and change its initial height from `0.75` to `0.74`.
2. Add a converter regression test that distinguishes row reversal from column reversal and verifies both converter entry points produce the same 32-D state.
3. Define the bridge's online history as the most recently executed command `[roll, pitch, yaw, height]`, initialized to `[0, 0, 0, 0.74]`.
4. Mark the currently inspected checkpoint `UNCERTIFIED`. It may be used only in `shadow` mode until its processed dataset is proven to use the corrected layout. If it learned the row-reversed layout, the data must be regenerated and PSI0 retrained.
5. Require a certified policy-contract document and matching server metadata before `sim-control` can activate. A fake contract is accepted only by the test-only fake server.

Certification compares a processed episode against the same raw episode under both candidate transformations. It succeeds only when every stored `state[:, 28:31]` matches chronological `[roll, pitch, yaw]`, index `31` is `0.74`, and neither the legacy row-reversed candidate nor an off-by-one history candidate matches. The certification artifact records the checkpoint hash, dataset-manifest hash, converter commit, and result.

### WBC defaults are incompatible

The WBC default is `enable_waist=False`, which puts the waist in the lower-body group and leaves a 28-element upper-body goal. This bridge requires a 31-element upper-body goal containing waist, arms, and hands. `sim-control` therefore refuses the default WBC configuration. The WBC must be launched explicitly with waist and hands enabled.

### Right-hand supplemental limits are incorrect

The current supplemental model negates the right index and middle ranges and gives the wrong signs/ranges for two right-thumb joints. The implementation corrects the supplemental values to the authoritative URDF values:

| Joint | Correct range in radians |
|---|---|
| `right_hand_thumb_0_joint` | `[-1.04719755, 1.04719755]` |
| `right_hand_thumb_1_joint` | `[-1.04719755, 0.72431163]` |
| `right_hand_thumb_2_joint` | `[-1.74532925, 0]` |
| `right_hand_index_0_joint` | `[0, 1.57079632]` |
| `right_hand_index_1_joint` | `[0, 1.74532925]` |
| `right_hand_middle_0_joint` | `[0, 1.57079632]` |
| `right_hand_middle_1_joint` | `[0, 1.74532925]` |

The bridge must use the corrected robot model. Bypassing supplemental limits only inside the bridge is prohibited because the WBC safety monitor uses the same model.

### The current 24-action server misses the scheduling budget

At 50 Hz, 24 actions provide 0.48 seconds of runway. The scheduler contract is:

```text
execution_horizon / control_frequency
    > inference_deadline + scheduling_margin
```

The initial exact values are:

- execution horizon: 24 actions;
- control frequency: 50 Hz;
- inference deadline: 0.40 seconds;
- scheduling margin: 0.06 seconds.

Thus `0.48 > 0.46` is valid mathematically. The inspected RTX 3060 server produced six warmed request latencies with p50 approximately 0.440 seconds, p95 approximately 0.469 seconds, and maximum approximately 0.474 seconds. It does not meet the 0.40-second deadline and is blocked from control use. A longer certified execution horizon, a faster server, or an optimized model must be selected and re-benchmarked; the bridge will not reduce the margin to accommodate the current measurements.

The five-second HTTP timeout is only a network/worker cleanup bound. It is not the control deadline.

## Files and components

The implementation will add or change:

- `src/simple/deploy/psi0_simple_bridge.py`: transport-independent state machine, named-joint mapping, validation, scheduler, and goal construction.
- `scripts/psi0_simple_real_bridge.py`: PC2 CLI, local keyboard input, WBC messaging adapters, composed-camera reader, policy client, and structured metrics.
- `src/simple/baselines/client.py`: optional HTTP timeout whose general default remains `None`; this bridge passes five seconds explicitly.
- `scripts/postprocess_psi0.py`: corrected chronological RPY history and `0.74` initial height.
- `third_party/decoupled_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py`: corrected right-hand limits.
- `third_party/decoupled_wbc/control/main/teleop/configs/configs.py`: explicit `domain_id` option copied into WBC `DOMAIN_ID` so simulation can be isolated.
- unit tests for the bridge, converter, joint limits, HTTP timeout, and WBC preflight.
- a fake policy server, fake composed-camera server, policy-contract fixture, and bounded simulation smoke-test driver under `scripts/tests/`.

The bridge uses the explicit SIMPLE 36-D action path. It does not use the 64-D SONIC token interface or the 78-D real SONIC PSI0 checkpoint contract.

## Architecture

```text
WBC config service ── bounded startup preflight ───────────────┐
Certified policy contract ── server-contract comparison ──────┤
                                                              ▼
Composed camera ZMQ ── owned polling reader ──┐          PAUSED/blocked
                                              ├─ latest valid snapshot
WBC state subscriber ── 50 Hz polling ───────┘          │
                                                        ▼
Local keyboard `p` ── generation state machine ── one HTTP worker
                                                        │
                                            current/staged chunks
                                                        │
                                                        ▼
                                  50 Hz validator/goal constructor
                                                        │
                                    sim-control only ────┘
                                                        ▼
                                      ControlPolicy/upper_body_pose
```

The core receives an injected clock, state source, camera snapshot store, action client, policy contract, and optional publisher. Unit tests use fakes and do not initialize ROS2, ZMQ, Unitree DDS, a GPU server, or MuJoCo.

## Startup preflight

Preflight completes before the bridge creates a control-goal publisher. Failure exits nonzero without offering activation.

### WBC contract

The CLI queries `WBCPolicy/robot_config` with a three-second overall startup deadline. It must not use the existing indefinitely waiting ROS service-client constructor. A bridge-specific bounded adapter waits in short intervals, aborts at the deadline, and destroys its temporary client cleanly.

For `sim-control`, the returned configuration must match all of the following:

| Field | Required value |
|---|---|
| `env_type` | `sim` |
| `interface` | `lo` |
| `simulator` | `mujoco` |
| `messaging_backend` | `ros2` |
| `control_frequency` | `50` |
| `enable_waist` | `true` |
| `with_hands` | `true` |
| `wbc_version` | `gear_wbc` |
| `wbc_policy_class` | `G1DecoupledWholeBodyPolicy` |
| `wbc_model_path` | `policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx` |
| `domain_id` | the CLI's dedicated simulation domain, initially `42` |

The bridge then instantiates the same `lower_and_upper_body` G1 model and asserts:

- the full named state has 43 joints;
- the upper-body target group has exactly 31 unique names;
- it contains three waist, fourteen arm, and fourteen hand joints;
- the two configured ONNX model files exist and have the expected 516-element input and 15-element output signatures.

For `shadow`, the configuration is reported but differences do not create a publisher. Joint names and dimensions must still be known before validation can run.

### Publisher ownership

In `sim-control`, ROS graph inspection must report zero existing publishers and exactly one WBC subscription on `ControlPolicy/upper_body_pose`. Only then does the bridge create the sole publisher and verify the count becomes one publisher/one subscription. A teleoperation or second policy publisher causes startup failure.

`shadow` never creates a publisher and asserts that publisher count is unchanged across its lifetime.

### Policy contract

In `sim-control`, the CLI requires a certified local JSON contract and fetches `/contract` from the selected PSI0 server with a two-second startup timeout. The two documents must match on:

- schema version `simple.psi0.policy-contract.v1`;
- checkpoint SHA-256 and dataset-manifest SHA-256;
- converter layout `g1_simple_32_rpyh_v2`;
- observation dimension 32 and action dimension 36;
- model horizon 30 and an integer execution horizon H in `[24, 30]`, with the local and server values exactly equal;
- RTC enabled and RTC maximum delay;
- image key `rgb_head_stereo_left` and RGB numeric channel order.

The contract must also satisfy `30 - H <= rtc_max_delay` and the scheduling-runway equation below. The current server does not expose this endpoint, so live-server control integration remains blocked until that server dependency is added. The fake server exposes an explicitly test-only contract. `sim-control` accepts the test-only contract only when the WBC reports `env_type=sim`.

In `shadow`, the CLI attempts the same comparison but may continue after a missing or mismatched contract. Every preview and metric record is then labeled `policy_certified=false`; this exception never enables a publisher.

## Camera contract

The OAK producer converts its OpenCV BGR frame to RGB before serialization. Although JPEG encode/decode uses OpenCV on both sides, an actual codec round trip preserves the numeric channel positions. The composed-camera client output is therefore RGB-by-number, and the bridge default is identity, not an unconditional BGR-to-RGB conversion.

The CLI option `--camera-source-key` selects one exact composed-camera image key; there is no fallback to the first available image. `--camera-color-order rgb|bgr` defaults to `rgb`, and `bgr` performs one channel swap before policy submission. In either case, the policy receives a contiguous `uint8` HWC array under:

```python
{"rgb_head_stereo_left": rgb_image}
```

The fake-camera test sends spatially separate red and blue patches through the real `ImageMessageSchema` JPEG codec. The bridge verifies that red remains dominant in the red patch and blue remains dominant in the blue patch, allowing for JPEG tolerance. Real-camera deployment separately requires a saved visual validation for the selected key, viewpoint, and color-order option.

The camera ZMQ SUB socket is created, polled, read, and closed only by its reader thread. Polling/`RCVTIMEO` is 100 ms, so a stop event bounds reader shutdown. Each accepted image stores its local monotonic receive time; producer timestamps are diagnostic only.

## Robot-state and observation contract

The bridge consumes `G1Env/env_state_act`. A state sample is accepted only when `q`:

- has shape `(43,)`;
- contains only finite values;
- maps one-to-one to the preflighted 43 joint names;
- lies within the corrected URDF/model position bounds with a measured-state tolerance of `0.05` radians.

An invalid sample never replaces `last_valid_q`. In `ACTIVE`, it causes an immediate atomic fault. In `PAUSED` or `shadow`, it is reported and ignored. Holds are constructed only from `last_valid_q`, never from an invalid current sample.

The PSI0 observation has shape `(1, 32)`:

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

Raw numeric slices are not used for named joints. The last four values are command history, not measured waist state.

Freshness uses local monotonic receive times:

- maximum state age: 0.10 seconds;
- maximum camera age: 0.25 seconds;
- maximum absolute camera/state receive-time skew: 0.10 seconds.

Activation and every inference snapshot require all three conditions. While active, an age or skew violation faults on the next 50 Hz tick. The snapshot copies state and image atomically under the latest-input lock before releasing it to the worker.

## Operator and fault state machine

The bridge has four states:

- `PAUSED`: startup state; does not query PSI0. After the first valid state it republishes a fixed measured-pose, zero-navigation hold in `sim-control`.
- `ACTIVE`: fixed-horizon requests and validated actions may execute.
- `FAULT`: latched safe state; all policy actions are discarded and a fixed zero-navigation hold is published in `sim-control`.
- `STOPPED`: terminal shutdown state.

Local `p` is the only production activation toggle:

- `PAUSED` + `p`: activate only after successful preflight, fresh synchronized inputs, no fault, and a physically idle HTTP worker.
- `ACTIVE` + `p`: atomically pause, invalidate the generation, clear both action buffers and request bookkeeping, and capture a hold.
- `FAULT` + `p`: acknowledge and rearm only with fresh synchronized inputs and a physically idle worker. If an invalidated request is still blocked, rearm is refused with a visible `worker busy` diagnostic; the operator presses `p` again after it returns or times out.

Every pause, fault, rearm, and stop increments a monotonically increasing generation. Worker results carry the generation with which they were launched and are discarded on mismatch.

`enter_fault(reason)` is one locked transition that:

1. latches `FAULT` and records the first reason/time;
2. increments the generation;
3. clears the current and staged chunks and their indices;
4. clears logical request/deadline/history bookkeeping;
5. captures a hold from `last_valid_q` and last safe height;
6. leaves the separate physical-worker-busy flag set until the blocked request really returns.

No new worker thread is spawned to work around a blocked request. The one worker either returns or reaches its explicit HTTP timeout; its late result is discarded.

## Fixed-horizon inference scheduler

The bridge sends:

- the instruction supplied on the CLI and copied into the metrics record;
- image `{"rgb_head_stereo_left": image}`;
- state `{"states": state_32}`;
- an empty condition dictionary;
- dataset name `simple`;
- `history={"reset": True}` only on the first request of an activation generation, then `{}`.

Only the inference worker performs HTTP. `HttpActionClient` retains `timeout=None` as its general backward-compatible default; this bridge constructs it with a five-second timeout.

The scheduler is a fixed-horizon double buffer, not a low-water queue:

1. On activation, dispatch request R0 and publish hold while it runs.
2. R0 must return exactly `(H, 36)` within the 0.40-second control deadline. After whole-chunk validation, it becomes `current`.
3. On the same tick that execution of `current[0]` starts, snapshot the newest inputs and dispatch exactly one successor request.
4. A valid successor becomes `staged`; it never replaces or appends to the partly executed current chunk.
5. After exactly H current actions, atomically swap the complete staged chunk to current and immediately dispatch the next successor from a new snapshot.
6. Missing staged data at the swap, any response after its independent 0.40-second deadline, or any invalid response faults immediately. The five-second HTTP operation may remain blocked in the worker, but cannot delay the main-loop fault.

The returned shape must be exactly `(H, 36)`, where H equals the certified server execution horizon. Arbitrary `T >= 1` responses are rejected. The worker timestamps completion with the shared monotonic clock. Each tick drains a completed result before evaluating the deadline, but accepts it only when its recorded completion time is at or before the request deadline. The startup contract equation is checked numerically, and runtime uses absolute monotonic request deadlines rather than accumulated sleep intervals.

## 36-D action mapping and goal construction

Each action is:

| Indices | Meaning |
|---|---|
| `0:7` | left hand: thumb 3, middle 2, index 2 |
| `7:14` | right hand: thumb 3, index 2, middle 2 |
| `14:21` | left arm |
| `21:28` | right arm |
| `28:31` | waist roll, pitch, yaw |
| `31` | base height |
| `32:36` | `vx`, `vy`, turning flag, target yaw |

The goal's `target_upper_body_pose` follows the exact 31-name order returned by the preflighted WBC model. For the selected model that order is:

1. waist yaw, roll, pitch;
2. left arm;
3. left index, middle, thumb;
4. right arm;
5. right index, middle, thumb.

Mapping is by joint name, including the waist reordering from action RPY to WBC yaw/roll/pitch. Every goal contains:

- `target_upper_body_pose`: finite float32 shape `(31,)`;
- `base_height_command`: finite float32 shape `(1,)`;
- `navigate_cmd`: finite float32 shape `(4,)`;
- `timestamp`: current monotonic time;
- `target_time`: current monotonic time plus 0.02 seconds.

The command-history values for the next observation update only after an action is actually selected for publication/execution, not when a chunk arrives or while shadow validation merely previews it.

In `shadow`, the same mapped goal is recorded as a non-published preview and command history advances on the simulated 50 Hz execution schedule. It is always labeled `published=false`.

## Action safety envelope

The entire H-action response is rejected before staging if any action has the wrong shape, contains a non-finite value, or violates an absolute bound. Validation does not clip absolute violations.

Absolute bounds are:

- exact corrected model/URDF position limits for every named waist, arm, and hand target;
- base height `[0.20, 0.74]` metres;
- `vx` and `vy` each `[-0.5, 0.5]` metres per second;
- turning flag `[-1, 1]`;
- target yaw `[-pi, pi]`.

After whole-chunk absolute validation, per-tick slew limits apply relative to the last published command, or the last simulated command in shadow:

- arms: `1.0 rad/s`;
- hands: `2.0 rad/s`;
- waist: `0.5 rad/s`;
- base height: `0.1 m/s`;
- planar navigation: `0.5 m/s^2`;
- turning flag: `2.0/s`;
- target yaw: shortest-path change limited to `1.0 rad/s`.

These are simulation defaults, not approved real-robot gains or rates.

## Hold and WBC ownership behavior

Before any valid state, the bridge publishes nothing. It never invents an upper-body initial pose.

Entering `PAUSED` or `FAULT` captures the upper-body portion of `last_valid_q` once and republishes that fixed pose at 50 Hz with zero navigation and the last safe height (`0.74` before any action). It does not continuously follow measured joints.

When Ctrl-C is received in `sim-control`, the bridge first enters paused hold and publishes 25 final hold ticks (0.5 seconds), then transitions to `STOPPED` and closes its publisher. This leaves the WBC's last received goal as a zero-navigation hold, but does not guarantee balance after the bridge exits. The WBC and its lower-body balance/walk policy are separate processes whose current activation state is unchanged by `p` or Ctrl-C. Operators must not terminate the WBC or remove support based on the bridge's last command.

## Concurrency and bounded shutdown

The main thread owns the 50 Hz state machine, keyboard events, deadlines, action-buffer swaps, and publication. One camera reader owns its ZMQ socket. One inference worker owns its HTTP session. Shared inputs/results use immutable copies behind short-held locks.

Shutdown is bounded as follows:

1. latch stop, invalidate the generation, clear both action buffers, and publish the final 0.5-second hold in `sim-control`;
2. signal the camera reader, whose 100 ms poll exits and whose own thread closes its socket;
3. prevent new HTTP work and wait at most 5.5 seconds for the existing five-second request timeout;
4. close messaging resources and restore the terminal in `finally` blocks;
5. report any daemon worker that did not exit, while allowing the process to terminate; no non-daemon worker may remain.

The bridge never kills the PSI0 server, WBC, simulator, or any robot process.

## Verification

### Deterministic tests

Tests must cover:

1. Converter sentinels prove chronological row order, yaw/pitch/roll-to-roll/pitch/yaw column order, `0.74` height, and agreement between both converter paths.
2. A 43-value named sentinel maps every hand and arm joint to the correct 32-D index; the final four values are executed-command history.
3. A 36-value named sentinel maps every waist, arm, hand, height, and navigation value to the correct 31-D WBC goal position.
4. Every commanded joint's effective model limit matches its URDF limit; its midpoint and endpoints are accepted, and values just below/above are rejected.
5. Invalid measured-state shape, non-finite values, out-of-bounds values, stale age, and excessive camera/state skew cannot replace `last_valid_q` or activate control.
6. RGB and BGR options pass red/blue sentinels through the actual composed-camera JPEG schema with tolerance.
7. The bridge starts paused; local `p` activates only after both preflights and synchronized inputs.
8. Fixed H-action responses stage and swap whole; short, long, late, malformed, or invalid chunks fault without partial execution.
9. A blocked fake request never blocks 50 Hz publication, cannot be bypassed by a second worker, and prevents rearm until physically idle.
10. Pause, fault, and stop atomically clear current/staged actions; all late generations are discarded; the first fault goal has zero navigation.
11. The general HTTP client remains unbounded by default while the bridge passes five seconds explicitly.
12. WBC config mismatches for waist, hands, frequency, models, dimensions, environment, interface, domain, or publisher ownership fail before publisher creation.
13. Camera polling and terminal cleanup finish within their stated shutdown bounds.

### Isolated 15-second simulation smoke test

After unit tests pass, the automated smoke test is run from the repository root:

```bash
ROS_DOMAIN_ID=42 uv run python scripts/tests/smoke_psi0_simple_bridge.py \
  --duration-s 15 \
  --unitree-domain-id 42 \
  --camera-port 15555 \
  --policy-port 22086
```

The driver launches:

- MuJoCo WBC with `--interface sim --enable-waist --with-hands --domain-id 42`, offscreen and onscreen rendering disabled;
- a fake composed-camera publisher using the real codec and red/blue sentinel image;
- a fake `/contract` plus `/act` server returning exact `(24, 36)` chunks at 50 ms latency;
- the bridge in a pseudo-terminal so the test sends the same local `p` byte as an operator.

The phases are:

- seconds 0-3: paused hold;
- seconds 3-11: active fake inference;
- at second 11: the fake policy stalls its next request beyond the 0.40-second deadline;
- seconds 11-13: observe latched fault and hold;
- seconds 13-15: Ctrl-C and cleanup.

Pass criteria are all of:

- before bridge start, goal-topic publisher/subscription counts are `0/1`; while running they are `1/1`; after exit publishers return to zero;
- during each steady publication phase, mean rate is 49-51 Hz and maximum inter-message gap is at most 60 ms;
- the main loop continues publishing while HTTP is blocked;
- fault is recorded no later than request start plus 0.42 seconds (deadline plus one tick);
- the first goal at or after the fault transition has navigation exactly `[0, 0, 0, 0]` and no queued policy action executes afterward;
- all bridge threads exit, child PIDs are reaped, terminal settings are restored, and all test ports can be rebound;
- the generated metrics file records zero real-interface connections and zero extra goal publishers.

This smoke test does initialize Unitree SDK2 DDS inside the simulator. Safety comes from loopback interface plus dedicated Unitree domain 42, together with ROS domain 42; it does not claim to avoid DDS.

### Live policy integration gate

The fake server is replaced only after a certified corrected checkpoint exists and at least 100 representative warmed requests show p99 latency at or below 0.40 seconds with no timeout or shape failure. The latency report, policy contract, and checkpoint hash are saved together. The currently measured RTX 3060 server fails this gate.

## Later real-robot deployment gates

No `real-control` entry point is added until all of the following are separately reviewed:

1. Run `shadow` on PC2 with the real 43-D state and camera while proving it creates no goal publisher.
2. Validate camera viewpoint, selected key, RGB order, resolution, and state/camera skew using saved samples.
3. Validate every observed joint and every mapped goal index by name on a supported G1.
4. Resolve checkpoint provenance or retrain from corrected data, certify its contract, and pass the latency gate.
5. Perform supported-robot tests with physical support, low gains/rates, staffed E-stop, and a clear safety zone.
6. Add and review an explicit `real-control` mode whose preflight requires the approved real interface and deployment profile.
7. Only then launch the real WBC DDS loop and let the operator separately activate its balance policy with `]`.

The C++ SONIC controller and Python decoupled WBC controller must never publish real low-level commands concurrently.
