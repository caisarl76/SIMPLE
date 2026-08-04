# PSI0 SIMPLE PC2 Bridge Design

Date: 2026-08-03
Revised: 2026-08-04
Status: revised after third runtime-contract review

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

The shoulder-roll differences are treated separately and retained deliberately as the bridge's conservative no-cross-torso control envelope:

| Joint | URDF range | Effective control range |
|---|---|---|
| `left_shoulder_roll_joint` | `[-1.5882, 2.2515]` | `[0.19, 2.2515]` |
| `right_shoulder_roll_joint` | `[-2.2515, 1.5882]` | `[-2.2515, -0.19]` |

These two effective ranges are strict subsets of the URDF ranges. They are not corrected to URDF equality. After fixing the hands, an audit of all 31 commanded joints must find exactly these two allowlisted discrepancies and no others. Action and measured-state validation use the effective model limits, so a shoulder-roll target may be URDF-valid yet intentionally rejected by the control envelope.

### The current server cannot satisfy the RTC timing contract

This design follows the notation and timing model in the [original RTC paper](https://www.physicalintelligence.company/download/real_time_chunking.pdf) and the [LeRobot RTC documentation](https://huggingface.co/docs/lerobot/en/rtc):

- `P`: prediction horizon in 50 Hz action ticks;
- `s`: execution horizon, the number of new actions contributed by each completed RTC response;
- `d`: a fixed, certified inference-delay bound in 50 Hz action ticks;
- `Delta = 0.02 s`: one action tick.

The required constraints are:

```text
2 <= d <= s
d + s <= P
d < rtc_training_max_delay
```

The first inequality permits inference to start `s - d` ticks after a chunk begins. The second guarantees that a length-`P` prediction contains the `d` committed prefix actions plus all `s` newly executable actions. The third matches the checkpoint's training-time RTC implementation, whose delay upper bound is exclusive.

For the current `P=30`, `s=24` checkpoint, `d` can be at most 6 ticks, so the runtime response must be available by 0.12 seconds. Certification reserves one scheduling tick and therefore requires warmed p99 latency at or below 0.10 seconds. The inspected RTX 3060 server produced six warmed latencies with p50 approximately 0.440 seconds, p95 approximately 0.469 seconds, and maximum approximately 0.474 seconds. It misses the six-tick bound by a wide margin.

The previously proposed 0.40-second deadline is 20 ticks, and `20 + 24 <= 30` is false. Supporting `s=24, d=20` requires at least `P=44` and a checkpoint trained with `rtc_training_max_delay > 20`; it is not a client-only configuration change.

The deployed `/act` implementation is also not a correct RTC transport for this bridge. It hardcodes `d=P-s=6`, conditions on a server-side raw prediction shifted by `s`, returns `prediction[:s]`, and cannot receive the exact post-slew actions that PC2 will commit during inference. A versioned RTC request/response protocol defined below is required. The current checkpoint and server remain blocked from `sim-control`; only the fake protocol server is eligible for this milestone.

The five-second HTTP timeout is only a network/worker cleanup bound. It is not the RTC deadline.

## Files and components

The implementation will add or change:

- `src/simple/deploy/psi0_simple_bridge.py`: transport-independent state machine, named-joint mapping, validation, scheduler, and goal construction.
- `scripts/psi0_simple_real_bridge.py`: PC2 CLI, local keyboard input, WBC messaging adapters, composed-camera reader, policy client, and structured metrics.
- `src/simple/baselines/client.py`: optional HTTP timeout whose general default remains `None`, plus a versioned RTC query/metadata path; this bridge passes five seconds explicitly.
- `scripts/postprocess_psi0.py`: corrected chronological RPY history and `0.74` initial height.
- `third_party/decoupled_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py`: corrected right-hand limits while retaining the explicitly classified conservative shoulder-roll envelope.
- `third_party/decoupled_wbc/control/main/teleop/configs/configs.py`: serialized `env_type` plus an explicit `domain_id` copied into WBC `DOMAIN_ID` so simulation can be isolated.
- `third_party/decoupled_wbc/control/main/teleop/run_g1_control_loop.py` and a nested helper: publish the config service only after model construction and include the canonical connected-model attestation.
- unit tests for the bridge, converter, joint limits, HTTP timeout, and WBC preflight.
- a fake policy server, fake composed-camera server, policy-contract fixture, and bounded simulation smoke-test driver under `scripts/tests/`.
- `.gitmodules` only if an accessible decoupled-WBC fork must replace the current fetch URL.

A companion change in the separately checked-out PSI0 repository must add the versioned RTC endpoint before a live policy can replace the fake server. The policy contract records the exact PSI0 server commit. That external server change is not silently treated as part of this SIMPLE repository commit.

The bridge uses the explicit SIMPLE 36-D action path. It does not use the 64-D SONIC token interface or the 78-D real SONIC PSI0 checkpoint contract.

## Submodule delivery contract

`third_party/decoupled_wbc` is a Git submodule, so editing files beneath it without publishing the nested commit would produce an unusable root gitlink. Delivery requires this order:

1. Create one nested decoupled-WBC commit containing the config serialization, domain ID, right-hand limits, and their nested tests.
2. Push that exact nested commit to a fetchable remote before recording it in SIMPLE. If the current upstream is not writable, use an accessible fork and update `.gitmodules` to its HTTPS fetch URL.
3. Push a named delivery branch and verify that `git ls-remote --exit-code <submodule-fetch-url> refs/heads/<delivery-branch>` reports the exact nested SHA from the deployment environment.
4. Update and commit the SIMPLE root gitlink only after step 3 succeeds.
5. Clone the local SIMPLE commit with `git clone --no-local`, run `git submodule update --init --recursive`, assert the nested SHA, and run the nested plus root tests from that clean checkout.

A local-only nested commit, a dirty submodule, or a root commit pointing to an unreachable object is not a deliverable. Publishing the nested commit is an external write and requires explicit repository authority when implementation reaches that step.

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

`BaseConfig.__post_init__` currently assigns `env_type` dynamically, while `ArgsConfig.to_dict()` uses `dataclasses.asdict()` and consequently omits it. The nested WBC change declares:

```python
env_type: Literal["sim", "real"] = field(init=False)
```

`__post_init__` continues assigning the value returned by `resolve_interface`. The regression test constructs the real `ControlLoopConfig(interface="sim", domain_id=42)`, calls `to_dict()`, and asserts the payload contains `env_type="sim"`, normalized `interface="lo"` on Linux, and `domain_id=42`. The test inspects the actual service payload rather than a bridge fake.

The WBC currently publishes its config service before constructing `robot_model` or `wbc_policy`. The nested change moves service publication until both objects have constructed successfully and adds `model_contract` plus `model_contract_sha256` to the response. `model_contract` uses schema `decoupled_wbc.g1-model-contract.v1` and contains:

- the exact decoupled-WBC Git SHA and `working_tree_clean=true`;
- robot-model name;
- all 43 ordered joint names;
- all 43 effective lower and upper position-limit arrays after supplemental overrides;
- the exact 31 ordered `upper_body` joint names;
- the selected URDF SHA-256;
- both resolved WBC ONNX paths, file SHA-256 values, and 516-input/15-output signatures.

The digest is SHA-256 over UTF-8 canonical JSON of `model_contract` using sorted keys and compact separators; the digest field itself is excluded. In `sim-control`, PC2:

1. recomputes the response digest and rejects malformed content;
2. obtains the expected nested SHA from the SIMPLE root gitlink and requires an exact match plus a clean WBC tree;
3. builds the expected contract from its clean checked-out submodule;
4. compares the full ordered names, effective limits, URDF hash, ONNX hashes/signatures, and final digest.

This attests the model actually used by the connected WBC. A process running another checkout, the old signed hand limits, reordered joints, or different ONNX files cannot pass merely by returning compatible top-level configuration fields.

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

The attested connected model and the locally built expected `lower_and_upper_body` G1 model must both assert:

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

- schema version `simple.psi0.policy-contract.v2`;
- checkpoint SHA-256 and dataset-manifest SHA-256;
- exact PSI0 server Git commit;
- converter layout `g1_simple_32_rpyh_v2`;
- observation dimension 32 and action dimension 36;
- action frequency 50 Hz, prediction horizon `P`, execution horizon `s`, and fixed delay bound `d`;
- RTC enabled and exclusive `rtc_training_max_delay`;
- RTC endpoint `/act-rtc-v1`, committed-prefix semantics, and executable-suffix response semantics;
- image key `rgb_head_stereo_left` and RGB numeric channel order.

The contract must satisfy `2 <= d <= s`, `d + s <= P`, and `d < rtc_training_max_delay`. Requiring at least two delay ticks permits the certification rule to reserve one full scheduling tick. The current server exposes neither this metadata nor `/act-rtc-v1`, so live-server control integration remains blocked until the companion server change is delivered. The fake server exposes an explicitly test-only v2 contract. `sim-control` accepts the test-only contract only when the WBC reports `env_type=sim`.

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
- lies within the attested effective model position bounds with a measured-state tolerance of `0.05` radians.

An invalid sample never replaces `last_valid_q`. In `ACTIVE`, it causes an immediate atomic fault. In `PAUSED` or `shadow`, it is reported and ignored. The tolerance absorbs small measurement overshoot for state validity only; it never expands a commanded or held target limit.

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

- `PAUSED`: startup state; does not query PSI0. After the first valid state it republishes a fixed bounded-pose, zero-navigation hold in `sim-control`.
- `ACTIVE`: fixed-horizon requests and validated actions may execute.
- `FAULT`: latched safe state; all policy actions are discarded and a fixed zero-navigation hold is published in `sim-control`.
- `STOPPED`: terminal shutdown state.

Local `p` is the only production activation toggle:

- `PAUSED` + `p`: activate only after successful preflight, fresh synchronized inputs, no fault, and a physically idle HTTP worker.
- `ACTIVE` + `p`: atomically pause, invalidate the generation, clear current, staged, and committed action buffers plus request bookkeeping, and capture a hold.
- `FAULT` + `p`: acknowledge and rearm only with fresh synchronized inputs and a physically idle worker. If an invalidated request is still blocked, rearm is refused with a visible `worker busy` diagnostic; the operator presses `p` again after it returns or times out.

Every pause, fault, rearm, and stop increments a monotonically increasing generation. Worker results carry the generation with which they were launched and are discarded on mismatch.

`enter_fault(reason)` is one locked transition that:

1. latches `FAULT` and records the first reason/time;
2. increments the generation;
3. clears the current, staged, and immutable committed-prefix buffers and their indices;
4. clears logical request/deadline/history bookkeeping;
5. constructs the bounded hold defined below from fresh `last_valid_q` or the last safe published goal;
6. leaves the separate physical-worker-busy flag set until the blocked request really returns.

No new worker thread is spawned to work around a blocked request. The one worker either returns or reaches its explicit HTTP timeout; its late result is discarded.

## RTC request/response and scheduler

### Tick and action semantics

Controller tick `k` denotes the boundary at which command `a[k-1]` has just been consumed and command `a[k]` will be selected for the next 0.02-second interval. The tick order is:

1. Poll and validate the newest WBC state and camera.
2. Build observation `o[k]`; its command-history tail is the final post-slew command successfully published at tick `k-1`.
3. If the RTC trigger is due, reserve the next `d` commands, snapshot `o[k]`, and dispatch the request.
4. Select and publish `a[k]`.
5. Only after successful publication, update command history to the final post-slew RPY/height from `a[k]`.

In `shadow`, steps 4-5 advance the explicitly labeled simulated command timeline without constructing a publisher.

For a request launched at observation tick `r`, the generated length-`P` plan is aligned so plan index `j` represents global action tick `r+j`. Plan indices `0:d` are committed commands already guaranteed to execute while inference runs. The server returns only plan indices `d:d+s`, exactly `s` newly executable actions whose first execution tick is `r+d`.

### Versioned HTTP payload

The bridge calls `/act-rtc-v1` with the normal SIMPLE fields:

- instruction supplied on the CLI and copied into the metrics record;
- image `{"rgb_head_stereo_left": image}`;
- state `{"states": state_32}`;
- empty condition dictionary;
- dataset name `simple`.

Every request, including R0, uses the same RTC schema. R0 carries `reset=True`; successors omit it:

```python
history = {
    "reset": True,
    "session_id": session_id,
    "request_seq": 0,
    "observation_tick": r0,
    "rtc_delay_steps": d,
    "committed_actions": committed_hold_actions,  # shape (d, 36)
}
```

Each successor carries:

```python
history = {
    "session_id": session_id,
    "request_seq": request_seq,
    "observation_tick": r,
    "rtc_delay_steps": d,
    "committed_actions": committed_post_slew_actions,  # shape (d, 36)
}
```

The server first resets the named session when requested, then normalizes the supplied committed actions, uses them as RTC plan prefix `0:d`, generates a length-`P` plan, and returns denormalized plan suffix `d:d+s`. Every response action shape is exactly `(s, 36)`, including R0. Its RTC metadata echoes `session_id`, `request_seq`, `observation_tick`, `P`, `s`, `d`, and `first_action_tick=r+d`. Missing or mismatched metadata is a whole-response fault; there is no alternate reset response schema.

This protocol makes the actual post-slew PC2 commands, rather than the server's previous raw prediction, the RTC conditioning source. The current `/act` endpoint does not implement these semantics and is ineligible for control.

### Scheduling algorithm

Only one inference worker performs HTTP. `HttpActionClient` retains `timeout=None` as its general backward-compatible default; this bridge constructs it with a five-second timeout.

1. Before activation is permitted, at least one bounded paused-hold goal must have published successfully. At activation tick `r0`, repeat that exact zero-navigation 36-D hold action `d` times, mark the prefix immutable, and dispatch R0 with `o[r0]` and the full tick metadata above.
2. Continue publishing the committed hold prefix during ticks `r0` through `r0+d-1`. R0 must return and validate before action selection at tick `r0+d`; an early valid suffix is staged.
3. At tick `r0+d`, require the staged `(s, 36)` suffix, make it current, and publish its element zero. The response metadata fixes this element's global tick as `first_action_tick=r0+d`.
4. At the next request tick `r=r0+s`, exactly `d` current actions remain. Starting from the last published command, run the slew limiter forward across those `d` raw actions, replace them with the resulting effective commands, mark them immutable, and send that exact `(d, 36)` committed prefix with `o[r]`.
5. Continue publishing the immutable committed prefix during ticks `r` through `r+d-1`. An early valid response is stored as one staged suffix and does not alter these commands.
6. At tick `r+d`, after all committed commands have been consumed, require the staged suffix and atomically make it current. Its element zero is the command for tick `r+d`.
7. Every later request starts `s` ticks after the previous request and after `s-d` actions from the current suffix have executed. Thus every observation-to-first-new-action delay is exactly `d` ticks.
8. If a response fails validation or HTTP before handoff, fault immediately. If no valid response is available before action selection at tick `r+d`, atomically fault and publish zero-navigation hold instead of another policy action.

The worker records completion with the shared monotonic clock and request tick. Each main-loop tick drains a completed result before checking handoff, but a result stamped after its `r+d` deadline is rejected. No response appends behind an arbitrary queue, replaces partly executed commands, or changes a committed prefix.

For the initial fake contract `P=30, s=24, d=6`, R0 begins with six committed hold actions and every successor begins with six committed policy actions remaining. Each runtime handoff is 0.12 seconds later, and latency certification requires warmed p99 at or below 0.10 seconds. The generic five-second HTTP timeout only bounds the physical worker after a logical RTC fault.

### Time-indexed sentinel example

The deterministic scheduler test uses `P=8, s=5, d=3` and activates at tick 100:

- `o[100]` contains post-slew paused-hold history from tick 99; R0 carries `observation_tick=100`, `d=3`, and exact committed hold commands for ticks 100-102;
- those hold commands publish unchanged at ticks 100-102 while R0 runs;
- R0 returns generated-plan indices `3:8`, tagged for global ticks 103-107, and must be available before tick 103 selection;
- tick 103 publishes R0 response element 0;
- tick 105 starts R1 with history from tick 104 and exact post-slew committed commands for ticks 105-107;
- R1 returns generated-plan indices `3:8`, tagged for global ticks 108-112, and tick 108 publishes R1 response element 0.

Unique per-tick and per-dimension sentinels prove there are no duplicates, skips, raw-versus-slew substitutions, stale-history updates, or off-by-one handoffs.

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

The command-history values for the next observation update only after the final post-slew action is successfully published. At request tick `r`, the observation therefore contains the RPY/height from tick `r-1`; reserving future committed actions does not advance history. In shadow, the equivalent update happens only when the simulated 50 Hz timeline consumes a preview action.

In `shadow`, the same mapped goal is recorded as a non-published preview and command history advances on the simulated 50 Hz execution schedule. It is always labeled `published=false`.

## Action safety envelope

The entire `s`-action executable suffix is rejected before staging if any action has the wrong shape, contains a non-finite value, or violates an absolute bound. The separately supplied `d`-action committed prefix was already validated and slew-limited before dispatch. Validation does not clip absolute violations.

Absolute bounds are:

- corrected effective model limits for every named waist, arm, and hand target, including the two conservative shoulder-roll subsets;
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

`build_bounded_hold(now)` has an exact source order:

1. If `last_valid_q` is no more than 0.10 seconds old, extract its named upper-body values and clamp each value to the attested effective lower/upper limit. Record the source as `measured_clamped`, including every clamped joint and delta.
2. Otherwise, if a previously successful bounded publication exists, copy its 31-D upper-body target and bounded base height, set navigation to zero, and record the source as `last_safe_published`.
3. Otherwise return no hold. Publish nothing and deny activation until a fresh valid state permits step 1.

The base height is independently clamped to `[0.20, 0.74]`; navigation is always exactly zero. Once `PAUSED` or `FAULT` captures a hold, that target is fixed and republished at 50 Hz rather than following later measurements. Every successful hold or policy publication updates the bounded, final-post-slew `last_safe_published` fallback. Named inverse mapping converts the bounded 31-D WBC hold into the 36-D PSI0 action order used by R0, with the same height and four zero navigation values.

Consequently, a measurement up to 0.05 radians outside an effective joint envelope may remain valid for monitoring, but the corresponding hold target is the nearest effective endpoint and is never published outside the envelope. R0's committed hold prefix is built from this already bounded, successfully published hold.

When Ctrl-C is received in `sim-control`, the bridge first enters paused hold and publishes 25 final hold ticks (0.5 seconds), then transitions to `STOPPED` and closes its publisher. This leaves the WBC's last received goal as a zero-navigation hold, but does not guarantee balance after the bridge exits. The WBC and its lower-body balance/walk policy are separate processes whose current activation state is unchanged by `p` or Ctrl-C. Operators must not terminate the WBC or remove support based on the bridge's last command.

## Concurrency and bounded shutdown

The main thread owns the 50 Hz state machine, keyboard events, deadlines, action-buffer swaps, and publication. One camera reader owns its ZMQ socket. One inference worker owns its HTTP session. Shared inputs/results use immutable copies behind short-held locks.

Shutdown is bounded as follows:

1. latch stop, invalidate the generation, clear current, staged, and committed-prefix buffers, and publish the final 0.5-second hold in `sim-control`;
2. signal the camera reader, whose 100 ms poll exits and whose own thread closes its socket;
3. prevent new HTTP work and wait at most 5.5 seconds for the existing five-second request timeout;
4. close messaging resources and restore the terminal in `finally` blocks;
5. report any daemon worker that did not exit, while allowing the process to terminate; no non-daemon worker may remain.

The explicit overall deadline from Ctrl-C receipt to bridge-process exit is 6.5 seconds: at most 0.5 seconds of final hold, 5.5 seconds of inference-worker join budget, and 0.5 seconds for camera/messaging/terminal cleanup. The long bound is tested separately and is not assumed to fit inside the smoke test's two-second cleanup phase.

The bridge never kills the PSI0 server, WBC, simulator, or any robot process.

## Verification

### Deterministic tests

Tests must cover:

1. Converter sentinels prove chronological row order, yaw/pitch/roll-to-roll/pitch/yaw column order, `0.74` height, and agreement between both converter paths.
2. A 43-value named sentinel maps every hand and arm joint to the correct 32-D index; the final four values are executed-command history.
3. A 36-value named sentinel maps every waist, arm, hand, height, and navigation value to the correct 31-D WBC goal position.
4. Corrected right-hand effective limits equal the URDF within `1e-7`, both shoulder-roll effective limits equal their documented conservative subsets, and no other commanded-joint discrepancy exists. Each effective midpoint and endpoint is accepted; values `1e-4` outside are rejected.
5. Invalid measured-state shape, non-finite values, out-of-tolerance values, stale age, and excessive camera/state skew cannot replace `last_valid_q` or activate control. A measurement 0.04 radians outside an effective limit is accepted but its hold is clamped to the endpoint; 0.051 radians outside is rejected. Stale state selects `last_safe_published`, and no fallback publishes nothing.
6. RGB and BGR options pass red/blue sentinels through the actual composed-camera JPEG schema with tolerance.
7. The bridge starts paused; local `p` activates only after both preflights, synchronized inputs, and one successful bounded hold publication. R0 contains `observation_tick`, `d` identical committed hold actions, and an echoed `first_action_tick=r0+d`.
8. The `P=8, s=5, d=3` time-indexed sentinel proves R0 at tick 100, hold-prefix ticks 100-102, first suffix tick 103, R1 at tick 105, observation history, post-slew committed prefixes, response indices, and subsequent handoff at tick 108 exactly.
9. RTC constraints reject `d<2`, `d>s`, `d+s>P`, `d>=rtc_training_max_delay`, legacy `/act` semantics, short/long suffixes, wrong metadata, and late responses before any partial suffix executes.
10. A blocked fake request never blocks 50 Hz publication, cannot be bypassed by a second worker, and prevents rearm until physically idle.
11. Pause, fault, and stop atomically clear current, staged, and committed actions; all late generations are discarded; the first fault goal has zero navigation.
12. The general HTTP client remains unbounded by default while the bridge passes five seconds explicitly.
13. The real `ControlLoopConfig(interface="sim", domain_id=42).to_dict()` payload contains `env_type="sim"`, Linux `interface="lo"`, and `domain_id=42`. An event-order test proves robot-model and WBC-policy construction both precede service publication. The real model-contract builder emits 43/31 ordered names, effective limits, nested SHA, clean-tree flag, URDF/ONNX hashes, and a reproducible digest; mutating any hand limit, joint order, hash, SHA, or cleanliness flag changes/rejects the contract before publisher creation.
14. Camera polling and terminal cleanup finish within their stated shutdown bounds.
15. A clean `git clone --no-local` plus recursive submodule initialization resolves the recorded decoupled-WBC SHA and passes its targeted tests; root tests run against that clean submodule, not the developer's dirty checkout.
16. A deterministic subprocess shutdown test starts a local HTTP handler that accepts `/act-rtc-v1` and withholds its response beyond the client's five-second read timeout. After the handler signals that the request is in flight, the test sends Ctrl-C immediately and requires bridge exit within 6.5 seconds, final zero-navigation hold publication, camera exit, terminal restoration, no live non-daemon bridge thread, and reusable ports.

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
- a fake v2 `/contract` plus `/act-rtc-v1` server with `P=30, s=24, d=6`, returning exact `(24, 36)` executable suffixes at 50 ms latency;
- the bridge in a pseudo-terminal so the test sends the same local `p` byte as an operator.

The phases are:

- seconds 0-3: paused hold;
- seconds 3-11: active fake inference;
- at second 11: the fake policy makes its next request take exactly 0.30 seconds, exceeding the six-tick/0.12-second handoff deadline but returning well before smoke cleanup;
- seconds 11-13: observe latched fault and hold;
- at second 13: require the invalidated worker to be physically idle, then send Ctrl-C;
- seconds 13-15: final hold and short-path cleanup.

Pass criteria are all of:

- before bridge start, goal-topic publisher/subscription counts are `0/1`; while running they are `1/1`; after exit publishers return to zero;
- during each steady publication phase, mean rate is 49-51 Hz and maximum inter-message gap is at most 60 ms;
- the main loop continues publishing while HTTP is blocked;
- successor requests start with exactly six committed post-slew actions remaining, and accepted handoffs introduce no duplicate or skipped tick;
- fault is recorded no later than request start plus 0.14 seconds (six-tick deadline plus one observation tick);
- the late 0.30-second response is discarded, and the worker reports idle before second 13;
- the first goal at or after the fault transition has navigation exactly `[0, 0, 0, 0]` and no queued policy action executes afterward;
- all bridge threads exit, child PIDs are reaped, terminal settings are restored, and all test ports can be rebound;
- the generated metrics file records zero real-interface connections and zero extra goal publishers.

This smoke test does initialize Unitree SDK2 DDS inside the simulator. Safety comes from loopback interface plus dedicated Unitree domain 42, together with ROS domain 42; it does not claim to avoid DDS.

The smoke test exercises the short shutdown path only. The deterministic test in item 16 owns the five-second in-flight-request case and its 6.5-second overall assertion.

### Live policy integration gate

The fake server is replaced only after a certified corrected checkpoint and `/act-rtc-v1` server exist. At least 100 representative warmed requests must show p99 latency at or below `(d-1)/50` seconds with no timeout, metadata, or shape failure; every runtime request still faults if unavailable at tick `r+d`. For the current candidate `d=6`, certification is at most 0.10 seconds and handoff is at 0.12 seconds. The latency report, policy contract, checkpoint hash, and PSI0 server commit are saved together. The currently measured RTX 3060 server fails this gate, and the current `P=30, s=24` checkpoint cannot instead certify a 20-tick delay.

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
