# Hexapod locomotion: what does reinforcement learning add to a kinematic gait?

Research code for the BumbleBee hexapod (18 Dynamixel AX servos, BNO055 IMU, Jetson Nano).

## Research question

A hexapod can already walk with inverse kinematics and a fixed tripod gait.
So when is learning worth it? This repo compares three controllers on the
same robot model, the same reward, the same terrains and the same metrics:

| Controller | What sets the 18 joint targets |
|---|---|
| **Kinematic** (baseline) | Open-loop tripod gait + inverse kinematics. No learning, no feedback. |
| **Residual RL** | The same gait, plus a learned correction of up to 0.3 rad per joint. |
| **Direct RL** | A learned policy only. No gait, no inverse kinematics. |

All three follow a velocity command (forward, sideways, turn rate). The
policies only see what the real robot can measure: IMU tilt, gyro and joint
angles. No simulator-only signals, so a trained policy can be moved to hardware.

## Layout

```
hexapod/robot.py    robot dimensions, MuJoCo model, inverse kinematics, tripod gait
hexapod/env.py      the environment: observation, action, reward, terrain, randomization
train.py            PPO training
evaluate.py         fixed test protocol -> results.csv
EXPERIMENTS.md      log of every training attempt: what changed, what came out
tests/              self-checks (IK against the simulator, baseline walks, Gym API)
papers/             reference papers
Report/             bachelor thesis and first article
mujoco-hexapod/     previous experiments; deploy/ holds the servo driver and calibration
Bumblebee/          original thesis code (PyBullet) and real-robot scripts
STL/                3D-print files
```

## Run

```
pip install -r requirements.txt
python -m tests.test_hexapod

python evaluate.py kinematic                    # baseline numbers
python train.py --tag v2 --mode residual --seed 0     # about 20-40 min on a 16-core CPU
python train.py --tag v2 --mode direct --seed 0
python evaluate.py runs/v2_residual_stairs_s0
python evaluate.py runs/v2_direct_stairs_s0
```

### Watching a run while it trains

Training saves the current policy after every update, so in a second terminal:

```
tensorboard --logdir runs                                                  # learning curves, open http://localhost:6006
python evaluate.py runs/v2_residual_stairs_s0 --render --episodes 1 --height 0.08  # the robot with the current policy
```

Run the second command again whenever you want to see the newest policy.
In TensorBoard, `rollout/ep_rew_mean` should rise and `rollout/ep_len_mean`
should stay near 500 (500 = the robot did not tip over for the whole 20 s).

## Test protocol

The main task is stair climbing, because that is where a fixed gait breaks.
`evaluate.py` runs 20 episodes of 20 s per stair height: 8 steps, 25 cm deep,
risers of 0, 4, 6, 8 and 10 cm, command 0.15 m/s forward. Start position and
heading vary a little per episode from fixed seeds, so every controller gets
exactly the same 20 trials. No training-time randomization during the test.

Metrics: climbed rate (reached the top without tipping over), fall rate,
distance, forward speed, velocity tracking error (RMSE), body tilt (RMS) and
mechanical cost of transport (energy / weight / distance).

The kinematic baseline is tuned: its foot lift (6 cm) is the best of 4, 6, 8 and
10 cm on this task. `--terrain bumps` tests random bumps instead of stairs.

## Results so far

Share of 20 trials that reached the top of the stairs (one training seed, simulation only):

| Riser | Kinematic gait | Residual RL (v2) | Direct RL (v2) |
|---|---|---|---|
| 0 cm | 100% | 90% | 0% |
| 4 cm | 100% | 80% | 0% |
| 6 cm | 60% | 100% | 0% |
| 8 cm | 0% | 100% | 0% |
| 10 cm | 0% | 80% | 0% |

Residual RL climbs stairs the tuned kinematic gait cannot. This is a first
result, not a final one: the v2 policies ignore the commanded speed and do not
hold their heading, and the direct policy walks in circles. `EXPERIMENTS.md`
has every attempt, including the failed ones, with full tables and diagnoses.
Numbers for this table are in `results_v2.csv`.

## Before real-robot experiments

The model in `hexapod/robot.py` uses published PhantomX dimensions and AX-18A
datasheet values. Lines marked `MEASURE-ME` are estimates. Check on BumbleBee:

1. Link lengths (joint axis to joint axis), hip positions and joint limits.
2. Masses: weigh one leg and the body.
3. Servo response: command a step on one joint, log the position, and tune
   `SERVO_KP`, `SERVO_DAMPING`, `SERVO_MAX_TORQUE` until simulation matches.
4. How fast the servo bus can read 18 positions and write 18 targets. The
   policy assumes 25 Hz (`CONTROL_DT`).
5. The offset and direction between each joint angle here and its servo tick
   (start from the table in `mujoco-hexapod/deploy/hexapod_real.py`).

Then retrain. A policy trained on the wrong geometry will not transfer.
