# Experiment log

One entry per training attempt: what was changed, why, and what came out.
Failed attempts stay in this file; they are part of the research record.

Rules:
- Every attempt gets a new `--tag`. `train.py` refuses to overwrite a run.
- The code of each attempt is marked with a git tag `exp-<tag>` (for example
  `git checkout exp-v2`), so its exact reward and observation can be recovered.
- Numbers come from `python evaluate.py <controller>` (20 trials per stair
  height, 20 s each, command 0.15 m/s) unless stated otherwise.

"Climbed" = share of trials that reached the top of 8 steps without tipping over.
"Distance" = mean distance travelled in 20 s (the top starts at 2.5 m).

## Baseline: tuned kinematic tripod gait

Foot lift 6 cm, the best overall of 4, 6, 8 and 10 cm on this task. Full rows in `results.csv`.

| Riser | Climbed | Distance |
|---|---|---|
| 0 cm | 100% | 2.86 m |
| 4 cm | 100% | 2.63 m |
| 6 cm | 60% | 2.45 m |
| 8 cm | 0% | 0.98 m |
| 10 cm | 0% | 0.38 m |

## v0 — random bumps, first working stack (2026-11-13)

Purpose: check that the new environment learns at all.
Setup: old Trossen-size model (tibia 95 mm), bumps up to 3 cm, PPO, seed 0.
Stopped by hand at about 1.0M of 3M steps. 10 trials per row.

| Controller | Bumps | Falls | Distance | Speed (commanded 0.15) |
|---|---|---|---|---|
| Residual | 0 cm | 0% | 3.15 m | 0.150 m/s |
| Residual | 3 cm | 0% | 3.04 m | 0.146 m/s |
| Direct | 0 cm | 0% | 4.44 m | 0.263 m/s |
| Direct | 3 cm | 0% | 4.15 m | 0.215 m/s |

Finding: direct joint-space RL learns to walk within 1M steps (it still overshot
the commanded speed when stopped). The kinematic gait also never falls on 3 cm
bumps, so bumps cannot show a benefit of RL. Task changed to stairs.
Run files: not kept.

## (discarded) — stairs with a collision bug (2026-11-13)

Only the first stair step was solid; the robot walked through the others.
Cause: stair blocks were moved by editing `geom_pos` after the model was built,
which leaves MuJoCo's collision lookup stale. Fixed by using mocap bodies, plus
a test that drops the robot on every step. All numbers from this attempt are invalid.

## v1 — stairs, first valid attempt (2026-11-13)

Setup: PhantomX-size model with AX-18A values, stairs with risers 0-10 cm in
training, 3M steps, seed 0. Observation 47 numbers (no blocked-leg signal).
Reward: `exp(-error^2 / 0.1^2)` on velocity, 0.5 x yaw tracking, belly contact -1.0 per step.

| Riser | Residual climbed / distance | Direct climbed / distance |
|---|---|---|
| 0 cm | 100% / 2.78 m | 0% / 0.03 m |
| 4 cm | 65% / 2.57 m | 0% / 0.03 m |
| 6 cm | 0% / 2.25 m | 0% / 0.03 m |
| 8 cm | 0% / 0.59 m | 0% / 0.03 m |
| 10 cm | 0% / 0.42 m | 0% / 0.03 m |

Finding: failed. Residual was worse than the baseline at every height. Direct
learned to stand still.
Diagnosis: the velocity reward used a fixed width of 0.1 m/s, so with slow
commands (0.05-0.2 m/s) standing still already earned up to 78% of it, plus the
full yaw bonus. Moving toward the stairs risked the -1.0 belly penalty.
Run files: deleted by mistake before v2 was started. Only this table remains.

## v2 — reward fix and blocked-leg signal (2026-11-13)

Changes from v1:
1. Velocity error measured relative to the commanded speed; added a linear
   progress term. Standing still now earns about 0.02 of the tracking reward.
2. Observation grows to 65: added (joint target - joint angle) for each joint,
   the robot's only sense of touch.
3. Yaw tracking weight 0.5 -> 0.3, level penalty 1.0 -> 0.5, belly penalty 1.0 -> 0.2.

Runs: `runs/v2_residual_stairs_s0`, `runs/v2_direct_stairs_s0`. Code: git tag `exp-v2`. 3M steps, seed 0.

| Riser | Kinematic climbed | Residual climbed / falls / distance | Direct climbed / falls / distance |
|---|---|---|---|
| 0 cm | 100% | 90% / 0% / 4.56 m | 0% / 0% / 1.05 m |
| 4 cm | 100% | 80% / 10% / 4.59 m | 0% / 0% / 1.43 m |
| 6 cm | 60% | 100% / 0% / 4.54 m | 0% / 0% / 1.62 m |
| 8 cm | 0% | 100% / 0% / 4.37 m | 0% / 0% / 1.43 m |
| 10 cm | 0% | 80% / 20% / 3.59 m | 0% / 0% / 1.24 m |

Finding: first positive result. Residual RL climbs 8 cm stairs in 20 of 20 trials
and 10 cm stairs in 16 of 20, where the tuned kinematic gait climbs 0 of 20.

Fairness check: the residual policy walks faster than commanded, and "climbed"
has a 20 s limit. So the kinematic gait was also tested with faster commands
(0.20 and 0.25 m/s, 20 trials each). It still climbed 0% at 8 cm and 0% at 10 cm
(6 cm: 70% and 85%). The gain at 8-10 cm is not a speed effect.

Problems found (reasons this is not a final result):
1. Neither policy obeys the speed command. Commanded 0.15 m/s; residual walks at
   about 0.23-0.27 m/s, direct at about 0.45 m/s. Cause: the progress term is
   capped at the target speed but does not punish going faster, and the tracking
   term is flat once the policy is far above the target.
2. Neither policy holds its heading. On flat ground the residual policy turned
   110 degrees in 20 s; the direct policy walked in circles (9.0 m of path, about
   1 m of net distance). That is why direct "climbed" 0%, and probably why residual lost
   some trials at 0 and 4 cm. Cause: the yaw reward barely changes for slow turning.
3. Residual tipped over in 2 of 20 trials at 4 cm and 4 of 20 at 10 cm. The
   kinematic gait never tipped over.
4. One seed only.

Next (v3): punish overshooting the commanded speed; hold the heading with the IMU.
