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

## v3 — obey the speed command, hold the heading (2026-11-13)

Changes from v2:
1. Progress term now peaks at the commanded speed and falls off on both sides
   (0 when standing still, 1 on target, 0 at double speed, -1 at triple speed).
2. Heading hold on stairs: every step the yaw-rate command is set from the IMU
   heading (`-1.0 x yaw`, limited to +/-0.4 rad/s). This is part of the task for
   all three controllers, so the kinematic baseline was measured again with it.
3. Yaw tracking weight 0.3 -> 0.5, and its width narrowed (0.1 -> 0.05) so slow turning costs reward.

Test protocol changed (heading hold), so v3 numbers go in a new `results.csv`.
The v2 numbers are kept in `results_v2.csv`.

Baseline under the v3 protocol: 0 cm 100%, 4 cm 90%, 6 cm 50%, 8 cm 0%, 10 cm 0%.

Runs: `runs/v3_residual_stairs_s0`, `runs/v3_direct_stairs_s0`. Code: git tag `exp-v3`. 3M steps, seed 0.

| Riser | Kinematic climbed / distance / speed | Residual climbed / distance / speed | Direct climbed / distance |
|---|---|---|---|
| 0 cm | 100% / 2.86 m / 0.136 | 100% / 3.47 m / 0.169 | 0% / 1.20 m |
| 4 cm | 90% / 2.58 m / 0.124 | 100% / 3.14 m / 0.153 | 0% / 0.91 m |
| 6 cm | 50% / 2.44 m / 0.118 | 100% / 2.91 m / 0.143 | 0% / 0.80 m |
| 8 cm | 0% / 0.99 m / 0.031 | 100% / 2.66 m / 0.132 | 0% / 0.83 m |
| 10 cm | 0% / 0.44 m / 0.010 | 0% / 2.14 m / 0.109 | 0% / 0.58 m |

Speed in m/s, commanded 0.15. Nobody tipped over in any trial (0 falls in all 300 trials).

Finding (residual): climbs every stair up to 8 cm in 20 of 20 trials, with no
falls, and now walks close to the commanded speed. Both v2 problems are fixed:
on flat ground the final heading error is about 4 degrees (v2: 110 degrees).
Velocity tracking error is lower than the kinematic gait at every height, and
so is cost of transport (for example 13.4 against 42.8 at 8 cm).

10 cm with more time (extra check, 10 trials, 40 s instead of 20 s): residual
reaches the top in 10 of 10 trials; it is slow, not stuck. The kinematic gait
with 40 s reaches the top in 0 of 10 at 8 cm and 0 of 10 at 10 cm: it does not
get over the first riser and slides sideways along it.

Finding (direct): still fails. It moves, but turns away from the stairs
(final heading error about 110 degrees on flat ground). Its training reward was
still rising when the run ended (-230 at the start, -7 at 3M steps), so it has
not converged. Learning to walk, turn and climb from nothing needs more than 3M steps.

Open points:
1. One seed. Seeds 1 and 2 of the residual run are needed before any claim.
2. Direct needs a longer run (`v3long`, 12M steps) before it can be judged.
3. Simulation only, with estimated masses and servo gains.

### v3, residual seeds 1 and 2, and a 40 s time limit (2026-11-13)

Runs: `runs/v3_residual_stairs_s1`, `runs/v3_residual_stairs_s2` (same code and settings as seed 0).

With the 20 s limit (`results.csv`) the seeds looked very different at 8 cm:
seed 0 climbed 100%, seeds 1 and 2 climbed 0%. But seed 1 had travelled 2.40 m
of the 2.5 m needed: it was slow, not stuck. The 20 s limit mixes up "slow" and
"cannot climb", so the comparison was repeated with 40 s per trial
(`python evaluate.py <controller> --seconds 40 --out results_40s.csv`).

Climbed within 40 s, 20 trials each. No trial ended with the robot tipped over.

| Riser | Kinematic | Residual seed 0 | Residual seed 1 | Residual seed 2 (first try, INVALID) |
|---|---|---|---|---|
| 0 cm | 100% | 100% | 100% | 100% |
| 4 cm | 100% | 100% | 100% | 100% |
| 6 cm | 100% | 100% | 100% | 100% |
| 8 cm | 0% | 100% | 100% | 0% |
| 10 cm | 0% | 100% | 100% | 0% |

**Correction (same day): the seed 2 column above is not a result.** That run
crashed with an out-of-memory error after its first update (8,192 of 3,000,000
steps; three trainings were running at once). The saved policy was untrained,
so it behaved like the kinematic gait. The earlier reading here ("seed 2
learned nothing, the method is not reliable across seeds") was wrong. The
user spotted it from the learning curve that stopped. The crashed run is kept
as `runs/v3_residual_stairs_s2_crashed`, its rows were removed from the CSV
files, and seed 2 was trained again (see below). `evaluate.py` now prints how
many steps a policy was trained for and warns when the run did not finish.

Seed 2 trained again, full 3M steps (`runs/v3_residual_stairs_s2`), 40 s, 20 trials:
climbed 100% at 0, 4, 6, 8 and 10 cm, no falls. It is slower and tilts more on
10 cm stairs than on the lower ones (0.112 m/s, tilt 0.27 rad RMS).

Finding: all three seeds climb 8 cm and 10 cm stairs in every trial, where the
kinematic gait never does. Three seeds is still a small number; a paper needs
at least 5.

Lesson: check that a run reached its planned step count before testing it.

Note: the first 40 s run showed every controller "falling" on flat ground at
exactly 5 m. The robot had walked off the edge of the simulated floor. The
floor is now 20 m wide and the table above is from the corrected run.

### v3long — direct RL, 12M steps (2026-11-13)

Run: `runs/v3long_direct_stairs_s0`. Same code as v3, 12M steps instead of 3M.

Interim check at 9.4M steps (10 trials, 40 s): climbed 0% at every height,
net distance 1.22 m on flat ground and 0.31-0.65 m on stairs.

What it does: it walks forward at the right speed in its own frame (0.12-0.14 m/s,
commanded 0.15), so the speed part is learned. But it cannot hold its heading.
Traced on flat ground: it turned 89 degrees in the first 2 s, slowly came back
while the compass loop commanded a turn, then swung away again as soon as it
faced forward. It follows large turn commands at about half the commanded rate
and drifts when the command is near zero. Net result: it walks in loops near the start.

Reading: with 4x the training, direct RL learned to walk but not to steer.
The kinematic gait gives straight walking and turning for free, which is the
practical advantage of the residual design in this comparison.

Final, after all 12M steps (20 trials, 40 s, `results_40s.csv`): climbed 0% at
every height, no falls. Body-frame speed 0.135-0.153 m/s (commanded 0.15), but
net distance only 1.6-1.8 m in 40 s (a straight walk would cover 6 m). Same
picture as the interim check: it walks, it does not steer.

Idea for v4 (not done): reward progress toward the stairs in the world frame,
instead of body-frame speed plus a yaw-rate command. Walking in loops would then
earn nothing, which should help the direct policy most.

### Second terrain: rubble the policies never trained on (2026-11-13)

Question: does a policy trained only on stairs help on a different obstacle?
Test terrain: `--terrain bumps`, a field of random bumps (rubble), heights
2 to 10 cm. No new training. Same v3 policies as above.

    python evaluate.py <controller> --terrain bumps --height 0.02 0.04 0.06 0.08 0.10 --out results_bumps.csv

Net distance in 20 s, command 0.15 m/s (a perfect walk is 3.0 m), 20 trials. No falls anywhere.

| Bump height | Kinematic | Residual s0 | Residual s1 | Residual s2 | Direct 12M |
|---|---|---|---|---|---|
| 2 cm | 2.85 m | 3.27 m | 2.93 m | 3.21 m | 0.44 m |
| 4 cm | 2.82 m | 3.18 m | 2.89 m | 3.15 m | 0.78 m |
| 6 cm | 2.74 m | 3.14 m | 2.85 m | 3.08 m | 0.78 m |
| 8 cm | 2.42 m | 3.10 m | 2.77 m | 2.88 m | 0.50 m |
| 10 cm | 1.59 m | 2.70 m | 2.65 m | 2.54 m | 0.35 m |

Finding: on low rubble everything with a gait is fine, as expected. On 10 cm
rubble the kinematic gait slows to half speed (0.075 m/s) and the three residual
policies keep 0.12-0.13 m/s, covering 60-70% more ground at about half the cost
of transport (12.7-15.7 against 25.1). They were never trained on this terrain.

Weak point found on the way: the first version of this test had no heading
hold on rubble (it was only switched on for stairs). Kinematic and residual s0
were fine, but residual s1 and s2 walked in curves: right speed in their own
frame, only 0.2-0.6 m net distance. So the learned corrections add a turning
bias that the compass loop hides. `HexapodEnv` now has `heading_hold`, and
`evaluate.py` switches it on for every controller on every terrain (the robot
has the IMU for it). The kinematic numbers did not change with it. The policies
need the compass loop; that must be said in the paper, and the loop must run on
the real robot.

### Third terrain: a smooth ramp, also never trained on (2026-11-13)

New terrain `--terrain slope`: a smooth ramp starting 0.5 m ahead, with the same
average incline as stairs of that riser (`--height 0.10` = 10 cm rise per 25 cm
= 21.8 degrees). Built from one tilted slab. No new training, same v3 policies.

    python evaluate.py <controller> --terrain slope --height 0.04 0.06 0.08 0.10 0.125 0.15 --out results_slope.csv

Net distance in 20 s, command 0.15 m/s, 20 trials. No falls anywhere.

| Incline | Kinematic | Residual s0 | Residual s1 | Residual s2 | Direct 12M |
|---|---|---|---|---|---|
| 9 deg (0.04) | 2.53 m | 2.75 m | 2.60 m | 2.65 m | 0.39 m |
| 13 deg (0.06) | 2.29 m | 2.34 m | 2.46 m | 2.43 m | 0.41 m |
| 18 deg (0.08) | 1.99 m | 1.89 m | 2.27 m | 2.23 m | 0.42 m |
| 22 deg (0.10) | 1.68 m | 1.48 m | 1.97 m | 2.00 m | 0.43 m |
| 27 deg (0.125) | 1.19 m | 1.14 m | 1.65 m | 1.44 m | 0.45 m |
| 31 deg (0.15) | 0.45 m | 0.47 m | 1.04 m | 1.21 m | 0.45 m |

Finding: on a smooth ramp the stairs-trained policies give little. Seeds 1 and 2
are 10-40% ahead of the gait on steep ramps, seed 0 is level with it or slightly
behind. Everything slows down with the incline, and nothing walks well at 31
degrees. This is the honest "RL does not add much here" case: a ramp has no
edges to catch a foot on, so there is nothing for the blocked-leg signal to
react to; the limit is servo torque and grip. The direct policy never reaches
the ramp (it loops on the flat start).

Summary over the three terrains: big gain on stairs (0% -> 100%), clear gain on
high rubble (unseen), small or no gain on ramps (unseen).
