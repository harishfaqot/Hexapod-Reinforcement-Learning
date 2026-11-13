"""Measure a controller on fixed test terrains and append the numbers to results.csv.

    python evaluate.py kinematic                       # open-loop tripod gait, the baseline
    python evaluate.py runs/v2_residual_stairs_s0
    python evaluate.py runs/v2_direct_stairs_s0 --render --episodes 1 --height 0.04

Every controller sees the same terrains and commands (same seeds), without
training-time randomization, so rows are directly comparable.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from hexapod import HexapodEnv, robot
from hexapod.env import CONTROL_DT, TERRAINS

FIELDS = ["controller", "terrain", "height_m", "episodes", "fall_rate", "climbed_rate", "distance_m", "speed_mps",
          "tracking_rmse_mps", "tilt_rms_rad", "cost_of_transport"]


def load_policy(run_dir: Path, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    for attempt in range(5):        # a running training rewrites these files every few seconds
        try:
            model = PPO.load(run_dir / "model", device="cpu")
            norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.5)
    planned = json.loads((run_dir / "config.json").read_text()).get("timesteps", 0)
    print(f"policy trained for {model.num_timesteps:,} steps" +
          (f"  WARNING: run planned {planned:,}, it is unfinished or crashed" if model.num_timesteps < planned else ""))
    norm.training = False
    return lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]


def evaluate(controller: str, terrain: str, height: float, episodes: int, command, render: bool, seconds: float = 20.0):
    kinematic = controller == "kinematic"
    mode = "residual" if kinematic else json.loads((Path(controller) / "config.json").read_text())["mode"]
    env = HexapodEnv(mode=mode, command=command, terrain=terrain, terrain_height=height, episode_seconds=seconds,
                     heading_hold=True,          # every controller gets the IMU compass loop, on every terrain
                     render_mode="human" if render else None)
    policy = (lambda obs: np.zeros(18)) if kinematic else load_policy(Path(controller), env)

    falls, climbed, speeds, sq_errors, tilts, energy, distance = 0, 0, [], [], [], 0.0, 0.0
    # x the robot must pass: the flat top (or bottom) after the stairs, or the end of the course
    stairs_top = robot.COURSE_FINISH if terrain == "course" else robot.STAIR_START + robot.STAIR_STEPS * 0.25
    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        start = env.data.qpos[:2].copy()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            v = info["body_velocity"]
            speeds.append(v[0])
            sq_errors.append(np.sum((np.asarray(command[:2]) - v[:2]) ** 2))
            tilts.append(np.arccos(np.clip(-info["gravity"][2], -1, 1)))
            energy += info["power"] * CONTROL_DT
            # the course ends at its finish line (the ramp behind it goes on for metres)
            done = terminated or truncated or (terrain == "course" and env.data.qpos[0] > stairs_top)
            if render:
                time.sleep(CONTROL_DT)
        falls += terminated
        # going up: really on the top, not inside the stairs. Going down: past the last step is enough.
        on_top = terrain != "stairs" or env.data.qpos[2] > robot.STAIR_STEPS * height + 0.5 * robot.HOME_HEIGHT
        climbed += bool(env.data.qpos[0] > stairs_top and on_top and not terminated)
        distance += float(np.linalg.norm(env.data.qpos[:2] - start))
    env.close()
    return {
        "controller": controller, "terrain": terrain, "height_m": height, "episodes": episodes,
        "fall_rate": falls / episodes,
        # stairs up or down, course: share of episodes that got past the whole thing without falling
        "climbed_rate": climbed / episodes if terrain in ("stairs", "descend", "course") else float("nan"),
        "distance_m": distance / episodes,
        "speed_mps": float(np.mean(speeds)),
        "tracking_rmse_mps": float(np.sqrt(np.mean(sq_errors))),
        "tilt_rms_rad": float(np.sqrt(np.mean(np.square(tilts)))),
        # mechanical energy per unit weight per metre; lower is more efficient
        "cost_of_transport": energy / (env.total_mass * 9.81 * max(distance, 1e-6)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("controller", help='"kinematic" or a run directory such as runs/residual_s0')
    p.add_argument("--terrain", choices=TERRAINS, default="stairs")
    p.add_argument("--height", type=float, nargs="+", default=[0.0, 0.04, 0.06, 0.08, 0.10],
                   help="bump heights or stair riser heights to test [m]")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--command", type=float, nargs=3, default=[0.15, 0.0, 0.0], metavar=("VX", "VY", "YAW_RATE"))
    p.add_argument("--seconds", type=float, default=20.0, help="time limit per episode")
    p.add_argument("--render", action="store_true")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    out = Path(args.out)
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for height in args.height:
            row = evaluate(args.controller, args.terrain, height, args.episodes, args.command, args.render, args.seconds)
            if not args.render:
                writer.writerow(row)
            print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
