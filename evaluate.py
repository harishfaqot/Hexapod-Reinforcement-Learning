"""Measure a controller on fixed test terrains and append the numbers to results.csv.

    python evaluate.py kinematic                       # open-loop tripod gait, the baseline
    python evaluate.py runs/residual_stairs_s0
    python evaluate.py runs/direct_stairs_s0 --render --episodes 1 --height 0.04

Every controller sees the same terrains and commands (same seeds), without
training-time randomization, so rows are directly comparable.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hexapod import HexapodEnv, robot
from hexapod.env import CONTROL_DT

FIELDS = ["controller", "terrain", "height_m", "episodes", "fall_rate", "climbed_rate", "distance_m", "speed_mps",
          "tracking_rmse_mps", "tilt_rms_rad", "cost_of_transport"]


def load_policy(run_dir: Path, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model = PPO.load(run_dir / "model", device="cpu")
    norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training = False
    return lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]


def evaluate(controller: str, terrain: str, height: float, episodes: int, command, render: bool):
    kinematic = controller == "kinematic"
    mode = "residual" if kinematic else json.loads((Path(controller) / "config.json").read_text())["mode"]
    env = HexapodEnv(mode=mode, command=command, terrain=terrain, terrain_height=height,
                     render_mode="human" if render else None)
    policy = (lambda obs: np.zeros(18)) if kinematic else load_policy(Path(controller), env)

    falls, climbed, speeds, sq_errors, tilts, energy, distance = 0, 0, [], [], [], 0.0, 0.0
    stairs_top = robot.STAIR_START + robot.STAIR_STEPS * 0.25      # x where the flat top begins
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
            done = terminated or truncated
            if render:
                import time
                time.sleep(CONTROL_DT)
        falls += terminated
        on_top = env.data.qpos[2] > robot.STAIR_STEPS * height + 0.5 * robot.HOME_HEIGHT    # really up there, not inside
        climbed += bool(env.data.qpos[0] > stairs_top and on_top and not terminated)
        distance += float(np.linalg.norm(env.data.qpos[:2] - start))
    env.close()
    return {
        "controller": controller, "terrain": terrain, "height_m": height, "episodes": episodes,
        "fall_rate": falls / episodes,
        # stairs only: share of episodes that reached the top without falling
        "climbed_rate": climbed / episodes if terrain == "stairs" else float("nan"),
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
    p.add_argument("--terrain", choices=["bumps", "stairs"], default="stairs")
    p.add_argument("--height", type=float, nargs="+", default=[0.0, 0.04, 0.06, 0.08, 0.10],
                   help="bump heights or stair riser heights to test [m]")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--command", type=float, nargs=3, default=[0.15, 0.0, 0.0], metavar=("VX", "VY", "YAW_RATE"))
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
            row = evaluate(args.controller, args.terrain, height, args.episodes, args.command, args.render)
            if not args.render:
                writer.writerow(row)
            print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
