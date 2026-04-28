"""
eval_ant.py — Evaluate a trained PPO model on Ant-v5.

Usage:
    python eval_ant.py --model-path PPO_ant/logs/ant_ppo_final.zip
    python eval_ant.py --model-path PPO_ant/logs/best/best_model.zip --episodes 10
    python eval_ant.py --model-path PPO_ant/logs/ant_ppo_final.zip --no-render
"""

import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",    type=str, required=True)
    p.add_argument("--stats-path",    type=str, default=None,
                   help="VecNormalize .pkl (default: auto-detect next to model)")
    p.add_argument("--episodes",      type=int, default=5)
    p.add_argument("--no-render",     action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--seed",          type=int, default=0)
    return p.parse_args()


def make_eval_env(stats_path: str | None, seed: int):
    env = DummyVecEnv([lambda: gym.make("Ant-v5", render_mode="human")])
    if stats_path:
        print(f"Loading VecNormalize stats: {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    return env


def run_episode(model, env, render: bool, deterministic: bool):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    rewards = []

    t0 = time.perf_counter()
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, info = env.step(action)
        total_reward += float(reward[0])
        rewards.append(float(reward[0]))
        steps += 1
        done = bool(terminated[0])

    elapsed = time.perf_counter() - t0
    return {
        "total_reward": total_reward,
        "mean_reward":  float(np.mean(rewards)),
        "steps":        steps,
        "fps":          steps / elapsed,
        "terminated":   done,
    }


def main():
    args = parse_args()

    # Auto-detect stats file next to model
    stats_path = args.stats_path
    if stats_path is None:
        import os
        candidate = os.path.join(os.path.dirname(args.model_path), "..", "vec_normalize.pkl")
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            stats_path = candidate
            print(f"Auto-detected VecNormalize stats: {stats_path}")
        else:
            print("No VecNormalize stats found — running without normalization")

    render = not args.no_render
    env = make_eval_env(stats_path, args.seed)

    print(f"\nLoading: {args.model_path}")
    model = PPO.load(args.model_path, env=env, device=args.device)
    print(f"Obs : {env.observation_space.shape}  |  Act : {env.action_space.shape}\n")

    all_stats = []
    for ep in range(1, args.episodes + 1):
        stats = run_episode(model, env, render, args.deterministic)
        all_stats.append(stats)
        print(
            f"Ep {ep:>3} | steps={stats['steps']:>4} | "
            f"reward={stats['total_reward']:>8.1f} | "
            f"fps={stats['fps']:>5.0f} | "
            f"terminated={'yes' if stats['terminated'] else 'no ':>3}"
        )

    print(f"\n── Summary ({args.episodes} episodes) ──────────────────")
    rewards = [s["total_reward"] for s in all_stats]
    steps   = [s["steps"]        for s in all_stats]
    print(f"  Reward  mean ± std : {np.mean(rewards):>8.1f} ± {np.std(rewards):.1f}")
    print(f"  Reward  min / max  : {np.min(rewards):>8.1f} / {np.max(rewards):.1f}")
    print(f"  Steps   mean       : {np.mean(steps):>8.0f}")
    survival = sum(1 for s in all_stats if not s["terminated"]) / len(all_stats)
    print(f"  Survival rate      : {survival*100:.0f}%")
    print()

    env.close()


if __name__ == "__main__":
    main()
