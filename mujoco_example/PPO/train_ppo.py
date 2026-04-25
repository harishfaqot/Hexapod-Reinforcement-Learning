import argparse
import os
import sys

import gym
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PPO.hexapod_ppo_env import HexapodPPOEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--log-dir", type=str, default="PPO/logs")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--command-mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--vcmd-x", type=float, default=0.1)
    parser.add_argument("--vcmd-y", type=float, default=0.0)
    parser.add_argument("--wcmd-yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 not installed. Run: pip install stable-baselines3"
        ) from exc

    os.makedirs(args.log_dir, exist_ok=True)

    def make_env(rank: int):
        def _init():
            env = HexapodPPOEnv(
                model_path=args.model_path,
                max_steps=args.max_steps,
                command_mode=args.command_mode,
                vcmd_xy=(args.vcmd_x, args.vcmd_y),
                wcmd_yaw=args.wcmd_yaw,
                seed=args.seed + rank,
            )
            return Monitor(env)

        return _init

    if args.num_envs <= 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=args.log_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model_path = os.path.join(args.log_dir, "ppo_hexapod")
    model.save(model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
