"""
train_ant.py — Train PPO on Ant-v5 (MuJoCo gymnasium env).

Install deps:
    pip install stable-baselines3 gymnasium[mujoco]

Usage:
    python train_ant.py
    python train_ant.py --total-timesteps 5_000_000
    python train_ant.py --resume PPO_ant/logs/best_model.zip
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)


# ── Logging callback ──────────────────────────────────────────────────────────

class TrainingStatsCallback(BaseCallback):
    """Print a clean one-line summary every N rollouts."""

    def __init__(self, print_every_n_rollouts: int = 5):
        super().__init__()
        self.print_every = print_every_n_rollouts
        self.rollout_count = 0

    def _on_rollout_end(self):
        self.rollout_count += 1
        if self.rollout_count % self.print_every != 0:
            return

        fps       = self.locals.get("fps", 0)
        ep_info   = self.model.ep_info_buffer
        if len(ep_info) == 0:
            return

        mean_rew  = np.mean([e["r"] for e in ep_info])
        mean_len  = np.mean([e["l"] for e in ep_info])
        n_steps   = self.num_timesteps

        print(
            f"  [step {n_steps:>9,}]  "
            f"ep_rew={mean_rew:>8.1f}  "
            f"ep_len={mean_len:>6.0f}  "
            f"fps={fps:>5}"
        )

    def _on_step(self):
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int,   default=3_000_000)
    p.add_argument("--num-envs",        type=int,   default=8)
    p.add_argument("--log-dir",         type=str,   default="PPO_ant/logs")
    p.add_argument("--device",          type=str,   default="cpu",  choices=["cpu", "cuda", "auto"])
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--resume",          type=str,   default=None,   help="Resume from .zip checkpoint")
    p.add_argument("--n-steps",         type=int,   default=2048)
    p.add_argument("--batch-size",      type=int,   default=512)
    p.add_argument("--lr",              type=float, default=3e-4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "eval"), exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Ant-v5  |  device={args.device}  |  envs={args.num_envs}")
    print(f"  timesteps={args.total_timesteps:,}  |  seed={args.seed}")
    print(f"{'='*55}\n")

    # ── Training envs ─────────────────────────────────────────────────────────
    train_env = make_vec_env(
        "Ant-v5",
        n_envs=args.num_envs,
        seed=args.seed,
    )
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # ── Eval env (unnormalized reward so stats are interpretable) ─────────────
    eval_env = make_vec_env("Ant-v5", n_envs=1, seed=args.seed + 999)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # keep reward raw for eval
        clip_obs=10.0,
        training=False,      # don't update stats during eval
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=max(50_000 // args.num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ant_ppo",
        verbose=1,
    )

    stats_callback = TrainingStatsCallback(print_every_n_rollouts=5)

    callbacks = [eval_callback, checkpoint_callback, stats_callback]

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            device=args.device,
            tensorboard_log=args.log_dir,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            device=args.device,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,                         # silenced — stats_callback handles printing
            tensorboard_log=args.log_dir,
            seed=args.seed,
        )

    print(f"Obs  : {train_env.observation_space.shape}")
    print(f"Act  : {train_env.action_space.shape}")
    print(f"Params: {sum(p.numel() for p in model.policy.parameters()):,}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume),
        progress_bar=True,
    )

    # ── Save final model + VecNormalize stats ─────────────────────────────────
    model_path = os.path.join(args.log_dir, "ant_ppo_final")
    model.save(model_path)
    train_env.save(os.path.join(args.log_dir, "vec_normalize.pkl"))
    print(f"\nSaved model  : {model_path}.zip")
    print(f"Saved stats  : {args.log_dir}/vec_normalize.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
