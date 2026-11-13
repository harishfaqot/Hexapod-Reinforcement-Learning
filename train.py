"""Train a PPO policy.

    python train.py --mode residual --seed 0
    python train.py --mode direct   --seed 0

Each run writes to runs/<mode>_<terrain>_s<seed>/ : model.zip, vecnormalize.pkl, config.json
and TensorBoard logs.

Watch a run while it trains (second terminal):
    tensorboard --logdir runs                                  # learning curves
    python evaluate.py runs/residual_stairs_s0 --render --episodes 1 --height 0.04  # the robot, current policy
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from hexapod import HexapodEnv


class SaveLatest(BaseCallback):
    """Save the current policy after every update (about every 8000 steps), so
    `python evaluate.py <run_dir> --render` can show it while training is still
    running, and a crash loses almost nothing."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.save(self.run_dir / "model")
        self.training_env.save(str(self.run_dir / "vecnormalize.pkl"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["residual", "direct"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--terrain", choices=["bumps", "stairs"], default="stairs")
    p.add_argument("--height", type=float, default=0.10, help="max bump or stair riser height in training [m]")
    p.add_argument("--n-envs", type=int, default=16)
    args = p.parse_args()

    run_dir = Path("runs") / f"{args.mode}_{args.terrain}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    env = make_vec_env(
        HexapodEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(mode=args.mode, terrain=args.terrain, terrain_height=args.height, randomize=True))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env, seed=args.seed, device="cpu",   # small MLP: CPU is faster than GPU here
        n_steps=512, batch_size=1024, n_epochs=5,         # 8192 samples per update
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),   # start with small joint noise
        tensorboard_log=str(run_dir), verbose=1)

    try:
        model.learn(total_timesteps=args.timesteps, callback=SaveLatest(run_dir))
    finally:
        env.close()
        print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
