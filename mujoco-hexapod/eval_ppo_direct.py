"""
eval_direct.py — Evaluate a trained PPO model on HexapodEnvDirect.

Usage:
    python eval_direct.py --model-path PPO/logs/ppo_hexapod.zip
    python eval_direct.py --model-path PPO/logs/ppo_hexapod.zip --episodes 10 --no-render
    python eval_direct.py --model-path PPO/logs/ppo_hexapod.zip --command-mode random
"""

import argparse
import time
import os
import numpy as np
from stable_baselines3 import PPO
from envs.hexapod_env_direct import HexapodEnvDirect


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained hexapod policy")
    parser.add_argument("--model-path", type=str, default="PPO/logs/ppo_hexapod.zip")
    parser.add_argument("--xml-path",      type=str,   default=None,         help="Path to MuJoCo XML (default: asset in env)")
    parser.add_argument("--episodes",      type=int,   default=5,            help="Number of episodes to run")
    parser.add_argument("--max-steps",     type=int,   default=10000,         help="Max steps per episode")
    parser.add_argument("--frame-skip",    type=int,   default=1,            help="Frame skip (match training)")
    parser.add_argument("--command-mode",  type=str,   default="fixed",      choices=["fixed", "random"])
    parser.add_argument("--vcmd-x",        type=float, default=1.0,          help="Forward velocity command")
    parser.add_argument("--vcmd-y",        type=float, default=0.0,          help="Lateral velocity command")
    parser.add_argument("--wcmd-yaw",      type=float, default=0.0,          help="Yaw rate command")
    parser.add_argument("--no-render",     action="store_true",               help="Disable MuJoCo viewer")
    parser.add_argument("--render-every",  type=int,   default=1,            help="Render every N steps (1=every step)")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        type=str,   default="cpu",        choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def run_episode(model, env, render: bool, render_every: int, deterministic: bool):
    obs, info = env.reset()
    episode_reward = 0.0
    step = 0
    done = False

    rewards = []
    vx_actuals = []
    vy_actuals = []

    t_start = time.perf_counter()

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        step += 1
        done = terminated or truncated

        # Pull actual body velocity from obs (indices 7:10 = linear vel)
        vx_actuals.append(float(obs[7]))
        vy_actuals.append(float(obs[8]))

        if render and (step % render_every == 0):
            env.render()

    elapsed = time.perf_counter() - t_start
    fps = step / elapsed if elapsed > 0 else 0

    return {
        "total_reward":  episode_reward,
        "steps":         step,
        "fps":           fps,
        "mean_reward":   float(np.mean(rewards)),
        "min_reward":    float(np.min(rewards)),
        "max_reward":    float(np.max(rewards)),
        "mean_vx":       float(np.mean(vx_actuals)),
        "mean_vy":       float(np.mean(vy_actuals)),
        "terminated":    terminated,
    }


def print_episode_stats(ep_num: int, stats: dict, vcmd_xy):
    print(f"\n── Episode {ep_num} ──────────────────────────────────────")
    print(f"  Steps        : {stats['steps']}")
    print(f"  Total reward : {stats['total_reward']:+.2f}")
    print(f"  Mean reward  : {stats['mean_reward']:+.4f}  "
          f"[min {stats['min_reward']:+.4f} / max {stats['max_reward']:+.4f}]")
    print(f"  Cmd vx/vy    : {vcmd_xy[0]:.3f} / {vcmd_xy[1]:.3f}")
    print(f"  Actual vx/vy : {stats['mean_vx']:+.3f} / {stats['mean_vy']:+.3f}")
    vx_err = abs(vcmd_xy[0] - stats['mean_vx'])
    vy_err = abs(vcmd_xy[1] - stats['mean_vy'])
    print(f"  Vel error    : vx={vx_err:.3f}  vy={vy_err:.3f}")
    print(f"  Terminated   : {stats['terminated']}")
    print(f"  Sim FPS      : {stats['fps']:.0f}")


def print_summary(all_stats: list, vcmd_xy):
    print("\n══ Summary ═══════════════════════════════════════════")
    rewards      = [s["total_reward"] for s in all_stats]
    steps        = [s["steps"]        for s in all_stats]
    mean_vx      = [s["mean_vx"]      for s in all_stats]
    n_terminated = sum(1 for s in all_stats if s["terminated"])

    print(f"  Episodes         : {len(all_stats)}")
    print(f"  Reward  mean/std : {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  Reward  min/max  : {np.min(rewards):+.2f} / {np.max(rewards):+.2f}")
    print(f"  Steps   mean     : {np.mean(steps):.0f}")
    print(f"  Survival rate    : {(len(all_stats)-n_terminated)/len(all_stats)*100:.0f}%")
    print(f"  Mean actual vx   : {np.mean(mean_vx):+.3f}  (cmd={vcmd_xy[0]:.3f})")
    print(f"  Avg vel tracking : err = {abs(vcmd_xy[0]-np.mean(mean_vx)):.3f} m/s")
    print("══════════════════════════════════════════════════════\n")


def main():
    args = parse_args()

    render = not args.no_render

    # ── Build env ─────────────────────────────────────────────────────────────
    env = HexapodEnvDirect(
        model_path=args.xml_path,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        command_mode=args.command_mode,
        vcmd_xy=(args.vcmd_x, args.vcmd_y),
        wcmd_yaw=args.wcmd_yaw,
        seed=args.seed,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_path}")
    model = PPO.load(args.model_path, env=env, device=args.device)
    print(f"Policy: {model.policy}")
    print(f"Obs dim: {env.observation_space.shape}  |  Action dim: {env.action_space.shape}")
    print(f"Command: vx={args.vcmd_x}  vy={args.vcmd_y}  wyaw={args.wcmd_yaw}")
    print(f"Render : {render}")
    print()

    # ── Run episodes ──────────────────────────────────────────────────────────
    all_stats = []
    vcmd_xy = np.array([args.vcmd_x, args.vcmd_y], dtype=np.float32)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(
            model=model,
            env=env,
            render=render,
            render_every=args.render_every,
            deterministic=args.deterministic,
        )
        # Update vcmd_xy in case of random mode
        vcmd_xy = env.vcmd_xy.copy()
        all_stats.append(stats)
        print_episode_stats(ep, stats, vcmd_xy)

    print_summary(all_stats, np.array([args.vcmd_x, args.vcmd_y]))

    env.close()


if __name__ == "__main__":
    main()