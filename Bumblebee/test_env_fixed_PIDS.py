import argparse
import math
import time

import numpy as np
import pybullet as p

from hexapod_env import HexapodEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HexapodEnv with fixed actions (no RL model)."
    )
    parser.add_argument("--steps", type=int, default=300, help="Number of env steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reset")
    parser.add_argument("--save-log", action="store_true", help="Save env logs to CSV")
    parser.add_argument("--pid-kp", type=float, default=None, help="Override env yaw PID Kp")
    parser.add_argument("--pid-ki", type=float, default=None, help="Override env yaw PID Ki")
    parser.add_argument("--pid-kd", type=float, default=None, help="Override env yaw PID Kd")
    parser.add_argument("--pid-ray-weight", type=float, default=None, help="Override ray avoidance weight in yaw PID")

    # Fixed normalized residual actions in [-1, 1] (same order as env action space)
    parser.add_argument("--body-roll", type=float, default=0.0, help="Action[0] residual in [-1, 1]")
    parser.add_argument("--body-pitch", type=float, default=0.0, help="Action[1] residual in [-1, 1]")
    parser.add_argument("--step-height", type=float, default=0.0, help="Action[2] residual in [-1, 1]")
    parser.add_argument("--step-dur", type=float, default=0.0, help="Action[3] residual in [-1, 1]")
    parser.add_argument(
        "--gait-phase", type=float, default=0.0,
        help="Action[4] residual in [-1, 1]"
    )
    parser.add_argument("--duty-factor", type=float, default=0.0, help="Action[5] residual in [-1, 1]")
    parser.add_argument("--vx", type=float, default=0.06, help="Action[6] residual in [-1, 1] (ignored: env auto-controls vx from rays)")
    parser.add_argument("--vy", type=float, default=0.0, help="Action[7] residual in [-1, 1] (ignored: env auto-controls vy from rays)")
    parser.add_argument("--vrot", type=float, default=0.0, help="Action[8] residual in [-1, 1] (ignored: env auto-controls vrot from rays)")
    parser.add_argument("--body-x", type=float, default=0.0, help="Action[9] residual in [-1, 1]")
    parser.add_argument("--body-y", type=float, default=0.0, help="Action[10] residual in [-1, 1]")
    parser.add_argument("--body-z", type=float, default=0.0, help="Action[11] residual in [-1, 1]")

    return parser.parse_args()


def build_action(args):
    return np.array(
        [
            args.body_roll,
            args.body_pitch,
            args.step_height,
            args.step_dur,
            args.gait_phase,
            args.duty_factor,
            args.vx,
            args.vy,
            args.vrot,
            args.body_x,
            args.body_y,
            args.body_z,
        ],
        dtype=np.float32,
    )


def main():
    args = parse_args()

    env = HexapodEnv(is_train=False)
    env.set_yaw_pid_gains(
        kp=args.pid_kp,
        ki=args.pid_ki,
        kd=args.pid_kd,
        ray_weight=args.pid_ray_weight,
    )
    obs, _ = env.reset(seed=args.seed)

    action = build_action(args)

    # Keep action inside declared bounds to avoid invalid control inputs.
    action = np.clip(action, env.action_space.low, env.action_space.high)

    print("=== Fixed Action Env Test (No RL) ===")
    print(f"action used: {action.tolist()}")
    print(f"obs shape: {obs.shape}, action shape: {action.shape}")

    total_reward = 0.0
    started = time.time()

    for step in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if step % 10 == 0 or terminated or truncated:
            pos, orn = p.getBasePositionAndOrientation(env.robot)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            ray_count = getattr(env, "RAY_COUNT", 0)
            min_ray = float(np.min(obs[-ray_count:])) if ray_count > 0 else 1.0
            print(
                f"step={step:04d} reward={reward:+8.3f} total={total_reward:+10.3f} "
                f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) "
                f"rpy_deg=({math.degrees(roll):+.1f}, {math.degrees(pitch):+.1f}, {math.degrees(yaw):+.1f}) "
                f"min_ray={min_ray:.3f}"
            )

        if terminated or truncated:
            print(f"Episode ended at step={step} terminated={terminated} truncated={truncated}")
            break

    elapsed = time.time() - started
    print("=== Summary ===")
    print(f"steps_run: {step + 1}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"avg_reward: {total_reward / max(step + 1, 1):.3f}")
    print(f"wall_time_s: {elapsed:.2f}")

    if args.save_log:
        env.save_logs_to_csv("test_env_fixed_logs.csv")
        print("saved: test_env_fixed_logs.csv")

    env.close()


if __name__ == "__main__":
    main()
