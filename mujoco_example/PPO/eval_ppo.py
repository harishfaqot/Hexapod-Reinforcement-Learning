import argparse
import os
import sys
import time

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PPO.hexapod_ppo_env import HexapodPPOEnv
from envs.hexapod_simple import HexapodTkUI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PPO/logs/ppo_hexapod.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--tk-ui", type=bool, default=True)
    parser.add_argument("--command-mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--vcmd-x", type=float, default=-0.1)
    parser.add_argument("--vcmd-y", type=float, default=0.0)
    parser.add_argument("--wcmd-yaw", type=float, default=0.0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 not installed. Run: pip install stable-baselines3"
        ) from exc

    env = HexapodPPOEnv(
        max_steps=args.max_steps,
        command_mode=args.command_mode,
        vcmd_xy=(args.vcmd_x, args.vcmd_y),
        wcmd_yaw=args.wcmd_yaw,
    )

    ui = None
    if args.tk_ui:
        ui = HexapodTkUI(env.sim_env)
        ui.root.protocol("WM_DELETE_WINDOW", ui.root.destroy)
        # Expose the UI on the env so the env can read UI sliders (heading/v_rot)
        env.ui = ui

    model = PPO.load(args.model)

    rewards = []
    plot_enabled = bool(args.plot and plt is not None)
    if args.plot and plt is None:
        print("matplotlib not installed; realtime plots disabled")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        if plot_enabled:
            plt.ion()
            fig, (ax_vel, ax_yaw) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            fig.suptitle("Command vs Actual")

            timesteps = []
            vcmd_x = []
            vcmd_y = []
            vxy_x = []
            vxy_y = []
            wcmd = []
            wyaw = []

            (line_vcmd_x,) = ax_vel.plot([], [], label="vcmd_x")
            (line_vcmd_y,) = ax_vel.plot([], [], label="vcmd_y")
            (line_vxy_x,) = ax_vel.plot([], [], label="v_x")
            (line_vxy_y,) = ax_vel.plot([], [], label="v_y")
            ax_vel.set_ylabel("m/s")
            ax_vel.legend(loc="upper right")
            ax_vel.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

            (line_wcmd,) = ax_yaw.plot([], [], label="wcmd_yaw")
            (line_wyaw,) = ax_yaw.plot([], [], label="wyaw")
            ax_yaw.set_xlabel("timestep")
            ax_yaw.set_ylabel("rad/s")
            ax_yaw.legend(loc="upper right")
            ax_yaw.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if ui is not None and ui.root.winfo_exists():
                env.vcmd_xy = np.array([
                    float(ui.vx_slider.get()),
                    float(ui.vy_slider.get()),
                ], dtype=np.float32)
                env.wcmd_yaw = float(ui.v_rot_slider.get())
                ui.step_height_slider.set(float(action[0]))
                ui.cpg_slider.set(float(action[1]))
                ui.duty_slider.set(float(action[2]))
                ui.x_slider.set(float(action[3]))
                ui.y_slider.set(float(action[4]))
                ui.z_slider.set(float(action[5]))
                ui.r_slider.set(float(action[6]))
                ui.p_slider.set(float(action[7]))
                ui.yaw_slider.set(float(action[8]))

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

            if ui is not None and ui.root.winfo_exists():
                env.sim_env.camera_follow_enabled = bool(ui.camera_follow_var.get())
                if step_idx % 2 == 0:
                    ui._update_imu_plot()
                ui.root.update_idletasks()
                ui.root.update()

            if plot_enabled:
                body_id = int(env.body_id)
                v_world = np.asarray(env.sim_env.sim.data.body_xvelp[body_id], dtype=np.float32)
                w_world = np.asarray(env.sim_env.sim.data.body_xvelr[body_id], dtype=np.float32)
                rot = np.asarray(env.sim_env.sim.data.body_xmat[body_id], dtype=np.float32).reshape(3, 3)
                v_body = rot.T @ v_world
                w_body = rot.T @ w_world

                vxy = v_body[:2]
                wyaw_val = float(w_body[2])

                timesteps.append(step_idx)
                vcmd_x.append(float(env.vcmd_xy[0]))
                vcmd_y.append(float(env.vcmd_xy[1]))
                vxy_x.append(float(vxy[0]))
                vxy_y.append(float(vxy[1]))
                wcmd.append(float(env.wcmd_yaw))
                wyaw.append(wyaw_val)

                line_vcmd_x.set_data(timesteps, vcmd_x)
                line_vcmd_y.set_data(timesteps, vcmd_y)
                line_vxy_x.set_data(timesteps, vxy_x)
                line_vxy_y.set_data(timesteps, vxy_y)
                line_wcmd.set_data(timesteps, wcmd)
                line_wyaw.set_data(timesteps, wyaw)

                ax_vel.relim()
                ax_vel.autoscale_view()
                ax_yaw.relim()
                ax_yaw.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            step_idx += 1

            if args.render:
                env.render()

        rewards.append(ep_reward)
        if plot_enabled:
            plt.ioff()
            plt.close(fig)
        print(f"Episode {ep + 1}: reward={ep_reward:.3f}")

    print(f"Average reward over {args.episodes} episodes: {np.mean(rewards):.3f}")
    env.close()
    if ui is not None and ui.root.winfo_exists():
        ui.root.destroy()


if __name__ == "__main__":
    main()
