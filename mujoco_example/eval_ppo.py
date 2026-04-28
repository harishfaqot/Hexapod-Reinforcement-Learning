import argparse
import os
import sys
import time

import numpy as np
import mujoco

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from envs.hexapod_env import HexapodEnv, HexapodTkUI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PPO/logs/ppo_hexapod.zip")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--tk-ui", type=bool, default=True)
    parser.add_argument("--command-mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--vcmd-x", type=float, default=0)
    parser.add_argument("--vcmd-y", type=float, default=0.0)
    parser.add_argument("--wcmd-yaw", type=float, default=0.0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 not installed. Run: pip install stable-baselines3"
        ) from exc

    env = HexapodEnv(
        max_steps=args.max_steps,
        command_mode=args.command_mode,
        vcmd_xy=(args.vcmd_x, args.vcmd_y),
        wcmd_yaw=args.wcmd_yaw,
    )

    # Build UI for display only — DO NOT call ui.run() or ui._tick().
    # _tick() runs its own gait loop which would fight with RL action output.
    # We only use the UI for: reading vx/vy/heading sliders, and displaying
    # denormalized RL actions on the body/gait sliders.
    ui = None
    if args.tk_ui:
        ui = HexapodTkUI(env.sim_env)
        ui.root.protocol("WM_DELETE_WINDOW", ui.root.destroy)
        env.ui = ui

        # cpg and gait sliders stay enabled — user can adjust them live during eval.
        # The env reads them via _get_cpg_hz() / _get_gait_phase_lag() each step.

    model = PPO.load(args.model, device=args.device)

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
            vcmd_x_log = []
            vcmd_y_log = []
            vxy_x_log = []
            vxy_y_log = []
            wcmd_log = []
            wyaw_log = []

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
                # Read velocity commands from UI sliders (user controls these)
                env.vcmd_xy = np.array([
                    float(ui.vx_slider.get()),
                    float(ui.vy_slider.get()),
                ], dtype=np.float32)
                # heading is read inside env via _get_desired_heading_deg() → ui.heading_slider

                # Denormalize RL action to physical units before displaying on sliders.
                # action is [-1, 1]; denorm maps to physical ranges defined in HexapodEnv.__init__
                # [step_height, duty, x, y, z, roll, pitch, yaw]
                denorm = env._denormalize_action(action)
                step_height  = float(denorm[0])  # [0.03, 0.10]
                duty         = float(denorm[1])  # [0.3,  0.6]
                body_x       = float(denorm[2])  # [-0.05, 0.05]
                body_y       = float(denorm[3])  # [-0.05, 0.05]
                body_z       = float(denorm[4])  # [-0.05, 0.05]
                body_roll    = float(denorm[5])  # [-30,  30] deg
                body_pitch   = float(denorm[6])  # [-30,  30] deg
                body_yaw     = float(denorm[7])  # [-30,  30] deg

                # Display denormalized values on body/gait sliders (read-only visual feedback)
                ui.step_height_slider.set(step_height)
                ui.duty_slider.set(duty)
                ui.x_slider.set(body_x)
                ui.y_slider.set(body_y)
                ui.z_slider.set(body_z)
                ui.r_slider.set(body_roll)
                ui.p_slider.set(body_pitch)
                ui.yaw_slider.set(body_yaw)

                # cpg_hz and gait_phase_lag are read live from ui.cpg_slider / ui.gait_slider
                # inside env._get_cpg_hz() and env._get_gait_phase_lag() — user controls these freely.

            # env.step() calls _denormalize_action internally, so pass raw [-1,1] action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

            if ui is not None and ui.root.winfo_exists():
                env.sim_env.camera_follow_enabled = bool(ui.camera_follow_var.get())
                if step_idx % 2 == 0:
                    ui._update_imu_plot()
                # Pump Tk event loop to keep UI responsive — no _tick(), no gait loop
                ui.root.update_idletasks()
                ui.root.update()

            if plot_enabled:
                body_velocity_local = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(
                    env.sim_env.model,
                    env.sim_env.data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(env.body_id),
                    body_velocity_local,
                    1,
                )
                vxy = np.asarray(body_velocity_local[3:5], dtype=np.float32)
                wyaw_val = float(body_velocity_local[2])

                timesteps.append(step_idx)
                vcmd_x_log.append(float(env.vcmd_xy[0]))
                vcmd_y_log.append(float(env.vcmd_xy[1]))
                vxy_x_log.append(float(vxy[0]))
                vxy_y_log.append(float(vxy[1]))
                wcmd_log.append(float(env.wcmd_yaw))
                wyaw_log.append(wyaw_val)

                line_vcmd_x.set_data(timesteps, vcmd_x_log)
                line_vcmd_y.set_data(timesteps, vcmd_y_log)
                line_vxy_x.set_data(timesteps, vxy_x_log)
                line_vxy_y.set_data(timesteps, vxy_y_log)
                line_wcmd.set_data(timesteps, wcmd_log)
                line_wyaw.set_data(timesteps, wyaw_log)

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
