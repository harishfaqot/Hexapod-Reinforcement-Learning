import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.hexapod_simple import (
    HexapodSimple,
    LEG_ORDER,
    LEG_BASE_POSITIONS,
    _build_actuator_index_map,
    _compute_neutral_ik_targets,
    _joint_info_for_actuator,
    _joint_delta_sign_for_leg,
    _foot_home_offset_body,
    _body_target_to_leg_ik_frame,
    _clamp,
    body_kinematics,
    generate_movement,
    inverse_kinematics,
    HexapodTkUI,
)


class HexapodPPOEnv(gym.Env):
    """PPO training wrapper around HexapodSimple with gait-parameter actions."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        model_path: Optional[str] = None,
        frame_skip: int = 2,
        max_steps: int = 1000,
        command_mode: str = "fixed",
        vcmd_xy: Tuple[float, float] = (0.1, 0.0),
        wcmd_yaw: float = 0.0,
        command_range_xy: float = 0.2,
        command_range_yaw: float = 0.6,
        seed: Optional[int] = None,
        tk_ui: bool = False,
    ):
        super().__init__()

        self.sim_env = HexapodSimple(model_path=model_path, frame_skip=frame_skip)
        self.max_steps = int(max_steps)
        self.command_mode = command_mode
        self.vcmd_xy = np.array(vcmd_xy, dtype=np.float32)
        self.wcmd_yaw = float(wcmd_yaw)
        self.command_range_xy = float(command_range_xy)
        self.command_range_yaw = float(command_range_yaw)

        if seed is not None:
            np.random.seed(seed)

        # Optionally create a Tk UI (same UI used by hexapod_simple) so sliders
        # like heading and v_rot are available on the env instance.
        self.ui = None
        if tk_ui:
            try:
                self.ui = HexapodTkUI(self.sim_env)
                # expose ui sliders on the env for backward compatibility
                if hasattr(self.ui, "heading_slider"):
                    self.heading_slider = self.ui.heading_slider
                if hasattr(self.ui, "v_rot_slider"):
                    self.v_rot_slider = self.ui.v_rot_slider
            except Exception:
                # UI creation is best-effort; don't fail env construction if it
                # can't be created (e.g., headless server)
                self.ui = None

        self._build_joint_calibration()
        self.body_id = int(self.sim_env.camera_follow_body_id)

        # Action: step height, cpg hz, duty, body x,y,z, roll, pitch, yaw
        act_low = np.array([0.03, 0.3, 0.3, -0.05, -0.05, -0.05, -30.0, -30.0, -30.0], dtype=np.float32)
        act_high = np.array([0.1, 2.0, 0.6, 0.05, 0.05, 0.05, 30.0, 30.0, 30.0], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Observation: body quaternion (w, x, y, z)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.step_count = 0
        self.last_action = None

    def _build_joint_calibration(self):
        actuator_map = _build_actuator_index_map(self.sim_env)
        nominal_ik_targets = _compute_neutral_ik_targets()

        state0 = self.sim_env.sim.get_state()
        qpos0 = np.asarray(state0.qpos, dtype=np.float32)

        self.joint_calibration = {}
        for leg_i, leg in enumerate(LEG_ORDER):
            self.joint_calibration[leg] = {}
            for joint_name in ["coxa", "femur", "tibia"]:
                actuator_idx = actuator_map[leg][joint_name]
                if actuator_idx is None:
                    self.joint_calibration[leg][joint_name] = None
                    continue

                info = _joint_info_for_actuator(self.sim_env, actuator_idx)
                neutral_joint_qpos = float(qpos0[info["qpos_adr"]])
                self.joint_calibration[leg][joint_name] = {
                    "actuator_idx": actuator_idx,
                    "neutral_joint_qpos": neutral_joint_qpos,
                    "nominal_ik": float(nominal_ik_targets[leg][joint_name]),
                    "low": float(info["low"]),
                    "high": float(info["high"]),
                    "delta_sign": _joint_delta_sign_for_leg(leg_i, joint_name),
                }

    def _sample_command(self):
        if self.command_mode == "random":
            self.vcmd_xy = np.random.uniform(
                -self.command_range_xy, self.command_range_xy, size=(2,)
            ).astype(np.float32)
            self.wcmd_yaw = float(np.random.uniform(-self.command_range_yaw, self.command_range_yaw))

    def _get_obs(self) -> np.ndarray:
        return self.sim_env.get_imu_quat().astype(np.float32)

    def _compute_gait_action(self, action: np.ndarray) -> np.ndarray:
        step_height, cpg_hz, duty, x, y, z, roll, pitch, yaw = action
        # print(f"Action: step_height={step_height:.3f}, cpg_hz={cpg_hz:.3f}, duty={duty:.3f}, "
        #       f"x={x:.3f}, y={y:.3f}, z={z:.3f}, roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")
        
        cpg_hz = 1
        gait_phase_lag = math.pi
        body_position = (float(x), float(y), float(z))
        body_orientation = (float(roll), float(pitch), float(yaw))

        # Commanded velocity drives gait; policy adapts other gait/body parameters.
        vx = float(self.vcmd_xy[0])
        vy = float(self.vcmd_xy[1])
        # Heading control: v_rot is proportional to heading error (simple P controller)
        # Prefer a UI-provided heading slider if available (eval_ppo creates a UI
        # as `ui = HexapodTkUI(env.sim_env)` and can set `env.ui = ui`), otherwise
        # fall back to any heading_slider attribute if present. If none exist,
        # default to zero heading command.
        desired_heading = 0.0
        ui = getattr(self, "ui", None)
        if ui is not None and hasattr(ui, "heading_slider"):
            desired_heading = float(ui.heading_slider.get())
        elif hasattr(self, "heading_slider"):
            desired_heading = float(self.heading_slider.get())

        if vx == 0.0 and vy == 0.0 and desired_heading == 0.0:
            v_rot = 0.0
        else:
            current_yaw = float(self.sim_env.get_imu_rpy()[2]) * 180.0 / np.pi  # degrees
            heading_error = -(desired_heading - current_yaw + 180) % 360 - 180  # shortest path
            k_p = 0.005
            v_rot = k_p * heading_error
            v_rot = _clamp(v_rot, -0.15, 0.15)

        # Update UI v_rot display if available (prefer ui, then env attribute)
        if ui is not None and hasattr(ui, "v_rot_slider"):
            ui.v_rot_slider.config(state="normal")
            ui.v_rot_slider.set(v_rot)
            ui.v_rot_slider.config(state="disabled")
        elif hasattr(self, "v_rot_slider"):
            self.v_rot_slider.config(state="normal")
            self.v_rot_slider.set(v_rot)
            self.v_rot_slider.config(state="disabled")

        t = float(self.sim_env.sim.data.time)
        body_leg_positions = body_kinematics(body_position, body_orientation)

        action_ctrl = np.asarray(self.sim_env.sim.data.ctrl.copy(), dtype=np.float32)

        for i, leg in enumerate(LEG_ORDER):
            movement, _ = generate_movement(
                t=t,
                freq_hz=float(cpg_hz),
                phase_lag_rad=gait_phase_lag,
                step_height=float(step_height),
                body_movement=(vx, vy, v_rot),
                leg_index=i,
                duty_factor=float(duty),
            )

            leg_base = np.array(body_leg_positions[i], dtype=np.float32)
            nominal_leg_base = np.array(LEG_BASE_POSITIONS[i], dtype=np.float32)

            target_foot = leg_base + _foot_home_offset_body(i) + movement
            ik_x, ik_y, ik_z = _body_target_to_leg_ik_frame(target_foot, nominal_leg_base, i)

            coxa, femur, tibia = inverse_kinematics(
                ik_x,
                ik_y,
                ik_z,
            )

            desired_ik = {
                "coxa": coxa,
                "femur": femur,
                "tibia": tibia,
            }

            for joint_name, ik_value in desired_ik.items():
                calib = self.joint_calibration[leg][joint_name]
                if calib is None:
                    continue

                delta = (float(ik_value) - calib["nominal_ik"]) * calib["delta_sign"]
                target_joint = calib["neutral_joint_qpos"] + delta
                target_joint = _clamp(target_joint, calib["low"], calib["high"])
                action_ctrl[calib["actuator_idx"]] = target_joint

        return action_ctrl

    def _compute_reward(self) -> float:
        v_world = np.asarray(self.sim_env.sim.data.body_xvelp[self.body_id], dtype=np.float32)
        w_world = np.asarray(self.sim_env.sim.data.body_xvelr[self.body_id], dtype=np.float32)
        rot = np.asarray(self.sim_env.sim.data.body_xmat[self.body_id], dtype=np.float32).reshape(3, 3)
        v_body = rot.T @ v_world
        w_body = rot.T @ w_world
        vxy = v_body[:2]
        wyaw = float(w_body[2])

        # Use signed component errors so forward/backward and left/right are rewarded correctly.
        vx_err = float(self.vcmd_xy[0] - vxy[0])
        vy_err = float(self.vcmd_xy[1] - vxy[1])
        vel_err = vx_err * vx_err + vy_err * vy_err
        yaw_err = (self.wcmd_yaw - wyaw) ** 2

        # print(f"Velocity: vxy={vxy}, wyaw={wyaw:.3f}, vel_err={vel_err:.3f}, yaw_err={yaw_err:.3f}")

        r_vel = math.exp(-4.0 * vel_err)
        r_yaw = math.exp(-4.0 * yaw_err)

        posture_penalty = 0.0
        if self.last_action is not None and len(self.last_action) >= 9:
            _, _, _, x, y, z, roll, pitch, _ = self.last_action

            # Dead-zone penalty: no cost inside small band, quadratic outside.
            pos_deadzone = 0.01
            rot_deadzone = 5.0
            pos_weight = 100.0
            rot_weight = 0.01

            dx = max(0.0, abs(float(x)) - pos_deadzone)
            dy = max(0.0, abs(float(y)) - pos_deadzone)
            dz = max(0.0, abs(float(z)) - pos_deadzone)
            droll = max(0.0, abs(float(roll)) - rot_deadzone)
            dpitch = max(0.0, abs(float(pitch)) - rot_deadzone)

            posture_penalty = pos_weight * (dx + dy + dz)
            posture_penalty += rot_weight * (droll + dpitch)

        return 1.0 * r_vel + 0.5 * r_yaw - posture_penalty

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.sim_env.sim.reset()
        self.step_count = 0
        self._sample_command()
        obs = self._get_obs()
        info = {
            "vcmd_xy": self.vcmd_xy.copy(),
            "wcmd_yaw": float(self.wcmd_yaw),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        ctrl = self._compute_gait_action(action)
        self.sim_env.step(ctrl)

        obs = self._get_obs()
        reward = self._compute_reward()

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Check for flipped robot: large roll/pitch or very low body height
        try:
            rpy = self.sim_env.get_imu_rpy()
            if rpy is not None:
                roll = float(rpy[0])
                pitch = float(rpy[1])
                # If roll or pitch exceed threshold (radians), consider flipped.
                if abs(roll) > 1.2 or abs(pitch) > 1.2:
                    terminated = True
        except Exception:
            # If IMU read fails, don't flip based on IMU
            pass

        # If terminated and auto_reset requested, perform reset now and return
        if terminated:
            obs, info = self.reset()
            return obs, reward, terminated, truncated, info

        info = {
            "vcmd_xy": self.vcmd_xy.copy(),
            "wcmd_yaw": float(self.wcmd_yaw),
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.sim_env.render(mode=mode)

    def close(self):
        self.sim_env.close()
