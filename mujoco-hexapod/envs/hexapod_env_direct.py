"""
HexapodEnvDirect — Direct joint position control, no IK/kinematics.

Action  : 18 joint target positions (rad), clipped to joint limits
           Order: coxa_fl, femur_fl, tibia_fl,
                  coxa_fr, femur_fr, tibia_fr,
                  coxa_rr, femur_rr, tibia_rr,
                  coxa_rl, femur_rl, tibia_rl,
                  coxa_mr, femur_mr, tibia_mr,
                  coxa_ml, femur_ml, tibia_ml

Observation (39-dim):
    [0:4]   body quaternion (w, x, y, z)              — orientation
    [4:7]   body angular velocity (roll, pitch, yaw)  — from qvel[3:6]
    [7:10]  body linear velocity  (x, y, z)           — from qvel[0:3]
    [10:28] joint positions (18)                       — current qpos
    [28:46] joint velocities (18)                      — current qvel  ← wait, obs is 46 not 39
    [46:48] velocity command (vcmd_x, vcmd_y)          — task command

Total: 4 + 3 + 3 + 18 + 18 + 2 = 48 dims
"""

from typing import Optional, Tuple
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


# ── Joint ordering (matches XML actuator order) ───────────────────────────────
JOINT_NAMES = [
    "coxa_fl", "femur_fl", "tibia_fl",
    "coxa_fr", "femur_fr", "tibia_fr",
    "coxa_rr", "femur_rr", "tibia_rr",
    "coxa_rl", "femur_rl", "tibia_rl",
    "coxa_mr", "femur_mr", "tibia_mr",
    "coxa_ml", "femur_ml", "tibia_ml",
]
N_JOINTS = len(JOINT_NAMES)  # 18

# Default standing pose — all joints at 0 except femur slightly down and tibia bent
_DEFAULT_FEMUR_ANGLE = 0.25  # rad, positive = leg pointing down
_DEFAULT_TIBIA_ANGLE = -0.8  # rad


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _quat_wxyz_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = np.sign(sinp) * np.pi/2 if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


class HexapodEnvDirect(gym.Env):
    """
    Hexapod env with direct 18-DOF joint position control.
    No IK, no CPG, no kinematics layer — policy outputs joint angles directly.
    """

    metadata = {"render.modes": ["human"]}

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "assets",
        "hexapod_trossen_new.xml",
        # "hexapod_trossen_rails.xml"
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        frame_skip: int = 4,
        max_steps: int = 10000,
        command_mode: str = "fixed",          # "fixed" or "random"
        vcmd_xy: Tuple[float, float] = (0.2, 0.0),
        wcmd_yaw: float = 0.0,
        command_range_xy: float = 0.3,
        command_range_yaw: float = 0.5,
        terminate_on_flip: bool = True,
        flip_threshold_rad: float = 1.0,      # ~57° roll or pitch → terminated
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.frame_skip = max(1, int(frame_skip))
        self.max_steps = int(max_steps)
        self.command_mode = command_mode
        self.vcmd_xy = np.array(vcmd_xy, dtype=np.float32)
        self.wcmd_yaw = float(wcmd_yaw)
        self.command_range_xy = float(command_range_xy)
        self.command_range_yaw = float(command_range_yaw)
        self.terminate_on_flip = terminate_on_flip
        self.flip_threshold_rad = float(flip_threshold_rad)

        if seed is not None:
            np.random.seed(seed)

        # ── Load MuJoCo model ─────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data  = mujoco.MjData(self.model)
        self.viewer = None

        # ── Resolve body / sensor IDs ─────────────────────────────────────────
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        imu_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat"
        )
        self.imu_sensor_id = int(imu_id) if imu_id >= 0 else None

        # ── Resolve joint qpos addresses (for reset & obs) ────────────────────
        self._joint_qpos_adr = np.array([
            int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ])
            for jn in JOINT_NAMES
        ], dtype=np.int32)

        self._joint_qvel_adr = np.array([
            int(self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ])
            for jn in JOINT_NAMES
        ], dtype=np.int32)

        # free-joint (root) occupies qpos[0:7], qvel[0:6]
        self._root_qpos_adr = 0
        self._root_qvel_adr = 0

        # ── Action space: read per-joint limits from XML ──────────────────────
        _joint_low  = np.empty(N_JOINTS, dtype=np.float32)
        _joint_high = np.empty(N_JOINTS, dtype=np.float32)
        for i, jn in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            _joint_low[i]  = float(self.model.jnt_range[jid][0])
            _joint_high[i] = float(self.model.jnt_range[jid][1])
        self._action_low  = _joint_low
        self._action_high = _joint_high
        self.action_space = spaces.Box(
            low=_joint_low, high=_joint_high, dtype=np.float32
        )

        # ── Observation space: 48-dim ─────────────────────────────────────────
        # [quat(4), ang_vel(3), lin_vel(3), joint_pos(18), joint_vel(18), cmd(2)]
        obs_dim = 4 + 3 + 3 + N_JOINTS + N_JOINTS + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Internal state ────────────────────────────────────────────────────
        self.step_count = 0
        self.prev_torso_xy = np.zeros(2, dtype=np.float32)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_imu_quat(self) -> np.ndarray:
        if self.imu_sensor_id is not None:
            adr = int(self.model.sensor_adr[self.imu_sensor_id])
            dim = int(self.model.sensor_dim[self.imu_sensor_id])
            if dim >= 4:
                return np.asarray(self.data.sensordata[adr:adr+4], dtype=np.float32)
        return np.asarray(self.data.xquat[self.torso_body_id], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        quat     = self._get_imu_quat()                          # (4,)
        ang_vel  = self.data.qvel[3:6].astype(np.float32)        # (3,) body angular vel
        lin_vel  = self.data.qvel[0:3].astype(np.float32)        # (3,) body linear vel
        jpos     = self.data.qpos[self._joint_qpos_adr].astype(np.float32)  # (18,)
        jvel     = self.data.qvel[self._joint_qvel_adr].astype(np.float32)  # (18,)
        cmd      = np.array([self.vcmd_xy[0], self.vcmd_xy[1]], dtype=np.float32)  # (2,)
        return np.concatenate([quat, ang_vel, lin_vel, jpos, jvel, cmd])

    def _sample_command(self):
        if self.command_mode == "random":
            self.vcmd_xy = np.random.uniform(
                -self.command_range_xy, self.command_range_xy, size=(2,)
            ).astype(np.float32)
            self.wcmd_yaw = float(
                np.random.uniform(-self.command_range_yaw, self.command_range_yaw)
            )

    def _compute_reward(self, action: np.ndarray) -> float:
        # ── Local body velocity (body frame) ──────────────────────────────────
        body_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body_id,
            body_vel,
            1,  # local frame
        )
        vx_actual   = float(body_vel[0])
        vy_actual   = float(body_vel[1])
        wyaw_actual = float(body_vel[5])

        # ── 1. Velocity reward — paper eq.(4) style ───────────────────────────
        # Peaks at target velocity, does NOT simply reward "go fast forever".
        # r_v = 1/(|vx - v_tar| + 1) - 1/(v_tar + 1)
        # This gives 0 reward at v=0 and positive gradient toward v_tar.
        vtar_x = float(self.vcmd_xy[0])
        vtar_y = float(self.vcmd_xy[1])
        r_vx = 1.0 / (abs(vx_actual - vtar_x) + 1.0) - 1.0 / (abs(vtar_x) + 1.0)
        r_vy = 1.0 / (abs(vy_actual - vtar_y) + 1.0) - 1.0 / (abs(vtar_y) + 1.0)
        r_vel = r_vx + r_vy  # [0, 1] range each; agent gets 0 at v=0

        print(f"DEBUG: vx={vx_actual:.3f} vtar_x={vtar_x:.3f} r_vx={r_vx:.3f} | "
              f"vy={vy_actual:.3f} vtar_y={vtar_y:.3f} r_vy={r_vy:.3f}")    

        # ── 2. Heading correction — paper eq.(5) style ───────────────────────
        # Reward agent for CORRECTING yaw error rather than penalising deviation.
        # r_c = |yaw_err_prev| - |yaw_err_now|  → positive when closing the gap.
        yaw_err = self.wcmd_yaw - wyaw_actual
        r_yaw = float(np.exp(-2.0 * yaw_err**2))  # soft tracking

        # ── 3. Posture — small weight, only fires on large angles ─────────────
        # Reduced weight vs original so it doesn't dominate early learning.
        rpy = _quat_wxyz_to_rpy(self._get_imu_quat())
        r_posture = -0.05 * float(rpy[0]**2 + rpy[1]**2)

        # ── 4. Smoothness — penalise torso z-accel (paper: penalise ż²) ───────
        vz_body = float(body_vel[2])
        r_smooth = -0.05 * vz_body**2

        # ── 5. Energy / torque penalty — paper penalises τ² ──────────────────
        # ctrl = position targets; use joint velocity as proxy for effort
        jvel = self.data.qvel[self._joint_qvel_adr].astype(np.float32)
        r_energy = -5e-5 * float(np.sum(jvel**2))

        # ── 6. Alive bonus — scaled to not overwhelm velocity signal ──────────
        r_alive = 0.05

        return r_vel + 0.3 * r_yaw + r_posture + r_smooth + r_energy + r_alive

    def _is_terminated(self) -> bool:
        if not self.terminate_on_flip:
            return False
        rpy = _quat_wxyz_to_rpy(self._get_imu_quat())
        return bool(abs(float(rpy[0])) > self.flip_threshold_rad or
                    abs(float(rpy[1])) > self.flip_threshold_rad)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def _default_qpos(self) -> np.ndarray:
        qpos = np.zeros(N_JOINTS, dtype=np.float32)
        for i in range(N_JOINTS):
            lo = float(self._action_low[i])
            hi = float(self._action_high[i])
            if i % 3 == 1:  # femur: slight crouch
                qpos[i] = _clamp(_DEFAULT_FEMUR_ANGLE, lo, hi)
            elif i % 3 == 2:  # tibia: bent
                qpos[i] = _clamp(_DEFAULT_TIBIA_ANGLE, lo, hi)
        return qpos

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set default standing pose (within joint limits)
        self.data.qpos[self._joint_qpos_adr] = self._default_qpos()
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_torso_xy = np.asarray(self.data.xpos[self.torso_body_id, :2], dtype=np.float32)
        self._sample_command()

        obs  = self._get_obs()
        info = {"vcmd_xy": self.vcmd_xy.copy(), "wcmd_yaw": float(self.wcmd_yaw)}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self._action_low, self._action_high)

        # Apply action (position targets) and step physics
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated()
        self.step_count += 1
        truncated  = self.step_count >= self.max_steps

        info = {"vcmd_xy": self.vcmd_xy.copy(), "wcmd_yaw": float(self.wcmd_yaw)}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.6
            self.viewer.cam.elevation = -25
            self.viewer.cam.trackbodyid = self.torso_body_id
        self.viewer.cam.lookat[:] = self.data.xpos[self.torso_body_id]
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    env = HexapodEnvDirect(frame_skip=4)
    obs, _ = env.reset()
    print(f"obs shape : {obs.shape}")          # (48,)
    print(f"action dim: {env.action_space.shape}")  # (18,)

    # Benchmark FPS
    N = 2000
    t0 = time.perf_counter()
    for i in range(N):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    dt = time.perf_counter() - t0
    print(f"FPS (frame_skip=4): {N * 4 / dt:.0f}  ({N / dt:.0f} policy steps/s)")
    env.close()