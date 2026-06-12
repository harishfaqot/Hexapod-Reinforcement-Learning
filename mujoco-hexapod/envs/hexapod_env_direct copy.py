"""
HexapodEnvDirect — Direct joint position control, no IK/kinematics.

Action  : 18 joint target positions (rad), clipped to joint limits
           Order: coxa_fl, femur_fl, tibia_fl,
                  coxa_fr, femur_fr, tibia_fr,
                  coxa_rr, femur_rr, tibia_rr,
                  coxa_rl, femur_rl, tibia_rl,
                  coxa_mr, femur_mr, tibia_mr,
                  coxa_ml, femur_ml, tibia_ml

Observation (54-dim):
    [0:4]   body quaternion (w, x, y, z)
    [4:7]   body angular velocity (roll, pitch, yaw)
    [7:10]  body linear velocity  (x, y, z)
    [10:28] joint positions (18)
    [28:46] joint velocities (18)
    [46:52] binary foot contacts (6) ← NEW: fl, fr, rr, rl, mr, ml
    [52:54] velocity command (vcmd_x, vcmd_y)

Total: 4 + 3 + 3 + 18 + 18 + 6 + 2 = 54 dims
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

# Foot tip body names — used for contact detection.
# Falls back to geom-based contact scan if body-based misses.
FOOT_BODY_NAMES = ["tibia_fl", "tibia_fr", "tibia_rr", "tibia_rl", "tibia_mr", "tibia_ml"]
N_FEET = len(FOOT_BODY_NAMES)

_DEFAULT_FEMUR_ANGLE = 0.25   # rad
_DEFAULT_TIBIA_ANGLE = -0.8   # rad


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _quat_wxyz_to_rpy(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = np.sign(sinp) * np.pi/2 if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


class HexapodEnvDirect(gym.Env):
    metadata = {"render.modes": ["human"]}

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "assets",
        "hexapod_trossen_new.xml",
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        frame_skip: int = 4,
        max_steps: int = 1000,
        command_mode: str = "fixed",
        vcmd_xy: Tuple[float, float] = (0.2, 0.0),
        wcmd_yaw: float = 0.0,
        command_range_xy: float = 0.3,
        command_range_yaw: float = 0.5,
        terminate_on_flip: bool = True,
        flip_threshold_rad: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.model_path     = model_path or self.DEFAULT_MODEL_PATH
        self.frame_skip     = max(1, int(frame_skip))
        self.max_steps      = int(max_steps)
        self.command_mode   = command_mode
        self.vcmd_xy        = np.array(vcmd_xy, dtype=np.float32)
        self.wcmd_yaw       = float(wcmd_yaw)
        self.command_range_xy  = float(command_range_xy)
        self.command_range_yaw = float(command_range_yaw)
        self.terminate_on_flip = terminate_on_flip
        self.flip_threshold_rad = float(flip_threshold_rad)

        if seed is not None:
            np.random.seed(seed)

        # ── Load model ────────────────────────────────────────────────────────
        self.model  = mujoco.MjModel.from_xml_path(self.model_path)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        # ── Torso body ────────────────────────────────────────────────────────
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # ── IMU sensor (optional) ─────────────────────────────────────────────
        imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self.imu_sensor_id = int(imu_id) if imu_id >= 0 else None

        # ── Joint addresses ───────────────────────────────────────────────────
        self._joint_qpos_adr = np.array([
            int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]) for jn in JOINT_NAMES
        ], dtype=np.int32)

        self._joint_qvel_adr = np.array([
            int(self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]) for jn in JOINT_NAMES
        ], dtype=np.int32)

        # ── Foot contact bodies / geoms ───────────────────────────────────────
        # Try body-level contact first; fall back to geom scan.
        self._foot_body_ids = []
        for name in FOOT_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._foot_body_ids.append(int(bid))  # -1 if not found

        # Precompute geom → foot index map for fast contact scanning
        # Geoms belonging to each foot body (including child geoms)
        self._foot_geom_sets = []
        for bid in self._foot_body_ids:
            geom_set = set()
            if bid >= 0:
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == bid:
                        geom_set.add(gid)
            self._foot_geom_sets.append(geom_set)

        # ── Contact force sensors (if defined in XML) ─────────────────────────
        CONTACT_SENSOR_NAMES = [
            "contact_fl", "contact_fr", "contact_rr",
            "contact_rl", "contact_mr", "contact_ml",
        ]
        self._contact_sensor_adrs = []
        self._use_contact_sensors = True
        for sname in CONTACT_SENSOR_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid < 0:
                self._use_contact_sensors = False
                break
            self._contact_sensor_adrs.append(int(self.model.sensor_adr[sid]))

        # ── Action space ──────────────────────────────────────────────────────
        _jlow  = np.empty(N_JOINTS, dtype=np.float32)
        _jhigh = np.empty(N_JOINTS, dtype=np.float32)
        for i, jn in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            _jlow[i]  = float(self.model.jnt_range[jid][0])
            _jhigh[i] = float(self.model.jnt_range[jid][1])
        self._action_low  = _jlow
        self._action_high = _jhigh
        self.action_space = spaces.Box(low=_jlow, high=_jhigh, dtype=np.float32)

        # ── Observation space: 54-dim ─────────────────────────────────────────
        # quat(4) + ang_vel(3) + lin_vel(3) + jpos(18) + jvel(18) + contacts(6) + cmd(2)
        obs_dim = 4 + 3 + 3 + N_JOINTS + N_JOINTS + N_FEET + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Internal state ────────────────────────────────────────────────────
        self.step_count   = 0
        self.prev_torso_x = 0.0   # for displacement reward

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_imu_quat(self) -> np.ndarray:
        if self.imu_sensor_id is not None:
            adr = int(self.model.sensor_adr[self.imu_sensor_id])
            if int(self.model.sensor_dim[self.imu_sensor_id]) >= 4:
                return np.asarray(self.data.sensordata[adr:adr+4], dtype=np.float32)
        return np.asarray(self.data.xquat[self.torso_body_id], dtype=np.float32)

    def _get_foot_contacts(self) -> np.ndarray:
        """
        Returns binary (0/1) contact state for each of the 6 feet.

        Priority:
          1. XML touch/force sensors named contact_fl … contact_ml
          2. Geometry contact scan — check if any active contact involves
             a geom belonging to that foot's body
        """
        contacts = np.zeros(N_FEET, dtype=np.float32)

        if self._use_contact_sensors:
            for i, adr in enumerate(self._contact_sensor_adrs):
                contacts[i] = 1.0 if abs(float(self.data.sensordata[adr])) > 1e-6 else 0.0
            return contacts

        # Fallback: scan active contacts
        for c_idx in range(self.data.ncon):
            g1 = self.data.contact[c_idx].geom1
            g2 = self.data.contact[c_idx].geom2
            for foot_i, geom_set in enumerate(self._foot_geom_sets):
                if g1 in geom_set or g2 in geom_set:
                    contacts[foot_i] = 1.0
        return contacts

    def _get_obs(self) -> np.ndarray:
        quat     = self._get_imu_quat()                                          # (4,)
        ang_vel  = self.data.qvel[3:6].astype(np.float32)                        # (3,)
        lin_vel  = self.data.qvel[0:3].astype(np.float32)                        # (3,)
        jpos     = self.data.qpos[self._joint_qpos_adr].astype(np.float32)       # (18,)
        jvel     = self.data.qvel[self._joint_qvel_adr].astype(np.float32)       # (18,)
        contacts = self._get_foot_contacts()                                      # (6,)
        cmd      = np.array([self.vcmd_xy[0], self.vcmd_xy[1]], dtype=np.float32) # (2,)
        return np.concatenate([quat, ang_vel, lin_vel, jpos, jvel, contacts, cmd])

    def _sample_command(self):
        if self.command_mode == "random":
            self.vcmd_xy = np.random.uniform(
                -self.command_range_xy, self.command_range_xy, size=(2,)
            ).astype(np.float32)
            self.wcmd_yaw = float(
                np.random.uniform(-self.command_range_yaw, self.command_range_yaw)
            )

    def _compute_reward(self, action: np.ndarray) -> float:
        # ── Body velocity (local frame) ───────────────────────────────────────
        body_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_body_id,
            body_vel, 1,
        )
        vx_actual   = float(body_vel[0])
        vy_actual   = float(body_vel[1])
        vz_body     = float(body_vel[2])
        wyaw_actual = float(body_vel[5])

        # ── 1. Velocity reward — paper eq.(4) ────────────────────────────────
        # r_v = 1/(|v - v_tar| + 1) - 1/(v_tar + 1)
        # = 0 when v=0, peaks at 1 - 1/(v_tar+1) when v=v_tar
        vtar_x = float(self.vcmd_xy[0])
        vtar_y = float(self.vcmd_xy[1])
        r_vx = 1.0 / (abs(vx_actual - vtar_x) + 1.0) - 1.0 / (abs(vtar_x) + 1.0)
        r_vy = 1.0 / (abs(vy_actual - vtar_y) + 1.0) - 1.0 / (abs(vtar_y) + 1.0)
        r_vel = r_vx + r_vy  # 0 at rest, positive when moving toward target

        print(f"DEBUG: vx={vx_actual:.3f} (tar {vtar_x:.3f}) (vrx {r_vx:.3f})  vy={vy_actual:.3f} (tar {vtar_y:.3f}) (vry {r_vy:.3f}) r_vel={r_vel:.3f}")

        # ── 2. Displacement bonus — directly reward forward progress ──────────
        # This is the critical signal for "jalan ditempat": if body doesn't
        # translate, this reward is exactly 0. Scales with actual displacement.
        torso_x_now = float(self.data.xpos[self.torso_body_id, 0])
        dx = torso_x_now - self.prev_torso_x
        self.prev_torso_x = torso_x_now
        # Reward forward motion, penalise backward motion
        r_displacement = np.clip(dx / (self.frame_skip * self.model.opt.timestep), -1.0, 1.0)

        # ── 3. Yaw heading — paper eq.(5) style ──────────────────────────────
        yaw_err = self.wcmd_yaw - wyaw_actual
        r_yaw = float(np.exp(-2.0 * yaw_err**2))

        # ── 4. Contact reward — reward alternating stance (not all feet up) ───
        # Tripod gait: 3 feet down, 3 up. Reward non-trivial contact patterns.
        contacts = self._get_foot_contacts()
        n_contact = float(np.sum(contacts))
        # Penalise all feet up (0) or all feet grounded (6) — both are bad.
        # Peak reward at 3 feet, which is tripod.
        r_contact = -abs(n_contact - 3.0) * 0.05

        # ── 5. Posture ────────────────────────────────────────────────────────
        rpy = _quat_wxyz_to_rpy(self._get_imu_quat())
        r_posture = -0.05 * float(rpy[0]**2 + rpy[1]**2)

        # ── 6. Vertical bounce penalty ────────────────────────────────────────
        r_bounce = -0.05 * vz_body**2

        # ── 7. Energy (joint velocity proxy) ─────────────────────────────────
        jvel = self.data.qvel[self._joint_qvel_adr].astype(np.float32)
        r_energy = -5e-5 * float(np.sum(jvel**2))

        # ── 8. Alive bonus ────────────────────────────────────────────────────
        r_alive = 0.05

        total = (
            1.5 * r_vel          # main velocity tracking signal
            + 2.0 * r_displacement  # direct body-translation signal — key fix
            + 0.3 * r_yaw
            + r_contact
            + r_posture
            + r_bounce
            + r_energy
            + r_alive
        )
        return float(total)

    def _is_terminated(self) -> bool:
        if not self.terminate_on_flip:
            return False
        rpy = _quat_wxyz_to_rpy(self._get_imu_quat())
        return bool(
            abs(float(rpy[0])) > self.flip_threshold_rad or
            abs(float(rpy[1])) > self.flip_threshold_rad
        )

    # ── Gym API ───────────────────────────────────────────────────────────────

    def _default_qpos(self) -> np.ndarray:
        qpos = np.zeros(N_JOINTS, dtype=np.float32)
        for i in range(N_JOINTS):
            lo, hi = float(self._action_low[i]), float(self._action_high[i])
            if i % 3 == 1:
                qpos[i] = _clamp(_DEFAULT_FEMUR_ANGLE, lo, hi)
            elif i % 3 == 2:
                qpos[i] = _clamp(_DEFAULT_TIBIA_ANGLE, lo, hi)
        return qpos

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_adr] = self._default_qpos()
        mujoco.mj_forward(self.model, self.data)

        self.step_count   = 0
        self.prev_torso_x = float(self.data.xpos[self.torso_body_id, 0])
        self._sample_command()

        obs  = self._get_obs()
        info = {"vcmd_xy": self.vcmd_xy.copy(), "wcmd_yaw": float(self.wcmd_yaw)}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            self._action_low, self._action_high
        )
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


if __name__ == "__main__":
    import time
    env = HexapodEnvDirect(frame_skip=4)
    obs, _ = env.reset()
    print(f"obs shape : {obs.shape}")         # expect (54,)
    print(f"action dim: {env.action_space.shape}")

    N = 2000
    t0 = time.perf_counter()
    for i in range(N):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    dt = time.perf_counter() - t0
    print(f"FPS: {N * 4 / dt:.0f}  ({N / dt:.0f} policy steps/s)")
    env.close()