"""
HexapodEnvDirect — Direct joint position control, no IK/kinematics.

Observation (54-dim):
    [0:4]   body quaternion (w, x, y, z)
    [4:7]   body angular velocity (roll, pitch, yaw)
    [7:10]  body linear velocity  (x, y, z)
    [10:28] joint positions (18)
    [28:46] joint velocities (18)
    [46:52] binary foot contacts (6)  fl, fr, rr, rl, mr, ml
    [52:54] velocity command (vcmd_x, vcmd_y)
"""

from typing import Optional, Tuple
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

JOINT_NAMES = [
    "coxa_fl", "femur_fl", "tibia_fl",
    "coxa_fr", "femur_fr", "tibia_fr",
    "coxa_rr", "femur_rr", "tibia_rr",
    "coxa_rl", "femur_rl", "tibia_rl",
    "coxa_mr", "femur_mr", "tibia_mr",
    "coxa_ml", "femur_ml", "tibia_ml",
]
N_JOINTS = len(JOINT_NAMES)  # 18

FOOT_BODY_NAMES = ["tibia_fl", "tibia_fr", "tibia_rr", "tibia_rl", "tibia_mr", "tibia_ml"]
N_FEET = len(FOOT_BODY_NAMES)  # 6

_DEFAULT_FEMUR_ANGLE =  0.25   # rad
_DEFAULT_TIBIA_ANGLE = -0.80   # rad

# ── Nominal torso height when standing ─────────────────────────────────────────
# Measured from model at default pose. Jumping detection uses this.
# Adjust if your robot stands at a different height.
_NOMINAL_TORSO_HEIGHT = 0.10   # metres — tune to match your XML


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
        flip_threshold_rad: float = 0.6,   # tighter: ~34° — was 1.0 (~57°)
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.model_path          = model_path or self.DEFAULT_MODEL_PATH
        self.frame_skip          = max(1, int(frame_skip))
        self.max_steps           = int(max_steps)
        self.command_mode        = command_mode
        self.vcmd_xy             = np.array(vcmd_xy, dtype=np.float32)
        self.wcmd_yaw            = float(wcmd_yaw)
        self.command_range_xy    = float(command_range_xy)
        self.command_range_yaw   = float(command_range_yaw)
        self.terminate_on_flip   = terminate_on_flip
        self.flip_threshold_rad  = float(flip_threshold_rad)

        if seed is not None:
            np.random.seed(seed)

        self.model  = mujoco.MjModel.from_xml_path(self.model_path)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self.imu_sensor_id = int(imu_id) if imu_id >= 0 else None

        self._joint_qpos_adr = np.array([
            int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in JOINT_NAMES], dtype=np.int32)

        self._joint_qvel_adr = np.array([
            int(self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in JOINT_NAMES], dtype=np.int32)

        # ── Foot contact setup ────────────────────────────────────────────────
        self._foot_body_ids = [
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n))
            for n in FOOT_BODY_NAMES
        ]
        self._foot_geom_sets = []
        for bid in self._foot_body_ids:
            s = set()
            if bid >= 0:
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == bid:
                        s.add(gid)
            self._foot_geom_sets.append(s)

        CONTACT_SENSOR_NAMES = [
            "contact_fl", "contact_fr", "contact_rr",
            "contact_rl", "contact_mr", "contact_ml"]
        self._contact_sensor_adrs = []
        self._use_contact_sensors = True
        for sn in CONTACT_SENSOR_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sn)
            if sid < 0:
                self._use_contact_sensors = False
                break
            self._contact_sensor_adrs.append(int(self.model.sensor_adr[sid]))

        # ── Action space ──────────────────────────────────────────────────────
        jlow  = np.empty(N_JOINTS, dtype=np.float32)
        jhigh = np.empty(N_JOINTS, dtype=np.float32)
        for i, jn in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            jlow[i]  = float(self.model.jnt_range[jid][0])
            jhigh[i] = float(self.model.jnt_range[jid][1])
        self._action_low  = jlow
        self._action_high = jhigh
        self.action_space = spaces.Box(low=jlow, high=jhigh, dtype=np.float32)

        # ── Observation space: 54-dim ─────────────────────────────────────────
        obs_dim = 4 + 3 + 3 + N_JOINTS + N_JOINTS + N_FEET + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ── Measure nominal torso height at default pose ───────────────────────
        # Used at runtime to detect jumping.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_adr] = self._default_qpos()
        mujoco.mj_forward(self.model, self.data)
        self._nominal_height = float(self.data.xpos[self.torso_body_id, 2])
        # Reset back to clean state
        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0

    # ─────────────────────────────────────────────────────────────────────────

    def _get_imu_quat(self):
        if self.imu_sensor_id is not None:
            adr = int(self.model.sensor_adr[self.imu_sensor_id])
            if int(self.model.sensor_dim[self.imu_sensor_id]) >= 4:
                return np.asarray(self.data.sensordata[adr:adr+4], dtype=np.float32)
        return np.asarray(self.data.xquat[self.torso_body_id], dtype=np.float32)

    def _get_foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(N_FEET, dtype=np.float32)
        if self._use_contact_sensors:
            for i, adr in enumerate(self._contact_sensor_adrs):
                contacts[i] = 1.0 if abs(float(self.data.sensordata[adr])) > 1e-6 else 0.0
            return contacts
        for c_idx in range(self.data.ncon):
            g1 = self.data.contact[c_idx].geom1
            g2 = self.data.contact[c_idx].geom2
            for fi, gs in enumerate(self._foot_geom_sets):
                if g1 in gs or g2 in gs:
                    contacts[fi] = 1.0
        return contacts

    def _get_obs(self) -> np.ndarray:
        quat     = self._get_imu_quat()
        ang_vel  = self.data.qvel[3:6].astype(np.float32)
        lin_vel  = self.data.qvel[0:3].astype(np.float32)
        jpos     = self.data.qpos[self._joint_qpos_adr].astype(np.float32)
        jvel     = self.data.qvel[self._joint_qvel_adr].astype(np.float32)
        contacts = self._get_foot_contacts()
        cmd      = np.array([self.vcmd_xy[0], self.vcmd_xy[1]], dtype=np.float32)
        return np.concatenate([quat, ang_vel, lin_vel, jpos, jvel, contacts, cmd])

    def _sample_command(self):
        if self.command_mode == "random":
            self.vcmd_xy = np.random.uniform(
                -self.command_range_xy, self.command_range_xy, size=(2,)
            ).astype(np.float32)
            self.wcmd_yaw = float(
                np.random.uniform(-self.command_range_yaw, self.command_range_yaw))

    def _compute_reward(self, action: np.ndarray) -> float:
        # ── Body velocity (local frame) ───────────────────────────────────────
        body_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_body_id,
            body_vel, 1)
        vx_actual   = float(body_vel[0])
        vy_actual   = float(body_vel[1])
        vz_body     = float(body_vel[2])
        wyaw_actual = float(body_vel[5])

        # ── 1. Velocity tracking — asymmetric with dead zone ─────────────────
        # Problem: small negative velocity (from gait oscillation) is normal.
        # Symmetric reward punishes it equally to being far from target,
        # so robot learns "stay still" is safer than "try to walk and risk v<0".
        #
        # Solution: split into two terms per axis:
        #   (a) Progress term  — reward moving in the right direction at all.
        #       Uses dot product sign: positive if v and v_tar same direction.
        #       Small negative v during swing phase still gets partial reward.
        #   (b) Tracking term  — reward being close to the exact target speed.
        #       Only kicks in when already moving in correct direction.
        #
        # Net effect: small negative v → small positive progress reward (not 0).
        # Staying still → 0 progress reward.  Moving at v_tar → max reward.

        vtar_x = float(self.vcmd_xy[0])
        vtar_y = float(self.vcmd_xy[1])

        def _vel_reward(v_actual: float, v_tar: float) -> float:
            if abs(v_tar) < 1e-3:
                # target is ~zero: just penalise moving away from zero
                return -abs(v_actual) * 0.5

            # (a) Progress: how much velocity is in the right direction
            #     = v_actual projected onto v_tar direction, normalised to [0,1]
            #     Clipped so large opposite velocity still gives -1 max
            direction = v_tar / abs(v_tar)          # +1 or -1
            progress  = v_actual * direction        # positive = correct dir
            r_progress = np.tanh(progress * 3.0)   # smooth, [-1, +1]
            #   tanh(0)   = 0   → standing still = 0 reward
            #   tanh(-0.1*3) ≈ -0.29  → small negative v = small penalty
            #   tanh(+0.05*3) ≈ +0.14 → tiny positive v already rewarded
            #   tanh(+1*3)   ≈ +1.0  → at or above target = full reward

            # (b) Tracking: bonus for being close to exact target
            #     Only adds on top when already in correct direction
            r_track = 1.0 / (abs(v_actual - v_tar) + 1.0) - 1.0 / (abs(v_tar) + 1.0)

            return 0.6 * r_progress + 0.4 * r_track

        r_vel = _vel_reward(vx_actual, vtar_x) + _vel_reward(vy_actual, vtar_y)

        # ── 2. Yaw tracking ───────────────────────────────────────────────────
        yaw_err = self.wcmd_yaw - wyaw_actual
        r_yaw = float(np.exp(-2.0 * yaw_err**2))

        # ── 3. Height penalty — ANTI-JUMP ─────────────────────────────────────
        # Penalise torso height deviating from nominal standing height.
        # Jumping raises torso → large positive deviation → heavy penalty.
        # Crouching raises penalty too — robot learns to stay at correct height.
        torso_z = float(self.data.xpos[self.torso_body_id, 2])
        height_dev = torso_z - self._nominal_height
        r_height = -3.0 * height_dev**2   # strong: coefficient 3.0

        # ── 4. Vertical velocity penalty — ANTI-JUMP ─────────────────────────
        # Jumping produces large positive vz. Penalise it hard.
        r_vz = -2.0 * vz_body**2

        # ── 5. Posture — penalise roll/pitch ──────────────────────────────────
        rpy = _quat_wxyz_to_rpy(self._get_imu_quat())
        r_posture = -0.3 * float(rpy[0]**2 + rpy[1]**2)

        # ── 6. Contact reward — reward tripod pattern, punish all-feet-off ────
        contacts = self._get_foot_contacts()
        n_contact = float(np.sum(contacts))
        # Peak at 3 feet grounded (tripod). Extra penalty when ALL feet are off
        # ground simultaneously — that is a jump.
        r_contact = -0.1 * abs(n_contact - 3.0)
        if n_contact == 0:
            r_contact -= 0.5   # hard penalty for airborne

        # ── 7. Energy ─────────────────────────────────────────────────────────
        jvel = self.data.qvel[self._joint_qvel_adr].astype(np.float32)
        r_energy = -1e-4 * float(np.sum(jvel**2))

        # ── 8. Alive bonus ────────────────────────────────────────────────────
        r_alive = 0.1

        total = (
            2.0 * r_vel        # primary: velocity tracking
            + 0.3 * r_yaw
            + r_height         # anti-jump 1: torso height deviation
            + r_vz             # anti-jump 2: vertical velocity
            + r_posture
            + r_contact        # gait structure: tripod preferred
            + r_energy
            + r_alive
        )
        return float(total)

    def _is_terminated(self) -> bool:
        if not self.terminate_on_flip:
            return False
        rpy = _quat_wxyz_to_rpy(self._get_imu_quat())
        if bool(abs(float(rpy[0])) > self.flip_threshold_rad or
                abs(float(rpy[1])) > self.flip_threshold_rad):
            return True
        # Also terminate if robot jumps too high (> 2x nominal height)
        torso_z = float(self.data.xpos[self.torso_body_id, 2])
        if torso_z > self._nominal_height * 2.5:
            return True
        return False

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

        self.step_count = 0
        self._sample_command()

        return self._get_obs(), {"vcmd_xy": self.vcmd_xy.copy(), "wcmd_yaw": float(self.wcmd_yaw)}

    def step(self, action: np.ndarray):
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            self._action_low, self._action_high)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated()
        self.step_count += 1
        truncated  = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, \
               {"vcmd_xy": self.vcmd_xy.copy(), "wcmd_yaw": float(self.wcmd_yaw)}

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
    print(f"obs shape       : {obs.shape}")          # (54,)
    print(f"action dim      : {env.action_space.shape}")
    print(f"nominal height  : {env._nominal_height:.4f} m")
    N = 2000
    t0 = time.perf_counter()
    for i in range(N):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            env.reset()
    dt = time.perf_counter() - t0
    print(f"FPS: {N * 4 / dt:.0f}")
    env.close()