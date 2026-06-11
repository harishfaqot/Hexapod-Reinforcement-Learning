from typing import Optional, Tuple
import os
import tkinter as tk
from tkinter import ttk

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvasTkAgg = Figure = None

# ─── Constants ────────────────────────────────────────────────────────────────

LEG_ORDER = ["fl", "ml", "rl", "fr", "mr", "rr"]

LEG_BASE_POSITIONS = np.array([
    [0.12, 0.06, 0.0], [0.00, 0.10, 0.0], [-0.12, 0.06, 0.0],
    [0.12, -0.06, 0.0], [0.00, -0.10, 0.0], [-0.12, -0.06, 0.0],
], dtype=np.float32)

_coxa_dir = np.array([
    [0.03676, 0.03676], [0.0, 0.05200], [-0.03676, 0.03676],
    [0.03676, -0.03676], [0.0, -0.05200], [-0.03676, -0.03676],
], dtype=np.float32)

LEG_RADIAL_UNITS = _coxa_dir / np.linalg.norm(_coxa_dir, axis=1, keepdims=True)
LEG_TANGENT_UNITS = np.stack([LEG_RADIAL_UNITS[:, 1], -LEG_RADIAL_UNITS[:, 0]], axis=1).astype(np.float32)

IK_SWAP_XY = {0, 2, 3, 5}
IK_INVERT_X = {1, 4}

FOOT_HOME_RADIAL = 0.105
FOOT_HOME_Z      = -0.12
COXA_LENGTH      = 0.052
FEMUR_LENGTH     = 0.066
TIBIA_LENGTH     = 0.095
DISABLE_SOFT_LIMITS = True

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "hexapod_trossen.xml")

# ─── Servo hardware config ────────────────────────────────────────────────────
#
# AX-12/18: range 0–1023, center=512, total 300° → 1023/300° = 3.41 ticks/deg
# 1 rad = 195.38 ticks
#
# Servo IDs 1–18 map sequentially:
#   ID  1– 3 : fl  → coxa, femur, tibia
#   ID  4– 6 : ml  → coxa, femur, tibia
#   ID  7– 9 : rl  → coxa, femur, tibia
#   ID 10–12 : fr  → coxa, femur, tibia
#   ID 13–15 : mr  → coxa, femur, tibia
#   ID 16–18 : rr  → coxa, femur, tibia
#
# SERVO_DIRECTION: +1 = same as MuJoCo sign, -1 = physically inverted.
# Edit values below to match your robot. Index 0 = servo ID 1.
#
#              fl              ml              rl              fr              mr              rr
#           cx  fe  ti     cx  fe  ti     cx  fe  ti     cx  fe  ti     cx  fe  ti     cx  fe  ti
SERVO_DIRECTION = [
    +1, -1, +1,   +1, -1, +1,   +1, -1, +1,   +1, -1, +1,   +1, -1, +1,   +1, -1, +1,
]

# SERVO_OFFSET: add this angle (radians) to each servo BEFORE converting to ticks.
# Positive = shift tick value up; negative = shift down.
# Tune each value until the real joint matches the sim neutral pose.
SERVO_OFFSET = [
    -3.14, -2.04, 0.34,   -1.57, -2.04, 0.34,   -3.14, -2.04, 0.34,   -3.14, -2.04, 0.34,   -1.57, -2.04, 0.34,   -3.14, -2.04, 0.34,
]  # index 0 = servo ID 1

AX_CENTER        = 512
AX_TICKS_PER_RAD = 195.38   # 1023 / 300° * (180°/π)
AX_MIN_TICK      = 0
AX_MAX_TICK      = 1023
SERVO_IDS        = list(range(1, 19))


# ─── Servo angle converter ────────────────────────────────────────────────────

def rad_to_ax_tick(angle_rad: float, servo_idx: int) -> int:
    """Convert a joint angle in radians (MuJoCo frame) to an AX-series tick.
    servo_idx is 0-based (= servo_id - 1)."""
    direction = SERVO_DIRECTION[servo_idx]
    adjusted  = angle_rad + SERVO_OFFSET[servo_idx]
    tick = AX_CENTER + round(direction * adjusted * AX_TICKS_PER_RAD)
    return int(np.clip(tick, AX_MIN_TICK, AX_MAX_TICK))


class RealRobot:
    """
    Thin wrapper around DynamixelServo.
    Translates MuJoCo joint angles (radians) → AX ticks and sends via sync-write.

    Usage:
        robot = RealRobot(device="COM5")
        robot.enable()
        robot.send_angles([...18 floats in radians...])
        robot.disable()
        robot.close()
    """

    def __init__(self, device: str = "COM5", baudrate: int = 1_000_000):
        from servo import DynamixelServo
        self._servo   = DynamixelServo(device_name=device, baudrate=baudrate)
        self._enabled = False
        print(f"[RealRobot] connected on {device}")

    def enable(self):
        self._servo.enable_torque(SERVO_IDS)
        self._enabled = True
        print("[RealRobot] torque ON")

    def disable(self):
        self._servo.disable_torque(SERVO_IDS)
        self._enabled = False
        print("[RealRobot] torque OFF")

    def send_angles(self, angles_rad: list):
        """18 joint angles in radians, ordered by servo ID (1–18). Skips if torque is off."""
        if not self._enabled:
            return
        ticks = [rad_to_ax_tick(a, i) for i, a in enumerate(angles_rad)]
        self._servo.write(SERVO_IDS, ticks)
        print(f"[RealRobot] sent angles (ticks): {ticks}")  

    def go_home(self):
        """Send all servos to center (512 ticks = mechanical neutral)."""
        if not self._enabled:
            self.enable()
        self._servo.write(SERVO_IDS, [AX_CENTER] * 18)

    def close(self):
        if self._enabled:
            self.disable()
        self._servo.close()
        print("[RealRobot] port closed")


# ─── Math helpers ─────────────────────────────────────────────────────────────

def _clamp(x, lo, hi): return max(lo, min(hi, x))

def _quat_to_rpy(q):
    w, x, y, z = [float(v) for v in q]
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = np.sign(sinp)*np.pi/2 if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)

def _angle_diff(target, current):
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


# ─── Kinematics ───────────────────────────────────────────────────────────────

def inverse_kinematics(x, y, z):
    coxa_angle  = np.pi/2 + np.arctan2(x, y)
    r           = np.sqrt(x*x + y*y) - COXA_LENGTH
    d           = _clamp(np.sqrt(r*r + z*z), 1e-6, FEMUR_LENGTH + TIBIA_LENGTH - 1e-6)
    a1          = np.arctan2(z, r + 1e-9)
    cA          = _clamp((d*d + FEMUR_LENGTH**2 - TIBIA_LENGTH**2) / (2*d*FEMUR_LENGTH), -1, 1)
    femur_angle = np.pi/2 - (np.arccos(cA) + a1)
    cB          = _clamp((FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - d*d) / (2*FEMUR_LENGTH*TIBIA_LENGTH), -1, 1)
    return float(coxa_angle), float(femur_angle), float(np.pi - np.arccos(cB))

def _foot_home_offset(leg_i):
    r = LEG_RADIAL_UNITS[leg_i]
    return np.array([r[0]*FOOT_HOME_RADIAL, r[1]*FOOT_HOME_RADIAL, FOOT_HOME_Z], dtype=np.float32)

def _to_leg_frame(target, leg_base, leg_i):
    rel = np.asarray(target, dtype=np.float32) - np.asarray(leg_base, dtype=np.float32)
    x   = float(np.dot(rel[:2], LEG_TANGENT_UNITS[leg_i]))
    y   = float(np.dot(rel[:2], LEG_RADIAL_UNITS[leg_i]))
    if leg_i in IK_SWAP_XY: x, y = y, x
    if leg_i in IK_INVERT_X: x = -x
    return x, y, float(rel[2])

def body_kinematics(pos, orient_deg):
    x, y, z          = pos
    roll, pitch, yaw = [np.radians(v) for v in orient_deg]
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ], dtype=np.float32)
    return [tuple((R @ b + [x, y, z]).tolist()) for b in LEG_BASE_POSITIONS]

def generate_movement(t, freq_hz, phase_lag, step_height, vx, vy, vrot, leg_i, duty):
    phase  = (2*np.pi*freq_hz*t + leg_i*phase_lag) % (2*np.pi)
    p      = phase / (2*np.pi)
    angle  = np.pi/2 + np.arctan2(LEG_BASE_POSITIONS[leg_i][0], LEG_BASE_POSITIONS[leg_i][1])
    ox, oy = vrot*np.sin(angle), vrot*np.cos(angle)
    duty   = _clamp(duty, 0.05, 0.95)
    if p < duty:
        s = p / duty
        x, y, z = (vx+ox)*(0.5-s), (vy+oy)*(0.5-s), 0.0
    else:
        s = (p - duty) / (1 - duty)
        x, y, z = (vx+ox)*(s-0.5), (vy+oy)*(s-0.5), step_height*np.sin(np.pi*s)
    if vx == vy == vrot == 0: z = 0.0
    return np.array([x, y, z], dtype=np.float32)

def compute_heading_control(desired_deg, current_yaw_rad, vx, vy, k_p=0.003, max_rate=0.15):
    if vx == vy == 0 and desired_deg == 0: return 0.0
    err = -(desired_deg - np.degrees(current_yaw_rad) + 180) % 360 - 180
    return _clamp(k_p * err, -max_rate, max_rate)


# ─── Actuator / joint calibration ─────────────────────────────────────────────

def _build_actuator_map(model, nu):
    name_to_idx = {}
    for i in range(nu):
        jid = int(model.actuator_trnid[i][0])
        if jid >= 0:
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if n: name_to_idx[n] = i
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n: name_to_idx[n] = i
    result = {}
    for leg in LEG_ORDER:
        result[leg] = {}
        for jnt in ["coxa", "femur", "tibia"]:
            found = next((name_to_idx[n] for n in [
                f"{jnt}_{leg}", f"{leg}_{jnt}",
                f"{jnt}_{leg}_motor", f"{leg}_{jnt}_motor",
            ] if n in name_to_idx), None)
            result[leg][jnt] = found
    return result

def _compute_neutral_ik():
    neutral = {}
    for i, leg in enumerate(LEG_ORDER):
        base     = np.zeros(3, dtype=np.float32)
        target   = base + _foot_home_offset(i)
        x, y, z  = _to_leg_frame(target, base, i)
        c, f, t  = inverse_kinematics(x, y, z)
        neutral[leg] = {"coxa": c, "femur": f, "tibia": t}
    return neutral

def _build_joint_calibration(model, data, nu):
    act_map    = _build_actuator_map(model, nu)
    neutral_ik = _compute_neutral_ik()
    qpos0      = np.asarray(data.qpos, dtype=np.float32)
    calib      = {}
    for leg in LEG_ORDER:
        calib[leg] = {}
        for jnt in ["coxa", "femur", "tibia"]:
            idx = act_map[leg][jnt]
            if idx is None:
                calib[leg][jnt] = None
                continue
            jid     = int(model.actuator_trnid[idx][0])
            adr     = int(model.jnt_qposadr[jid])
            limited = int(model.jnt_limited[jid]) == 1 and not DISABLE_SOFT_LIMITS
            lo, hi  = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])) if limited else (-np.inf, np.inf)
            calib[leg][jnt] = {
                "actuator_idx":       idx,
                "neutral_joint_qpos": float(qpos0[adr]),
                "nominal_ik":         float(neutral_ik[leg][jnt]),
                "low": lo, "high": hi,
                "delta_sign": 1.0,
            }
    return calib

def _apply_ik_to_ctrl(ctrl, calib, leg, coxa, femur, tibia):
    for jnt, val in [("coxa", coxa), ("femur", femur), ("tibia", tibia)]:
        c = calib[leg][jnt]
        if c is None: continue
        delta = (float(val) - c["nominal_ik"]) * c["delta_sign"]
        ctrl[c["actuator_idx"]] = _clamp(c["neutral_joint_qpos"] + delta, c["low"], c["high"])

def compute_gait_ctrl(data, calib, t, vx, vy, v_rot, step_height, duty,
                      cpg_hz, phase_lag, body_pos=(0,0,0), body_orient=(0,0,0)):
    ctrl         = np.asarray(data.ctrl.copy(), dtype=np.float32)
    body_leg_pos = body_kinematics(body_pos, body_orient)
    for i, leg in enumerate(LEG_ORDER):
        mv       = generate_movement(t, cpg_hz, phase_lag, step_height, vx, vy, v_rot, i, duty)
        leg_base = np.array(body_leg_pos[i], dtype=np.float32)
        target   = leg_base + _foot_home_offset(i) + mv
        x, y, z  = _to_leg_frame(target, LEG_BASE_POSITIONS[i], i)
        _apply_ik_to_ctrl(ctrl, calib, leg, *inverse_kinematics(x, y, z))
    return ctrl

def _ctrl_to_servo_angles(ctrl, calib) -> list:
    """
    Extract 18 IK joint angles (radians, MuJoCo frame) from a ctrl vector,
    in servo-ID order: fl_coxa, fl_femur, fl_tibia, ml_coxa, …, rr_tibia.
    """
    angles = [0.0] * 18
    for leg_i, leg in enumerate(LEG_ORDER):
        for jnt_i, jnt in enumerate(["coxa", "femur", "tibia"]):
            c = calib[leg][jnt]
            if c is None: continue
            delta_qpos        = float(ctrl[c["actuator_idx"]]) - c["neutral_joint_qpos"]
            ik_angle          = c["nominal_ik"] + delta_qpos / c["delta_sign"]
            angles[leg_i*3 + jnt_i] = ik_angle
    return angles


# ─── HexapodEnv ───────────────────────────────────────────────────────────────

class HexapodEnv(gym.Env):
    """
    Hexapod MuJoCo environment.
    Pass real_robot=RealRobot(...) to mirror every joint command to the physical robot.

    Direct IK test (sim + real):
        robot = RealRobot("COM5")
        env   = HexapodEnv(randomize_spawn=False, real_robot=robot)
        env.reset()
        ui = HexapodTkUI(env); ui.run()

    RL training (sim only):
        env = HexapodEnv()
        obs, _ = env.reset()
        obs, reward, term, trunc, info = env.step(action)
    """

    metadata = {"render.modes": ["human"]}
    DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH

    def __init__(
        self,
        model_path:           Optional[str]       = None,
        frame_skip:           int                 = 1,
        max_steps:            int                 = 1000,
        command_mode:         str                 = "fixed",
        vcmd_xy:              Tuple[float, float] = (0.1, 0.0),
        wcmd_yaw:             float               = 0.0,
        command_range_xy:     float               = 0.2,
        command_range_yaw:    float               = 0.6,
        seed:                 Optional[int]       = None,
        randomize_spawn:      bool                = True,
        spawn_range_xy:       float               = 10.0,
        randomize_spawn_yaw:  bool                = True,
        stuck_limit:          int                 = 40,
        stuck_penalty:        float               = -50.0,
        enable_debug:         bool                = False,
        real_robot:           Optional[RealRobot] = None,
    ):
        super().__init__()
        self.model      = mujoco.MjModel.from_xml_path(model_path or self.DEFAULT_MODEL_PATH)
        self.data       = mujoco.MjData(self.model)
        self.viewer     = None
        self.frame_skip = max(1, int(frame_skip))
        self.nu         = int(self.model.nu)

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.body_id = int(body_id) if body_id >= 0 else 1
        imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self.imu_sensor_id = int(imu_id) if imu_id >= 0 else None

        self.camera_follow = True
        self.joint_calib   = _build_joint_calibration(self.model, self.data, self.nu)
        self.real_robot: Optional[RealRobot] = real_robot

        self.max_steps           = int(max_steps)
        self.command_mode        = command_mode
        self.vcmd_xy             = np.array(vcmd_xy, dtype=np.float32)
        self.wcmd_yaw            = float(wcmd_yaw)
        self.command_range_xy    = float(command_range_xy)
        self.command_range_yaw   = float(command_range_yaw)
        self.randomize_spawn     = bool(randomize_spawn)
        self.spawn_range_xy      = float(spawn_range_xy)
        self.randomize_spawn_yaw = bool(randomize_spawn_yaw)
        self.stuck_limit         = max(1, int(stuck_limit))
        self.stuck_penalty       = float(stuck_penalty)
        self.enable_debug        = bool(enable_debug)
        if seed is not None: np.random.seed(seed)

        act_low  = np.array([0.03, 0.3, -0.05, -0.05, -0.05, -30, -30, -30], dtype=np.float32)
        act_high = np.array([0.10, 0.6,  0.05,  0.05,  0.05,  30,  30,  30], dtype=np.float32)
        self._act_scale        = 0.5 * (act_high - act_low)
        self._act_bias         = 0.5 * (act_high + act_low)
        self.action_space      = spaces.Box(low=-np.ones(8, dtype=np.float32), high=np.ones(8, dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.step_count    = 0
        self.neg_dir_count = 0
        self.last_action   = None
        self.ui: Optional["HexapodTkUI"] = None

    # ── Core ──

    def get_imu_quat(self):
        if self.imu_sensor_id is not None:
            adr = int(self.model.sensor_adr[self.imu_sensor_id])
            if int(self.model.sensor_dim[self.imu_sensor_id]) >= 4:
                return np.asarray(self.data.sensordata[adr:adr+4], dtype=np.float32)
        return np.asarray(self.data.xquat[self.body_id], dtype=np.float32)

    def get_imu_rpy(self): return _quat_to_rpy(self.get_imu_quat())

    def _mj_step(self, ctrl):
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def _send_to_real(self, ctrl):
        """Mirror ctrl vector to physical robot. Silent on error so sim keeps running."""
        if self.real_robot is None or not self.real_robot._enabled:
            return
        try:
            self.real_robot.send_angles(_ctrl_to_servo_angles(ctrl, self.joint_calib))
        except Exception as e:
            print(f"[RealRobot] send error: {e}")

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 1.6
            self.viewer.cam.elevation = -25
        if self.camera_follow:
            self.viewer.cam.trackbodyid = self.body_id
            self.viewer.cam.lookat[:]   = self.data.xpos[self.body_id]
        self.viewer.sync()

    def close(self):
        if self.viewer:     self.viewer.close(); self.viewer = None
        if self.real_robot: self.real_robot.close()

    # ── RL interface ──

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.randomize_spawn:
            r = self.spawn_range_xy
            self.data.qpos[0] = float(np.random.uniform(-r, r))
            self.data.qpos[1] = float(np.random.uniform(-r, r))
        if self.randomize_spawn_yaw:
            yaw = float(np.random.uniform(-np.pi, np.pi))
            self.data.qpos[3] = np.cos(yaw/2); self.data.qpos[4:7] = [0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0; self.neg_dir_count = 0; self.last_action = None
        if self.command_mode == "random":
            self.vcmd_xy  = np.random.uniform(-self.command_range_xy, self.command_range_xy, 2).astype(np.float32)
            self.wcmd_yaw = float(np.random.uniform(-self.command_range_yaw, self.command_range_yaw))
        return self.get_imu_quat(), {"vcmd_xy": self.vcmd_xy.copy(), "wcmd_yaw": self.wcmd_yaw}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        raw    = (self._act_bias + action * self._act_scale).astype(np.float32)
        self.last_action = raw.copy()
        step_height, duty, x, y, z, roll, pitch, yaw_body = raw

        ui        = self.ui
        cpg_hz    = max(0.1, float(ui.cpg_slider.get()))   if ui and hasattr(ui, "cpg_slider")    else 1.0
        phase_lag = float(ui.gait_slider.get())             if ui and hasattr(ui, "gait_slider")   else float(np.pi)
        desired_heading = float(ui.heading_slider.get())    if ui and hasattr(ui, "heading_slider") else 0.0

        vx, vy = float(self.vcmd_xy[0]), float(self.vcmd_xy[1])
        v_rot  = compute_heading_control(desired_heading, float(self.get_imu_rpy()[2]), vx, vy)

        ctrl = compute_gait_ctrl(
            self.data, self.joint_calib, float(self.data.time),
            vx, vy, v_rot, step_height, duty, cpg_hz, phase_lag,
            body_pos=(x, y, z), body_orient=(roll, pitch, yaw_body),
        )
        self._mj_step(ctrl)
        self._send_to_real(ctrl)

        obs    = self.get_imu_quat()
        reward, stuck = self._compute_reward(vx, vy, desired_heading)
        self.step_count += 1
        rpy        = self.get_imu_rpy()
        terminated = stuck or abs(float(rpy[0])) > 1.5 or abs(float(rpy[1])) > 1.5
        truncated  = self.step_count >= self.max_steps

        if terminated:
            obs, info = self.reset()
            info["stuck_detected"] = bool(stuck)
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, {
            "vcmd_xy": self.vcmd_xy.copy(), "stuck_detected": bool(stuck),
        }

    def _compute_reward(self, vx_cmd, vy_cmd, desired_heading_deg):
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, vel, 1)
        dir_r = 10 * (vx_cmd * vel[0] + vy_cmd * vel[1])
        if dir_r < -0.01:
            dir_r *= 10; self.neg_dir_count += 1
            if self.enable_debug: print(f"neg dir r={dir_r:.3f} count={self.neg_dir_count}")
        elif dir_r > 0.01:
            self.neg_dir_count = max(0, self.neg_dir_count - 1)
        stuck          = self.neg_dir_count >= self.stuck_limit
        roll, pitch, yaw = [float(v) for v in self.get_imu_rpy()]
        yaw_err        = _angle_diff(np.deg2rad(desired_heading_deg), yaw)
        reward         = dir_r + np.exp(-3*(roll**2 + pitch**2)) + np.exp(-2*yaw_err**2)
        if stuck:
            reward += self.stuck_penalty
            if self.enable_debug: print(f"Stuck! penalty={self.stuck_penalty}")
        return reward, stuck


# ─── Tkinter UI ───────────────────────────────────────────────────────────────

def _slider(parent, label, lo, hi, init, row, col, res=0.001, length=250):
    f = tk.Frame(parent); f.grid(row=row, column=col, padx=8, pady=4, sticky="we")
    tk.Label(f, text=label, anchor="w").pack(fill="x")
    s = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal", length=length)
    s.set(init); s.pack(fill="x")
    return s


class _IMUPlot:
    def __init__(self, parent, names, colors, y_range, max_pts=240):
        if Figure is None: raise RuntimeError("matplotlib required")
        self.names, self.max_pts = list(names), int(max_pts)
        self.vals   = {n: [] for n in names}
        self.fig    = Figure(figsize=(5.2, 3.2), dpi=100)
        self.ax     = self.fig.add_subplot(111)
        self.ax.set_ylim(*y_range); self.ax.set_xlim(0, max_pts-1)
        self.ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4); self.ax.set_xticks([])
        self.lines  = {n: self.ax.plot([], [], color=c, linewidth=1.5)[0] for n, c in zip(names, colors)}
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()

    def update(self, vals):
        for n, v in zip(self.names, vals):
            self.vals[n].append(float(v))
            if len(self.vals[n]) > self.max_pts: self.vals[n].pop(0)
        for n in self.names:
            self.lines[n].set_data(range(len(self.vals[n])), self.vals[n])
        self.ax.set_xlim(0, self.max_pts-1)

    def draw(self): self.canvas.draw_idle()


class HexapodTkUI:
    """
    Tkinter controller. Drives MuJoCo sim and real robot simultaneously.
    Adds Torque ON/OFF button and Go Home button.
    """

    def __init__(self, env: HexapodEnv):
        self.env  = env
        self.root = tk.Tk()
        self.root.title("Hexapod Controller — Sim + Real")
        self.root.geometry("1100x800")
        self.root.grid_columnconfigure(0, weight=1); self.root.grid_rowconfigure(0, weight=1)
        self.root.lift(); self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))

        self.leg_offset_controls = {}
        self.joint_deg_controls  = {}
        self.control_mode        = tk.StringVar(value="ik")
        self.camera_follow_var   = tk.BooleanVar(value=True)

        self._build_ui()
        self.tick_ms    = 8
        self.tick_count = 0
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        nb = ttk.Notebook(self.root); nb.grid(row=0, column=0, sticky="nsew")
        t1, t2, t3 = tk.Frame(nb), tk.Frame(nb), tk.Frame(nb)
        nb.add(t1, text="Movement + IMU"); nb.add(t2, text="Leg offsets"); nb.add(t3, text="Foot params")
        for t in [t1, t2, t3]:
            t.grid_columnconfigure(0, weight=1); t.grid_columnconfigure(1, weight=1)

        # ── Movement tab ──
        tk.Label(t1, text="Movement controls", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(8,4), sticky="w")
        self.vx_slider          = _slider(t1, "vx",              -0.15, 0.15,      0.0,       1, 0)
        self.vy_slider          = _slider(t1, "vy",              -0.15, 0.15,      0.0,       2, 0)
        self.heading_slider     = _slider(t1, "heading (deg)",   -180,  180,       0.0,       3, 0, res=1.0)
        self.v_rot_slider       = _slider(t1, "v_rot (feedback)",-0.15, 0.15,      0.0,       4, 0)
        self.v_rot_slider.config(state="disabled")
        self.step_height_slider = _slider(t1, "step height",      0.0,  0.3,       0.05,      5, 0)
        self.cpg_slider         = _slider(t1, "cpg freq (Hz)",    0.1,  4.0,       1.0,       6, 0)
        self.gait_slider        = _slider(t1, "gait phase lag",   np.pi/3, np.pi,  np.pi,     7, 0)
        self.duty_slider        = _slider(t1, "duty factor",      0.2,  0.8,       0.5,       8, 0)

        tk.Label(t1, text="Body controls", font=("Arial", 11, "bold")).grid(
            row=0, column=1, columnspan=2, padx=8, pady=(8,4), sticky="w")
        self.x_slider   = _slider(t1, "pos x", -0.1, 0.1, 0.0, 1, 1)
        self.y_slider   = _slider(t1, "pos y", -0.1, 0.1, 0.0, 2, 1)
        self.z_slider   = _slider(t1, "pos z", -0.1, 0.1, 0.0, 3, 1)
        self.r_slider   = _slider(t1, "roll",  -60,  60,  0.0, 4, 1, res=0.1)
        self.p_slider   = _slider(t1, "pitch", -60,  60,  0.0, 5, 1, res=0.1)
        self.yaw_slider = _slider(t1, "yaw",   -60,  60,  0.0, 6, 1, res=0.1)

        # ── Control bar ──
        bar = tk.Frame(t1); bar.grid(row=9, column=0, columnspan=2, padx=8, pady=10, sticky="we")
        tk.Button(bar, text="Reset leg offsets", command=self._reset_leg_offsets).pack(side="left", padx=(0,6))
        tk.Button(bar, text="Reset sliders",     command=self._reset_sliders).pack(side="left", padx=(0,16))
        tk.Label(bar, text="Mode:").pack(side="left")
        tk.Radiobutton(bar, text="IK",        variable=self.control_mode, value="ik").pack(side="left")
        tk.Radiobutton(bar, text="Joint deg", variable=self.control_mode, value="joint").pack(side="left")
        tk.Checkbutton(bar, text="Camera follow", variable=self.camera_follow_var).pack(side="left", padx=(16,0))

        # ── Real robot bar ──
        rbar = tk.Frame(t1); rbar.grid(row=10, column=0, columnspan=2, padx=8, pady=(0,6), sticky="we")
        self._torque_btn = tk.Button(
            rbar, text="⚡ Torque ON", width=14,
            bg="#2a9d8f", fg="white", font=("Arial", 10, "bold"),
            command=self._toggle_torque,
        )
        self._torque_btn.pack(side="left", padx=(0,8))
        tk.Button(rbar, text="🏠 Go Home", width=10, command=self._go_home).pack(side="left", padx=(0,8))
        self._robot_status_var = tk.StringVar(value=self._status_text())
        tk.Label(rbar, textvariable=self._robot_status_var, fg="#555", font=("Arial", 9)).pack(side="left", padx=(8,0))

        # ── IMU ──
        tk.Label(t1, text="IMU telemetry", font=("Arial", 11, "bold")).grid(
            row=11, column=0, columnspan=2, padx=8, pady=(10,4), sticky="w")
        imu_f = tk.Frame(t1); imu_f.grid(row=12, column=0, columnspan=2, padx=8, pady=(0,6), sticky="we")
        imu_f.grid_columnconfigure(0, weight=1); imu_f.grid_columnconfigure(1, weight=1)
        self.imu_quat_var = tk.StringVar(value="quat: --")
        self.imu_rpy_var  = tk.StringVar(value="rpy: --")
        self.quat_plot = self.rpy_plot = None
        if Figure is not None:
            for col, (names, colors, yr, attr) in enumerate([
                (["w","x","y","z"], ["#FFD166","#EF476F","#06D6A0","#118AB2"], (-1.1, 1.1), "quat_plot"),
                (["roll","pitch","yaw"], ["#FFD166","#06D6A0","#118AB2"],       (-0.5, 0.5), "rpy_plot"),
            ]):
                f = tk.Frame(imu_f); f.grid(row=0, column=col, padx=6, pady=2, sticky="we")
                p = _IMUPlot(f, names, colors, yr); p.widget.pack(fill="both", expand=True)
                setattr(self, attr, p)
        tk.Label(imu_f, textvariable=self.imu_quat_var, anchor="w").grid(row=1, column=0, sticky="w", padx=2)
        tk.Label(imu_f, textvariable=self.imu_rpy_var,  anchor="w").grid(row=1, column=1, sticky="w", padx=2)

        # ── Leg offsets tab ──
        c = tk.Frame(t2); c.pack(fill="both", expand=True, padx=8, pady=8)
        tk.Label(c, text="Per-leg offset controls", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Label(c, text="fl  ml  rl  fr  mr  rr").pack(anchor="w", pady=(0,6))
        nb2 = ttk.Notebook(c); nb2.pack(fill="both", expand=True)
        for i, leg in enumerate(LEG_ORDER):
            tab = tk.Frame(nb2); nb2.add(tab, text=f"Leg {i+1}")
            tk.Label(tab, text=f"Leg {i+1} ({leg})", font=("Arial",10,"bold")).grid(row=0, column=0, columnspan=2, padx=8, pady=(6,2), sticky="w")
            tk.Label(tab, text="IK offset", font=("Arial",10,"bold")).grid(row=1, column=0, padx=8, sticky="w")
            tk.Label(tab, text="Joint deg", font=("Arial",10,"bold")).grid(row=1, column=1, padx=8, sticky="w")
            self.leg_offset_controls[leg] = {
                ax: _slider(tab, f"L{i+1} d{ax}", -0.08, 0.08, 0.0, 2+j, 0, length=190)
                for j, ax in enumerate("xyz")
            }
            self.joint_deg_controls[leg] = {
                jnt: _slider(tab, f"L{i+1} {jnt}", lo, hi, 0.0, 2+j, 1, res=0.5, length=190)
                for j, (jnt, lo, hi) in enumerate([("coxa",-90,90),("femur",-90,90),("tibia",-120,120)])
            }

        # ── Foot params tab ──
        tk.Label(t3, text="Foot home tuning", font=("Arial",11,"bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(8,4), sticky="w")
        self.foot_radial_slider = _slider(t3, "FOOT_HOME_RADIAL", 0.05, 0.18, FOOT_HOME_RADIAL, 1, 0)
        self.foot_z_slider      = _slider(t3, "FOOT_HOME_Z",     -0.20, -0.05, FOOT_HOME_Z,     1, 1)

    # ── Real robot controls ──

    def _status_text(self):
        r = self.env.real_robot
        if r is None:    return "Real robot: not connected"
        if r._enabled:   return "Real robot: TORQUE ON ✓"
        return "Real robot: connected, torque OFF"

    def _toggle_torque(self):
        r = self.env.real_robot
        if r is None:
            self._robot_status_var.set("Real robot: not connected"); return
        if r._enabled:
            r.disable()
            self._torque_btn.config(text="⚡ Torque ON",  bg="#2a9d8f")
        else:
            r.enable()
            self._torque_btn.config(text="🛑 Torque OFF", bg="#e76f51")
        self._robot_status_var.set(self._status_text())

    def _go_home(self):
        r = self.env.real_robot
        if r is None: return
        if not r._enabled: r.enable()
        r.go_home()

    # ── Slider resets ──

    def _reset_sliders(self):
        defaults = {
            self.cpg_slider: 1.0, self.gait_slider: float(np.pi), self.duty_slider: 0.5,
            self.step_height_slider: 0.05, self.foot_radial_slider: 0.105, self.foot_z_slider: -0.12,
        }
        for s in [self.vx_slider, self.vy_slider, self.heading_slider, self.v_rot_slider,
                  self.step_height_slider, self.cpg_slider, self.gait_slider, self.duty_slider,
                  self.x_slider, self.y_slider, self.z_slider, self.r_slider, self.p_slider,
                  self.yaw_slider, self.foot_radial_slider, self.foot_z_slider]:
            s.set(defaults.get(s, 0.0))

    def _reset_leg_offsets(self):
        for leg in LEG_ORDER:
            for ax in "xyz": self.leg_offset_controls[leg][ax].set(0.0)
            for jnt in ["coxa","femur","tibia"]: self.joint_deg_controls[leg][jnt].set(0.0)

    # ── Main loop ──

    def _tick(self):
        if not self.root.winfo_exists(): return
        env = self.env
        env.camera_follow = bool(self.camera_follow_var.get())

        global FOOT_HOME_RADIAL, FOOT_HOME_Z
        FOOT_HOME_RADIAL = float(self.foot_radial_slider.get())
        FOOT_HOME_Z      = float(self.foot_z_slider.get())

        calib = env.joint_calib
        ctrl  = np.asarray(env.data.ctrl.copy(), dtype=np.float32)

        if self.control_mode.get() == "joint":
            for leg in LEG_ORDER:
                for jnt in ["coxa","femur","tibia"]:
                    c = calib[leg][jnt]
                    if c is None: continue
                    target = _clamp(
                        c["neutral_joint_qpos"] + np.radians(float(self.joint_deg_controls[leg][jnt].get())) * c["delta_sign"],
                        c["low"], c["high"],
                    )
                    ctrl[c["actuator_idx"]] = target
        else:
            vx  = float(self.vx_slider.get()); vy = float(self.vy_slider.get())
            desired_heading = float(self.heading_slider.get())
            v_rot = compute_heading_control(desired_heading, float(env.get_imu_rpy()[2]), vx, vy)
            self.v_rot_slider.config(state="normal"); self.v_rot_slider.set(v_rot); self.v_rot_slider.config(state="disabled")

            bp = (float(self.x_slider.get()), float(self.y_slider.get()), float(self.z_slider.get()))
            bo = (float(self.r_slider.get()), float(self.p_slider.get()), float(self.yaw_slider.get()))

            ctrl = compute_gait_ctrl(
                env.data, calib, float(env.data.time),
                vx, vy, v_rot,
                float(self.step_height_slider.get()),
                float(self.duty_slider.get()),
                max(0.1, float(self.cpg_slider.get())),
                float(self.gait_slider.get()),
                body_pos=bp, body_orient=bo,
            )
            # Per-leg IK offsets
            body_leg_pos = body_kinematics(bp, bo)
            for i, leg in enumerate(LEG_ORDER):
                offset = np.array([float(self.leg_offset_controls[leg][ax].get()) for ax in "xyz"], dtype=np.float32)
                if np.any(offset != 0):
                    mv       = generate_movement(float(env.data.time), max(0.1, float(self.cpg_slider.get())),
                                                 float(self.gait_slider.get()), float(self.step_height_slider.get()),
                                                 vx, vy, v_rot, i, float(self.duty_slider.get()))
                    leg_base = np.array(body_leg_pos[i], dtype=np.float32)
                    target   = leg_base + _foot_home_offset(i) + mv + offset
                    x, y, z  = _to_leg_frame(target, LEG_BASE_POSITIONS[i], i)
                    _apply_ik_to_ctrl(ctrl, calib, leg, *inverse_kinematics(x, y, z))

        env._mj_step(ctrl)
        env._send_to_real(ctrl)   # ← simultaneous real robot mirror

        self.tick_count += 1
        if self.tick_count % 2 == 0:
            env.render()
            quat = env.get_imu_quat(); rpy = env.get_imu_rpy()
            if self.quat_plot: self.quat_plot.update(quat); self.quat_plot.draw()
            if self.rpy_plot:  self.rpy_plot.update(rpy);  self.rpy_plot.draw()
            self.imu_quat_var.set(f"quat: w={quat[0]:+.3f} x={quat[1]:+.3f} y={quat[2]:+.3f} z={quat[3]:+.3f}")
            self.imu_rpy_var.set( f"rpy:  r={rpy[0]:+.3f}  p={rpy[1]:+.3f}  y={rpy[2]:+.3f}")
            self._robot_status_var.set(self._status_text())

        self.root.after(self.tick_ms, self._tick)

    def _on_close(self):
        self.env.close()
        self.root.destroy()

    def run(self):
        self.root.after(0, self._tick)
        self.root.mainloop()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=True, action="store_true", help="Connect to real robot")
    parser.add_argument("--port", default="COM5",    help="Serial port  e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", default=1_000_000, type=int)
    args = parser.parse_args()

    real_robot = None
    if args.real:
        real_robot = RealRobot(device=args.port, baudrate=args.baud)

    env = HexapodEnv(randomize_spawn=False, real_robot=real_robot)
    env.reset()
    print(f"HexapodEnv ready | nu={env.nu} | real={'yes' if real_robot else 'no'}")

    ui     = HexapodTkUI(env)
    env.ui = ui
    ui.run()