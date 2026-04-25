import os
import sys

nexabots_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if nexabots_project_root not in sys.path:
    sys.path.insert(0, nexabots_project_root)

if os.name == "nt":
    mujoco_bin = os.path.join(os.path.expanduser("~"), ".mujoco", "mujoco210", "bin")
    if os.path.isdir(mujoco_bin):
        os.environ["PATH"] = mujoco_bin + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(mujoco_bin)
        except (AttributeError, FileNotFoundError):
            pass

from typing import Optional
import tkinter as tk
from tkinter import ttk

import gym
import mujoco_py
import numpy as np
from gym import spaces

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional dependency for research-grade plots
    FigureCanvasTkAgg = None
    Figure = None


LEG_ORDER = ["fl", "ml", "rl", "fr", "mr", "rr"]

# Leg base anchor positions taken directly from the MuJoCo XML body pos attributes.
LEG_BASE_POSITIONS = np.array([
    [0.12, 0.06, 0.0],
    [0.00, 0.10, 0.0],
    [-0.12, 0.06, 0.0],
    [0.12, -0.06, 0.0],
    [0.00, -0.10, 0.0],
    [-0.12, -0.06, 0.0],
], dtype=np.float32)

# Coxa direction vectors taken from the MuJoCo XML coxa geom fromto values.
LEG_COXA_DIRECTIONS = np.array([
    [0.03676, 0.03676],
    [0.0, 0.05200],
    [-0.03676, 0.03676],
    [0.03676, -0.03676],
    [0.0, -0.05200],
    [-0.03676, -0.03676],
], dtype=np.float32)

LEG_RADIAL_UNITS = LEG_COXA_DIRECTIONS / np.linalg.norm(LEG_COXA_DIRECTIONS, axis=1, keepdims=True)
LEG_TANGENT_UNITS = np.stack([LEG_RADIAL_UNITS[:, 1], -LEG_RADIAL_UNITS[:, 0]], axis=1).astype(np.float32)

# Legs with diagonal coxa orientation need x/y swapped in IK local frame.
IK_SWAP_XY_LEG_INDICES = {0, 2, 3, 5}

# Legs that need x-axis inversion in IK local frame.
IK_INVERT_X_LEG_INDICES = {1, 4}

# Nominal foot location in each leg-local frame.
FOOT_HOME_RADIAL = 0.105
FOOT_HOME_Z = -0.12

# Link lengths from the MuJoCo XML.
COXA_LENGTH = 0.052
FEMUR_LENGTH = 0.066
TIBIA_LENGTH = 0.095
DISABLE_SOFT_LIMITS = True


def _foot_home_offset_body(leg_index: int) -> np.ndarray:
    radial = LEG_RADIAL_UNITS[leg_index]
    return np.array([
        float(radial[0]) * FOOT_HOME_RADIAL,
        float(radial[1]) * FOOT_HOME_RADIAL,
        FOOT_HOME_Z,
    ], dtype=np.float32)


def _body_target_to_leg_ik_frame(target_body: np.ndarray, leg_base_body: np.ndarray, leg_index: int):
    rel = np.asarray(target_body, dtype=np.float32) - np.asarray(leg_base_body, dtype=np.float32)
    rel_xy = rel[:2]
    radial = LEG_RADIAL_UNITS[leg_index]
    tangent = LEG_TANGENT_UNITS[leg_index]
    x_local = float(np.dot(rel_xy, tangent))
    y_local = float(np.dot(rel_xy, radial))
    if leg_index in IK_SWAP_XY_LEG_INDICES:
        x_local, y_local = y_local, x_local
    if leg_index in IK_INVERT_X_LEG_INDICES:
        x_local = -x_local
    z_local = float(rel[2])
    return x_local, y_local, z_local


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_movement(
    t: float,
    freq_hz: float,
    phase_lag_rad: float,
    step_height: float,
    body_movement,
    leg_index: int,
    duty_factor: float,
):
    vx, vy, vrot = body_movement

    phase = (2.0 * np.pi * freq_hz * t + leg_index * phase_lag_rad) % (2.0 * np.pi)
    progress = phase / (2.0 * np.pi)

    angle = np.pi / 2.0 + np.arctan2(LEG_BASE_POSITIONS[leg_index][0], LEG_BASE_POSITIONS[leg_index][1])
    offset_x = vrot * np.sin(angle)
    offset_y = vrot * np.cos(angle)

    duty = _clamp(float(duty_factor), 0.05, 0.95)
    if progress < duty:
        p = progress / duty
        x = (vx + offset_x) * (0.5 - p)
        y = (vy + offset_y) * (0.5 - p)
        z = 0.0
    else:
        p = (progress - duty) / (1.0 - duty)
        x = (vx + offset_x) * (p - 0.5)
        y = (vy + offset_y) * (p - 0.5)
        z = step_height * np.sin(np.pi * p)

    if vx == 0.0 and vy == 0.0 and vrot == 0.0:
        z = 0.0

    return np.array([x, y, z], dtype=np.float32), progress


def inverse_kinematics(x: float, y: float, z: float):
    """Inverse kinematics matching the MuJoCo hexapod geometry."""
    coxa_angle = np.pi / 2.0 + np.arctan2(x, y)

    r_total = np.sqrt(x * x + y * y)
    r = r_total - COXA_LENGTH

    d = np.sqrt(r * r + z * z)
    d = _clamp(d, 1e-6, FEMUR_LENGTH + TIBIA_LENGTH - 1e-6)

    a1 = np.arctan2(z, r + 1e-9)

    cA = (d * d + FEMUR_LENGTH * FEMUR_LENGTH - TIBIA_LENGTH * TIBIA_LENGTH) / (2.0 * d * FEMUR_LENGTH)
    cA = _clamp(cA, -1.0, 1.0)
    A = np.arccos(cA)
    femur_angle = np.pi / 2.0 - (A + a1)

    cB = (FEMUR_LENGTH * FEMUR_LENGTH + TIBIA_LENGTH * TIBIA_LENGTH - d * d) / (2.0 * FEMUR_LENGTH * TIBIA_LENGTH)
    cB = _clamp(cB, -1.0, 1.0)
    B = np.arccos(cB)
    tibia_angle = np.pi - B

    return float(coxa_angle), float(femur_angle), float(tibia_angle)


def body_kinematics(body_position, body_orientation_deg):
    x_trans, y_trans, z_trans = body_position
    roll_deg, pitch_deg, yaw_deg = body_orientation_deg

    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    rotation_matrix = np.array([
        [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
        [-s_p, c_p * s_r, c_p * c_r],
    ], dtype=np.float32)

    leg_positions = []
    for leg_base in LEG_BASE_POSITIONS:
        rotated_leg_base = rotation_matrix.dot(np.array(leg_base, dtype=np.float32))
        leg_x = rotated_leg_base[0] + x_trans
        leg_y = rotated_leg_base[1] + y_trans
        leg_z = rotated_leg_base[2] + z_trans
        leg_positions.append((float(leg_x), float(leg_y), float(leg_z)))

    return leg_positions


def _make_slider(root, label, minimum, maximum, initial, row, col, resolution=0.001, length=250):
    frame = tk.Frame(root)
    frame.grid(row=row, column=col, padx=8, pady=4, sticky="we")
    tk.Label(frame, text=label, anchor="w").pack(fill="x")
    slider = tk.Scale(
        frame,
        from_=minimum,
        to=maximum,
        resolution=resolution,
        orient="horizontal",
        length=length,
    )
    slider.set(initial)
    slider.pack(fill="x")
    return slider


def _make_leg_offset_controls(root):
    container = tk.Frame(root)
    container.grid(row=0, column=0, columnspan=2, padx=8, pady=(12, 4), sticky="nsew")

    tk.Label(container, text="Per-leg offset controls", font=("Arial", 11, "bold")).pack(anchor="w")
    tk.Label(container, text="Leg 1=fl  Leg 2=ml  Leg 3=rl  Leg 4=fr  Leg 5=mr  Leg 6=rr").pack(anchor="w", pady=(0, 6))

    notebook = ttk.Notebook(container)
    notebook.pack(fill="both", expand=True)

    controls = {}
    joint_deg_controls = {}
    for i, leg in enumerate(LEG_ORDER):
        leg_num = i + 1
        tab = tk.Frame(notebook)
        notebook.add(tab, text=f"Leg {leg_num}")

        tk.Label(tab, text=f"Leg {leg_num} ({leg})", font=("Arial", 10, "bold")).grid(
            row=0,
            column=0,
            columnspan=2,
            padx=8,
            pady=(6, 2),
            sticky="w",
        )

        tk.Label(tab, text="IK control", font=("Arial", 10, "bold")).grid(
            row=1,
            column=0,
            padx=8,
            pady=(4, 0),
            sticky="w",
        )

        tk.Label(tab, text="Joint control (degrees)", font=("Arial", 10, "bold")).grid(
            row=1,
            column=1,
            padx=8,
            pady=(4, 0),
            sticky="w",
        )

        x_slider = _make_slider(tab, f"L{leg_num} dx", -0.08, 0.08, 0.0, row=2, col=0, length=190)
        y_slider = _make_slider(tab, f"L{leg_num} dy", -0.08, 0.08, 0.0, row=3, col=0, length=190)
        z_slider = _make_slider(tab, f"L{leg_num} dz", -0.08, 0.08, 0.0, row=4, col=0, length=190)
        controls[leg] = {"x": x_slider, "y": y_slider, "z": z_slider}

        coxa_deg = _make_slider(tab, f"L{leg_num} coxa deg", -90.0, 90.0, 0.0, row=2, col=1, resolution=0.5, length=190)
        femur_deg = _make_slider(tab, f"L{leg_num} femur deg", -90.0, 90.0, 0.0, row=3, col=1, resolution=0.5, length=190)
        tibia_deg = _make_slider(tab, f"L{leg_num} tibia deg", -120.0, 120.0, 0.0, row=4, col=1, resolution=0.5, length=190)
        joint_deg_controls[leg] = {
            "coxa": coxa_deg,
            "femur": femur_deg,
            "tibia": tibia_deg,
        }

    return controls, joint_deg_controls


def _build_actuator_index_map(env: "HexapodSimple"):
    name_to_index = {}
    for i in range(env.nu):
        joint_id = int(env.model.actuator_trnid[i][0])
        if joint_id >= 0:
            joint_name = env.model.joint_id2name(joint_id)
            if joint_name:
                name_to_index[joint_name] = i

        actuator_name = env.model.actuator_id2name(i)
        if actuator_name:
            name_to_index[actuator_name] = i

    result = {}
    for leg in LEG_ORDER:
        leg_map = {}
        candidates = {
            "coxa": [f"coxa_{leg}", f"{leg}_coxa", f"coxa_{leg}_motor", f"{leg}_coxa_motor"],
            "femur": [f"femur_{leg}", f"{leg}_femur", f"femur_{leg}_motor", f"{leg}_femur_motor"],
            "tibia": [f"tibia_{leg}", f"{leg}_tibia", f"tibia_{leg}_motor", f"{leg}_tibia_motor"],
        }
        for joint_name, names in candidates.items():
            found = None
            for n in names:
                if n in name_to_index:
                    found = name_to_index[n]
                    break
            leg_map[joint_name] = found
        result[leg] = leg_map
    return result


def _joint_info_for_actuator(env: "HexapodSimple", actuator_idx: int):
    joint_id = int(env.model.actuator_trnid[actuator_idx][0])
    qpos_adr = int(env.model.jnt_qposadr[joint_id])
    limited = (int(env.model.jnt_limited[joint_id]) == 1) and (not DISABLE_SOFT_LIMITS)
    if limited:
        low = float(env.model.jnt_range[joint_id][0])
        high = float(env.model.jnt_range[joint_id][1])
    else:
        low, high = -np.inf, np.inf
    return {
        "joint_id": joint_id,
        "qpos_adr": qpos_adr,
        "limited": limited,
        "low": low,
        "high": high,
    }


def _compute_neutral_ik_targets():
    neutral_leg_positions = body_kinematics((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    neutral = {}
    for i, leg in enumerate(LEG_ORDER):
        leg_base = np.array(neutral_leg_positions[i], dtype=np.float32)
        neutral_target = leg_base + _foot_home_offset_body(i)
        x, y, z = _body_target_to_leg_ik_frame(neutral_target, leg_base, i)

        coxa, femur, tibia = inverse_kinematics(
            float(x),
            float(y),
            float(z),
        )
        neutral[leg] = {
            "coxa": coxa,
            "femur": femur,
            "tibia": tibia,
        }
    return neutral


def _joint_delta_sign_for_leg(leg_index: int, joint_name: str) -> float:
    return 1.0


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


class _MatplotlibPlot:
    def __init__(self, parent, series_names, colors, y_range, max_points=240):
        if Figure is None or FigureCanvasTkAgg is None:
            raise RuntimeError("matplotlib is required for IMU plots")

        self.series_names = list(series_names)
        self.colors = list(colors)
        self.max_points = int(max_points)
        self.values = {name: [] for name in self.series_names}

        self.figure = Figure(figsize=(5.2, 3.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(float(y_range[0]), float(y_range[1]))
        self.ax.set_xlim(0, max(1, self.max_points - 1))
        self.ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        self.ax.set_xticks([])

        self.lines = {}
        for name, color in zip(self.series_names, self.colors):
            (line,) = self.ax.plot([], [], color=color, linewidth=1.5)
            self.lines[name] = line

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()

    def update(self, values):
        for name, value in zip(self.series_names, values):
            series = self.values[name]
            series.append(float(value))
            if len(series) > self.max_points:
                series.pop(0)

        for name in self.series_names:
            series = self.values[name]
            x_vals = list(range(len(series)))
            self.lines[name].set_data(x_vals, series)

        self.ax.set_xlim(0, max(1, self.max_points - 1))

    def draw(self):
        self.canvas.draw_idle()


class HexapodTkUI:
    def __init__(self, env: "HexapodSimple"):
        self.env = env
        self.root = tk.Tk()
        self.root.title("Hexapod IK + Body Kinematics Controller")
        self.root.geometry("1100x760")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))

        self.actuator_map = _build_actuator_index_map(env)
        self.nominal_ik_targets = _compute_neutral_ik_targets()

        state0 = env.sim.get_state()
        qpos0 = np.asarray(state0.qpos, dtype=np.float32)

        self.joint_calibration = {}
        for leg_i, leg in enumerate(LEG_ORDER):
            self.joint_calibration[leg] = {}
            for joint_name in ["coxa", "femur", "tibia"]:
                actuator_idx = self.actuator_map[leg][joint_name]
                if actuator_idx is None:
                    self.joint_calibration[leg][joint_name] = None
                    continue

                info = _joint_info_for_actuator(env, actuator_idx)
                neutral_joint_qpos = float(qpos0[info["qpos_adr"]])
                self.joint_calibration[leg][joint_name] = {
                    "actuator_idx": actuator_idx,
                    "neutral_joint_qpos": neutral_joint_qpos,
                    "nominal_ik": float(self.nominal_ik_targets[leg][joint_name]),
                    "low": float(info["low"]),
                    "high": float(info["high"]),
                    "delta_sign": _joint_delta_sign_for_leg(leg_i, joint_name),
                }

        self.air_mode = False
        self.air_height = 0.42
        self.air_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.control_mode = tk.StringVar(value="ik")
        self.camera_follow_var = tk.BooleanVar(value=True)

        self._build_tabs()
        self._build_controls()
        self._build_leg_offsets()
        self._build_foot_params()

        self.ui_update_ms = 8
        self.sim_steps_per_tick = 2
        self.render_every_n_ticks = 2
        self.plot_every_n_ticks = 2
        self.tick_counter = 0

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_controls = tk.Frame(self.notebook)
        self.tab_leg_offsets = tk.Frame(self.notebook)
        self.tab_foot_params = tk.Frame(self.notebook)

        self.notebook.add(self.tab_controls, text="Movement + IMU")
        self.notebook.add(self.tab_leg_offsets, text="Leg offsets")
        self.notebook.add(self.tab_foot_params, text="Foot params")

        self.tab_controls.grid_columnconfigure(0, weight=1)
        self.tab_controls.grid_columnconfigure(1, weight=1)
        self.tab_leg_offsets.grid_columnconfigure(0, weight=1)
        self.tab_foot_params.grid_columnconfigure(0, weight=1)
        self.tab_foot_params.grid_columnconfigure(1, weight=1)

    def _build_controls(self):
        tab = self.tab_controls

        tk.Label(tab, text="Movement controls", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(8, 4), sticky="w"
        )
        self.vx_slider = _make_slider(tab, "vx", -0.15, 0.15, 0.0, row=1, col=0)
        self.vy_slider = _make_slider(tab, "vy", -0.15, 0.15, 0.0, row=2, col=0)
        self.heading_slider = _make_slider(tab, "heading (deg)", -180, 180, 0.0, row=3, col=0, resolution=1.0)
        self.v_rot_slider = _make_slider(tab, "v_rot (feedback)", -0.15, 0.15, 0.0, row=4, col=0)
        self.v_rot_slider.config(state="disabled")
        self.step_height_slider = _make_slider(tab, "step height", 0.0, 0.3, 0.05, row=5, col=0)
        self.cpg_slider = _make_slider(tab, "cpg freq (Hz)", 0.1, 4.0, 1.0, row=6, col=0)
        self.gait_slider = _make_slider(tab, "gait phase lag (rad)", float(np.pi / 3.0), float(np.pi), float(np.pi), row=7, col=0)
        self.duty_slider = _make_slider(tab, "duty factor", 0.2, 0.8, 0.5, row=8, col=0)

        tk.Label(tab, text="Body controls", font=("Arial", 11, "bold")).grid(
            row=0, column=1, columnspan=2, padx=8, pady=(8, 4), sticky="w"
        )
        self.x_slider = _make_slider(tab, "pos x", -0.1, 0.1, 0.0, row=1, col=1)
        self.y_slider = _make_slider(tab, "pos y", -0.1, 0.1, 0.0, row=2, col=1)
        self.z_slider = _make_slider(tab, "pos z", -0.1, 0.1, 0.0, row=3, col=1)
        self.r_slider = _make_slider(tab, "roll", -60.0, 60.0, 0.0, row=4, col=1, resolution=0.1)
        self.p_slider = _make_slider(tab, "pitch", -60.0, 60.0, 0.0, row=5, col=1, resolution=0.1)
        self.yaw_slider = _make_slider(tab, "yaw", -60.0, 60.0, 0.0, row=6, col=1, resolution=0.1)

        control_bar = tk.Frame(tab)
        control_bar.grid(row=8, column=0, columnspan=2, padx=8, pady=10, sticky="we")
        tk.Button(control_bar, text="Reset leg offsets", command=self._reset_leg_offsets).pack(side="left", padx=(0, 6))
        tk.Button(control_bar, text="Reset sliders", command=self._reset_sliders).pack(side="left", padx=(0, 16))
        tk.Label(control_bar, text="Leg control mode:").pack(side="left", padx=(0, 6))
        tk.Radiobutton(control_bar, text="IK", variable=self.control_mode, value="ik").pack(side="left")
        tk.Radiobutton(control_bar, text="Joint deg", variable=self.control_mode, value="joint").pack(side="left")
        tk.Checkbutton(control_bar, text="Camera follow", variable=self.camera_follow_var).pack(side="left", padx=(16, 0))

        tk.Label(tab, text="IMU telemetry (rad)", font=("Arial", 11, "bold")).grid(
            row=9, column=0, columnspan=2, padx=8, pady=(10, 4), sticky="w"
        )
        imu_frame = tk.Frame(tab)
        imu_frame.grid(row=10, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="we")
        imu_frame.grid_columnconfigure(0, weight=1)
        imu_frame.grid_columnconfigure(1, weight=1)

        self.imu_quat_var = tk.StringVar(value="quat: --")
        self.imu_rpy_var = tk.StringVar(value="rpy: --")

        if Figure is None or FigureCanvasTkAgg is None:
            tk.Label(imu_frame, text="matplotlib not installed; IMU plots disabled", fg="#b00020").grid(
                row=0, column=0, columnspan=2, sticky="w", padx=2
            )
            self.quat_plot = None
            self.rpy_plot = None
        else:
            quat_frame = tk.Frame(imu_frame)
            quat_frame.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="we")
            rpy_frame = tk.Frame(imu_frame)
            rpy_frame.grid(row=0, column=1, padx=(6, 0), pady=2, sticky="we")

            self.quat_plot = _MatplotlibPlot(
                quat_frame,
                ["w", "x", "y", "z"],
                ["#FFD166", "#EF476F", "#06D6A0", "#118AB2"],
                y_range=(-1.1, 1.1),
            )
            self.rpy_plot = _MatplotlibPlot(
                rpy_frame,
                ["roll", "pitch", "yaw"],
                ["#FFD166", "#06D6A0", "#118AB2"],
                y_range=(-0.5, 0.5),
            )

            self.quat_plot.widget.pack(fill="both", expand=True)
            self.rpy_plot.widget.pack(fill="both", expand=True)

        tk.Label(imu_frame, textvariable=self.imu_quat_var, anchor="w").grid(row=1, column=0, sticky="w", padx=2)
        tk.Label(imu_frame, textvariable=self.imu_rpy_var, anchor="w").grid(row=1, column=1, sticky="w", padx=2)

    def _build_leg_offsets(self):
        tab = self.tab_leg_offsets
        self.leg_offset_controls, self.joint_deg_controls = _make_leg_offset_controls(tab)

    def _build_foot_params(self):
        tab = self.tab_foot_params
        tk.Label(tab, text="Foot home tuning", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(8, 4), sticky="w"
        )
        self.foot_home_radial_slider = _make_slider(
            tab, "FOOT_HOME_RADIAL", 0.05, 0.18, FOOT_HOME_RADIAL, row=1, col=0, resolution=0.001
        )
        self.foot_home_z_slider = _make_slider(
            tab, "FOOT_HOME_Z", -0.20, -0.05, FOOT_HOME_Z, row=1, col=1, resolution=0.001
        )

    def _reset_sliders(self):
        for s in [
            self.vx_slider, self.vy_slider, self.heading_slider, self.v_rot_slider,
            self.step_height_slider, self.cpg_slider, self.gait_slider, self.duty_slider,
            self.x_slider, self.y_slider, self.z_slider,
            self.r_slider, self.p_slider, self.yaw_slider,
            self.foot_home_radial_slider, self.foot_home_z_slider,
        ]:
            if s is self.cpg_slider:
                s.set(1.0)
            elif s is self.gait_slider:
                s.set(float(np.pi))
            elif s is self.duty_slider:
                s.set(0.5)
            elif s is self.step_height_slider:
                s.set(0.05)
            elif s is self.foot_home_radial_slider:
                s.set(0.105)
            elif s is self.foot_home_z_slider:
                s.set(-0.12)
            else:
                s.set(0.0)

    def _reset_leg_offsets(self):
        for leg in LEG_ORDER:
            self.leg_offset_controls[leg]["x"].set(0.0)
            self.leg_offset_controls[leg]["y"].set(0.0)
            self.leg_offset_controls[leg]["z"].set(0.0)
            self.joint_deg_controls[leg]["coxa"].set(0.0)
            self.joint_deg_controls[leg]["femur"].set(0.0)
            self.joint_deg_controls[leg]["tibia"].set(0.0)

    def _update_imu_plot(self):
        quat = self.env.get_imu_quat()
        rpy = self.env.get_imu_rpy()
        if self.quat_plot is not None and self.rpy_plot is not None:
            self.quat_plot.update(quat)
            self.rpy_plot.update(rpy)
            self.quat_plot.draw()
            self.rpy_plot.draw()
        self.imu_quat_var.set(
            f"quat: w={quat[0]:+.3f} x={quat[1]:+.3f} y={quat[2]:+.3f} z={quat[3]:+.3f}"
        )
        self.imu_rpy_var.set(
            f"rpy: r={rpy[0]:+.3f} p={rpy[1]:+.3f} y={rpy[2]:+.3f}"
        )

    def _step_sim(self, action):
        for _ in range(self.sim_steps_per_tick):
            self.env.step(action)

    def _tick(self):
        if not self.root.winfo_exists():
            return

        self.env.camera_follow_enabled = bool(self.camera_follow_var.get())

        if self.air_mode:
            self.env.sim.data.qpos[0:3] = np.array([0.0, 0.0, self.air_height], dtype=np.float32)
            self.env.sim.data.qpos[3:7] = self.air_quat
            self.env.sim.data.qvel[0:6] = 0.0
            self.env.sim.forward()

        global FOOT_HOME_RADIAL, FOOT_HOME_Z
        FOOT_HOME_RADIAL = float(self.foot_home_radial_slider.get())
        FOOT_HOME_Z = float(self.foot_home_z_slider.get())

        t = float(self.env.sim.data.time)
        vx = float(self.vx_slider.get())
        vy = float(self.vy_slider.get())
        # Heading control: v_rot is proportional to heading error (simple P controller)
        desired_heading = float(self.heading_slider.get())
        if vx == 0.0 and vy == 0.0 and desired_heading == 0.0:
            v_rot = 0.0
        else:  
            current_yaw = float(self.env.get_imu_rpy()[2]) * 180.0 / np.pi  # degrees
            heading_error = -(desired_heading - current_yaw + 180) % 360 - 180  # shortest path
            k_p = 0.01  # proportional gain, tune as needed
            v_rot = k_p * heading_error
            v_rot = _clamp(v_rot, -0.15, 0.15)

        self.v_rot_slider.config(state="normal")
        self.v_rot_slider.set(v_rot)
        self.v_rot_slider.config(state="disabled")
        step_height = float(self.step_height_slider.get())
        cpg_hz = max(0.1, float(self.cpg_slider.get()))
        gait_phase_lag = float(self.gait_slider.get())
        duty_factor = float(self.duty_slider.get())

        body_position = (
            float(self.x_slider.get()),
            float(self.y_slider.get()),
            float(self.z_slider.get()),
        )
        body_orientation = (
            float(self.r_slider.get()),
            float(self.p_slider.get()),
            float(self.yaw_slider.get()),
        )

        action = np.asarray(self.env.sim.data.ctrl.copy(), dtype=np.float32)
        if self.control_mode.get() == "joint":
            for leg in LEG_ORDER:
                for joint_name in ["coxa", "femur", "tibia"]:
                    calib = self.joint_calibration[leg][joint_name]
                    if calib is None:
                        continue

                    joint_deg = float(self.joint_deg_controls[leg][joint_name].get())
                    joint_rad = np.radians(joint_deg)
                    target_joint = calib["neutral_joint_qpos"] + joint_rad * calib["delta_sign"]
                    target_joint = _clamp(target_joint, calib["low"], calib["high"])
                    action[calib["actuator_idx"]] = target_joint

            self._step_sim(action)
            self._after_step()
            self.root.after(self.ui_update_ms, self._tick)
            return

        body_leg_positions = body_kinematics(body_position, body_orientation)

        for i, leg in enumerate(LEG_ORDER):
            movement, _ = generate_movement(
                t=t,
                freq_hz=cpg_hz,
                phase_lag_rad=gait_phase_lag,
                step_height=step_height,
                body_movement=(vx, vy, v_rot),
                leg_index=i,
                duty_factor=duty_factor,
            )

            leg_base = np.array(body_leg_positions[i], dtype=np.float32)
            nominal_leg_base = np.array(LEG_BASE_POSITIONS[i], dtype=np.float32)

            leg_offset = np.array([
                float(self.leg_offset_controls[leg]["x"].get()),
                float(self.leg_offset_controls[leg]["y"].get()),
                float(self.leg_offset_controls[leg]["z"].get()),
            ], dtype=np.float32)

            target_foot = leg_base + _foot_home_offset_body(i) + movement + leg_offset
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
                action[calib["actuator_idx"]] = target_joint

        self._step_sim(action)
        self._after_step()
        self.root.after(self.ui_update_ms, self._tick)

    def _after_step(self):
        self.tick_counter += 1
        if self.tick_counter % self.render_every_n_ticks == 0:
            self.env.render()

        if self.tick_counter % self.plot_every_n_ticks == 0:
            self._update_imu_plot()

    def _on_close(self):
        self.env.close()
        self.root.destroy()

    def run(self):
        self.root.after(0, self._tick)
        self.root.mainloop()


class HexapodSimple(gym.Env):
    """
    Minimal hexapod environment for direct joint-position control.

    Action:
        np.ndarray shape (nu,) in actuator units (target joint positions).
    Observation:
        np.ndarray of concatenated qpos and qvel.
    """

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "assets",
        "hexapod_trossen_new.xml",
        # "hexapod_trossen_terrain_2.xml",
    )

    metadata = {"render.modes": ["human"]}

    def __init__(self, model_path: Optional[str] = None, frame_skip: int = 1):
        super().__init__()

        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.frame_skip = int(max(1, frame_skip))

        self.model = mujoco_py.load_model_from_path(self.model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        self.camera_follow_enabled = True
        self.camera_follow_body_id = self._resolve_body_id("torso")
        self.imu_sensor_id = self._resolve_sensor_id("imu_quat")

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)

        if (not DISABLE_SOFT_LIMITS) and self.model.actuator_ctrllimited is not None and np.any(self.model.actuator_ctrllimited):
            ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
            ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        else:
            ctrl_low = -np.full(self.nu, 1e6, dtype=np.float32)
            ctrl_high = np.full(self.nu, 1e6, dtype=np.float32)

        self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)

        obs_dim = self.nq + self.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _resolve_sensor_id(self, sensor_name: str) -> Optional[int]:
        try:
            return int(self.model.sensor_name2id(sensor_name))
        except Exception:
            return None

    def _resolve_body_id(self, body_name: str) -> int:
        try:
            return int(self.model.body_name2id(body_name))
        except Exception:
            return 1 if self.model.nbody > 1 else 0

    def _update_camera_follow(self):
        if self.viewer is None:
            return

        if self.camera_follow_enabled:
            body_id = int(self.camera_follow_body_id)
            self.viewer.cam.trackbodyid = body_id
            body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=np.float32)
            self.viewer.cam.lookat[:] = body_pos
        else:
            self.viewer.cam.trackbodyid = -1

    def _get_obs(self) -> np.ndarray:
        state = self.sim.get_state()
        qpos = np.asarray(state.qpos, dtype=np.float32)
        qvel = np.asarray(state.qvel, dtype=np.float32)
        return np.concatenate([qpos, qvel], axis=0)

    def get_imu_quat(self) -> np.ndarray:
        if self.imu_sensor_id is not None:
            adr = int(self.model.sensor_adr[self.imu_sensor_id])
            dim = int(self.model.sensor_dim[self.imu_sensor_id])
            if dim >= 4:
                return np.asarray(self.sim.data.sensordata[adr: adr + 4], dtype=np.float32)

        body_id = int(self.camera_follow_body_id)
        return np.asarray(self.sim.data.body_xquat[body_id], dtype=np.float32)

    def get_imu_rpy(self) -> np.ndarray:
        quat = self.get_imu_quat()
        return _quat_wxyz_to_rpy(quat)

    def reset(self):
        self.sim.reset()
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.nu:
            raise ValueError(f"Expected action shape ({self.nu},), got {action.shape}")

        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        self.sim.data.ctrl[:] = clipped

        for _ in range(self.frame_skip):
            self.sim.step()

        obs = self._get_obs()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def set_joint_positions(self, joint_positions):
        return self.step(joint_positions)

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Only human render mode is supported.")
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.cam.distance = 1.6
            self.viewer.cam.elevation = -25
        self._update_camera_follow()
        self.viewer.render()

    def close(self):
        self.viewer = None


if __name__ == "__main__":
    env = HexapodSimple()
    env.reset()

    print("HexapodSimple ready")
    print(f"Action size (nu): {env.nu}")
    print("Tkinter IK controller ready. Use sliders to drive body and gait.")

    ui = HexapodTkUI(env)
    ui.run()
