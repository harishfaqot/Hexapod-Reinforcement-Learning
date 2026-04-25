import math
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path(__file__).parent / "models" / "hexapod.xml"
LEG_ORDER = ["lf", "lm", "lr", "rf", "rm", "rr"]
LEFT_LEGS = {"lf", "lm", "lr"}
TRIPOD_A = {"lf", "rm", "lr"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def make_actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    actuator_ids: dict[str, int] = {}
    for leg in LEG_ORDER:
        for joint in ("hip", "knee"):
            name = f"{leg}_{joint}_motor"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise RuntimeError(f"Actuator not found: {name}")
            actuator_ids[name] = actuator_id
    return actuator_ids


def print_help() -> None:
    print("\nManual Hexapod Control")
    print("W/S: increase/decrease forward command")
    print("A/D: turn left/right")
    print("Arrow Up/Down: increase/decrease step frequency")
    print("Arrow Right/Left: increase/decrease step amplitude")
    print("Space: zero forward/turn")
    print("R: reset robot state")
    print("H: show help\n")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    actuator_ids = make_actuator_ids(model)

    cmd = {
        "forward": 0.0,
        "turn": 0.0,
        "freq": 1.4,
        "amp": 0.35,
    }

    hip_base = 0.0
    knee_base = -0.65

    def show_state() -> None:
        print(
            f"forward={cmd['forward']:+.2f}, turn={cmd['turn']:+.2f}, "
            f"freq={cmd['freq']:.2f}Hz, amp={cmd['amp']:.2f}"
        )

    def key_callback(keycode: int) -> None:
        changed = True

        if keycode == glfw.KEY_W:
            cmd["forward"] = clamp(cmd["forward"] + 0.15, -1.0, 1.0)
        elif keycode == glfw.KEY_S:
            cmd["forward"] = clamp(cmd["forward"] - 0.15, -1.0, 1.0)
        elif keycode == glfw.KEY_A:
            cmd["turn"] = clamp(cmd["turn"] + 0.15, -1.0, 1.0)
        elif keycode == glfw.KEY_D:
            cmd["turn"] = clamp(cmd["turn"] - 0.15, -1.0, 1.0)
        elif keycode == glfw.KEY_UP:
            cmd["freq"] = clamp(cmd["freq"] + 0.1, 0.3, 3.0)
        elif keycode == glfw.KEY_DOWN:
            cmd["freq"] = clamp(cmd["freq"] - 0.1, 0.3, 3.0)
        elif keycode == glfw.KEY_RIGHT:
            cmd["amp"] = clamp(cmd["amp"] + 0.03, 0.05, 0.7)
        elif keycode == glfw.KEY_LEFT:
            cmd["amp"] = clamp(cmd["amp"] - 0.03, 0.05, 0.7)
        elif keycode == glfw.KEY_SPACE:
            cmd["forward"] = 0.0
            cmd["turn"] = 0.0
        elif keycode == glfw.KEY_R:
            mujoco.mj_resetData(model, data)
        elif keycode == glfw.KEY_H:
            print_help()
        else:
            changed = False

        if changed:
            show_state()

    print_help()
    show_state()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            t = data.time
            cycle = 2.0 * math.pi * cmd["freq"] * t

            for leg in LEG_ORDER:
                in_group_a = leg in TRIPOD_A
                phase = cycle + (0.0 if in_group_a else math.pi)

                stride = cmd["forward"] * cmd["amp"] * math.sin(phase)
                turn_sign = -1.0 if leg in LEFT_LEGS else 1.0
                turn_bias = cmd["turn"] * 0.18 * turn_sign

                hip_target = clamp(hip_base + stride + turn_bias, -0.7, 0.7)

                lift = max(0.0, math.sin(phase + math.pi / 2.0))
                knee_target = clamp(knee_base + 0.45 * lift, -1.1, 0.25)

                data.ctrl[actuator_ids[f"{leg}_hip_motor"]] = hip_target
                data.ctrl[actuator_ids[f"{leg}_knee_motor"]] = knee_target

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
