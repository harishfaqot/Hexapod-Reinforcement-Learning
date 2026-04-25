from pathlib import Path
import math
import os
import time
import tkinter as tk

MODEL_PATH = Path(__file__).parent / "models" / "hex.xml"


def _setup_mujoco_py_windows() -> Path:
    mujoco_root = Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", Path.home() / ".mujoco" / "mujoco210"))
    mujoco_bin = mujoco_root / "bin"

    if not mujoco_bin.exists():
        raise FileNotFoundError(
            f"MuJoCo bin path not found: {mujoco_bin}. Set MUJOCO_PY_MUJOCO_PATH to your MuJoCo install."
        )

    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco_root))
    os.environ["PATH"] = f"{mujoco_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(mujoco_bin))

    return mujoco_root


def _build_joint_window(model, joint_names, joint_limits):
    root = tk.Tk()
    root.title("Hexapod Direct Joint Control")

    sliders = {}
    auto = tk.IntVar(value=0)
    amp = tk.DoubleVar(value=0.3)
    freq = tk.DoubleVar(value=0.6)

    header = tk.Label(root, text=f"Model file: {MODEL_PATH.name} | joints: {len(joint_names)}")
    header.pack(fill="x", padx=8, pady=4)

    for joint in joint_names:
        low, high = joint_limits[joint]
        row = tk.Frame(root)
        row.pack(fill="x", padx=8, pady=2)

        label = tk.Label(row, text=joint, width=14, anchor="w")
        label.pack(side="left")

        scale = tk.Scale(
            row,
            from_=low,
            to=high,
            orient="horizontal",
            resolution=0.01,
            length=320,
        )
        scale.set(0.0)
        scale.pack(side="left", fill="x", expand=True)

        sliders[joint] = scale

    wave_frame = tk.Frame(root)
    wave_frame.pack(fill="x", padx=8, pady=6)

    tk.Checkbutton(wave_frame, text="Auto wave all joints", variable=auto).pack(side="left")
    tk.Label(wave_frame, text="amp").pack(side="left", padx=(8, 2))
    tk.Scale(wave_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=amp, length=130).pack(side="left")
    tk.Label(wave_frame, text="freq").pack(side="left", padx=(8, 2))
    tk.Scale(wave_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", variable=freq, length=130).pack(side="left")

    reset_btn = tk.Button(
        root,
        text="Reset All Joints",
        command=lambda: [sliders[name].set(0.0) for name in joint_names],
    )
    reset_btn.pack(fill="x", padx=8, pady=6)

    return root, sliders, auto, amp, freq


def _actuator_index_map(model):
    idx = {}
    for i in range(model.nu):
        name = model.actuator_id2name(i)
        if name:
            idx[name] = i

        # For XMLs with unnamed actuators, map by driven joint name.
        joint_id = int(model.actuator_trnid[i][0])
        if joint_id >= 0:
            joint_name = model.joint_id2name(joint_id)
            if joint_name:
                idx[joint_name] = i
    return idx


def _joint_limits(model, joint_name: str):
    j_id = model.joint_name2id(joint_name)
    if int(model.jnt_limited[j_id]) == 1:
        return float(model.jnt_range[j_id][0]), float(model.jnt_range[j_id][1])
    return -1.0, 1.0


def main() -> None:
    _setup_mujoco_py_windows()
    import mujoco_py

    model = mujoco_py.load_model_from_path(str(MODEL_PATH))
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    actuator_idx = _actuator_index_map(model)
    joint_names = [
        "coxa_fl", "femur_fl", "tibia_fl",
        "coxa_fr", "femur_fr", "tibia_fr",
        "coxa_rr", "femur_rr", "tibia_rr",
        "coxa_rl", "femur_rl", "tibia_rl",
        "coxa_mr", "femur_mr", "tibia_mr",
        "coxa_ml", "femur_ml", "tibia_ml",
    ]
    joint_limits = {name: _joint_limits(model, name) for name in joint_names}
    root, sliders, auto_wave, wave_amp, wave_freq = _build_joint_window(model, joint_names, joint_limits)
    running = True
    t = 0.0

    def on_close() -> None:
        nonlocal running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    print("mujoco_py direct joint mode")
    print("Move each joint with sliders. No IK/body/gait is used.")
    print("Close the slider window or press Ctrl+C in terminal to exit.")

    while running:
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break

        dt = sim.model.opt.timestep
        t += dt

        for idx, name in enumerate(joint_names):
            if name not in actuator_idx:
                continue

            i = actuator_idx[name]
            low, high = joint_limits[name]

            if auto_wave.get() == 1:
                center = 0.5 * (low + high)
                span = 0.5 * (high - low) * float(wave_amp.get())
                cmd = center + span * math.sin(2.0 * math.pi * float(wave_freq.get()) * t + idx * 0.35)
            else:
                cmd = float(sliders[name].get())

            sim.data.ctrl[i] = max(low, min(high, cmd))

        sim.step()
        viewer.render()
        time.sleep(dt)


if __name__ == "__main__":
    main()
