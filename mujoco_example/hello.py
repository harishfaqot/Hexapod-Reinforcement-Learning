import mujoco.viewer
import mujoco
import time

model = mujoco.MjModel.from_xml_path("envs/assets/hexapod_trossen_rails.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)   # 🔴 critical