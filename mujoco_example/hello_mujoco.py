import time
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.08" rgba="0.1 0.4 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 10:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)