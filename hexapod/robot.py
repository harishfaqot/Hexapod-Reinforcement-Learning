"""Robot description: geometry, MuJoCo model, inverse kinematics and the tripod gait.

The robot is BumbleBee: a 3D-printed hexapod with the Trossen PhantomX layout
and 18 Dynamixel AX-18A servos. Everything that depends on its dimensions lives
here, so the simulated model and the kinematics can never disagree.

Where the numbers come from:
  - hip positions and link lengths: published PhantomX hexapod dimensions
    (coxa 52, femur 66, tibia 133 mm), consistent with the parts in STL/
    (body plate 262 x 202 mm, tibia plate 172 mm long)
  - servo: AX-18A datasheet (55.9 g, 1.8 Nm stall, 97 rpm no-load at 12 V)
  - total mass 2.0 kg: thesis table 4.1
MEASURE-ME marks values that are estimates. Check them on the real robot
before sim-to-real experiments.

Joint angle convention (not servo ticks): all zeros = femur horizontal and
pointing outward, tibia pointing straight down. Positive femur lowers the
foot, positive tibia swings the foot toward the body.
"""
import numpy as np

# name: (hip x, hip y, yaw of the leg's outward direction), body frame, metres/radians
LEGS = {
    "fl": (0.12, 0.06, np.pi / 4),
    "ml": (0.00, 0.10, np.pi / 2),
    "rl": (-0.12, 0.06, 3 * np.pi / 4),
    "fr": (0.12, -0.06, -np.pi / 4),
    "mr": (0.00, -0.10, -np.pi / 2),
    "rr": (-0.12, -0.06, -3 * np.pi / 4),
}
LEG_NAMES = list(LEGS)
LEG_BASE = np.array([[x, y, 0.0] for x, y, _ in LEGS.values()])
LEG_YAW = np.array([yaw for _, _, yaw in LEGS.values()])

COXA, FEMUR, TIBIA = 0.052, 0.066, 0.133            # joint-to-joint link lengths [m]
FOOT_RADIUS = 0.012
TORSO_HALF_SIZE = (0.13, 0.07, 0.021)                # two body plates with a 37 mm spacer
JOINT_RANGE = {"coxa": 0.6, "femur": 1.3, "tibia": 1.3}     # +/- limits [rad]  MEASURE-ME

# Masses [kg], total 2.0. One AX-18A is 0.056.  MEASURE-ME (weigh one leg and the body)
TORSO_MASS = 0.83                                    # plates, 6 coxa servos, Jetson, battery
LINK_MASS = {"coxa": 0.075,                          # femur servo + bracket
             "femur": 0.020,                         # two printed links
             "tibia": 0.100}                         # tibia servo + two arms + foot

# Servo model: AX-18A in position mode.  MEASURE-ME (step response of one joint)
SERVO_KP = 10.0          # Nm/rad: full torque at about 9 deg of error (compliance slope 32)
SERVO_DAMPING = 0.15     # Nms/rad: gives the 10 rad/s no-load speed at max torque
SERVO_MAX_TORQUE = 1.5   # Nm, a little under the 1.8 Nm stall torque

# Tripod gait
GAIT_FREQ = 1.5          # cycles per second
DUTY = 0.5               # fraction of the cycle a foot is on the ground
STEP_HEIGHT = 0.06       # m. Tuned: best overall of 4/6/8/10 cm for this gait on stairs
PHASE_OFFSET = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])    # fl, rl, mr step together
HOME_RADIUS, HOME_HEIGHT = 0.16, 0.11                # neutral foot position, measured from the hip

HOME = LEG_BASE + np.column_stack([
    HOME_RADIUS * np.cos(LEG_YAW), HOME_RADIUS * np.sin(LEG_YAW), np.full(6, -HOME_HEIGHT)])

TERRAIN_SIZE = 5.0       # half-width of the terrain square [m]
TERRAIN_CELLS = 200      # grid cells per side (5 cm cells)
TERRAIN_MAX_HEIGHT = 0.15
STAIR_STEPS = 8          # steps going up along +x, then a long flat top
STAIR_START = 0.5        # x of the first riser [m]
STAIR_SLAB = (2.5, 2.5, 0.25)   # half-sizes of one slab; slab i is step i and everything behind it


def inverse_kinematics(feet: np.ndarray) -> np.ndarray:
    """Foot positions in the body frame, shape (6, 3) -> 18 joint angles
    ordered leg by leg as (coxa, femur, tibia). Unreachable targets are pulled
    to the nearest reachable distance."""
    p = feet - LEG_BASE
    c, s = np.cos(LEG_YAW), np.sin(LEG_YAW)
    x = c * p[:, 0] + s * p[:, 1]          # outward
    y = -s * p[:, 0] + c * p[:, 1]         # sideways
    z = p[:, 2]
    coxa = np.arctan2(y, x)
    r = np.hypot(x, y) - COXA
    d = np.clip(np.hypot(r, z), abs(FEMUR - TIBIA) + 1e-6, FEMUR + TIBIA - 1e-6)
    knee = -np.arccos(np.clip((d**2 - FEMUR**2 - TIBIA**2) / (2 * FEMUR * TIBIA), -1, 1))
    femur = np.arctan2(TIBIA * np.sin(knee), FEMUR + TIBIA * np.cos(knee)) - np.arctan2(z, r)
    tibia = -knee - np.pi / 2
    return np.column_stack([coxa, femur, tibia]).ravel()


def gait_foot_targets(t: float, command) -> np.ndarray:
    """Open-loop tripod gait. command = (vx, vy, yaw_rate) in the body frame.
    Returns foot targets in the body frame, shape (6, 3)."""
    vx, vy, wz = command
    # how far each foot travels over the ground during one stance
    stride = np.column_stack([vx - wz * HOME[:, 1], vy + wz * HOME[:, 0]]) * (DUTY / GAIT_FREQ)
    phase = (GAIT_FREQ * t + PHASE_OFFSET) % 1.0
    stance = phase < DUTY
    s = np.where(stance, phase / DUTY, (phase - DUTY) / (1 - DUTY))
    feet = HOME.copy()
    feet[:, :2] += stride * np.where(stance, 0.5 - s, s - 0.5)[:, None]
    lift = ~stance & (np.linalg.norm(stride, axis=1) > 1e-4)
    feet[:, 2] += np.where(lift, STEP_HEIGHT * np.sin(np.pi * s), 0.0)
    return feet


def gait_joint_targets(t: float, command) -> np.ndarray:
    return inverse_kinematics(gait_foot_targets(t, command))


DEFAULT_POSE = inverse_kinematics(HOME)
JOINT_LIMIT = np.tile([JOINT_RANGE["coxa"], JOINT_RANGE["femur"], JOINT_RANGE["tibia"]], 6)


def build_xml() -> str:
    """MuJoCo model. Each leg lives in its own frame with x pointing outward,
    so all six legs share the same local description."""
    legs, actuators = [], []
    for name, (bx, by, yaw) in LEGS.items():
        legs.append(f"""
      <body name="leg_{name}" pos="{bx} {by} 0" euler="0 0 {yaw}">
        <joint name="coxa_{name}" axis="0 0 1" range="{-JOINT_RANGE['coxa']} {JOINT_RANGE['coxa']}"/>
        <geom type="capsule" fromto="0 0 0 {COXA} 0 0" size="{FOOT_RADIUS}" mass="{LINK_MASS['coxa']}"/>
        <body pos="{COXA} 0 0">
          <joint name="femur_{name}" axis="0 1 0" range="{-JOINT_RANGE['femur']} {JOINT_RANGE['femur']}"/>
          <geom type="capsule" fromto="0 0 0 {FEMUR} 0 0" size="{FOOT_RADIUS}" mass="{LINK_MASS['femur']}"/>
          <body name="tibia_{name}" pos="{FEMUR} 0 0">
            <joint name="tibia_{name}" axis="0 1 0" range="{-JOINT_RANGE['tibia']} {JOINT_RANGE['tibia']}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-TIBIA}" size="{FOOT_RADIUS}" mass="{LINK_MASS['tibia']}" rgba="0 .7 .7 1"/>
          </body>
        </body>
      </body>""")
        actuators += [f'<position joint="{j}_{name}"/>' for j in ("coxa", "femur", "tibia")]
    stairs = "".join(
        # mocap bodies: the supported way to reposition fixed objects after the model is built.
        # (Editing geom_pos instead leaves MuJoCo's collision lookup stale and feet fall through.)
        f'<body name="stair_{i}" mocap="true" pos="0 0 -1"><geom type="box" conaffinity="1" rgba=".6 .6 .65 1" '
        f'size="{STAIR_SLAB[0]} {STAIR_SLAB[1]} {STAIR_SLAB[2]}"/></body>' for i in range(STAIR_STEPS))
    return f"""
<mujoco model="hexapod">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <default>
    <joint armature="0.01" damping="{SERVO_DAMPING}" limited="true"/>
    <geom conaffinity="0" condim="3" friction="1 0.005 0.0001" rgba=".2 .2 .2 1"/>
    <position kp="{SERVO_KP}" forcerange="{-SERVO_MAX_TORQUE} {SERVO_MAX_TORQUE}"/>
  </default>
  <asset>
    <hfield name="terrain" nrow="{TERRAIN_CELLS}" ncol="{TERRAIN_CELLS}"
            size="{TERRAIN_SIZE} {TERRAIN_SIZE} {TERRAIN_MAX_HEIGHT} 0.05"/>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1=".75 .8 .75" rgb2=".65 .7 .65"/>
    <material name="grid" texture="grid" texrepeat="20 20"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="hfield" hfield="terrain" conaffinity="1" material="grid" rgba="1 1 1 1"/>
    {stairs}
    <body name="torso" pos="0 0 0.15">
      <camera name="track" mode="track" pos="0 -1.0 0.5" xyaxes="1 0 0 0 0.45 1"/>
      <freejoint/>
      <geom name="torso" type="box" size="{TORSO_HALF_SIZE[0]} {TORSO_HALF_SIZE[1]} {TORSO_HALF_SIZE[2]}" mass="{TORSO_MASS}" rgba=".9 .7 .1 1"/>{''.join(legs)}
    </body>
  </worldbody>
  <actuator>
    {' '.join(actuators)}
  </actuator>
</mujoco>"""
