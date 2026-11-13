"""Run with:  python -m tests.test_hexapod   (or pytest)"""
import mujoco
import numpy as np

from hexapod import HexapodEnv, robot


def test_ik_matches_simulated_model():
    """Put the joints where IK says and check the simulated feet land on the targets."""
    env = HexapodEnv()
    rng = np.random.default_rng(0)
    for _ in range(20):
        targets = robot.HOME + rng.uniform(-0.015, 0.015, (6, 3))   # stays inside the leg's reach
        env.data.qpos[:7] = [0, 0, 0.3, 1, 0, 0, 0]
        env.data.qpos[7:] = robot.inverse_kinematics(targets)
        mujoco.mj_forward(env.model, env.data)
        for i, name in enumerate(robot.LEG_NAMES):
            tibia = env.data.body(f"tibia_{name}")
            foot = tibia.xpos + tibia.xmat.reshape(3, 3) @ [0, 0, -robot.TIBIA] - [0, 0, 0.3]
            assert np.allclose(foot, targets[i], atol=1e-6), (name, foot, targets[i])


def test_kinematic_gait_walks_forward():
    """Baseline sanity check, and a guard on the sign and frame of the velocity signal."""
    env = HexapodEnv(mode="residual", command=(0.15, 0.0, 0.0), episode_seconds=10)
    env.reset(seed=0)
    speeds = []
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.zeros(18))
        speeds.append(info["body_velocity"][0])
        assert not terminated
        done = truncated
    distance = env.data.qpos[0]
    assert 0.9 < distance < 1.8, distance                         # commanded 1.5 m
    assert abs(np.mean(speeds) * 10 - distance) < 0.15            # body velocity agrees with real travel


def test_env_follows_gym_api():
    from stable_baselines3.common.env_checker import check_env
    from hexapod.env import TERRAINS
    for mode, terrain in [("residual", t) for t in TERRAINS] + [("direct", "stairs")]:
        check_env(HexapodEnv(mode=mode, terrain=terrain, terrain_height=0.02, randomize=True))


def test_stairs_are_where_they_should_be():
    """Measure the ground height with a ray: 0 on the flat start, 8 risers on the top."""
    env = HexapodEnv(terrain="stairs", terrain_height=0.04)
    env.reset(seed=0)

    def ground(x):
        hit = np.zeros(1, np.int32)
        return 1.0 - mujoco.mj_ray(env.model, env.data, [x, 1.0, 1.0], [0, 0, -1], None, 1, -1, hit)

    assert abs(ground(0.3)) < 1e-6
    assert abs(ground(0.5 + 0.125) - 0.04) < 1e-6        # middle of the first step
    assert abs(ground(3.5) - 8 * 0.04) < 1e-6

    # Rays are not enough: also check that every step is solid for the robot's feet.
    standing = robot.HOME_HEIGHT + robot.FOOT_RADIUS
    for step in range(8):
        env = HexapodEnv(terrain="stairs", terrain_height=0.04, command=(0, 0, 0))
        env.reset(seed=0)
        env.data.qpos[:7] = [0.5 + (step + 0.5) * 0.25, 0, (step + 1) * 0.04 + standing + 0.02, 1, 0, 0, 0]
        for _ in range(40):
            env.step(np.zeros(18))
        # the robot is wider than one step, so it rests between this step's height and the next one's
        assert (step + 0.3) * 0.04 < env.data.qpos[2] - standing < (step + 2) * 0.04, (step, env.data.qpos[2])


def test_slope_is_a_solid_ramp():
    env = HexapodEnv(terrain="slope", terrain_height=0.10, command=(0, 0, 0))
    env.reset(seed=0)
    hit = np.zeros(1, np.int32)
    ground = lambda x: 2.0 - mujoco.mj_ray(env.model, env.data, [x, 1.0, 2.0], [0, 0, -1], None, 1, -1, hit)
    assert abs(ground(0.3)) < 1e-6
    assert abs(ground(1.5) - 1.0 * 0.10 / 0.25) < 1e-6          # 1 m up the ramp
    standing = robot.HOME_HEIGHT + robot.FOOT_RADIUS
    env.data.qpos[:3] = [1.5, 0, 0.4 + standing + 0.05]         # drop the robot on the ramp: it must rest on it
    for _ in range(40):
        env.step(np.zeros(18))
    assert 0.2 < env.data.qpos[2] - standing < 0.5, env.data.qpos[:3]


def test_descend_and_bars():
    hit = np.zeros(1, np.int32)
    env = HexapodEnv(terrain="descend", terrain_height=0.05, command=(0, 0, 0))
    env.reset(seed=0)
    ground = lambda x: 2.0 - mujoco.mj_ray(env.model, env.data, [x, 1.0, 2.0], [0, 0, -1], None, 1, -1, hit)
    assert abs(ground(0.3) - 0.40) < 1e-6 and abs(ground(0.5 + 0.125) - 0.35) < 1e-6 and abs(ground(3.0)) < 1e-6
    for _ in range(40):
        env.step(np.zeros(18))
    assert abs(env.data.qpos[2] - 0.40 - robot.HOME_HEIGHT - robot.FOOT_RADIUS) < 0.02     # stands on the platform

    env = HexapodEnv(terrain="bars", terrain_height=0.05)
    env.reset(seed=0)
    heights = [ground(x) for x in np.arange(0.5, 3.0, 0.01)]
    assert abs(ground(0.3)) < 1e-6 and abs(max(heights) - 0.05) < 1e-3 and min(heights) < 1e-6


def test_course_layout():
    env = HexapodEnv(terrain="course", terrain_height=0.05)
    env.reset(seed=0)
    hit = np.zeros(1, np.int32)
    ground = lambda x: 2.0 - mujoco.mj_ray(env.model, env.data, [x, 1.0, 2.0], [0, 0, -1], None, 1, -1, hit)
    assert abs(ground(0.2)) < 1e-6                              # start pad
    assert max(ground(x) for x in np.arange(0.5, 2.4, 0.01)) > 0.049      # bars
    assert max(ground(x) for x in np.arange(2.6, 4.4, 0.01)) > 0.03       # rubble
    assert abs(ground(4.8)) < 1e-6 and abs(ground(8.2)) < 1e-6  # flat before and after the stairs
    assert abs(ground(5.1) - 0.05) < 1e-6 and abs(ground(6.5) - 0.20) < 1e-6 and abs(ground(7.9) - 0.05) < 1e-6
    assert abs(ground(9.5) - 1.0 * 0.05 / 0.25) < 1e-6          # finish line, 1 m up the ramp
    for side in (1, -1):                                        # walls 1.5 m to each side
        assert abs(mujoco.mj_ray(env.model, env.data, [1.0, 0, 0.3], [0, side, 0], None, 1, -1, hit) - 1.5) < 1e-6
    env = HexapodEnv(terrain="stairs", terrain_height=0.05)     # and no walls anywhere else
    env.reset(seed=0)
    assert mujoco.mj_ray(env.model, env.data, [0.0, 0, 0.3], [0, 1, 0], None, 1, -1, hit) < 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
