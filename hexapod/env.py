"""Velocity-tracking hexapod environment.

One environment, three controllers that can be compared fairly:

    mode="residual"  joint target = tripod gait (IK) + learned correction
    mode="direct"    joint target = standing pose   + learned offset (no gait, no IK)
    kinematic        mode="residual" driven with all-zero actions (the baseline)

Observation (65 numbers), limited to what the real robot can measure with its
IMU and servos, so a trained policy can be moved to hardware:
    gravity direction in body frame (3), gyro (3), joint angles minus default (18),
    joint target minus joint angle (18)  <- the robot's only sense of touch: a leg
                                            that is blocked cannot reach its target
    previous action (18), command vx vy yaw_rate (3), gait clock sin cos (2)
Action: 18 numbers in [-1, 1], one per joint.

On stairs the yaw-rate command is not free: it is set every step from the IMU
heading so the robot keeps facing the stairs (a simple compass loop, the same
for all three controllers, and available on the real robot from the BNO055).
"""
import gymnasium as gym
import mujoco
import numpy as np

from hexapod import robot

CONTROL_DT = 0.04            # 25 Hz: a rate the AX servo bus can keep up with
RESIDUAL_SCALE = 0.3         # rad of correction on top of the gait
DIRECT_SCALE = 0.6           # rad of offset around the standing pose
COMMAND_LOW = np.array([0.05, -0.05, -0.4])     # vx, vy [m/s], yaw rate [rad/s]
COMMAND_HIGH = np.array([0.20, 0.05, 0.4])


class HexapodEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode="residual", command=None, terrain="bumps", terrain_height=0.0,
                 randomize=False, episode_seconds=20.0, render_mode=None):
        """command: fixed (vx, vy, yaw_rate), or None to sample a new one each episode.
        terrain: "bumps" (random bumps everywhere) or "stairs" (8 steps up along +x,
        starting 0.5 m ahead of the robot, then a flat top).
        terrain_height: bump height or stair riser height in metres. With
        randomize=True each episode draws a height between 0 and this value.
        randomize: training-time randomization of friction, payload, servo gain,
        action delay, sensor noise and pushes."""
        assert mode in ("residual", "direct")
        assert terrain in ("bumps", "stairs")
        assert 0.0 <= terrain_height <= robot.TERRAIN_MAX_HEIGHT
        self.mode = mode
        self.terrain = terrain
        self.fixed_command = None if command is None else np.asarray(command, dtype=float)
        self.terrain_height = terrain_height
        self.randomize = randomize
        self.max_steps = int(episode_seconds / CONTROL_DT)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(robot.build_xml())
        self.data = mujoco.MjData(self.model)
        self.frame_skip = round(CONTROL_DT / self.model.opt.timestep)
        self.torso_body = self.model.body("torso").id
        self.torso_geom = self.model.geom("torso").id
        self.floor_geom = self.model.geom("floor").id
        self.stair_mocap = [int(self.model.body(f"stair_{i}").mocapid[0]) for i in range(robot.STAIR_STEPS)]
        self.total_mass = float(self.model.body_subtreemass[self.torso_body])
        self._nominal_torso_mass = float(self.model.body_mass[self.torso_body])
        self.viewer = None

        self.action_space = gym.spaces.Box(-1.0, 1.0, (18,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (65,), np.float32)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rotation(self):
        return self.data.xmat[self.torso_body].reshape(3, 3)

    def body_velocity(self):
        """Linear velocity of the torso expressed in its own frame."""
        return self._rotation().T @ self.data.qvel[0:3]

    def _obs(self):
        gravity = -self._rotation()[2]              # world "down" seen from the body
        gyro = self.data.qvel[3:6].copy()           # free-joint angular velocity is already body frame
        joints = self.data.qpos[7:] - robot.DEFAULT_POSE
        if self.randomize:
            gravity = gravity + self.np_random.normal(0, 0.03, 3)
            gyro += self.np_random.normal(0, 0.1, 3)
            joints = joints + self.np_random.normal(0, 0.02, 18)
        clock = 2 * np.pi * robot.GAIT_FREQ * self.steps * CONTROL_DT
        blocked = self.data.ctrl - robot.DEFAULT_POSE - joints      # last target minus measured angle
        return np.concatenate([gravity, gyro, joints, blocked, self.last_action, self.command,
                               [np.sin(clock), np.cos(clock)]]).astype(np.float32)

    def _new_terrain(self):
        height = self.terrain_height * (self.np_random.uniform() if self.randomize else 1.0)
        n = robot.TERRAIN_CELLS
        field = np.zeros((n, n))
        if self.terrain == "bumps":
            field = self.np_random.uniform(0, height / robot.TERRAIN_MAX_HEIGHT, (n, n))
            field[n // 2 - 8: n // 2 + 8, n // 2 - 8: n // 2 + 8] = 0.0     # flat 0.8 m start pad
        elif height > 0:
            tread = self.np_random.uniform(0.20, 0.30) if self.randomize else 0.25
            for i, mocap in enumerate(self.stair_mocap):
                self.data.mocap_pos[mocap] = [robot.STAIR_START + i * tread + robot.STAIR_SLAB[0], 0.0,
                                             (i + 1) * height - robot.STAIR_SLAB[2]]
        self.model.hfield_data[:] = field.ravel()
        if self.viewer is not None:
            self.viewer.update_hfield(0)

    def _randomize_physics(self):
        rng = self.np_random
        self.model.geom_friction[self.floor_geom, 0] = rng.uniform(0.5, 1.2)
        self.model.body_mass[self.torso_body] = self._nominal_torso_mass * rng.uniform(0.8, 1.3)
        kp = robot.SERVO_KP * rng.uniform(0.8, 1.2)
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp
        self.delay = int(rng.integers(0, 2))        # 0 or 1 control step of latency

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.delay = 0
        if self.randomize:
            self._randomize_physics()
        mujoco.mj_resetData(self.model, self.data)          # also parks the stairs below the ground
        self._new_terrain()
        yaw = self.np_random.uniform(-0.15, 0.15)       # no two episodes meet the terrain the same way
        self.data.qpos[0:2] = self.np_random.uniform(-0.1, 0.1, 2)
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qpos[2] = robot.HOME_HEIGHT + robot.FOOT_RADIUS + 0.01
        self.data.qpos[7:] = robot.DEFAULT_POSE
        self.data.ctrl[:] = robot.DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.last_action = np.zeros(18)
        self.command = (self.fixed_command.copy() if self.fixed_command is not None
                        else self.np_random.uniform(COMMAND_LOW, COMMAND_HIGH))
        if self.terrain == "stairs" and self.fixed_command is None:
            self.command[1:] = 0.0                  # walk straight at the stairs
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.terrain == "stairs":                # compass loop: turn back toward heading 0
            w_, x_, y_, z_ = self.data.qpos[3:7]
            yaw = np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
            self.command[2] = np.clip(-1.0 * yaw, COMMAND_LOW[2], COMMAND_HIGH[2])
        applied = self.last_action if self.delay else action
        if self.mode == "direct":
            target = robot.DEFAULT_POSE + DIRECT_SCALE * applied
        else:
            target = robot.gait_joint_targets(self.steps * CONTROL_DT, self.command) + RESIDUAL_SCALE * applied
        self.data.ctrl[:] = np.clip(target, -robot.JOINT_LIMIT, robot.JOINT_LIMIT)

        if self.randomize and self.steps % 100 == 99:                   # a shove every 4 s
            self.data.qvel[0:2] += self.np_random.uniform(-0.3, 0.3, 2)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self.steps += 1

        v = self.body_velocity()
        w = self.data.qvel[3:6]
        gravity = -self._rotation()[2]
        power = float(np.sum(np.abs(self.data.actuator_force * self.data.qvel[6:])))   # watts

        belly_hit = any(c.geom1 == self.torso_geom or c.geom2 == self.torso_geom
                        for c in self.data.contact[: self.data.ncon])

        # Velocity errors are measured relative to the commanded speed. Otherwise standing
        # still earns most of the reward whenever the command is slow, and the policy learns to stand.
        speed = max(np.linalg.norm(self.command[:2]), 1e-6)
        error = np.linalg.norm(self.command[:2] - v[:2]) / speed
        track_lin = np.exp(-4.0 * error**2)                            # 0.02 standing still, 1 on target
        along = v[:2] @ self.command[:2] / speed**2                     # 1 = exactly the commanded speed
        progress = np.clip(1.0 - abs(along - 1.0), -1.0, 1.0)          # 0 standing still, 1 on target, 0 at double speed
        track_yaw = np.exp(-(self.command[2] - w[2]) ** 2 / 0.05)
        reward = (
            1.0 * track_lin
            + 1.0 * progress                               # keeps a slope toward the goal when far from it
            + 0.5 * track_yaw
            - 2.0 * v[2] ** 2                              # no bouncing
            - 0.05 * (w[0] ** 2 + w[1] ** 2)               # no rocking
            - 0.5 * (gravity[0] ** 2 + gravity[1] ** 2)    # stay level
            - 0.01 * np.sum((action - self.last_action) ** 2)   # smooth commands, kind to servos
            - 0.005 * power
            - 0.2 * belly_hit                              # body resting on the ground or a stair edge
        )
        self.last_action = action

        terminated = bool(gravity[2] > -0.5)               # tipped past 60 degrees
        truncated = self.steps >= self.max_steps
        info = {"body_velocity": v, "power": power, "gravity": gravity}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.torso_body
            self.viewer.cam.distance = 1.2
            self.viewer.update_hfield(0)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
