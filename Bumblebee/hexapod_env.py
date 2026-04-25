import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
from lib.hexapod_constant import *
from hexapod_control import *
from terrain import add_terrain, add_field
import pandas as pd
import threading

HOME_POSITION   = [2, 0, 0.5]
TARGET_POSITION = [-5, 0, 0.1]
DISTANCE = math.sqrt((TARGET_POSITION[0] - HOME_POSITION[0]) ** 2 +
                     (TARGET_POSITION[1] - HOME_POSITION[1]) ** 2)
TIMEOUT = 180

global_position    = np.array([0, 0, 0])
global_orientation = np.array([0, 0, 0, 0])
width  = 512
height = 512
fov    = 90
aspect = width / height
near   = 0.02
far    = 15
camera_offset = [0, 0, 0.5]


def simple_leg_trajectory(t, leg_index, freq_hz, phase_lag, duty_factor,
                          step_height, vx, vy, v_rot):
    """Direct per-leg phase trajectory matching simulation.py behavior."""
    phase = (2 * math.pi * freq_hz * t + leg_index * phase_lag) % (2 * math.pi)
    phi = phase / (2 * math.pi)
    duty = float(np.clip(duty_factor, 1e-4, 1.0 - 1e-4))

    angle = math.pi / 2 + math.atan2(leg_base_positions[leg_index][0],
                                      leg_base_positions[leg_index][1])
    offset_x = v_rot * math.sin(angle)
    offset_y = v_rot * math.cos(angle)
    step_span_x = vx + offset_x
    step_span_y = vy + offset_y

    if phi < duty:
        s = phi / duty
        x = step_span_x * (0.5 - s)
        y = step_span_y * (0.5 - s)
        z = 0.0
    else:
        s = (phi - duty) / (1.0 - duty)
        x = step_span_x * (s - 0.5)
        y = step_span_y * (s - 0.5)
        z = step_height * math.sin(math.pi * s)

    if vx == 0 and vy == 0 and v_rot == 0:
        z = 0.0

    return x, y, z


def get_robot_cam():
    global global_position, global_orientation
    prev_position       = np.array([0, 0, 0])
    alpha               = 0.2
    camera_distance_set = False
    while True:
        camera_info     = p.getDebugVisualizerCamera()
        manual_yaw      = camera_info[8]
        manual_pitch    = camera_info[9]
        manual_distance = 1.0 if not camera_distance_set else camera_info[10]
        if not camera_distance_set:
            camera_distance_set = True
        smoothed_position = alpha * global_position + (1 - alpha) * prev_position
        prev_position     = smoothed_position
        p.resetDebugVisualizerCamera(manual_distance, manual_yaw,
                                     manual_pitch, smoothed_position.tolist())
        pos, orientation   = global_position, global_orientation
        orientation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        x, y, z   = pos
        pos_cam   = (x - 0.2, y, z - 0.4)
        cam_pos   = np.array(pos_cam) + np.dot(orientation_matrix, camera_offset)
        cam_tgt   = np.array(pos_cam) + np.dot(orientation_matrix, [-5, 0, -1])
        vm        = p.computeViewMatrix(cam_pos.tolist(), cam_tgt.tolist(), [0, 0, 1])
        pm        = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        p.getCameraImage(width // 2, height // 2, vm, pm,
                         shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        time.sleep(1 / 240)


# ===========================================================================
class HexapodEnv(gym.Env):
# ===========================================================================

    # How many simulation substeps per RL step
    # One full gait cycle = SUBSTEPS_PER_CYCLE substeps
    SUBSTEPS_PER_CYCLE = 120   # at 240Hz this = 0.5s per RL step

    # Smoothing alpha for each action dimension
    # Slower alpha = smoother but less responsive
    ACTION_ALPHA = np.array([
        0.15,   # [0] body_roll    — slow, affects all legs
        0.15,   # [1] body_pitch   — slow, affects all legs
        0.20,   # [2] step_height  — medium
        0.08,   # [3] step_dur     — very slow, rhythm change
        0.06,   # [4] gait_phase   — slowest, gait pattern change
        0.12,   # [5] duty_factor  — medium, swing/stance split
        0.20,   # [6] vx           — reserved (overridden by auto ray steering)
        0.20,   # [7] vy           — reserved (overridden by auto ray steering)
        0.20,   # [8] vrot         — reserved (overridden by auto ray steering)
        0.15,   # [9] body_x       — slow body translation
        0.15,   # [10] body_y      — slow body translation
        0.15,   # [11] body_z      — slow body translation
    ], dtype=np.float32)

    # Physical parameter bounds used by low-level controller.
    # [body_roll, body_pitch, step_height, step_dur, gait_phase, duty_factor,
    #  vx, vy, vrot, body_x, body_y, body_z]
    PARAM_LOW = np.array([
        -1.0, -1.0, 0.03, 0.10, math.pi / 3, 0.20,
        -0.1, -0.1, -0.1,
        -1.0, -1.0, -1.0,
    ], dtype=np.float32)
    PARAM_HIGH = np.array([
        1.0, 1.0, 0.12, 0.50, math.pi, 0.80,
        0.1, 0.1, 0.1,
        0.5, 0.5, 0.5,
    ], dtype=np.float32)

    # Hand-tuned baseline. RL outputs residuals around this point.
    BASE_ACTION = np.array([
        0.0, 0.0, 0.06, 0.10, math.pi, 0.50,
        0.00, 0.00, 0.00,
        0.00, 0.00, 0.00,
    ], dtype=np.float32)

    # Lightweight obstacle sensing (front arc raycasts).
    RAY_COUNT = 11
    RAY_LENGTH = 1.25
    RAY_ANGLE_MIN = -1.0
    RAY_ANGLE_MAX = 1.0
    VISUALIZE_RAYS = True
    YAW_DEADBAND = 0.03
    RAY_AVOID_WEIGHT = 0.40
    PID_KP = 0.0
    PID_KI = 0.03
    PID_KD = 0.0
    PID_I_CLAMP = 0.6
    VX_MAX = 0.085
    VX_MIN = 0.020
    VX_CLEAR_FREE = 0.88
    VX_CLEAR_BLOCKED = 0.38
    VY_SIDE_GAIN = 0.1
    VXY_FILTER_ALPHA = 0.25

    def __init__(self, is_train=False):
        self.train        = is_train
        self.start_time   = time.time()
        self.time_elapsed = 0.0

        self.gait_time = 0.0

        self.stuck_counter        = 0
        self.body_touching_ground = False

        # Smoothed action state — interpolated toward target each substep.
        self._smooth_action = self.BASE_ACTION.copy()

        self.logs     = []
        self.step_num = 0
        self._ray_debug_ids = []
        self._latest_rays = np.ones(self.RAY_COUNT, dtype=np.float32)
        self._vx_cmd = float(self.BASE_ACTION[6])
        self._vy_cmd = float(self.BASE_ACTION[7])
        self._vrot_cmd = 0.0
        self._yaw_pid_i = 0.0
        self._yaw_pid_prev_e = 0.0

        super(HexapodEnv, self).__init__()

        if is_train:
            self.client = p.connect(p.DIRECT)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.GUI)
            threading.Thread(target=get_robot_cam, daemon=True).start()

        self.robot = self.load_hexapod_model()

        # -------------------------------------------------------------------
        # ACTION SPACE  (12 values, normalized residuals)
        # The policy predicts values in [-1, 1], then we map to residuals
        # around BASE_ACTION and clamp to physical bounds.
        # -------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

                # -------------------------------------------------------------------
                # OBSERVATION SPACE
                #   [0]    roll             body roll (rad)
                #   [1]    pitch            body pitch (rad)
                #   [2]    yaw_error        heading error (rad)
                #   [3]    lin_vx           base linear velocity x
                #   [4]    lin_vy           base linear velocity y
                #   [5]    ang_vz           base angular velocity z
                #   [6-11] foot_0..5        foot contact flags
                #   [12]   dist_norm        normalized distance-to-target
                #   [13]   time_norm        normalized remaining time
                #   [14-25]smooth_action    current low-level params (incl. vx, vy, vrot, body_x/y/z)
                #   [26-36]ray_front        normalized front ray distances
                # -------------------------------------------------------------------
        obs_low  = np.array(
                        [-math.pi, -math.pi, -math.pi,
                         -2.0, -2.0, -5.0,
              0, 0, 0, 0, 0, 0,
                            0.0, 0.0,
                         *self.PARAM_LOW.tolist(),
                         *([0.0] * self.RAY_COUNT)],
            dtype=np.float32
        )
        obs_high = np.array(
                        [ math.pi,  math.pi,  math.pi,
                            2.0, 2.0, 5.0,
              1, 1, 1, 1, 1, 1,
                            1.0, 1.0,
                         *self.PARAM_HIGH.tolist(),
                         *([1.0] * self.RAY_COUNT)],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._prev_pos  = np.array(HOME_POSITION[:2], dtype=np.float32)
        self._prev_time = time.time()

    # -----------------------------------------------------------------------

    def load_hexapod_model(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ground = p.loadURDF("plane.urdf")
        add_field()
        p.changeDynamics(ground, -1, lateralFriction=5)
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0 if self.train else 1)
        robot_id = p.loadURDF(
            "models/hexapod.urdf",
            basePosition=HOME_POSITION,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        return robot_id

    def save_logs_to_csv(self, filename=None):
        if filename is None:
            filename = f"hexapod_logs_{self.step_num}.csv"
        pd.DataFrame(self.logs).to_csv(filename, index=False)
        print(f"📁 Logs saved to {filename}")

    def set_yaw_pid_gains(self, kp=None, ki=None, kd=None, ray_weight=None):
        if kp is not None:
            self.PID_KP = float(kp)
        if ki is not None:
            self.PID_KI = float(ki)
        if kd is not None:
            self.PID_KD = float(kd)
        if ray_weight is not None:
            self.RAY_AVOID_WEIGHT = float(ray_weight)

    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        print("Robot RESET...")
        p.resetSimulation()
        self.robot = self.load_hexapod_model()

        self.stuck_counter  = 0
        self.gait_time      = 0.0
        self.time_elapsed   = 0.0
        self.start_time     = time.time()
        self._prev_pos      = np.array(HOME_POSITION[:2], dtype=np.float32)
        self._prev_time     = time.time()
        self._smooth_action = self.BASE_ACTION.copy()
        self._ray_debug_ids = []
        self._latest_rays   = np.ones(self.RAY_COUNT, dtype=np.float32)
        self._vx_cmd = float(self.BASE_ACTION[6])
        self._vy_cmd = float(self.BASE_ACTION[7])
        self._vrot_cmd = float(self.BASE_ACTION[8])
        self._yaw_pid_i = 0.0
        self._yaw_pid_prev_e = 0.0

        return self._get_obs(), {}

    # -----------------------------------------------------------------------

    def _get_foot_contacts(self):
        contacts      = p.getContactPoints(bodyA=self.robot)
        foot_contacts = [0] * 6
        for c in contacts:
            link_idx = c[3]
            for leg in range(6):
                if link_idx == leg * 3 + 3:
                    foot_contacts[leg] = 1
        return foot_contacts

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler    = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)

        target_angle = math.atan2(TARGET_POSITION[1] - pos[1],
                                  TARGET_POSITION[0] - pos[0])
        yaw_error    = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        foot_contacts = self._get_foot_contacts()

        dist_to_target = math.sqrt((TARGET_POSITION[0] - pos[0]) ** 2 +
                                   (TARGET_POSITION[1] - pos[1]) ** 2)
        dist_norm  = float(np.clip(dist_to_target / DISTANCE, 0.0, 1.0))
        elapsed    = self.gait_time
        time_norm  = float(np.clip(1.0 - elapsed / TIMEOUT, 0.0, 1.0))
        ray_front = self._latest_rays
        if ray_front.shape[0] != self.RAY_COUNT:
            ray_front = self._get_front_ray_distances(pos, orn)

        return np.array(
            [roll, pitch, yaw_error,
             float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])] +
            foot_contacts +
            [dist_norm, time_norm] +
            self._smooth_action.tolist() +
            ray_front.tolist(),
            dtype=np.float32
        )

    def _get_front_ray_distances(self, pos, orn):
        ray_angles = np.linspace(self.RAY_ANGLE_MIN, self.RAY_ANGLE_MAX, self.RAY_COUNT)
        ray_from = []
        ray_to = []

        rot = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)

        # Model forward axis is opposite local +X in this URDF setup.
        fwd_world = rot @ np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        right_world = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        fwd_world /= (np.linalg.norm(fwd_world) + 1e-8)
        right_world /= (np.linalg.norm(right_world) + 1e-8)
        up_world /= (np.linalg.norm(up_world) + 1e-8)

        origin = (np.array(pos, dtype=np.float32) +
                  0.15 * fwd_world +
                  0.03 * up_world)

        for da in ray_angles:
            direction = math.cos(float(da)) * fwd_world + math.sin(float(da)) * right_world
            direction /= (np.linalg.norm(direction) + 1e-8)
            start = origin.tolist()
            end = (origin + self.RAY_LENGTH * direction).tolist()
            ray_from.append(start)
            ray_to.append(end)

        hits = p.rayTestBatch(ray_from, ray_to)
        dists = []
        for i, hit in enumerate(hits):
            hit_obj = hit[0]
            hit_frac = hit[2]
            dist_norm = 1.0 if hit_obj < 0 else float(np.clip(hit_frac, 0.0, 1.0))
            dists.append(dist_norm)

            if (not self.train) and self.VISUALIZE_RAYS:
                start = ray_from[i]
                full_end = ray_to[i]
                hit_end = [
                    start[0] + (full_end[0] - start[0]) * dist_norm,
                    start[1] + (full_end[1] - start[1]) * dist_norm,
                    start[2] + (full_end[2] - start[2]) * dist_norm,
                ]
                color = [1.0, 0.2, 0.2] if hit_obj >= 0 else [0.2, 0.9, 0.2]

                if i < len(self._ray_debug_ids):
                    self._ray_debug_ids[i] = p.addUserDebugLine(
                        start,
                        hit_end,
                        color,
                        lineWidth=1.8,
                        lifeTime=0,
                        replaceItemUniqueId=self._ray_debug_ids[i],
                    )
                else:
                    debug_id = p.addUserDebugLine(
                        start,
                        hit_end,
                        color,
                        lineWidth=1.8,
                        lifeTime=0,
                    )
                    self._ray_debug_ids.append(debug_id)

        return np.array(dists, dtype=np.float32)

    # -----------------------------------------------------------------------

    def step(self, action):
        action = np.array(action, dtype=np.float32)

        # Residual control around a hand-tuned baseline keeps exploration safe.
        target_action = self.BASE_ACTION +  np.clip(action, -1.0, 1.0)
        target_action = np.clip(target_action, self.PARAM_LOW, self.PARAM_HIGH)

        pos_before, _ = p.getBasePositionAndOrientation(self.robot)
        dist_before   = math.sqrt((TARGET_POSITION[0] - pos_before[0]) ** 2 +
                                  (TARGET_POSITION[1] - pos_before[1]) ** 2)

        # Run SUBSTEPS_PER_CYCLE simulation substeps.
        # The smooth action is interpolated toward the target action
        # incrementally each substep — this is what eliminates jerkiness.
        self._run_substeps(target_action)

        obs            = self._get_obs()
        pos_after, orn = p.getBasePositionAndOrientation(self.robot)
        euler          = p.getEulerFromQuaternion(orn)
        roll_deg       = math.degrees(euler[0])
        pitch_deg      = math.degrees(euler[1])

        dist_after     = math.sqrt((TARGET_POSITION[0] - pos_after[0]) ** 2 +
                                   (TARGET_POSITION[1] - pos_after[1]) ** 2)
        distance_moved = dist_before - dist_after

        if distance_moved < 0.003:
            self.stuck_counter += 1
        elif distance_moved > 0.015 and self.stuck_counter > 0:
            self.stuck_counter = max(self.stuck_counter - 2, 0)

        contacts = p.getContactPoints(bodyA=self.robot)
        self.body_touching_ground = any(c[3] == 0 for c in contacts)

        self.time_elapsed = self.gait_time

        reward, done = self._compute_reward(distance_moved, dist_after,
                                            roll_deg, pitch_deg)

        if not done and self.time_elapsed > TIMEOUT:
            print("⏱️ Timeout!")
            reward -= 50
            done    = True

        self.logs.append({
            'step'              : self.step_num,
            'time'              : self.time_elapsed,
            'action_body_roll'  : float(self._smooth_action[0]),
            'action_body_pitch' : float(self._smooth_action[1]),
            'action_step_height': float(self._smooth_action[2]),
            'action_step_dur'   : float(self._smooth_action[3]),
            'action_gait'       : float(self._smooth_action[4]),
            'action_duty'       : float(self._smooth_action[5]),
            'action_vx'         : float(self._smooth_action[6]),
            'action_vy'         : float(self._smooth_action[7]),
            'action_vrot'       : float(self._smooth_action[8]),
            'action_body_x'     : float(self._smooth_action[9]),
            'action_body_y'     : float(self._smooth_action[10]),
            'action_body_z'     : float(self._smooth_action[11]),
            'reward'            : reward,
            'dist_to_target'    : dist_after,
            'roll'              : roll_deg,
            'pitch'             : pitch_deg,
            'stuck_counter'     : self.stuck_counter,
        })
        self.step_num += 1

        return obs, reward, done, False, {}

    # -----------------------------------------------------------------------

    def _run_substeps(self, target_action):
        """
        Run SUBSTEPS_PER_CYCLE simulation steps.

          Key design decisions:
             1. A simple gait clock is continuous across RL steps,
                 so rhythm is never interrupted.
             2. _smooth_action interpolates toward target_action each substep —
           CPG params change gradually, never jump.
                         3. body_kinematics is recomputed only when smooth_action changes.
                         4. vx/vy/vrot are auto-steered from ray clearance + goal heading.
        """
        for _ in range(self.SUBSTEPS_PER_CYCLE):

            # --- 1. Smoothly interpolate action toward target ---
            self._smooth_action = (self.ACTION_ALPHA * target_action +
                                   (1.0 - self.ACTION_ALPHA) * self._smooth_action)

            sim_dt = 1.0 / 240.0

            body_roll, body_pitch, step_height, step_dur, gait_phase, duty_factor, _, _, _, body_x, body_y, body_z = self._smooth_action

            pos_now, orn_now = p.getBasePositionAndOrientation(self.robot)
            # Ray-driven turning: steer toward the side with more free space.
            ray_front = self._get_front_ray_distances(pos_now, orn_now)
            self._latest_rays = ray_front

            half = self.RAY_COUNT // 2
            right_clear = float(np.mean(ray_front[:half]))
            left_clear = float(np.mean(ray_front[half + 1:]))
            center_clear = float(np.mean(ray_front[max(0, half - 1):min(self.RAY_COUNT, half + 2)]))
            obstacle_scale = float(np.clip((0.85 - center_clear) / 0.35, 0.0, 1.0))
            avoid_error = (left_clear - right_clear) * obstacle_scale

            speed_progress = float(np.clip(
                (center_clear - self.VX_CLEAR_BLOCKED) /
                max(self.VX_CLEAR_FREE - self.VX_CLEAR_BLOCKED, 1e-6),
                0.0,
                1.0,
            ))
            vx_target = self.VX_MIN + (self.VX_MAX - self.VX_MIN) * speed_progress
            vy_target = float(np.clip(
                self.VY_SIDE_GAIN * (right_clear - left_clear) * obstacle_scale,
                self.PARAM_LOW[7],
                self.PARAM_HIGH[7],
            ))

            self._vx_cmd = (
                self.VXY_FILTER_ALPHA * vx_target +
                (1.0 - self.VXY_FILTER_ALPHA) * self._vx_cmd
            )
            self._vy_cmd = (
                self.VXY_FILTER_ALPHA * vy_target +
                (1.0 - self.VXY_FILTER_ALPHA) * self._vy_cmd
            )
            self._vx_cmd = float(np.clip(self._vx_cmd, self.PARAM_LOW[6], self.PARAM_HIGH[6]))
            self._vy_cmd = float(np.clip(self._vy_cmd, self.PARAM_LOW[7], self.PARAM_HIGH[7]))
            self._smooth_action[6] = self._vx_cmd
            self._smooth_action[7] = self._vy_cmd

            target_angle = math.atan2(TARGET_POSITION[1] - pos_now[1],
                                      TARGET_POSITION[0] - pos_now[0])
            yaw_now = p.getEulerFromQuaternion(orn_now)[2]
            yaw_error = (target_angle - yaw_now + math.pi) % (2 * math.pi) - math.pi

            pid_error = yaw_error + self.RAY_AVOID_WEIGHT * avoid_error
            if abs(pid_error) < self.YAW_DEADBAND:
                pid_error = 0.0

            self._yaw_pid_i += pid_error * sim_dt
            self._yaw_pid_i = float(np.clip(self._yaw_pid_i, -self.PID_I_CLAMP, self.PID_I_CLAMP))
            pid_d = (pid_error - self._yaw_pid_prev_e) / sim_dt
            self._yaw_pid_prev_e = pid_error

            vrot_pid = self.PID_KP * pid_error + self.PID_KI * self._yaw_pid_i + self.PID_KD * pid_d
            self._vrot_cmd = float(np.clip(vrot_pid, self.PARAM_LOW[8], self.PARAM_HIGH[8]))
            self._smooth_action[8] = self._vrot_cmd

            # --- 2. Locomotion command from auto ray steering ---
            body_movement = [self._vx_cmd, -self._vy_cmd, -self._vrot_cmd]

            freq_hz = 1.0 / max(step_dur, 1e-3)
            self.gait_time += sim_dt

            # --- 3. Body kinematics (roll/pitch tilt) ---
            leg_pos_body = body_kinematics(
                [body_x, body_y, body_z],
                [body_roll * 20, body_pitch * 20, 0]
            )

            # --- 4. Apply direct phase trajectory to all 6 legs ---
            for leg_index in range(6):
                step_x, step_y, step_z = simple_leg_trajectory(
                    t=self.gait_time,
                    leg_index=leg_index,
                    freq_hz=freq_hz,
                    phase_lag=gait_phase,
                    duty_factor=duty_factor,
                    step_height=step_height,
                    vx=body_movement[0],
                    vy=body_movement[1],
                    v_rot=body_movement[2],
                )

                base = leg_pos_body[leg_index]
                lx, ly, lz = base[0] + step_x, base[1] + step_y, base[2] + step_z

                joint_index = leg_index * 3 + 1
                if joint_index >= 10:
                    lx *= -1
                    ly *= -1

                coxa_angle, femur_angle, tibia_angle = inverse_kinematics(lx, ly, lz)

                p.setJointMotorControl2(self.robot, joint_index,
                                        p.POSITION_CONTROL,
                                        targetPosition=coxa_angle)
                p.setJointMotorControl2(self.robot, joint_index + 1,
                                        p.POSITION_CONTROL,
                                        targetPosition=femur_angle)
                p.setJointMotorControl2(self.robot, joint_index + 2,
                                        p.POSITION_CONTROL,
                                        targetPosition=tibia_angle)

            # --- 5. Update camera global state ---
            global global_position, global_orientation
            pos_now, orn_now   = p.getBasePositionAndOrientation(self.robot)
            global_position    = np.array(pos_now)
            global_orientation = np.array(orn_now)

            p.stepSimulation()
            if not self.train:
                time.sleep(sim_dt)

    # -----------------------------------------------------------------------

    def _calc_energy(self):
        total_power = 0.0
        for j in range(p.getNumJoints(self.robot)):
            js           = p.getJointState(self.robot, j)
            total_power += abs(js[3] * js[1])
        return total_power * (1.0 / 60.0)

    def _compute_reward(self, distance_moved, dist_to_target, roll, pitch):
        done = False

        progress_reward = float(np.clip(distance_moved, -0.05, 0.08)) * 160.0
        forward_bonus   = max(distance_moved, 0.0) * 40.0
        backward_penalty = -max(-distance_moved, 0.0) * 80.0

        # Keep posture stable without dominating the objective.
        stability_penalty = -0.18 * (abs(roll) + abs(pitch))
        stability_penalty = float(np.clip(stability_penalty, -25.0, 0.0))

        energy_penalty = -self._calc_energy() * 0.03
        energy_penalty = float(np.clip(energy_penalty, -8.0, 0.0))

        stuck_penalty = -0.5 * min(self.stuck_counter, 30)
        body_penalty  = -2.0 if self.body_touching_ground else 0.0

        reward = (progress_reward + forward_bonus + backward_penalty +
                  stability_penalty + energy_penalty + stuck_penalty + body_penalty)

        if abs(roll) > 70 or abs(pitch) > 70:
            reward -= 150
            done    = True
            print(f"❌ Flipped! reward={reward:.1f} Roll:{roll:.1f}° Pitch:{pitch:.1f}° time={self.time_elapsed:.1f}s Distance to target: {dist_to_target:.2f}")
        elif self.stuck_counter > 60:
            reward -= 50
            done    = True
            print(f"🛑 Stuck! reward={reward:.1f} counter={self.stuck_counter} time={self.time_elapsed:.1f}s Distance to target: {dist_to_target:.2f}")
        elif dist_to_target < 0.25:
            finish_bonus = 500 + (TIMEOUT - self.time_elapsed) * 1.5
            reward      += finish_bonus
            print(f"✅ Success! reward={reward:.1f} time={self.time_elapsed:.1f}s Distance to target: {dist_to_target:.2f}")
            done = True

        # print(f"Distance to target: {dist_to_target:.2f}")

        reward = float(np.clip(reward, -100, 100))
        # print(f"r: progress={progress_reward:+.2f}  stability={stability_penalty:+.2f}  "
        #       f"energy={energy_penalty:+.2f}  stuck={stuck_penalty:+.2f}  "
        #       f"body={body_penalty:+.2f}  | total={reward:+.2f}")

        return reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()