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
    ], dtype=np.float32)

    # Physical gait parameter bounds used by low-level controller.
    PARAM_LOW = np.array([-1.0, -1.0, 0.03, 0.10, math.pi / 3, 0.20], dtype=np.float32)
    PARAM_HIGH = np.array([1.0, 1.0, 0.12, 0.50, math.pi, 0.80], dtype=np.float32)

    # Hand-tuned baseline gait. RL outputs residuals around this point.
    BASE_ACTION = np.array([0.0, 0.0, 0.06, 0.30, math.pi, 0.50], dtype=np.float32)
    RESIDUAL_SCALE = np.array([0.35, 0.35, 0.02, 0.08, 0.45, 0.12], dtype=np.float32)

    def __init__(self, is_train=False):
        self.train        = is_train
        self.start_time   = time.time()
        self.time_elapsed = 0.0

        self.gait_time = 0.0

        self.stuck_counter        = 0
        self.body_touching_ground = False

        # Smoothed action state — interpolated toward target each substep
        # [body_roll, body_pitch, step_height, step_dur, gait_phase, duty_factor]
        self._smooth_action = np.array([0.0, 0.0, 0.06, 0.1, 2.0, 0.5], dtype=np.float32)

        # Fixed locomotion params (RL does NOT control these)
        self.VX   = 0.07    # constant forward speed
        self.VROT_SCALE = 0.15   # auto yaw correction scale

        self.logs     = []
        self.step_num = 0

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
        # ACTION SPACE  (6 values, normalized residuals)
        # The policy predicts values in [-1, 1], then we map to residuals
        # around BASE_ACTION and clamp to physical bounds.
        # -------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

                # -------------------------------------------------------------------
                # OBSERVATION SPACE  (20 values)
                #   [0]    roll             body roll (rad)
                #   [1]    pitch            body pitch (rad)
                #   [2]    yaw_error        heading error (rad)
                #   [3]    lin_vx           base linear velocity x
                #   [4]    lin_vy           base linear velocity y
                #   [5]    ang_vz           base angular velocity z
                #   [6-11] foot_0..5        foot contact flags
                #   [12]   dist_norm        normalized distance-to-target
                #   [13]   time_norm        normalized remaining time
                #   [14-19]smooth_action    current low-level gait params
                # -------------------------------------------------------------------
        obs_low  = np.array(
                        [-math.pi, -math.pi, -math.pi,
                         -2.0, -2.0, -5.0,
              0, 0, 0, 0, 0, 0,
                            0.0, 0.0,
                         *self.PARAM_LOW.tolist()],
            dtype=np.float32
        )
        obs_high = np.array(
                        [ math.pi,  math.pi,  math.pi,
                            2.0, 2.0, 5.0,
              1, 1, 1, 1, 1, 1,
                            1.0, 1.0,
                         *self.PARAM_HIGH.tolist()],
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

        return np.array(
            [roll, pitch, yaw_error,
             float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])] +
            foot_contacts +
            [dist_norm, time_norm] +
            self._smooth_action.tolist(),
            dtype=np.float32
        )

    # -----------------------------------------------------------------------

    def step(self, action):
        action = np.array(action, dtype=np.float32)

        # Residual control around a hand-tuned baseline keeps exploration safe.
        target_action = self.BASE_ACTION + self.RESIDUAL_SCALE * np.clip(action, -1.0, 1.0)
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
             3. body_kinematics is recomputed only when smooth_action changes,
           not every substep unnecessarily.
             4. steering (vrot) computed from yaw_error, forward speed is fixed.
        """
        for _ in range(self.SUBSTEPS_PER_CYCLE):

            # --- 1. Smoothly interpolate action toward target ---
            self._smooth_action = (self.ACTION_ALPHA * target_action +
                                   (1.0 - self.ACTION_ALPHA) * self._smooth_action)

            body_roll, body_pitch, step_height, step_dur, gait_phase, duty_factor = self._smooth_action

            # --- 2. Auto steering toward target (fixed, not RL) ---
            pos, orn = p.getBasePositionAndOrientation(self.robot)
            euler    = p.getEulerFromQuaternion(orn)
            yaw      = euler[2]

            target_angle = math.atan2(TARGET_POSITION[1] - pos[1],
                                      TARGET_POSITION[0] - pos[0])
            yaw_error    = (target_angle - yaw) % (2 * math.pi) - math.pi
            vrot         = yaw_error * self.VROT_SCALE

            body_movement = [self.VX, 0.0, vrot]

            sim_dt = 1.0 / 240.0
            freq_hz = 1.0 / max(step_dur, 1e-3)
            self.gait_time += sim_dt

            # --- 3. Body kinematics (roll/pitch tilt) ---
            leg_pos_body = body_kinematics(
                [0, 0, 0],
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
            print(f"❌ Flipped! Roll:{roll:.1f}° Pitch:{pitch:.1f}°")
            reward -= 150
            done    = True
        elif self.stuck_counter > 60:
            print(f"🛑 Stuck! counter={self.stuck_counter}")
            reward -= 50
            done    = True
        elif dist_to_target < 0.1:
            finish_bonus = 500 + (TIMEOUT - self.time_elapsed) * 1.5
            reward      += finish_bonus
            print(f"✅ Success! reward={reward:.1f} time={self.time_elapsed:.1f}s")
            done = True

        reward = float(np.clip(reward, -100, 100))
        # print(f"r: progress={progress_reward:+.2f}  stability={stability_penalty:+.2f}  "
        #       f"energy={energy_penalty:+.2f}  stuck={stuck_penalty:+.2f}  "
        #       f"body={body_penalty:+.2f}  | total={reward:+.2f}")

        return reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()