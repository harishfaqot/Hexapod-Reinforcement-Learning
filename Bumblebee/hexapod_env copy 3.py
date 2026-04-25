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
TIMEOUT  = 180

global_position    = np.array([0, 0, 0])
global_orientation = np.array([0, 0, 0, 0])
width  = 512
height = 512
fov    = 90
aspect = width / height
near   = 0.02
far    = 15
camera_offset = [0, 0, 0.5]


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

        x, y, z = pos
        pos_cam = (x - 0.2, y, z - 0.4)
        camera_pos    = np.array(pos_cam) + np.dot(orientation_matrix, camera_offset)
        camera_target = np.array(pos_cam) + np.dot(orientation_matrix, [-5, 0, -1])

        view_matrix       = p.computeViewMatrix(camera_pos.tolist(), camera_target.tolist(), [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        p.getCameraImage(width // 2, height // 2, view_matrix, projection_matrix,
                         shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        time.sleep(1 / 240)


# ===========================================================================
class HexapodEnv(gym.Env):
# ===========================================================================

    # Action indices — for readability
    IDX_VX           = 0
    IDX_VY           = 1   # lateral (usually small)
    IDX_ROLL         = 2   # body roll tilt  (body kinematics)
    IDX_PITCH        = 3   # body pitch tilt (body kinematics)
    IDX_STEP_HEIGHT  = 4
    IDX_STEP_DUR     = 5
    IDX_GAIT_PHASE   = 6
    IDX_VROT         = 7

    # Smoothing factor for action interpolation (0=no change, 1=instant)
    # Lower = smoother movement but slower response
    ACTION_ALPHA = 0.25

    def __init__(self, is_train=False):
        self.train        = is_train
        self.start_time   = time.time()
        self.time_elapsed = 0

        # Internal gait state
        self.progress          = 0.0
        self.progress_old      = 0.0
        self.cycle_count       = 0      # how many full cycles completed
        self.body_touching_ground = False
        self.stuck_counter     = 0

        # Smoothed action — prevents patah-patah by interpolating between steps
        self._smooth_action = np.array(
            [0.07, 0.0, 0.0, 0.0, 0.06, 0.5, 2.0, 0.0], dtype=np.float32
        )

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
        # ACTION SPACE  (8 values)
        #
        #   [0] vx           forward speed          [0.00,  0.15]  m/s
        #   [1] vy           lateral speed          [-0.05, 0.05]  m/s
        #   [2] body_roll    body tilt roll         [-1.0,  1.0]   (×20°)
        #   [3] body_pitch   body tilt pitch        [-1.0,  1.0]   (×20°)
        #   [4] step_height  leg lift               [0.03,  0.12]  m
        #   [5] step_dur     gait cycle period      [0.30,  0.80]  s
        #   [6] gait_phase   tripod(1.5)↔wave(6.0) [1.5,   6.0]
        #   [7] vrot         yaw/steering           [-0.30, 0.30]
        # -------------------------------------------------------------------
        self.action_space = spaces.Box(
            low  = np.array([ 0.00, -0.05, -1.0, -1.0, 0.03, 0.30, 1.5, -0.30], dtype=np.float32),
            high = np.array([ 0.15,  0.05,  1.0,  1.0, 0.12, 0.80, 6.0,  0.30], dtype=np.float32),
            dtype=np.float32
        )

        # -------------------------------------------------------------------
        # OBSERVATION SPACE  (13 values)
        #
        #   [0]    roll         body roll angle       [-π, π]
        #   [1]    pitch        body pitch angle      [-π, π]
        #   [2]    yaw_error    heading error         [-π, π]
        #   [3–8]  foot_0..5    foot contact flags    [0,  1]
        #   [9]    vx_actual    forward velocity      [-1, 1]
        #   [10]   vy_actual    lateral velocity      [-1, 1]
        #   [11]   dist_norm    normalised dist to target [0, 1]
        #   [12]   time_norm    normalised time remaining [0, 1]
        # -------------------------------------------------------------------
        obs_low  = np.array(
            [-math.pi, -math.pi, -math.pi,
              0, 0, 0, 0, 0, 0,
             -1.0, -1.0,
              0.0,  0.0],
            dtype=np.float32
        )
        obs_high = np.array(
            [ math.pi,  math.pi,  math.pi,
              1, 1, 1, 1, 1, 1,
              1.0,  1.0,
              1.0,  1.0],
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
        p.setRealTimeSimulation(1)
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
        self.progress       = 0.0
        self.progress_old   = 0.0
        self.cycle_count    = 0
        self.start_time     = time.time()
        self._prev_pos      = np.array(HOME_POSITION[:2], dtype=np.float32)
        self._prev_time     = time.time()

        # Reset smooth action to neutral walking params
        self._smooth_action = np.array(
            [0.07, 0.0, 0.0, 0.0, 0.06, 0.5, 2.0, 0.0], dtype=np.float32
        )

        return self._get_obs(), {}

    # -----------------------------------------------------------------------

    def _get_foot_contacts(self):
        """Binary contact flag per leg — tibia link is leg*3+3."""
        contacts      = p.getContactPoints(bodyA=self.robot)
        foot_contacts = [0] * 6
        for c in contacts:
            link_idx = c[3]
            for leg in range(6):
                if link_idx == leg * 3 + 3:
                    foot_contacts[leg] = 1
        return foot_contacts

    def _get_velocity(self):
        """Estimate planar velocity from position delta."""
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        now    = time.time()
        dt     = max(now - self._prev_time, 1e-6)
        vx     = (pos[0] - float(self._prev_pos[0])) / dt
        vy     = (pos[1] - float(self._prev_pos[1])) / dt
        self._prev_pos  = np.array(pos[:2], dtype=np.float32)
        self._prev_time = now
        return float(np.clip(vx, -1.0, 1.0)), float(np.clip(vy, -1.0, 1.0))

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler    = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler

        # Heading error to target, normalised to [-π, π]
        target_angle = math.atan2(TARGET_POSITION[1] - pos[1],
                                  TARGET_POSITION[0] - pos[0])
        yaw_error    = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        foot_contacts        = self._get_foot_contacts()
        vx_actual, vy_actual = self._get_velocity()

        dist_to_target = math.sqrt((TARGET_POSITION[0] - pos[0]) ** 2 +
                                   (TARGET_POSITION[1] - pos[1]) ** 2)
        dist_norm = float(np.clip(dist_to_target / DISTANCE, 0.0, 1.0))

        elapsed    = time.time() - self.start_time
        time_norm  = float(np.clip(1.0 - elapsed / TIMEOUT, 0.0, 1.0))

        obs = np.array(
            [roll, pitch, yaw_error] +
            foot_contacts +
            [vx_actual, vy_actual, dist_norm, time_norm],
            dtype=np.float32
        )
        return obs   # shape (13,)

    # -----------------------------------------------------------------------

    def step(self, action):
        pos_before, _ = p.getBasePositionAndOrientation(self.robot)
        dist_before   = math.sqrt((TARGET_POSITION[0] - pos_before[0]) ** 2 +
                                  (TARGET_POSITION[1] - pos_before[1]) ** 2)

        # ---- Smooth the incoming action to avoid jerky transitions ----
        # Interpolate: smooth = alpha*new + (1-alpha)*old
        # CPG params (step_dur, gait_phase) use slower alpha — they affect
        # leg rhythm and need more time to transition without stumbling
        alpha        = np.full(8, self.ACTION_ALPHA, dtype=np.float32)
        alpha[5]     = 0.10   # step_duration  — slow transition
        alpha[6]     = 0.08   # gait_phase     — slowest, rhythm change
        self._smooth_action = alpha * np.array(action, dtype=np.float32) + \
                              (1 - alpha) * self._smooth_action

        # Run simulation until exactly ONE full gait cycle completes
        self._run_one_cycle(self._smooth_action)

        obs            = self._get_obs()
        pos_after, orn = p.getBasePositionAndOrientation(self.robot)
        euler          = p.getEulerFromQuaternion(orn)
        roll_deg       = math.degrees(euler[0])
        pitch_deg      = math.degrees(euler[1])

        dist_after     = math.sqrt((TARGET_POSITION[0] - pos_after[0]) ** 2 +
                                   (TARGET_POSITION[1] - pos_after[1]) ** 2)
        distance_moved = dist_before - dist_after   # positive = toward target

        # Stuck detection
        if distance_moved < 0.01:
            self.stuck_counter += 1
        elif distance_moved > 0.05 and self.stuck_counter > 0:
            self.stuck_counter -= 1

        # Body-ground contact (link index 0 = body)
        contacts = p.getContactPoints(bodyA=self.robot)
        self.body_touching_ground = any(c[3] == 0 for c in contacts)

        self.time_elapsed = -(time.time() - self.start_time)

        reward, done = self._compute_reward(distance_moved, dist_after,
                                            roll_deg, pitch_deg)

        if not done and self.time_elapsed < -TIMEOUT:
            print("⏱️ Timeout!")
            reward -= 50
            done    = True

        self.logs.append({
            'step'              : self.step_num,
            'time'              : time.time() - self.start_time,
            'action_vx'         : float(self._smooth_action[0]),
            'action_vy'         : float(self._smooth_action[1]),
            'action_body_roll'  : float(self._smooth_action[2]),
            'action_body_pitch' : float(self._smooth_action[3]),
            'action_step_height': float(self._smooth_action[4]),
            'action_step_dur'   : float(self._smooth_action[5]),
            'action_gait'       : float(self._smooth_action[6]),
            'action_vrot'       : float(self._smooth_action[7]),
            'reward'            : reward,
            'dist_to_target'    : dist_after,
            'roll'              : roll_deg,
            'pitch'             : pitch_deg,
            'stuck_counter'     : self.stuck_counter,
        })
        self.step_num += 1

        return obs, reward, done, False, {}

    # -----------------------------------------------------------------------

    def _run_one_cycle(self, action):
        """
        Run simulation substeps until exactly one full gait cycle completes.
        A cycle is detected when progress wraps: progress < progress_old
        (i.e., it crossed 0 again after being near 1).

        This ensures every RL step = one complete gait cycle,
        regardless of simulation speed — no more patah-patah.
        """
        vx, vy, body_roll, body_pitch, step_height, step_dur, gait_phase, vrot = action

        # Safety clamp after smoothing
        vx         = float(np.clip(vx,         0.00,  0.15))
        vy         = float(np.clip(vy,        -0.05,  0.05))
        body_roll  = float(np.clip(body_roll, -1.0,   1.0))
        body_pitch = float(np.clip(body_pitch,-1.0,   1.0))
        step_height= float(np.clip(step_height,0.03,  0.12))
        step_dur   = float(np.clip(step_dur,   0.30,  0.80))
        gait_phase = float(np.clip(gait_phase, 1.5,   6.0))
        vrot       = float(np.clip(vrot,      -0.30,  0.30))

        body_movement = [vx, vy, vrot]

        # Pre-compute body kinematics ONCE per RL step (not per substep)
        # body_roll/pitch are [-1,1] mapped to [-20°, 20°] inside body_kinematics
        leg_pos_body = body_kinematics([0, 0, 0], [body_roll * 20, body_pitch * 20, 0])

        cycle_started = False
        max_substeps  = 2000   # safety limit to avoid infinite loop

        for _ in range(max_substeps):
            t = time.time() - self.start_time

            prev_progress = self.progress

            for leg_index in range(6):
                leg_pos_delta, progress = generate_movement(
                    t, gait_phase, step_dur, step_height,
                    body_movement, leg_index
                )
                if leg_index == 0:
                    self.progress = progress

            # Detect cycle wrap-around: progress jumps from ~1.0 back to ~0.0
            if cycle_started and self.progress < 0.1 and prev_progress > 0.8:
                self.cycle_count += 1
                break

            if self.progress > 0.1:
                cycle_started = True

            # Apply leg positions
            for leg_index in range(6):
                leg_pos_delta, _ = generate_movement(
                    t, gait_phase, step_dur, step_height,
                    body_movement, leg_index
                )
                leg_base = leg_pos_body[leg_index]
                lx = leg_base[0] + leg_pos_delta[0]
                ly = leg_base[1] + leg_pos_delta[1]
                lz = leg_base[2] + leg_pos_delta[2]

                joint_index = leg_index * 3 + 1
                if joint_index >= 10:
                    lx *= -1
                    ly *= -1

                coxa_angle, femur_angle, tibia_angle = inverse_kinematics(lx, ly, lz)

                p.setJointMotorControl2(self.robot, jointIndex=joint_index,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=coxa_angle)
                p.setJointMotorControl2(self.robot, jointIndex=joint_index + 1,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=femur_angle)
                p.setJointMotorControl2(self.robot, jointIndex=joint_index + 2,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=tibia_angle)

            # Update global camera state
            pos_now, orn_now = p.getBasePositionAndOrientation(self.robot)
            global global_position, global_orientation
            global_position    = np.array(pos_now)
            global_orientation = np.array(orn_now)

            p.stepSimulation()
            if not self.train:
                time.sleep(1 / 240)

    # -----------------------------------------------------------------------

    def _calc_energy(self):
        total_power = 0.0
        for j in range(p.getNumJoints(self.robot)):
            js           = p.getJointState(self.robot, j)
            total_power += abs(js[3] * js[1])
        return total_power * (1 / 60)

    def _compute_reward(self, distance_moved, dist_to_target, roll, pitch):
        done = False

        w_progress  = 5.0
        w_stability = 2.0
        w_energy    = 0.5
        w_stuck     = 2.0
        w_body      = 1.0

        progress_reward   =  distance_moved * w_progress
        stability_penalty = -((abs(roll) + abs(pitch))) * w_stability
        stability_penalty =  float(np.clip(stability_penalty, -50, 50))
        energy_penalty    = -self._calc_energy() * w_energy
        energy_penalty    =  float(np.clip(energy_penalty, -50, 50))
        stuck_penalty     = -self.stuck_counter * w_stuck
        body_penalty      = -w_body if self.body_touching_ground else 0.0

        reward = (progress_reward + stability_penalty +
                  energy_penalty  + stuck_penalty + body_penalty)

        if abs(roll) > 60 or abs(pitch) > 60:
            print(f"❌ Flipped! Roll:{roll:.1f}° Pitch:{pitch:.1f}°")
            reward -= 1000
            done    = True
        elif self.stuck_counter > 20:
            print(f"🛑 Stuck! counter={self.stuck_counter}")
            reward -= 50
            done    = True
        elif dist_to_target < 0.1:
            finish_bonus = 1000 + (TIMEOUT + self.time_elapsed) * 2
            reward      += finish_bonus
            print(f"✅ Success! reward={reward:.1f} time={self.time_elapsed:.1f}s")
            done = True

        reward = float(np.clip(reward, -200, 200))

        print(f"r: progress={progress_reward:+.2f}  stability={stability_penalty:+.2f}  "
              f"energy={energy_penalty:+.2f}  stuck={stuck_penalty:+.2f}  "
              f"body={body_penalty:+.2f}  | total={reward:+.2f}")

        return reward, done

    # -----------------------------------------------------------------------

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()