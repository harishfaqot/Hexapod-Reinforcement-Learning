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

# Simulation parameters
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
        t_start = time.time()

        camera_info     = p.getDebugVisualizerCamera()
        manual_yaw      = camera_info[8]
        manual_pitch    = camera_info[9]
        manual_distance = 1.0 if not camera_distance_set else camera_info[10]

        if not camera_distance_set:
            manual_distance     = 1.0
            camera_distance_set = True

        smoothed_position = alpha * global_position + (1 - alpha) * prev_position
        prev_position     = smoothed_position

        p.resetDebugVisualizerCamera(manual_distance, manual_yaw,
                                     manual_pitch, smoothed_position.tolist())

        pos, orientation   = global_position, global_orientation
        orientation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        x, y, z = pos
        pos = x - 0.2, y, z - 0.4

        camera_pos           = np.array(pos) + np.dot(orientation_matrix, camera_offset)
        camera_target_offset = [-5, 0, -1]
        camera_target        = np.array(pos) + np.dot(orientation_matrix, camera_target_offset)

        view_matrix       = p.computeViewMatrix(camera_pos.tolist(), camera_target.tolist(), [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        p.getCameraImage(
            width  // 2,
            height // 2,
            view_matrix,
            projection_matrix,
            shadow=True,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        time.sleep(1 / 240)


# ===========================================================================
class HexapodEnv(gym.Env):
# ===========================================================================

    def __init__(self, is_train=False):
        self.train             = is_train
        self.progress          = 0
        self.progress_old      = 0
        self.stuck_counter     = 0
        self.start_time        = time.time()
        self.time_elapsed      = 0
        self.body_touching_ground = False

        self.logs              = []
        self.step_num          = 0

        super(HexapodEnv, self).__init__()

        if is_train:
            self.client = p.connect(p.DIRECT)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.GUI)
            t = threading.Thread(target=get_robot_cam, daemon=True)
            t.start()

        self.robot = self.load_hexapod_model()

        # -------------------------------------------------------------------
        # ACTION SPACE
        # RL now controls all meaningful CPG parameters:
        #   [0] vx            — forward speed        [0.00 – 0.15 m/s]
        #   [1] step_height   — leg lift height       [0.03 – 0.12 m]
        #   [2] step_duration — gait cycle period     [0.30 – 0.80 s]
        #   [3] gait_phase    — tripod(1.5) ↔ wave(6) [1.5  – 6.0]
        #   [4] vrot          — yaw/steering rate     [-0.30 – 0.30]
        # -------------------------------------------------------------------
        self.action_space = spaces.Box(
            low  = np.array([0.00, 0.03, 0.30, 1.5, -0.30], dtype=np.float32),
            high = np.array([0.15, 0.12, 0.80, 6.0,  0.30], dtype=np.float32),
            dtype=np.float32
        )

        # -------------------------------------------------------------------
        # OBSERVATION SPACE  (11 values)
        #   [0]     roll        — body roll angle      [-π, π]
        #   [1]     pitch       — body pitch angle     [-π, π]
        #   [2]     yaw_error   — heading error to target [-π, π]
        #   [3–8]   foot_0..5   — foot contact flags   [0, 1]
        #   [9]     vx_actual   — forward velocity     [-1, 1]
        #   [10]    vy_actual   — lateral velocity     [-1, 1]
        # -------------------------------------------------------------------
        obs_low  = np.array(
            [-math.pi, -math.pi, -math.pi,
              0, 0, 0, 0, 0, 0,
             -1.0, -1.0],
            dtype=np.float32
        )
        obs_high = np.array(
            [ math.pi,  math.pi,  math.pi,
              1, 1, 1, 1, 1, 1,
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

        self.stuck_counter = 0
        self.progress      = 0
        self.progress_old  = 0
        self.start_time    = time.time()
        self._prev_pos     = np.array(HOME_POSITION[:2], dtype=np.float32)
        self._prev_time    = time.time()

        return self._get_obs(), {}

    # -----------------------------------------------------------------------

    def _get_foot_contacts(self):
        """Binary contact flag per leg (1 = foot touching something)."""
        contacts      = p.getContactPoints(bodyA=self.robot)
        foot_contacts = [0] * 6
        for c in contacts:
            link_idx = c[3]  # linkIndexA
            for leg in range(6):
                tibia_link = leg * 3 + 3   # tibia is 3rd link per leg
                if link_idx == tibia_link:
                    foot_contacts[leg] = 1
        return foot_contacts

    def _get_velocity(self):
        """Estimate planar body velocity from position delta."""
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        now    = time.time()
        dt     = max(now - self._prev_time, 1e-6)
        vx     = (pos[0] - self._prev_pos[0]) / dt
        vy     = (pos[1] - self._prev_pos[1]) / dt
        self._prev_pos  = np.array(pos[:2])
        self._prev_time = now
        return float(np.clip(vx, -1.0, 1.0)), float(np.clip(vy, -1.0, 1.0))

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler    = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler

        # Heading error: positive = need to turn left, negative = turn right
        target_angle = math.atan2(TARGET_POSITION[1] - pos[1],
                                  TARGET_POSITION[0] - pos[0])
        yaw_error    = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        foot_contacts        = self._get_foot_contacts()          # 6 values
        vx_actual, vy_actual = self._get_velocity()               # 2 values

        obs = np.array(
            [roll, pitch, yaw_error] + foot_contacts + [vx_actual, vy_actual],
            dtype=np.float32
        )
        return obs  # shape (11,)

    # -----------------------------------------------------------------------

    def step(self, action):
        pos_before, _ = p.getBasePositionAndOrientation(self.robot)
        dist_before   = math.sqrt((TARGET_POSITION[0] - pos_before[0]) ** 2 +
                                  (TARGET_POSITION[1] - pos_before[1]) ** 2)

        # Run simulation until one full gait cycle completes
        while True:
            if self.apply_action(action):
                break

        obs           = self._get_obs()
        pos_after, orn = p.getBasePositionAndOrientation(self.robot)
        euler          = p.getEulerFromQuaternion(orn)
        roll_deg, pitch_deg = math.degrees(euler[0]), math.degrees(euler[1])

        dist_after     = math.sqrt((TARGET_POSITION[0] - pos_after[0]) ** 2 +
                                   (TARGET_POSITION[1] - pos_after[1]) ** 2)
        distance_moved = dist_before - dist_after   # positive = moved toward target

        # Stuck detection
        if distance_moved < 0.01:
            self.stuck_counter += 1
        elif distance_moved > 0.05 and self.stuck_counter > 0:
            self.stuck_counter -= 1

        # Body-ground contact (link 0 = body)
        contacts = p.getContactPoints(bodyA=self.robot)
        self.body_touching_ground = any(c[3] == 0 for c in contacts)

        self.time_elapsed = -(time.time() - self.start_time)

        reward, done = self._compute_reward(distance_moved, dist_after,
                                            roll_deg, pitch_deg)

        # Timeout check
        if not done and self.time_elapsed < -TIMEOUT:
            print("⏱️ Timeout!")
            reward -= 50
            done    = True

        self.logs.append({
            'step'              : self.step_num,
            'time'              : time.time() - self.start_time,
            'action_vx'         : float(action[0]),
            'action_step_height': float(action[1]),
            'action_step_dur'   : float(action[2]),
            'action_gait'       : float(action[3]),
            'action_vrot'       : float(action[4]),
            'reward'            : reward,
            'dist_to_target'    : dist_after,
            'roll'              : roll_deg,
            'pitch'             : pitch_deg,
            'stuck_counter'     : self.stuck_counter,
        })
        self.step_num += 1

        return obs, reward, done, False, {}

    # -----------------------------------------------------------------------

    def apply_action(self, action):
        t = time.time() - self.start_time

        pos, orn = p.getBasePositionAndOrientation(self.robot)

        global global_position, global_orientation
        global_position    = np.array(pos)
        global_orientation = np.array(orn)

        # Unpack RL-controlled CPG parameters
        vx, step_height, step_duration, gait_phase, vrot = [float(a) for a in action]
        vy = 0.0

        body_movement = [vx, vy, vrot]

        # One gait cycle complete — signal step() to return
        if self.progress_old > self.progress:
            print(f"Action → vx={vx:.3f} h={step_height:.3f} "
                  f"dur={step_duration:.3f} gait={gait_phase:.2f} vrot={vrot:.3f}")
            self.progress_old = self.progress
            return True

        self.progress_old = self.progress

        for leg_index in range(6):
            leg_pos_delta, progress = generate_movement(
                t, gait_phase, step_duration, step_height,
                body_movement, leg_index
            )
            if leg_index == 0:
                self.progress = progress

            leg_base = leg_base_positions[leg_index]
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

        p.stepSimulation()
        if not self.train:
            time.sleep(1 / 240)

        return False

    # -----------------------------------------------------------------------

    def _calc_energy(self):
        total_power = 0.0
        for j in range(p.getNumJoints(self.robot)):
            js           = p.getJointState(self.robot, j)
            total_power += abs(js[3] * js[1])   # torque × velocity
        return total_power * (1 / 60)

    def _compute_reward(self, distance_moved, dist_to_target, roll, pitch):
        done = False

        # Component weights
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

        # Terminal conditions
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
            reward += finish_bonus
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