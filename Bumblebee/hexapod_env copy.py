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

HOME_POSITION = [2, 0, 0.5]
TARGET_POSITION = [-5, 0, 0.1]
DISTANCE = math.sqrt((TARGET_POSITION[0] - HOME_POSITION[0]) ** 2 + (TARGET_POSITION[1] - HOME_POSITION[1]) ** 2)
TIMEOUT = 180

# Simulation parameters
global_position = np.array([0, 0, 0])
global_orientation  = np.array([0, 0, 0, 0])
width = 512
height = 512
fov = 90
aspect = width / height
near = 0.02
far = 15
camera_offset = [0, 0, 0.5]

def get_robot_cam():
    global global_position, global_orientation

    prev_position = np.array([0, 0, 0])  # Initialize previous position
    alpha = 0.2  # Smoothing factor for interpolation
    camera_distance_set = False

    while True:
        t_start = time.time()

        # Get camera settings
        camera_info = p.getDebugVisualizerCamera()
        manual_yaw = camera_info[8]
        manual_pitch = camera_info[9]
        manual_distance = 1.0 if not camera_distance_set else camera_info[10]

        if not camera_distance_set:
            manual_distance = 1.0
            camera_distance_set = True

        smoothed_position = alpha * global_position + (1 - alpha) * prev_position
        prev_position = smoothed_position

        # Update the camera
        p.resetDebugVisualizerCamera(manual_distance, manual_yaw, manual_pitch, smoothed_position.tolist())

        #-----------------------Robot Camera---------------------------#
        # Get the robot's position and orientation
        pos, orientation = global_position, global_orientation

        # Convert orientation from quaternion to rotation matrix
        orientation_matrix = p.getMatrixFromQuaternion(orientation)
        orientation_matrix = np.array(orientation_matrix).reshape(3, 3)

        x, y, z = pos
        pos = x-0.2, y, z-0.4

        # Calculate camera position in world coordinates
        camera_pos = np.array(pos) + np.dot(orientation_matrix, camera_offset)

        # Set the camera target (in front of the robot)
        camera_target_offset = [-5, 0, -1]  # Forward offset relative to robot
        camera_target = np.array(pos) + np.dot(orientation_matrix, camera_target_offset)

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(camera_pos.tolist(), camera_target.tolist(), [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        p.getCameraImage(
            width // 2,  # Reduce resolution
            height // 2,
            view_matrix,
            projection_matrix,
            shadow=True,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        t_end = time.time()
        t_total = t_end-t_start + 1/1e10
        # print(f"POV Camera: time: {t_total:.2f} seconds, fps: {1/t_total:.2f}")
        time.sleep(1/240)  # 240 Hz update rate

class HexapodEnv(gym.Env):
    def __init__(self, is_train=False):
        self.train = is_train
        self.progress = 0
        self.progress_old = 0
        self.stuck_counter = 0
        self.land_clear = True
        self.start_time = time.time()
        self.time_elapsed = 0
        self.terrain = 0
        self.body_touching_ground = 0

        self.h_step = 0.08
        self.d_step = 0.5
        self.gait = 2

        self.logs = []  # For storing logs
        self.step_num = 0
        self.stability_penalty = 0
        self.energy_penalty = 0
        self.distance =  0
        self.stuck_counter_reward = 0

        super(HexapodEnv, self).__init__()

        self.start_time = time.time()
        # Set up PyBullet simulation
        if is_train:
            self.client = p.connect(p.DIRECT)
            # Disable all visualization during training
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.GUI)
            robot_camera_thread = threading.Thread(target=get_robot_cam)
            robot_camera_thread.start()

        self.robot = self.load_hexapod_model()

        # Define action space (x, y, roll, pitch, step_height)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, 0.03]),
                                       high=np.array([1,  1,  1,  1, 0.08]),
                                       dtype=np.float32)
        
        # Define observation space (roll, pitch, terrain_type)
        self.observation_space = spaces.Box(low=np.array([-1.54, -1.54, 0]),
                                            high=np.array([1.54,  1.54, 20]),
                                            dtype=np.float32)

    def load_hexapod_model(self):
        # Load your hexapod robot URDF here (make sure the model is available)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ground = p.loadURDF("plane.urdf")
        terrain = add_field()

        p.changeDynamics(ground, -1, lateralFriction=5)

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(1)

        model_path = "models/hexapod.urdf"
        robot_id = p.loadURDF(model_path, basePosition=HOME_POSITION, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

        return robot_id
    
    def save_logs_to_csv(self, filename=None):
        if filename is None:
            filename = f"hexapod_logs_episode_{self.episode_num}.csv"

        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"📁 Logs saved to {filename}")

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        print("Robot RESET...")
        p.resetSimulation()
        self.robot = self.load_hexapod_model()
        self.stuck_counter = 0
        self.start_time = time.time()

        initial_state = self.get_state()
        return np.array(initial_state, dtype=np.float32), {}

    def get_state(self):
        # Get robot position and orientation from PyBullet
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        roll_degrees = math.degrees(roll)
        pitch_degrees = math.degrees(pitch)

        if abs(roll_degrees)<5 and abs(pitch_degrees)<5:
            self.terrain = 0
        elif abs(roll_degrees)<10 and abs(pitch_degrees)<10:
            self.terrain = 1
        elif abs(roll_degrees)<15 and abs(pitch_degrees)<15:
            self.terrain = 2
        elif abs(roll_degrees)<20 and abs(pitch_degrees)<20:
            self.terrain = 3
        elif abs(roll_degrees)<25 and abs(pitch_degrees)<25:
            self.terrain = 4
        else:
            self.terrain = 5

        return [roll, pitch, self.terrain]

    def step(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        target = math.sqrt((TARGET_POSITION[0] - pos[0]) ** 2 + (TARGET_POSITION[1] - pos[1]) ** 2)

        # Apply action to the robot (action mapping to robot control)
        while True:
            # Scale only the first five dimensions (x, y, z, roll, pitch)
            scaled_action = action
            
            # Apply the scaled action to the robot
            action_done = self.apply_action(scaled_action)
            if action_done:
                break

        # Get new state (feedback)
        state = self.get_state()
        roll, pitch, terrain = math.degrees(state[0]), math.degrees(state[1]), state[2]
        
        # Calculate reward
        self.reward = self.calculate_reward(state)

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        target_current = math.sqrt((TARGET_POSITION[0] - pos[0]) ** 2 + (TARGET_POSITION[1] - pos[1]) ** 2)

        if not hasattr(self, "prev_position"):
            self.prev_position = target
        
        distance_moved = -target_current + target
        self.prev_position = target_current  # Update last position

        # print(distance_moved, target_current)

        if distance_moved < 0.01:  # If movement is too small
            self.stuck_counter += 1
        elif distance_moved > 0.05 and self.stuck_counter>0:
            self.stuck_counter -= 1
        
        contacts = p.getContactPoints(bodyA=self.robot)
        self.body_touching_ground = False
        for c in contacts:
            if c[3] == 0:  # c[3] = linkIndexA, 0 adalah index dari link "body"
                self.body_touching_ground = True
                break

        done = False
        # Check if robot is flipped or other
        if abs(roll) > 60 or abs(pitch) > 60:
            print(f"❌ Robot Flipped due to Roll exceeding limit! 🚨 Roll: {roll}° | Pitch: {pitch}°")
            self.reward -= 1000
            done = True
        elif self.time_elapsed < -TIMEOUT:
            print(f"⏱️ Robot Flipped due to Negative Time Elapsed! Time: {self.time_elapsed}s")
            self.reward -= 50
            done = True
        elif self.stuck_counter > 20:
            print(f"🛑 Robot Stuck! 🤖 Stuck Counter: {self.stuck_counter}")
            self.reward -= 50
            done = True
        elif distance_moved < -0.01 and terrain>=2:
            print(f"💥 Robot Fall at speed {distance_moved:.2f} m/s 📍Pos: {target_current} 🚧 Stuck Counter: {self.stuck_counter}")
            self.reward -= 50
        elif distance_moved > 0.05 and terrain>=2:
            print(f"⛰️ Robot Climbing at speed {distance_moved:.2f} m/s 💪 📍Pos: {target_current}")
            self.reward += 10
        elif self.body_touching_ground:
            print("⚠️ Robot body is touching the ground! 🚨")
            self.reward -= 1

        if target_current<0.1:
            done = True
            finish_reward = 1000 + (TIMEOUT + self.time_elapsed) * 2
            self.reward += finish_reward 
            print(f"\033[94m[INFO]\033[0m \033[92mRobot Success\033[0m | Reward: \033[93m{self.reward}\033[0m | Time: \033[93m{self.time_elapsed}\033[0m")
        
        # print(f"Reward: \033[93m{self.reward}\033[0m")

        # Log action, reward, and state
        log_data = {
            'step': self.step_num,
            'time': time.time() - self.start_time,
            'action_x': action[0],
            'action_y': action[1],
            'action_roll': action[2],
            'action_pitch': action[3],
            'action_step_height': action[4],
            'reward': self.reward,
            'stability_penalty': self.stability_penalty,
            'energy_penalty': self.energy_penalty,
            'distance_penalty': self.distance,
            'stuck_penalty': self.stuck_counter_reward,
            'state_roll': state[0],
            'state_pitch': state[1],
            'terrain': state[2]
        }
        self.step_num+=1

        self.logs.append(log_data)

        return np.array(state, dtype=np.float32), self.reward, done, False, {}

    def apply_action(self, action):
        t = time.time() - self.start_time

        # Heading alignment reward (robot should face the target)
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(orn)

        global global_position, global_orientation
        global_position = np.array(pos)
        global_orientation = np.array(orn)

        x_pos, y_pos, z_pos = pos
        yaw_pos = euler[2]

        target_angle = math.atan2(TARGET_POSITION[1] - y_pos, TARGET_POSITION[0] - x_pos)
        # Normalize angle difference to [-π, π]
        angle_diff = (target_angle - yaw_pos) % (2 * math.pi) - math.pi
        vz = angle_diff * 0.1  # Increase the scaling factor for better respons

        # Update the only if one cycle done
        if self.progress_old > self.progress:
            print(f"Action: {[f'{val:.3f}' for val in action]}")
            self.progress_old = self.progress
            return True

        # Use the smoothed action for control
        x, y, roll, pitch, self.h_step = action
        z = 0
        yaw = 0
        vx, vy = [0.07, 0]

        body_movement = [vx, vy, vz]
        step_h, step_duration, phase = [self.h_step, self.d_step, self.gait]

        leg_pos_body = body_kinematics([x,y,z], [roll*20, pitch*20, yaw])

        self.progress_old = self.progress
        for leg_index in range(6):  # Iterate over all 6 legs
            # Compute trajectory
            pos, progress = generate_movement(t, phase, step_duration, step_h, body_movement, leg_index) #Pos akan menghasilkan array [x,y,z]
            if leg_index==0:
                self.progress = progress

            leg_base = leg_pos_body[leg_index]
            leg_pos = (leg_base[0] + pos[0], leg_base[1] + pos[1], leg_base[2] + pos[2])
            
            # Control leg joint angles
            target_x, target_y, target_z = leg_pos

            # untuk kaki bagian kiri itu dikalikan negatif
            joint_index=leg_index * 3 + 1
            if joint_index>=10: 
                target_x*=-1
                target_y*=-1
            coxa_angle, femur_angle, tibia_angle = inverse_kinematics(target_x, target_y, target_z)

            p.setJointMotorControl2(self.robot, jointIndex=joint_index, controlMode=p.POSITION_CONTROL, targetPosition=coxa_angle)
            p.setJointMotorControl2(self.robot, jointIndex=joint_index + 1, controlMode=p.POSITION_CONTROL, targetPosition=femur_angle)
            p.setJointMotorControl2(self.robot, jointIndex=joint_index + 2, controlMode=p.POSITION_CONTROL, targetPosition=tibia_angle)

        p.stepSimulation()
        if not self.train:
            time.sleep(1 / 240)

    def calculate_energy_consumption(self):
        total_power = 0.0
        for joint_index in range(p.getNumJoints(self.robot)):
            joint_state = p.getJointState(self.robot, joint_index)
            torque = joint_state[3]  # Applied joint force (torque)
            velocity = joint_state[1]  # Joint velocity
            power = abs(torque * velocity)  # Instantaneous power
            total_power += power

        # Integrate power over time (approximate energy consumption)
        time_step = 1 / 60  # Assuming simulation runs at 240 Hz
        energy_consumption = total_power * time_step
        return energy_consumption

    def calculate_reward(self, state):
        # Weights for different components of the reward
        w1 = 2    # Stability weight (penalize large pitch/roll)
        w2 = 0.5   # Energy efficiency weight (penalize high energy consumption)
        w3 = 0      # Distance
        w4 = 2    # Stuck

        roll, pitch, terrain = math.degrees(state[0]), math.degrees(state[1]), state[2]
        print(f"Terrain Type: {terrain}")

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        distance = DISTANCE - math.sqrt((TARGET_POSITION[0] - pos[0]) ** 2 + (TARGET_POSITION[1] - pos[1]) ** 2)

        # Stability penalty (penalize large pitch/roll)
        stability_penalty = -((abs(roll) + abs(pitch)))
        stability_penalty = np.clip(stability_penalty, -50, 50)

        # Energy efficiency penalty (penalize high energy consumption)
        energy_consumption = self.calculate_energy_consumption()
        energy_penalty = -energy_consumption
        energy_penalty = np.clip(energy_penalty, -50, 50)

        self.time_elapsed = -(time.time() - self.start_time)

        # Total reward
        reward = w1 * stability_penalty + w2 * energy_penalty + w3 * distance + w4 * -self.stuck_counter

        self.stability_penalty = w1 * stability_penalty
        self.energy_penalty = w2 * energy_penalty
        self.distance =  distance
        self.stuck_counter_reward = w4 * -self.stuck_counter

        reward = np.clip(reward, -100, 100)

        print(f"Reward: roll={roll:+06.2f} pitch={pitch:+06.2f} "
            f"stability={stability_penalty:+06.2f}, energy={energy_penalty:+06.2f}, stuck={self.stuck_counter:+06.2f}, time={self.time_elapsed:+06.2f} , d={distance:+06.2f}, total={reward:+06.2f}")

        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()