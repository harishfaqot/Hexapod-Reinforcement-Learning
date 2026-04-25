import math
import os
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data



class HexapodPyBulletEnv:
    def __init__(
        self,
        gui=False,
        target_vel=0.2,
        episode_steps=800,
        frame_skip=8,
        model_path=None,
    ):
        self.gui = gui
        self.target_vel = target_vel
        self.episode_steps = episode_steps
        self.frame_skip = frame_skip

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        if model_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            model_path = repo_root / "nexabots_pybullet" / "assets" / "hex_locomotion" / "hex.xml"
        self.model_path = str(model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Hexapod model not found: {self.model_path}")

        model_dir = os.path.dirname(self.model_path)
        p.setAdditionalSearchPath(model_dir, physicsClientId=self.client_id)
        body_ids = p.loadMJCF(self.model_path, physicsClientId=self.client_id)

        if not body_ids:
            raise RuntimeError(f"loadMJCF returned no bodies for: {self.model_path}")

        # Pick the body with the most joints as the robot.
        self.robot_id = max(
            body_ids,
            key=lambda rid: p.getNumJoints(rid, physicsClientId=self.client_id),
        )

        # PyBullet's MJCF importer can produce a static base (mass=0) for this model.
        # Force a dynamic base so gravity and ground contacts behave correctly.
        base_mass = p.getDynamicsInfo(self.robot_id, -1, physicsClientId=self.client_id)[0]
        if base_mass <= 0.0:
            p.changeDynamics(self.robot_id, -1, mass=1.0, physicsClientId=self.client_id)

        self.joint_ids = []
        self.joint_lows = []
        self.joint_highs = []

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_ids.append(i)
                low, high = info[8], info[9]
                if low >= high:
                    low, high = -1.0, 1.0
                self.joint_lows.append(low)
                self.joint_highs.append(high)

        self.joint_lows = np.asarray(self.joint_lows, dtype=np.float32)
        self.joint_highs = np.asarray(self.joint_highs, dtype=np.float32)
        self.joint_ranges = self.joint_highs - self.joint_lows

        if len(self.joint_ids) == 0:
            raise RuntimeError(f"No actuated joints found in model: {self.model_path}")

        self.act_dim = len(self.joint_ids)
        self.obs_dim = 7 + 6 + self.act_dim * 2

        self.step_ctr = 0
        self.prev_x = 0.0

        self.reset()

    def _scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        return (action * 0.5 + 0.5) * self.joint_ranges + self.joint_lows

    def _get_obs(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        base_linvel, base_angvel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        joint_states = p.getJointStates(self.robot_id, self.joint_ids, physicsClientId=self.client_id)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]

        obs = np.concatenate(
            [
                np.asarray(base_pos, dtype=np.float32),
                np.asarray(base_quat, dtype=np.float32),
                np.asarray(base_linvel, dtype=np.float32),
                np.asarray(base_angvel, dtype=np.float32),
                np.asarray(joint_pos, dtype=np.float32),
                np.asarray(joint_vel, dtype=np.float32),
            ]
        )
        return obs

    def step(self, action):
        target_pos = self._scale_action(action)
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=target_pos.tolist(),
            forces=[3.0] * self.act_dim,
            physicsClientId=self.client_id,
        )

        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.gui:
                time.sleep(1.0 / 240.0)

        self.step_ctr += 1

        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        base_linvel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        roll, pitch, _ = p.getEulerFromQuaternion(base_quat)

        x = base_pos[0]
        forward_progress = (x - self.prev_x) * 20.0
        self.prev_x = x

        vel_error = abs(base_linvel[0] - self.target_vel)
        ctrl_pen = float(np.square(np.asarray(action, dtype=np.float32)).mean())
        tilt_pen = float(roll * roll + pitch * pitch)

        reward = forward_progress - 0.3 * vel_error - 0.02 * ctrl_pen - 0.2 * tilt_pen

        done = False
        if self.step_ctr >= self.episode_steps:
            done = True
        if base_pos[2] < 0.08:
            done = True
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            done = True

        return self._get_obs(), float(reward), done, {}

    def reset(self):
        self.step_ctr = 0

        yaw = np.random.uniform(-0.15, 0.15)
        start_quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0.0, 0.0, 0.18],
            start_quat,
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.robot_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

        for j in self.joint_ids:
            p.resetJointState(self.robot_id, j, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.client_id)

        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=[0.0] * self.act_dim,
            forces=[3.0] * self.act_dim,
            physicsClientId=self.client_id,
        )

        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client_id)

        self.prev_x = 0.0
        return self._get_obs()

    def render(self):
        # GUI mode renders automatically when stepping.
        return None

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
