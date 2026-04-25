import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import threading
from lib.hexapod_constant import *
from terrain import add_field

start_time = time.time()

# Connect to PyBullet
p.connect(p.GUI)

# Set the path for the URDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)

# Set gravity
p.setGravity(0, 0, -9.8)

# Load a ground plane
ground = p.loadURDF("plane.urdf")
p.changeDynamics(ground, -1, lateralFriction=1)
add_field()

# Load your URDF
robot_id = p.loadURDF("models/hexapod.urdf", basePosition=[0, 0, 1], useFixedBase=False)
# Simulation parameters
width = 512
height = 512
fov = 90
aspect = width / height
near = 0.02
far = 15
# Camera offset relative to the robot's base
camera_offset = [0, 0, 0.5]  # Offset (x, y, z) from the robot's base position

# Add sliders for controlling vx and vy
vx_slider = p.addUserDebugParameter("vx", -0.07, 0.07, 0.0)  # Range: -1 to 1, initial value: 0
vy_slider = p.addUserDebugParameter("vy", -0.1, 0.1, 0.0)  # Range: -1 to 1, initial value: 0
v_rot_slider = p.addUserDebugParameter("v_rot", -0.1, 0.1, 0.0)

step_height_slider = p.addUserDebugParameter("step height", 0, 0.3, 0.05)
cpg_slider = p.addUserDebugParameter("cpg freq (Hz)", 0.1, 4.0, 1.0)
gait_slider = p.addUserDebugParameter("gait phase lag (rad)", math.pi / 3, math.pi, math.pi)
duty_slider = p.addUserDebugParameter("duty factor", 0.2, 0.8, 0.5)

# User sliders for body position and orientation
x_slider = p.addUserDebugParameter("pos x", -0.1, 0.1, 0)
y_slider = p.addUserDebugParameter("pos y", -0.1, 0.1, 0)
z_slider = p.addUserDebugParameter("pos z", -0.1, 0.1, 0)

r_slider = p.addUserDebugParameter("roll", -60, 60, 0)
p_slider = p.addUserDebugParameter("pitch",-60, 60, 0)
yaw_slider = p.addUserDebugParameter("yaw", -60, 60, 0)

# Global variable for robot position
global_position = np.array([0, 0, 0])
global_orientation  = np.array([0, 0, 0, 0])
lock = threading.Lock()  # Lock for thread safety
def camera_update_thread():
    """Thread function for updating the camera."""
    global global_position
    prev_position = np.array([0, 0, 0])  # Initialize previous position
    alpha = 0.2  # Smoothing factor for interpolation

    while True:
        t_start = time.time()
        with lock:
            # Smooth the camera's movement
            smoothed_position = alpha * global_position + (1 - alpha) * prev_position
            prev_position = smoothed_position

        # Get camera settings
        camera_info = p.getDebugVisualizerCamera()
        manual_yaw = camera_info[8]
        manual_pitch = camera_info[9]
        manual_distance = camera_info[10]

        # Update the camera
        p.resetDebugVisualizerCamera(manual_distance, manual_yaw, manual_pitch, smoothed_position.tolist())

        # Limit the update frequency for the camera thread
        t_end = time.time()
        t_total = t_end-t_start + 1/1e10
        # print(f"Camera: time: {t_total:.2f} seconds, fps: {1/t_total:.2f}")
        time.sleep(1/240)  # 240 Hz camera update rate

def get_robot_cam():
    global global_position, global_orientation
    while True:
        t_start = time.time()
        # Get the robot's position and orientation
        pos, orientation = global_position, global_orientation

        # Convert orientation from quaternion to rotation matrix
        orientation_matrix = p.getMatrixFromQuaternion(orientation)
        orientation_matrix = np.array(orientation_matrix).reshape(3, 3)

        # Calculate camera position in world coordinates
        camera_pos = np.array(pos) + np.dot(orientation_matrix, camera_offset)

        # Set the camera target (in front of the robot)
        camera_target_offset = [-5, 0, 0]  # Forward offset relative to robot
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
    
# Start the camera thread
camera_thread = threading.Thread(target=camera_update_thread, daemon=True)
camera_thread.start()

robot_camera_thread = threading.Thread(target=get_robot_cam)
robot_camera_thread.start()

# Simple journal-style phase oscillator CPG (single function):
# theta_i(t) = 2*pi*f*t + i*DeltaPhi
def cpg_leg_trajectory(t, leg_index, freq_hz, phase_lag, duty_factor, step_height, vx, vy, v_rot):
    phase = (2 * math.pi * freq_hz * t + leg_index * phase_lag) % (2 * math.pi)
    phi = phase / (2 * math.pi)
    duty = min(max(duty_factor, 1e-4), 1.0 - 1e-4)

    # Rotational velocity contributes per-leg lateral offset.
    angle = math.pi / 2 + math.atan2(leg_base_positions[leg_index][0], leg_base_positions[leg_index][1])
    offset_x = v_rot * math.sin(angle)
    offset_y = v_rot * math.cos(angle)
    step_span_x = vx + offset_x
    step_span_y = vy + offset_y

    if phi < duty:
        # Stance phase
        s = phi / duty
        x = step_span_x * (0.5 - s)
        y = step_span_y * (0.5 - s)
        z = 0.0
    else:
        # Swing phase
        s = (phi - duty) / (1.0 - duty)
        x = step_span_x * (s - 0.5)
        y = step_span_y * (s - 0.5)
        z = step_height * math.sin(math.pi * s)

    if vx == 0 and vy == 0 and v_rot == 0:
        z = 0.0

    return x, y, z

# Define inverse kinematics function
def inverse_kinematics(x, y, z):
    coxa_angle = math.pi / 2 + math.atan2(x, y)
    r = math.sqrt(x**2 + y**2)
    d = math.sqrt(r**2 + z**2)

    if d > femur_length + tibia_length:
        d = femur_length + tibia_length

    a1 = math.atan2(z, r)
    A = math.acos((d**2 + femur_length**2 - tibia_length**2) / (2 * d * femur_length))
    femur_angle = math.pi / 2 - (A + a1)

    B = math.acos((femur_length**2 + tibia_length**2 - d**2) / (2 * femur_length * tibia_length))
    tibia_angle = math.pi - B

    return coxa_angle, femur_angle, tibia_angle

# Define body kinematics with rotation
def body_kinematics(body_position, body_orientation):
    x_trans, y_trans, z_trans = body_position
    r, p, y = body_orientation
    roll = math.radians(r)
    pitch = math.radians(p)
    yaw = math.radians(y)

    # Compute the rotation matrix for the given roll, pitch, and yaw
    c_r, s_r = math.cos(roll), math.sin(roll)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    c_y, s_y = math.cos(yaw), math.sin(yaw)

    # Rotation matrix combining roll, pitch, and yaw
    rotation_matrix = np.array([
        [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
        [-s_p, c_p * s_r, c_p * c_r]
    ])

    leg_positions = []

    for leg_base in leg_base_positions:
        # Apply rotation to the leg base position
        rotated_leg_base = np.dot(rotation_matrix, np.array(leg_base))

        # Apply translation to the rotated position
        leg_x = rotated_leg_base[0] + x_trans
        leg_y = rotated_leg_base[1] + y_trans
        leg_z = rotated_leg_base[2] + z_trans

        leg_positions.append((leg_x, leg_y, leg_z))

    return leg_positions

# Control each leg joint
def joint_control(joint_index, pos):
    target_x, target_y, target_z = pos

    # untuk kaki bagian kiri itu dikalikan negatif
    if joint_index>=10: 
        target_x*=-1
        target_y*=-1
    coxa_angle, femur_angle, tibia_angle = inverse_kinematics(target_x, target_y, target_z)

    p.setJointMotorControl2(robot_id, jointIndex=joint_index, controlMode=p.POSITION_CONTROL, targetPosition=coxa_angle)
    p.setJointMotorControl2(robot_id, jointIndex=joint_index + 1, controlMode=p.POSITION_CONTROL, targetPosition=femur_angle)
    p.setJointMotorControl2(robot_id, jointIndex=joint_index + 2, controlMode=p.POSITION_CONTROL, targetPosition=tibia_angle)

# Main simulation loop with slider controls
p.setRealTimeSimulation(1)
while True:
    try:
        t_start = time.time()
        t = time.time() - start_time
        dt = 1 / 240

        # Read slider values for vx and vy
        vx = p.readUserDebugParameter(vx_slider)
        vy = p.readUserDebugParameter(vy_slider)
        v_rot = p.readUserDebugParameter(v_rot_slider)

        step_height = p.readUserDebugParameter(step_height_slider)
        cpg_freq = p.readUserDebugParameter(cpg_slider)
        phase_lag = p.readUserDebugParameter(gait_slider)
        duty_factor = p.readUserDebugParameter(duty_slider)

        # Read debug slider values
        body_position = (
            p.readUserDebugParameter(x_slider),
            p.readUserDebugParameter(y_slider),
            p.readUserDebugParameter(z_slider),
        )
        body_orientation = (
            p.readUserDebugParameter(r_slider),
            p.readUserDebugParameter(p_slider),
            p.readUserDebugParameter(yaw_slider),
        )

        # Compute leg positions
        leg_pos_body = body_kinematics(body_position, body_orientation)

        for leg_index in range(6):  # Iterate over all 6 legs
            pos = cpg_leg_trajectory(
                t=t,
                leg_index=leg_index,
                freq_hz=cpg_freq,
                phase_lag=phase_lag,
                duty_factor=duty_factor,
                step_height=step_height,
                vx=vx,
                vy=vy,
                v_rot=v_rot,
            )
            leg_base = leg_pos_body[leg_index]
            leg_pos = (leg_base[0] + pos[0], leg_base[1] + pos[1], leg_base[2] + pos[2])
            
            # Control leg joint angles
            joint_control(joint_index=leg_index * 3 + 1, pos=leg_pos)

        p.stepSimulation()
        time.sleep(1 / 240)  # Adjust simulation speed if necessary
    
        # Update the global position
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        global_position = np.array(position)
        global_orientation = np.array(orientation)

        t_end = time.time()
        t_total = t_end-t_start + 1/1e10
        print(f"Robot: time: {t_total:.2f} seconds, fps: {1/t_total:.2f}")

    except Exception as e:
        print(e)
        break