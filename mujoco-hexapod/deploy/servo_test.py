from servo import *

# Initialize the servo
servo = DynamixelServo(device_name="COM5", baudrate=1000000) #ls /dev/ttyUSB* -- /dev/ttyUSB0
servo_ids = list(range(1, 19))  # Servo IDs from 1 to 18
goal_positions = [512 for _ in range(18)]  # All servos to position 512
servo.enable_torque(servo_ids)

servo.write(servo_ids, goal_positions)

servo.disable_torque(servo_ids)