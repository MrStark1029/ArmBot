import pybullet as p
import pybullet_data
import time
import math

# Start simulation
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 240)
p.setRealTimeSimulation(0)

# Load ground and Husky robot
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

start_pos = [0, 0, 0.1]
husky = p.loadURDF("husky/husky.urdf", start_pos, useFixedBase=False)

# Get wheel joint indices (husky has 4 wheels)
wheel_joints = [2, 3, 4, 5]  # FL, FR, RL, RR

# Parameters
wheel_velocity = 5.0
turn_velocity = 2.0
forward_duration = 2.0  # seconds
turn_duration = 1.5     # seconds (90 deg turn)
num_sides = 4

# Move in square path
for side in range(num_sides):
    # Move forward
    start_time = time.time()
    while time.time() - start_time < forward_duration:
        p.setJointMotorControlArray(
            husky,
            wheel_joints,
            p.VELOCITY_CONTROL,
            targetVelocities=[wheel_velocity] * 4
        )
        p.stepSimulation()
        time.sleep(1. / 240)

    # Stop briefly
    p.setJointMotorControlArray(
        husky,
        wheel_joints,
        p.VELOCITY_CONTROL,
        targetVelocities=[0] * 4
    )
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1. / 240)

    # Turn 90 degrees
    start_time = time.time()
    while time.time() - start_time < turn_duration:
        p.setJointMotorControlArray(
            husky,
            wheel_joints,
            p.VELOCITY_CONTROL,
            targetVelocities=[-turn_velocity, turn_velocity, -turn_velocity, turn_velocity]
        )
        p.stepSimulation()
        time.sleep(1. / 240)

# Stop at end
p.setJointMotorControlArray(
    husky,
    wheel_joints,
    p.VELOCITY_CONTROL,
    targetVelocities=[0] * 4
)

print("Square path complete.")
while True:
    p.stepSimulation()
    time.sleep(1. / 240)
