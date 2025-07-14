import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import math
import os
import gc
from teleop import HuskyTeleopController
from autonomous_nav import RobotController
from sensor import SensorManager

# === Connect and Setup ===
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Optimize PyBullet settings
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
p.setRealTimeSimulation(1)
p.setPhysicsEngineParameter(numSolverIterations=5)

# === Load House Visual-Only ===
house = "E:/ArmBot/Mythings/Models/house.stl"
mesh_scale = [0.001, 0.001, 0.001]

visual_shape = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=house,
    meshScale=mesh_scale,
    rgbaColor=[0.8, 0.6, 0.4, 1.0]
)
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=-1,
    baseVisualShapeIndex=visual_shape,
    basePosition=[0, 0, -0.5]
)

# === Load Fridge ===
fridge_path = "E:/ArmBot/Mythings/Models/Fridge/11299/mobility.urdf"
fridge_id = p.loadURDF(fridge_path, basePosition=[15, 18, 0.8], useFixedBase=True)

# === Load Husky Robot ===
husky_path = os.path.join(pybullet_data.getDataPath(), "husky/husky.urdf")
husky_pos = [2.0, 1.0, 0.1]
husky_id = p.loadURDF(husky_path, basePosition=husky_pos, useFixedBase=False)

# === Mount Franka on Husky ===
panda_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
panda_offset = [0.0, 0.0, 0.45]
panda_position = [husky_pos[i] + panda_offset[i] for i in range(3)]

panda_id = p.loadURDF(panda_path, basePosition=panda_position, useFixedBase=False)
p.createConstraint(
    parentBodyUniqueId=husky_id,
    parentLinkIndex=-1,
    childBodyUniqueId=panda_id,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=panda_offset,
    childFramePosition=[0, 0, 0]
)

# Lock Panda joints
for j in range(p.getNumJoints(panda_id)):
    joint_info = p.getJointInfo(panda_id, j)
    if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, targetPosition=0, force=500)

# === Build Rigid Table (1 body) ===
table_pos = [12, 1, 0]
table_top_size = [0.5, 0.3, 0.025]
leg_height = 0.675
leg_radius = 0.025

top_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_top_size)
top_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=table_top_size, rgbaColor=[0.5, 0.3, 0.1, 1])

leg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=leg_radius, height=leg_height)
leg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=leg_radius, length=leg_height, rgbaColor=[0.4, 0.2, 0.1, 1])

leg_offsets = [
    [0.45, 0.25],
    [-0.45, 0.25],
    [0.45, -0.25],
    [-0.45, -0.25]
]

link_masses = [1] * 4
link_collisions = [leg_col] * 4
link_visuals = [leg_vis] * 4
link_positions = [[x, y, -leg_height/2] for x, y in leg_offsets]
link_orientations = [[0, 0, 0, 1]] * 4
link_joint_types = [p.JOINT_FIXED] * 4
link_joint_axes = [[0, 0, 0]] * 4
link_parent_indices = [0] * 4
link_inertial_frame_positions = [[0, 0, 0]] * 4
link_inertial_frame_orientations = [[0, 0, 0, 1]] * 4

table_id = p.createMultiBody(
    baseMass=5,
    baseCollisionShapeIndex=top_col,
    baseVisualShapeIndex=top_vis,
    basePosition=[table_pos[0], table_pos[1], table_pos[2] + leg_height + table_top_size[2]],
    linkMasses=link_masses,
    linkCollisionShapeIndices=link_collisions,
    linkVisualShapeIndices=link_visuals,
    linkPositions=link_positions,
    linkOrientations=link_orientations,
    linkJointTypes=link_joint_types,
    linkJointAxis=link_joint_axes,
    linkParentIndices=link_parent_indices,
    linkInertialFramePositions=link_inertial_frame_positions,
    linkInertialFrameOrientations=link_inertial_frame_orientations
)

# === Add Bottle Inside Fridge ===
bottle_base_pos = [15, 18, 0.9]
bottle_radius = 0.04
bottle_height = 0.3
bottle_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=bottle_radius, height=bottle_height)
bottle_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=bottle_radius, length=bottle_height,
                                 rgbaColor=[0.1, 0.6, 0.8, 1])
bottle_id = p.createMultiBody(0.3, bottle_col, bottle_vis, bottle_base_pos)

# === Add Cup on Table ===
cup_pos = [12, 1, leg_height + 2 * table_top_size[2] + 0.06]
cup_radius = 0.035
cup_height = 0.12
cup_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cup_radius, height=cup_height)
cup_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cup_radius, length=cup_height,
                              rgbaColor=[1.0, 0.9, 0.7, 1])
cup_id = p.createMultiBody(0.2, cup_col, cup_vis, cup_pos)

# === Control Mode Setup ===
MANUAL_MODE = 'manual'
AUTONOMOUS_MODE = 'autonomous'
current_mode = MANUAL_MODE

# Initialize controllers
sensor_manager = SensorManager(husky_id)
teleop_controller = HuskyTeleopController(husky_id)
autonomous_controller = RobotController(husky_id, sensor_manager)

# Print controls
print("\n=== Robot Control Modes ===")
print("Press 'M' for manual control mode")
print("Press 'A' for autonomous mode")
print("Press 'ESC' to quit")
teleop_controller.print_controls()

# === Sensor Functions - Delegated to SensorManager ===

# === Camera View (full environment) ===
p.resetDebugVisualizerCamera(
    cameraDistance=20,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[8, 10, 1]
)

# === Main Loop ===
UPDATE_FREQ = 4  # Update sensors every N simulation steps
step_count = 0

# Enable garbage collection for better memory management
gc.enable()
last_collection = time.time()

try:
    while True:
        p.stepSimulation()
        
        # Process keyboard input for mode switching
        key = None
        if cv2.waitKey(1) & 0xFF == ord('m'):
            if current_mode != MANUAL_MODE:
                print("Switching to manual control mode")
                current_mode = MANUAL_MODE
                autonomous_controller.stop()  # Stop autonomous movement
                teleop_controller.print_controls()
        elif cv2.waitKey(1) & 0xFF == ord('x'):
            if current_mode != AUTONOMOUS_MODE:
                print("Switching to autonomous mode")
                current_mode = AUTONOMOUS_MODE
                teleop_controller.stop()  # Stop manual movement
        
        # Handle control based on current mode
        if current_mode == MANUAL_MODE:
            # Process manual control input
            if teleop_controller.process_input():
                break  # Exit if ESC pressed
        else:  # Autonomous mode
            # Run autonomous navigation step
            if not autonomous_controller.update():
                print("Autonomous navigation complete or failed")
                current_mode = MANUAL_MODE
                teleop_controller.print_controls()
        
        # Update sensor displays at reduced frequency
        if step_count % UPDATE_FREQ == 0:
            try:
                # Update all displays using SensorManager
                sensor_manager.update_displays()
                
            except Exception as e:
                print(f"Error updating sensor data: {e}")
        
        # Periodic garbage collection
        if time.time() - last_collection > 60:  # Every minute
            gc.collect()
            last_collection = time.time()
        
        # Handle window close
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
        
        step_count += 1
        time.sleep(1./240.)  # Adjusted for better performance

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    # Stop any movement
    teleop_controller.stop()
    autonomous_controller.stop()
    
    # Cleanup
    sensor_manager.destroy_windows()  # Use SensorManager cleanup
    p.disconnect()
    cv2.destroyAllWindows()
    gc.collect()  # Final cleanup