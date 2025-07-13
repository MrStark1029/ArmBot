import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import os
from teleop import HuskyTeleopController
from sensor import SensorManager

# === Connect and Setup ===
physicsClient = p.connect(p.GUI)
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

# === Initialize Controller ===
# This is the key change - controller handles all movement logic
controller = HuskyTeleopController(husky_id)

# === Initialize Sensor Manager ===
sensor_manager = SensorManager(husky_id)

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

# === Build Rigid Table ===
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

# === Camera View ===
p.resetDebugVisualizerCamera(
    cameraDistance=20,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[8, 10, 1]
)

# === Main Loop ===
UPDATE_FREQ = 2
step_count = 0

try:
    # Controller handles all input and movement logic
    controller.print_controls()
    
    while True:
        # Controller processes input and updates robot movement
        should_exit = controller.process_input()
        if should_exit:
            break
            
        # Step simulation
        p.stepSimulation()
        
        # Update sensor displays at reduced frequency
        if step_count % UPDATE_FREQ == 0:
            sensor_manager.update_displays()
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
        
        step_count += 1
        time.sleep(1./360.)

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    controller.stop()
    sensor_manager.destroy_windows()
    p.disconnect()