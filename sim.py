import pybullet as p
import pybullet_data
import time
import os

# Start simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load ground
planeId = p.loadURDF("plane.urdf")

# House parameters
wall_height = 2.5
room_width = 3.5
room_depth = 3.5
wall_thickness = 0.15
door_width = 0.8
hallway_width = 1.5

# House layout - more realistic room arrangement:
#     [Kitchen]   [Living Room]
#       [Hall]    
#     [Bedroom]   [Bathroom]

room_positions = {
    'kitchen': [-room_width/2 - hallway_width/2, room_depth/2 + hallway_width/2],
    'living_room': [room_width/2 + hallway_width/2, room_depth/2 + hallway_width/2],
    'hallway': [0, 0],
    'bedroom': [-room_width/2 - hallway_width/2, -room_depth/2 - hallway_width/2],
    'bathroom': [room_width/2 + hallway_width/2, -room_depth/2 - hallway_width/2]
}

def create_wall(x, y, length, orientation='horizontal', thickness=wall_thickness):
    """Create a wall with specified position, length, and orientation"""
    if orientation == 'horizontal':
        half_extents = [length/2, thickness/2, wall_height/2]
    else:  # vertical
        half_extents = [thickness/2, length/2, wall_height/2]
    
    wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0.8, 0.8, 1])
    return p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=wall,
                             baseVisualShapeIndex=visual,
                             basePosition=[x, y, wall_height/2])

def create_room_walls(x0, y0, w, d, doors=None):
    """Create walls for a room with optional door openings"""
    if doors is None:
        doors = []
    
    # Bottom wall (south)
    if 'south' not in doors:
        create_wall(x0, y0 - d/2, w, 'horizontal')
    else:
        # Create wall segments with door opening
        door_offset = door_width/2
        create_wall(x0 - w/4 - door_offset/2, y0 - d/2, w/2 - door_width, 'horizontal')
        create_wall(x0 + w/4 + door_offset/2, y0 - d/2, w/2 - door_width, 'horizontal')
    
    # Top wall (north)
    if 'north' not in doors:
        create_wall(x0, y0 + d/2, w, 'horizontal')
    else:
        door_offset = door_width/2
        create_wall(x0 - w/4 - door_offset/2, y0 + d/2, w/2 - door_width, 'horizontal')
        create_wall(x0 + w/4 + door_offset/2, y0 + d/2, w/2 - door_width, 'horizontal')
    
    # Left wall (west)
    if 'west' not in doors:
        create_wall(x0 - w/2, y0, d, 'vertical')
    else:
        door_offset = door_width/2
        create_wall(x0 - w/2, y0 - d/4 - door_offset/2, d/2 - door_width, 'vertical')
        create_wall(x0 - w/2, y0 + d/4 + door_offset/2, d/2 - door_width, 'vertical')
    
    # Right wall (east)
    if 'east' not in doors:
        create_wall(x0 + w/2, y0, d, 'vertical')
    else:
        door_offset = door_width/2
        create_wall(x0 + w/2, y0 - d/4 - door_offset/2, d/2 - door_width, 'vertical')
        create_wall(x0 + w/2, y0 + d/4 + door_offset/2, d/2 - door_width, 'vertical')

# Create exterior walls of the house
house_width = room_width * 2 + hallway_width + wall_thickness
house_depth = room_depth * 2 + hallway_width + wall_thickness

# Exterior walls
create_wall(0, house_depth/2, house_width, 'horizontal')  # North exterior
create_wall(0, -house_depth/2, house_width, 'horizontal')  # South exterior
create_wall(-house_width/2, 0, house_depth, 'vertical')   # West exterior
create_wall(house_width/2, 0, house_depth, 'vertical')    # East exterior

# Create individual rooms with doors
kitchen_x, kitchen_y = room_positions['kitchen']
create_room_walls(kitchen_x, kitchen_y, room_width, room_depth, doors=['east'])  # Door to hallway

living_room_x, living_room_y = room_positions['living_room']
create_room_walls(living_room_x, living_room_y, room_width, room_depth, doors=['west'])  # Door to hallway

bedroom_x, bedroom_y = room_positions['bedroom']
create_room_walls(bedroom_x, bedroom_y, room_width, room_depth, doors=['east'])  # Door to hallway

bathroom_x, bathroom_y = room_positions['bathroom']
create_room_walls(bathroom_x, bathroom_y, room_width, room_depth, doors=['west'])  # Door to hallway

# Create hallway walls (partial walls to connect rooms)
hallway_x, hallway_y = room_positions['hallway']

# Remove the hallway walls entirely or create them with proper gaps
# The room walls already have gaps, so we might not need hallway walls at all
# But if you want some separation, create minimal connecting walls:

# North hallway connecting walls (between kitchen and living room)
create_wall(-hallway_width/2 - door_width/4, hallway_width/2, hallway_width/2 - door_width, 'horizontal')  # Left segment
create_wall(hallway_width/2 + door_width/4, hallway_width/2, hallway_width/2 - door_width, 'horizontal')   # Right segment

# South hallway connecting walls (between bedroom and bathroom)  
create_wall(-hallway_width/2 - door_width/4, -hallway_width/2, hallway_width/2 - door_width, 'horizontal')  # Left segment
create_wall(hallway_width/2 + door_width/4, -hallway_width/2, hallway_width/2 - door_width, 'horizontal')   # Right segment

# Load kitchen appliances and furniture
# Fridge in kitchen
fridge = p.loadURDF("E:/moving-roboarm/Kitchen_world_models/Fridge/12248/mobility.urdf", 
                    [kitchen_x + 1.0, kitchen_y + 1, 1], useFixedBase=True)
CabinateTall = p.loadURDF("E:/moving-roboarm/Kitchen_world_models/CabinetTall/46896/mobility.urdf", 
                    [kitchen_x -1, kitchen_y -1, 0], useFixedBase=True)
sink = p.loadURDF("E:/moving-roboarm/Kitchen_world_models/Sink/1023790/mobility.urdf", 
                    [kitchen_x -1, kitchen_y +1, 0], useFixedBase=True)

# Wait for appliances to settle
#time.sleep(1.0)

# Create a shelf inside the fridge
shelf = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.3, 0.02])
shelf_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.3, 0.02], 
                                  rgbaColor=[0.9, 0.9, 0.9, 0.8])
shelf_body = p.createMultiBody(baseMass=0, 
                              baseCollisionShapeIndex=shelf, 
                              baseVisualShapeIndex=shelf_visual,
                              basePosition=[kitchen_x + 1.0, kitchen_y + 1.0, 0.3])

# Create cylindrical oil bottle shape
bottle_radius = 0.05  # 5cm radius
bottle_height = 0.15  # 15cm height

# Create collision and visual shapes for the cylinder
bottle_collision = p.createCollisionShape(p.GEOM_CYLINDER, 
                                         radius=bottle_radius, 
                                         height=bottle_height)
bottle_visual = p.createVisualShape(p.GEOM_CYLINDER, 
                                   radius=bottle_radius, 
                                   length=bottle_height,
                                   rgbaColor=[0.8, 0.6, 0.2, 1.0])  # Golden oil color

# Create the oil bottle as a multi-body
oil_bottle = p.createMultiBody(baseMass=0.5,  # 0.5 kg mass
                              baseCollisionShapeIndex=bottle_collision,
                              baseVisualShapeIndex=bottle_visual,
                              basePosition=[kitchen_x + 1.075, kitchen_y + 1.0, 0.3])

# Set physics properties
p.changeDynamics(oil_bottle, -1, 
                lateralFriction=0.8,  # Good friction
                rollingFriction=0.3,  # Prevent rolling
                spinningFriction=0.3,  # Prevent spinning
                restitution=0.2)      # Low bounce

# Table in living room
table = p.loadURDF("table/table.urdf", [bedroom_x, bedroom_y, 0], useFixedBase=True)

table = p.loadURDF("franka_panda/panda.urdf", [bathroom_x, bathroom_y, 0], useFixedBase=True)
# Load objects on table/

# Add a knife on the kitchen counter (if you have a counter URDF)
# knife = p.loadURDF("E:/moving-roboarm/Kitchen_world_models/Knife/knife.urdf", 
#                    [kitchen_x + 0.5, kitchen_y - 1.0, 0.9])

# Optional: Add some lighting
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

# Camera positioning for better view
p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

print("House simulation loaded successfully!")
print("Rooms:")
print(f"Kitchen: {kitchen_x:.1f}, {kitchen_y:.1f}")
print(f"Living Room: {living_room_x:.1f}, {living_room_y:.1f}")
print(f"Bedroom: {bedroom_x:.1f}, {bedroom_y:.1f}")
print(f"Bathroom: {bathroom_x:.1f}, {bathroom_y:.1f}")
print(f"Hallway: {hallway_x:.1f}, {hallway_y:.1f}")

# Let it run
while p.isConnected():
    p.stepSimulation()
    time.sleep(1.0 / 240)