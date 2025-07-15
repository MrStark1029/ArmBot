import pybullet as p
import pybullet_data
import time
import os
import json
import math
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class WallPlacementGUI:
    def __init__(self):
        # Initialize PyBullet in separate thread
        self.pybullet_thread = None
        self.running = False
        self.tool = None
        
        # Create main GUI window
        self.root = tk.Tk()
        self.root.title("🧱 Interactive Wall Placement Tool")
        self.root.geometry("400x800")
        self.root.resizable(True, True)
        
        # Variables for GUI controls
        self.wall_width = tk.DoubleVar(value=2.0)  # Full width (not half)
        self.wall_thickness = tk.DoubleVar(value=0.2)  # Full thickness
        self.wall_height = tk.DoubleVar(value=2.0)  # Full height
        self.wall_rotation = tk.DoubleVar(value=0.0)
        self.grid_snap = tk.BooleanVar(value=True)
        self.grid_size = tk.DoubleVar(value=0.5)
        self.wall_count = tk.StringVar(value="0")
        
        # Position variables
        self.wall_x = tk.DoubleVar(value=0.0)
        self.wall_y = tk.DoubleVar(value=0.0)
        self.manual_position = tk.BooleanVar(value=False)
        
        # Color variables
        self.wall_r = tk.DoubleVar(value=0.6)
        self.wall_g = tk.DoubleVar(value=0.2)
        self.wall_b = tk.DoubleVar(value=0.2)
        
        self.setup_gui()
        
    def setup_gui(self):
        """Create the GUI interface"""
        # Create main canvas and scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main frame inside scrollable area
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)
        
        # Title
        title_label = ttk.Label(main_frame, text="🧱 Wall Placement Tool", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # PyBullet Control Section
        control_frame = ttk.LabelFrame(main_frame, text="PyBullet Control", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="🚀 Start PyBullet", 
                  command=self.start_pybullet).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="🛑 Stop PyBullet", 
                  command=self.stop_pybullet).grid(row=0, column=1)
        
        # Wall Dimensions Section
        dim_frame = ttk.LabelFrame(main_frame, text="Wall Dimensions", padding="10")
        dim_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Width control
        ttk.Label(dim_frame, text="Width (m):").grid(row=0, column=0, sticky=tk.W)
        width_frame = ttk.Frame(dim_frame)
        width_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(width_frame, text="-", width=3, 
                  command=lambda: self.adjust_dimension('width', -0.1)).grid(row=0, column=0)
        width_entry = ttk.Entry(width_frame, textvariable=self.wall_width, width=8)
        width_entry.grid(row=0, column=1, padx=5)
        width_entry.bind('<Return>', lambda e: self.update_wall_dimensions())
        ttk.Button(width_frame, text="+", width=3, 
                  command=lambda: self.adjust_dimension('width', 0.1)).grid(row=0, column=2)
        
        # Thickness control
        ttk.Label(dim_frame, text="Thickness (m):").grid(row=1, column=0, sticky=tk.W)
        thickness_frame = ttk.Frame(dim_frame)
        thickness_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(thickness_frame, text="-", width=3, 
                  command=lambda: self.adjust_dimension('thickness', -0.05)).grid(row=0, column=0)
        thickness_entry = ttk.Entry(thickness_frame, textvariable=self.wall_thickness, width=8)
        thickness_entry.grid(row=0, column=1, padx=5)
        thickness_entry.bind('<Return>', lambda e: self.update_wall_dimensions())
        ttk.Button(thickness_frame, text="+", width=3, 
                  command=lambda: self.adjust_dimension('thickness', 0.05)).grid(row=0, column=2)
        
        # Height control
        ttk.Label(dim_frame, text="Height (m):").grid(row=2, column=0, sticky=tk.W)
        height_frame = ttk.Frame(dim_frame)
        height_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(height_frame, text="-", width=3, 
                  command=lambda: self.adjust_dimension('height', -0.1)).grid(row=0, column=0)
        height_entry = ttk.Entry(height_frame, textvariable=self.wall_height, width=8)
        height_entry.grid(row=0, column=1, padx=5)
        height_entry.bind('<Return>', lambda e: self.update_wall_dimensions())
        ttk.Button(height_frame, text="+", width=3, 
                  command=lambda: self.adjust_dimension('height', 0.1)).grid(row=0, column=2)
        
        # Rotation Section
        rot_frame = ttk.LabelFrame(main_frame, text="Wall Rotation", padding="10")
        rot_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(rot_frame, text="Rotation (°):").grid(row=0, column=0, sticky=tk.W)
        rotation_frame = ttk.Frame(rot_frame)
        rotation_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(rotation_frame, text="-45°", width=6, 
                  command=lambda: self.adjust_rotation(-45)).grid(row=0, column=0)
        rotation_entry = ttk.Entry(rotation_frame, textvariable=self.wall_rotation, width=8)
        rotation_entry.grid(row=0, column=1, padx=5)
        rotation_entry.bind('<Return>', lambda e: self.update_wall_dimensions())
        ttk.Button(rotation_frame, text="+45°", width=6, 
                  command=lambda: self.adjust_rotation(45)).grid(row=0, column=2)
        
        ttk.Button(rot_frame, text="Reset (0°)", 
                  command=lambda: self.set_rotation(0)).grid(row=1, column=1, pady=5)
        
        # Position Section
        pos_frame = ttk.LabelFrame(main_frame, text="Position Control", padding="10")
        pos_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Manual position checkbox
        ttk.Checkbutton(pos_frame, text="Manual Position (overrides mouse)", 
                       variable=self.manual_position, 
                       command=self.update_position_mode).grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        # X position control
        ttk.Label(pos_frame, text="X Position (m):").grid(row=1, column=0, sticky=tk.W)
        x_pos_frame = ttk.Frame(pos_frame)
        x_pos_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(x_pos_frame, text="-", width=3, 
                  command=lambda: self.adjust_position('x', -0.5)).grid(row=0, column=0)
        x_pos_entry = ttk.Entry(x_pos_frame, textvariable=self.wall_x, width=8)
        x_pos_entry.grid(row=0, column=1, padx=5)
        x_pos_entry.bind('<Return>', lambda e: self.update_manual_position())
        ttk.Button(x_pos_frame, text="+", width=3, 
                  command=lambda: self.adjust_position('x', 0.5)).grid(row=0, column=2)
        
        # Y position control
        ttk.Label(pos_frame, text="Y Position (m):").grid(row=2, column=0, sticky=tk.W)
        y_pos_frame = ttk.Frame(pos_frame)
        y_pos_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(y_pos_frame, text="-", width=3, 
                  command=lambda: self.adjust_position('y', -0.5)).grid(row=0, column=0)
        y_pos_entry = ttk.Entry(y_pos_frame, textvariable=self.wall_y, width=8)
        y_pos_entry.grid(row=0, column=1, padx=5)
        y_pos_entry.bind('<Return>', lambda e: self.update_manual_position())
        ttk.Button(y_pos_frame, text="+", width=3, 
                  command=lambda: self.adjust_position('y', 0.5)).grid(row=0, column=2)
        
        # Position presets
        pos_preset_frame = ttk.Frame(pos_frame)
        pos_preset_frame.grid(row=3, column=0, columnspan=3, pady=5)
        
        ttk.Button(pos_preset_frame, text="Origin (0,0)", 
                  command=lambda: self.set_position(0, 0)).grid(row=0, column=0, padx=2)
        ttk.Button(pos_preset_frame, text="Center View", 
                  command=self.center_position).grid(row=0, column=1, padx=2)
        ttk.Button(pos_preset_frame, text="Place Here", 
                  command=self.place_at_manual_position).grid(row=0, column=2, padx=2)

        # Grid Settings Section
        grid_frame = ttk.LabelFrame(main_frame, text="Grid Settings", padding="10")
        grid_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(grid_frame, text="Enable Grid Snap", 
                       variable=self.grid_snap, 
                       command=self.update_grid_settings).grid(row=0, column=0, columnspan=3)
        
        ttk.Label(grid_frame, text="Grid Size (m):").grid(row=1, column=0, sticky=tk.W)
        grid_size_frame = ttk.Frame(grid_frame)
        grid_size_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(grid_size_frame, text="-", width=3, 
                  command=lambda: self.adjust_grid_size(-0.1)).grid(row=0, column=0)
        grid_entry = ttk.Entry(grid_size_frame, textvariable=self.grid_size, width=8)
        grid_entry.grid(row=0, column=1, padx=5)
        grid_entry.bind('<Return>', lambda e: self.update_grid_settings())
        ttk.Button(grid_size_frame, text="+", width=3, 
                  command=lambda: self.adjust_grid_size(0.1)).grid(row=0, column=2)
        
        # Wall Color Section
        color_frame = ttk.LabelFrame(main_frame, text="Wall Color", padding="10")
        color_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Color sliders
        ttk.Label(color_frame, text="Red:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(color_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.wall_r, command=self.update_wall_color).grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(color_frame, text="Green:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(color_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.wall_g, command=self.update_wall_color).grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(color_frame, text="Blue:").grid(row=2, column=0, sticky=tk.W)
        ttk.Scale(color_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.wall_b, command=self.update_wall_color).grid(row=2, column=1, sticky=(tk.W, tk.E))
        
        # Color presets
        preset_frame = ttk.Frame(color_frame)
        preset_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Button(preset_frame, text="Red", 
                  command=lambda: self.set_color(0.8, 0.2, 0.2)).grid(row=0, column=0, padx=2)
        ttk.Button(preset_frame, text="Blue", 
                  command=lambda: self.set_color(0.2, 0.2, 0.8)).grid(row=0, column=1, padx=2)
        ttk.Button(preset_frame, text="Green", 
                  command=lambda: self.set_color(0.2, 0.8, 0.2)).grid(row=0, column=2, padx=2)
        ttk.Button(preset_frame, text="Gray", 
                  command=lambda: self.set_color(0.6, 0.6, 0.6)).grid(row=0, column=3, padx=2)
        
        # Actions Section
        actions_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        actions_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # First row of buttons
        actions_row1 = ttk.Frame(actions_frame)
        actions_row1.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(actions_row1, text="↩️ Undo Last", 
                  command=self.undo_last_wall).grid(row=0, column=0, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(actions_row1, text="🗑️ Clear All", 
                  command=self.clear_all_walls).grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(actions_row1, text="📄 Reset View", 
                  command=self.reset_camera).grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Second row of buttons
        actions_row2 = ttk.Frame(actions_frame)
        actions_row2.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(actions_row2, text="💾 Save Layout", 
                  command=self.save_walls).grid(row=0, column=0, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(actions_row2, text="📁 Load Layout", 
                  command=self.load_walls).grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(actions_row2, text="📊 Export Data", 
                  command=self.export_data).grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Status Section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(status_frame, text="Walls Placed:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.wall_count).grid(row=0, column=1, sticky=tk.W)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instructions_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        instructions_text = """🖱️ LEFT CLICK: Place wall at cursor position
🖱️ RIGHT CLICK: Delete nearest wall
📍 Manual Position: Use position controls to place walls
🎯 Use GUI controls to adjust dimensions and settings
📝 All changes update in real-time
🖱️ Mouse wheel: Scroll through controls"""
        
        ttk.Label(instructions_frame, text=instructions_text, 
                 justify=tk.LEFT, wraplength=350).grid(row=0, column=0)
        
        # Configure grid weights for proper resizing
        main_frame.columnconfigure(1, weight=1)
        
        # Configure inner frame weights
        for frame in [width_frame, thickness_frame, height_frame, rotation_frame, grid_size_frame, x_pos_frame, y_pos_frame]:
            frame.columnconfigure(1, weight=1)
        
        color_frame.columnconfigure(1, weight=1)
        actions_row1.columnconfigure((0, 1, 2), weight=1)
        actions_row2.columnconfigure((0, 1, 2), weight=1)
        
    def adjust_dimension(self, dim_type, delta):
        """Adjust wall dimensions"""
        if dim_type == 'width':
            new_val = max(0.1, self.wall_width.get() + delta)
            self.wall_width.set(round(new_val, 2))
        elif dim_type == 'thickness':
            new_val = max(0.05, self.wall_thickness.get() + delta)
            self.wall_thickness.set(round(new_val, 2))
        elif dim_type == 'height':
            new_val = max(0.1, self.wall_height.get() + delta)
            self.wall_height.set(round(new_val, 2))
        
        self.update_wall_dimensions()
    
    def adjust_rotation(self, delta):
        """Adjust wall rotation"""
        new_rotation = (self.wall_rotation.get() + delta) % 360
        self.wall_rotation.set(new_rotation)
        self.update_wall_dimensions()
    
    def set_rotation(self, angle):
        """Set specific rotation angle"""
        self.wall_rotation.set(angle)
        self.update_wall_dimensions()
    
    def adjust_grid_size(self, delta):
        """Adjust grid size"""
        new_size = max(0.1, self.grid_size.get() + delta)
        self.grid_size.set(round(new_size, 2))
        self.update_grid_settings()
    
    def set_color(self, r, g, b):
        """Set wall color preset"""
        self.wall_r.set(r)
        self.wall_g.set(g)
        self.wall_b.set(b)
        self.update_wall_color()
    
    def adjust_position(self, axis, delta):
        """Adjust position coordinates"""
        if axis == 'x':
            new_val = self.wall_x.get() + delta
            self.wall_x.set(round(new_val, 2))
        elif axis == 'y':
            new_val = self.wall_y.get() + delta
            self.wall_y.set(round(new_val, 2))
        
        self.update_manual_position()
    
    def set_position(self, x, y):
        """Set specific position"""
        self.wall_x.set(x)
        self.wall_y.set(y)
        self.update_manual_position()
    
    def center_position(self):
        """Center position to current camera view"""
        if self.tool:
            cam_info = p.getDebugVisualizerCamera()
            target = cam_info[11]
            self.set_position(target[0], target[1])
    
    def update_position_mode(self):
        """Update position mode in PyBullet tool"""
        if self.tool:
            self.tool.manual_position_mode = self.manual_position.get()
    
    def update_manual_position(self):
        """Update manual position in PyBullet tool"""
        if self.tool:
            self.tool.manual_position = [self.wall_x.get(), self.wall_y.get(), 0]
    
    def place_at_manual_position(self):
        """Place wall at manual position"""
        if self.tool:
            position = [self.wall_x.get(), self.wall_y.get(), 0]
            self.tool.place_wall(position)
    
    def update_wall_dimensions(self):
        """Update wall dimensions in PyBullet tool"""
        if self.tool:
            self.tool.wall_half_extents = [
                self.wall_width.get() / 2,
                self.wall_thickness.get() / 2,
                self.wall_height.get() / 2
            ]
            self.tool.wall_rotation = self.wall_rotation.get()
    
    def update_grid_settings(self):
        """Update grid settings in PyBullet tool"""
        if self.tool:
            self.tool.grid_snap = self.grid_snap.get()
            self.tool.grid_size = self.grid_size.get()
    
    def update_wall_color(self, *args):
        """Update wall color in PyBullet tool"""
        if self.tool:
            self.tool.wall_color = [
                self.wall_r.get(),
                self.wall_g.get(),
                self.wall_b.get(),
                1.0
            ]
    
    def start_pybullet(self):
        """Start PyBullet simulation in separate thread"""
        if not self.running:
            self.running = True
            self.pybullet_thread = threading.Thread(target=self.run_pybullet)
            self.pybullet_thread.daemon = True
            self.pybullet_thread.start()
    
    def stop_pybullet(self):
        """Stop PyBullet simulation"""
        self.running = False
        if self.tool:
            self.tool.save_walls()
    
    def run_pybullet(self):
        """Run PyBullet simulation"""
        try:
            self.tool = WallPlacementTool(self)
            self.tool.run()
        except Exception as e:
            messagebox.showerror("PyBullet Error", f"Error running PyBullet: {e}")
    
    def update_wall_count(self, count):
        """Update wall count display"""
        self.wall_count.set(str(count))
    
    def undo_last_wall(self):
        """Undo last wall placement"""
        if self.tool:
            self.tool.undo_last_wall()
    
    def clear_all_walls(self):
        """Clear all walls"""
        if self.tool:
            self.tool.clear_all_walls()
    
    def reset_camera(self):
        """Reset camera view"""
        if self.tool:
            self.tool.setup_camera()
    
    def save_walls(self):
        """Save wall layout"""
        if self.tool:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Wall Layout"
            )
            if filename:
                self.tool.save_walls(filename)
    
    def load_walls(self):
        """Load wall layout"""
        if self.tool:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Wall Layout"
            )
            if filename:
                self.tool.load_walls(filename)
    
    def export_data(self):
        """Export wall data in various formats"""
        if self.tool and self.tool.wall_list:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Wall Data"
            )
            if filename:
                self.tool.export_wall_data(filename)
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.tool:
            self.tool.save_walls()
        self.root.destroy()

class WallPlacementTool:
    def __init__(self, gui):
        self.gui = gui
        
        # PyBullet setup
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)
        
        # Tool state
        self.wall_half_extents = [1.0, 0.1, 1.0]
        self.wall_color = [0.6, 0.2, 0.2, 1.0]
        self.wall_list = []
        self.wall_objects = []
        self.wall_rotation = 0
        self.preview_wall = None
        self.grid_snap = True
        self.grid_size = 0.5
        self.last_mouse_pos = [0, 0, 0]
        
        # Position control
        self.manual_position_mode = False
        self.manual_position = [0, 0, 0]
        
        # Load house model
        self.load_house_model()
        
        # Setup camera
        self.setup_camera()
        
        # Create ground grid visualization
        self.create_grid_visualization()
    
    def load_house_model(self):
        """Load house model as visual reference"""
        house_stl = "E:/ArmBot/Mythings/Models/house.stl"
        mesh_scale = [0.001, 0.001, 0.001]
        
        if os.path.exists(house_stl):
            try:
                visual_shape = p.createVisualShape(
                    p.GEOM_MESH,
                    fileName=house_stl,
                    meshScale=mesh_scale,
                    rgbaColor=[0.8, 0.6, 0.4, 0.3]
                )
                
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[0, 0, -0.5]
                )
                print("✅ House model loaded")
            except Exception as e:
                print(f"⚠️ Could not load house model: {e}")
    
    def setup_camera(self):
        """Set up camera position for top-down view"""
        p.resetDebugVisualizerCamera(
            cameraDistance=15,  # Higher distance for better overview
            cameraYaw=0,        # Front-facing
            cameraPitch=-89,    # Nearly top-down (90 degrees would be completely top-down)
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Configure debug visualizer for better top-down view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    
    def create_grid_visualization(self):
        """Create visual grid on ground"""
        grid_size = 10
        for i in range(-grid_size, grid_size + 1):
            # X-direction lines
            p.addUserDebugLine(
                [i, -grid_size, 0.01],
                [i, grid_size, 0.01],
                [0.3, 0.3, 0.3],
                lineWidth=1
            )
            # Y-direction lines
            p.addUserDebugLine(
                [-grid_size, i, 0.01],
                [grid_size, i, 0.01],
                [0.3, 0.3, 0.3],
                lineWidth=1
            )
    
    def snap_to_grid(self, pos):
        """Snap position to grid if enabled"""
        if not self.grid_snap:
            return pos
        
        return [
            round(pos[0] / self.grid_size) * self.grid_size,
            round(pos[1] / self.grid_size) * self.grid_size,
            pos[2]
        ]
    
    def get_mouse_world_pos(self, mouse_event):
        """Convert mouse position to world coordinates"""
        cam_info = p.getDebugVisualizerCamera()
        cam_pos = cam_info[11]
        
        ray_start = [cam_pos[0], cam_pos[1], 10]
        ray_end = [cam_pos[0], cam_pos[1], -10]
        
        hit = p.rayTest(ray_start, ray_end)[0]
        
        if hit[0] != -1:
            return hit[3]
        return [0, 0, 0]
    
    def create_wall_preview(self, position):
        """Create or update wall preview"""
        if self.preview_wall is not None:
            p.removeBody(self.preview_wall)
        
        # Use manual position if enabled, otherwise use mouse position
        if self.manual_position_mode:
            position = self.manual_position
        
        preview_color = [0.2, 0.8, 0.2, 0.5]
        
        wall_vis = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=self.wall_half_extents, 
            rgbaColor=preview_color
        )
        
        rotation_quat = p.getQuaternionFromEuler([0, 0, math.radians(self.wall_rotation)])
        
        self.preview_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=wall_vis,
            basePosition=[position[0], position[1], self.wall_half_extents[2]],
            baseOrientation=rotation_quat
        )
    
    def place_wall(self, position):
        """Place a wall at the given position"""
        snapped_pos = self.snap_to_grid(position)
        
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.wall_half_extents)
        wall_vis = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=self.wall_half_extents, 
            rgbaColor=self.wall_color
        )
        
        rotation_quat = p.getQuaternionFromEuler([0, 0, math.radians(self.wall_rotation)])
        
        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            baseVisualShapeIndex=wall_vis,
            basePosition=[snapped_pos[0], snapped_pos[1], self.wall_half_extents[2]],
            baseOrientation=rotation_quat
        )
        
        wall_data = {
            "position": [snapped_pos[0], snapped_pos[1], self.wall_half_extents[2]],
            "half_extents": self.wall_half_extents.copy(),
            "rotation": self.wall_rotation,
            "color": self.wall_color.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.wall_list.append(wall_data)
        self.wall_objects.append(wall_id)
        
        # Update GUI
        self.gui.update_wall_count(len(self.wall_list))
        
        print(f"🧱 Wall #{len(self.wall_list)} placed at ({snapped_pos[0]:.1f}, {snapped_pos[1]:.1f})")
    
    def delete_wall_at_position(self, position):
        """Delete wall closest to the given position"""
        if not self.wall_objects:
            return
        
        min_distance = float('inf')
        closest_wall_index = -1
        
        for i, wall_data in enumerate(self.wall_list):
            wall_pos = wall_data["position"]
            distance = math.sqrt(
                (position[0] - wall_pos[0])**2 + 
                (position[1] - wall_pos[1])**2
            )
            
            if distance < min_distance and distance < 1.0:
                min_distance = distance
                closest_wall_index = i
        
        if closest_wall_index >= 0:
            p.removeBody(self.wall_objects[closest_wall_index])
            del self.wall_objects[closest_wall_index]
            del self.wall_list[closest_wall_index]
            
            self.gui.update_wall_count(len(self.wall_list))
            print(f"🗑️ Wall deleted")
    
    def undo_last_wall(self):
        """Remove the last placed wall"""
        if self.wall_objects:
            p.removeBody(self.wall_objects[-1])
            self.wall_objects.pop()
            self.wall_list.pop()
            self.gui.update_wall_count(len(self.wall_list))
            print("↩️ Last wall undone")
    
    def clear_all_walls(self):
        """Remove all walls"""
        for wall_id in self.wall_objects:
            p.removeBody(wall_id)
        
        self.wall_objects.clear()
        self.wall_list.clear()
        self.gui.update_wall_count(len(self.wall_list))
        print("🗑️ All walls cleared")
    
    def save_walls(self, filepath=None):
        """Save wall configuration to JSON file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"wall_layout_{timestamp}.json"
        
        wall_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "wall_count": len(self.wall_list),
                "tool_version": "3.0_GUI"
            },
            "walls": self.wall_list
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(wall_data, f, indent=4)
            print(f"✅ {len(self.wall_list)} walls saved to {filepath}")
            messagebox.showinfo("Success", f"Saved {len(self.wall_list)} walls to {filepath}")
        except Exception as e:
            print(f"❌ Error saving walls: {e}")
            messagebox.showerror("Error", f"Error saving walls: {e}")
    
    def load_walls(self, filepath):
        """Load wall configuration from JSON file"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            # Clear existing walls
            self.clear_all_walls()
            
            # Load walls from file
            walls = data.get("walls", data) if "walls" in data else data
            
            for wall_data in walls:
                # Set wall parameters
                old_extents = self.wall_half_extents.copy()
                old_rotation = self.wall_rotation
                old_color = self.wall_color.copy()
                
                self.wall_half_extents = wall_data["half_extents"]
                self.wall_rotation = wall_data.get("rotation", 0)
                self.wall_color = wall_data.get("color", [0.6, 0.2, 0.2, 1.0])
                
                # Place wall
                self.place_wall(wall_data["position"])
                
                # Restore original parameters
                self.wall_half_extents = old_extents
                self.wall_rotation = old_rotation
                self.wall_color = old_color
            
            print(f"✅ Loaded {len(walls)} walls from {filepath}")
            messagebox.showinfo("Success", f"Loaded {len(walls)} walls from {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading walls: {e}")
            messagebox.showerror("Error", f"Error loading walls: {e}")
    
    def export_wall_data(self, filepath):
        """Export wall data in various formats"""
        try:
            if filepath.lower().endswith('.csv'):
                # Export as CSV
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Wall_ID', 'X', 'Y', 'Z', 'Width', 'Thickness', 'Height', 'Rotation', 'R', 'G', 'B'])
                    
                    for i, wall in enumerate(self.wall_list, 1):
                        pos = wall['position']
                        extents = wall['half_extents']
                        color = wall.get('color', [0.6, 0.2, 0.2, 1.0])
                        writer.writerow([
                            i, pos[0], pos[1], pos[2],
                            extents[0]*2, extents[1]*2, extents[2]*2,
                            wall.get('rotation', 0),
                            color[0], color[1], color[2]
                        ])
                        
            else:
                # Export as text
                with open(filepath, 'w') as f:
                    f.write("Wall Layout Export\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Walls: {len(self.wall_list)}\n\n")
                    
                    for i, wall in enumerate(self.wall_list, 1):
                        pos = wall['position']
                        extents = wall['half_extents']
                        color = wall.get('color', [0.6, 0.2, 0.2, 1.0])
                        
                        f.write(f"Wall {i}:\n")
                        f.write(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n")
                        f.write(f"  Dimensions: {extents[0]*2:.2f} x {extents[1]*2:.2f} x {extents[2]*2:.2f} m\n")
                        f.write(f"  Rotation: {wall.get('rotation', 0):.1f}°\n")
                        f.write(f"  Color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})\n\n")
            
            print(f"✅ Wall data exported to {filepath}")
            messagebox.showinfo("Success", f"Wall data exported to {filepath}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            messagebox.showerror("Error", f"Error exporting data: {e}")
    
    def handle_mouse_events(self):
        """Process mouse input"""
        mouse_events = p.getMouseEvents()
        
        for event in mouse_events:
            if event[0] == 2:  # Mouse button press
                button_state = event[3]
                
                if button_state == 0:  # Left button pressed
                    if self.manual_position_mode:
                        # Use manual position
                        self.place_wall(self.manual_position)
                    else:
                        # Use mouse position
                        world_pos = self.get_mouse_world_pos(event)
                        self.place_wall(world_pos)
                
                elif button_state == 1:  # Right button pressed
                    world_pos = self.get_mouse_world_pos(event)
                    self.delete_wall_at_position(world_pos)
        
        # Update mouse position for preview (only if not in manual mode)
        if not self.manual_position_mode:
            mouse_pos = p.getMouseEvents()
            if mouse_pos:
                self.last_mouse_pos = self.get_mouse_world_pos(mouse_pos[-1])
                self.create_wall_preview(self.last_mouse_pos)
        else:
            # Show preview at manual position
            self.create_wall_preview(self.manual_position)
    
    def run(self):
        """Main execution loop"""
        print("🚀 GUI-Based Wall Placement Tool Started!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        try:
            while self.gui.running:
                # Handle mouse events
                self.handle_mouse_events()
                
                # Sleep for smooth operation
                time.sleep(1.0 / 60.0)  # 60 FPS
                
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        
        finally:
            # Clean up preview
            if self.preview_wall is not None:
                p.removeBody(self.preview_wall)
            
            # Save walls
            self.save_walls()
            
            # Disconnect
            p.disconnect()
            print("👋 PyBullet simulation closed!")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🎯 Starting GUI-Based Wall Placement Tool...")
    print("Use the GUI window to control all wall placement settings!")
    
    app = WallPlacementGUI()
    app.run()