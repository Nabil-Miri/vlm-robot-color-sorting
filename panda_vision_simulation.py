#!/usr/bin/env python3
"""
PyBullet Robotic Arm Simulation with Vision-Language Model (CLIP)

This script extends the basic Panda robotic arm simulation by integrating
CLIP (Contrastive Language-Image Pre-training) for intelligent object selection and color sorting.
The robot uses its camera and CLIP to identify, select, and sort multiple objects based on text prompts or color.

Features:
- Physics-based simulation using PyBullet
- Panda 7-DOF robotic arm with gripper
- Multiple objects for manipulation (supports 5+ objects)
- Camera image capture and processing
- Vision-language (CLIP) based object selection and color sorting
- Intelligent pick and place based on text prompts
- Interactive and automatic color sorting demonstrations
- Modular, well-commented code structure
- Real-time visualization of selected objects and sorting

Dependencies:
- pybullet
- numpy
- torch
- transformers
- PIL (Pillow)
- matplotlib
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Tuple, Optional


class VisionLanguagePandaSimulation:
    """
    A class that combines Panda robot simulation with CLIP vision-language understanding
    for intelligent object selection and manipulation.
    """
    
    def __init__(self, gui_mode=True, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the simulation with both PyBullet physics and CLIP vision-language model.
        
        Args:
            gui_mode (bool): Whether to run simulation with GUI or headless
            model_name (str): Name of the CLIP model to use
        """
        # Initialize PyBullet simulation
        if gui_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Initialize CLIP vision-language model
        self._initialize_clip(model_name)
        
        # Simulation objects
        self.robot_id = None
        self.table_id = None
        self.objects = {}  # Dictionary to store multiple objects
        self.camera_id = None
        
        # Robot parameters
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        
        # Camera parameters (wider resolution for better field of view)
        self.camera_position = [0.8, 0.0, 1.2]  # Camera position above the table
        self.camera_target = [0.5, 0.0, 0.65]   # Looking at table center
        self.camera_width = 800  # Increased width for wider view
        self.camera_height = 480
        
        # Workspace positions
        self.home_position = [0.3, 0.0, 0.85]
        self.place_position = [0.2, 0.3, 0.6]  # Where to place selected objects
        
        # Color sorting zones - left and right of robot
        self.sorting_zones = {
            'red_zone': {
                'position': [-0.1, 0.3, 0.65],  # Left side
                'color': [1, 0, 0, 0.3],        # Semi-transparent red
                'size': [0.15, 0.15, 0.02]      # 15cm x 15cm x 2cm
            },
            'blue_zone': {
                'position': [-0.1, -0.3, 0.65], # Left side, back
                'color': [0, 0, 1, 0.3],        # Semi-transparent blue
                'size': [0.15, 0.15, 0.02]
            },
            'green_zone': {
                'position': [0.7, 0.3, 0.65],   # Right side
                'color': [0, 1, 0, 0.3],        # Semi-transparent green
                'size': [0.15, 0.15, 0.02]
            },
            'yellow_zone': {
                'position': [0.7, -0.3, 0.65],  # Right side, back
                'color': [1, 1, 0, 0.3],        # Semi-transparent yellow
                'size': [0.15, 0.15, 0.02]
            }
        }
        
    def _initialize_clip(self, model_name):
        """
        Initialize CLIP model for vision-language matching.
        
        Args:
            model_name (str): Name of the CLIP model to use
        """
        print(f"Loading CLIP model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP model loaded successfully!")
        
    def setup_simulation(self):
        """
        Set up the complete simulation environment with physics, objects, and camera.
        """
        print("Setting up simulation environment...")
        
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        ground_plane_id = p.loadURDF("plane.urdf")
        
        # Load and position the table
        table_position = [0.5, 0, 0.0]
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.table_id = p.loadURDF("table/table.urdf", 
                                  table_position, 
                                  table_orientation,
                                  globalScaling=1.0)
        
        # Load the Panda robot arm
        robot_position = [0.0, 0.0, 0.626]  # On table surface
        robot_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                  robot_position, 
                                  robot_orientation,
                                  useFixedBase=True)
        
        # Create multiple objects for selection
        self.create_multiple_objects()
        
        # Create color sorting zones
        self.create_sorting_zones()
        
        # Set up camera for better viewing
        self.setup_camera()
        
        print("Simulation environment setup complete!")
        
    def create_multiple_objects(self):
        """
        Create multiple objects of different colors and shapes for the robot to choose from.
        """
        print("Creating multiple objects for manipulation...")
        
        # Object parameters
        object_size = 0.05  # 5cm
        object_mass = 0.1   # 100g
        table_surface_z = 0.65  # Height of table surface (slightly lower to ensure contact)
        
        # Define different objects with positions and colors
        object_configs = [
            {
                'name': 'red_cube',
                'shape': 'cube',
                'position': [0.6, -0.1, table_surface_z],
                'color': [1, 0, 0, 1],  # Red
                'size': object_size
            },
            {
                'name': 'red_sphere',
                'shape': 'sphere', 
                'position': [0.6, 0.1, table_surface_z],
                'color': [1, 0, 0, 1],  # Red
                'size': object_size
            },
            {
                'name': 'green_sphere',
                'shape': 'sphere',
                'position': [0.4, -0.1, table_surface_z],
                'color': [0, 1, 0, 1],  # Green
                'size': object_size
            },
            {
                'name': 'yellow_cylinder',
                'shape': 'cylinder',
                'position': [0.4, 0.1, table_surface_z],
                'color': [1, 1, 0, 1],  # Yellow
                'size': object_size
            },
            {
                'name': 'blue_cube',
                'shape': 'cube',
                'position': [0.5, -0.2, table_surface_z],
                'color': [0, 0, 1, 1],  # Blue
                'size': object_size
            }
        ]
        
        # Create each object
        for config in object_configs:
            object_id = self.create_object(config)
            self.objects[config['name']] = {
                'id': object_id,
                'config': config,
                'initial_position': config['position'].copy()
            }
            
        print(f"Created {len(self.objects)} objects: {list(self.objects.keys())}")
        
    def create_object(self, config):
        """
        Create a single object based on configuration.
        
        Args:
            config (dict): Object configuration with shape, position, color, etc.
            
        Returns:
            int: PyBullet object ID
        """
        size = config['size']
        
        # Create collision and visual shapes based on object type
        if config['shape'] == 'cube':
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2] * 3)
            visual_shape = p.createVisualShape(p.GEOM_BOX, 
                                             halfExtents=[size/2] * 3,
                                             rgbaColor=config['color'])
        elif config['shape'] == 'sphere':
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size/2)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE,
                                             radius=size/2,
                                             rgbaColor=config['color'])
        elif config['shape'] == 'cylinder':
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                   radius=size/2, height=size)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                             radius=size/2, length=size,
                                             rgbaColor=config['color'])
        else:
            raise ValueError(f"Unsupported shape: {config['shape']}")
        
        # Create the object in simulation
        object_id = p.createMultiBody(baseMass=0.1,
                                     baseCollisionShapeIndex=collision_shape,
                                     baseVisualShapeIndex=visual_shape,
                                     basePosition=config['position'],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        
        return object_id
        
    def create_sorting_zones(self):
        """
        Create colored sorting zones with borders for color-based object placement.
        These zones will serve as targets for color sorting tasks.
        """
        print("Creating color sorting zones with borders...")
        
        self.zone_ids = {}  # Store zone object IDs
        
        for zone_name, zone_config in self.sorting_zones.items():
            # Create a flat colored box to represent the sorting zone
            position = zone_config['position']
            color = zone_config['color']
            size = zone_config['size']
            
            # Create collision and visual shapes for the zone base
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=[size[0]/2, size[1]/2, size[2]/2]
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                rgbaColor=color
            )
            
            # Create the zone object (static, no mass)
            zone_id = p.createMultiBody(
                baseMass=0,  # Static object
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            
            self.zone_ids[zone_name] = zone_id
            
            # Create borders around the zone to prevent objects from falling off
            border_height = 0.03  # 3cm high borders
            border_thickness = 0.01  # 1cm thick borders
            
            # Border positions (front, back, left, right)
            border_positions = [
                # Front border
                [position[0], position[1] + size[1]/2 + border_thickness/2, position[2] + border_height/2],
                # Back border  
                [position[0], position[1] - size[1]/2 - border_thickness/2, position[2] + border_height/2],
                # Left border
                [position[0] - size[0]/2 - border_thickness/2, position[1], position[2] + border_height/2],
                # Right border
                [position[0] + size[0]/2 + border_thickness/2, position[1], position[2] + border_height/2]
            ]
            
            border_sizes = [
                # Front and back borders (span full width + border thickness)
                [size[0] + 2*border_thickness, border_thickness, border_height],
                [size[0] + 2*border_thickness, border_thickness, border_height],
                # Left and right borders (span just the zone height)  
                [border_thickness, size[1], border_height],
                [border_thickness, size[1], border_height]
            ]
            
            # Create borders with slightly darker color
            border_color = [c * 0.7 for c in color[:3]] + [0.8]  # Darker, more opaque
            
            for i, (border_pos, border_size) in enumerate(zip(border_positions, border_sizes)):
                border_collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[border_size[0]/2, border_size[1]/2, border_size[2]/2]
                )
                
                border_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[border_size[0]/2, border_size[1]/2, border_size[2]/2],
                    rgbaColor=border_color
                )
                # Actually create the border object in simulation
                border_id = p.createMultiBody(
                    baseMass=0,  # Static border
                    baseCollisionShapeIndex=border_collision,
                    baseVisualShapeIndex=border_visual,
                    basePosition=border_pos,
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
                )
            
            # Add text label above the zone
            zone_color_name = zone_name.replace('_zone', '').upper()
            print(f"   Created {zone_color_name} sorting zone with borders at {position}")
        
        print(f"Created {len(self.zone_ids)} sorting zones: {list(self.zone_ids.keys())}")
        
    def setup_camera(self):
        """
        Set up the simulation camera for optimal viewing and image capture.
        """
        # Set debug camera for GUI viewing (wider distance for better coverage)
        p.resetDebugVisualizerCamera(cameraDistance=2.0,  # Increased distance for wider view
                                   cameraYaw=45,
                                   cameraPitch=-25,  # Slightly less steep angle
                                   cameraTargetPosition=[0.5, 0, 0.5])
        
    def capture_camera_image(self):
        """
        Capture an image from the robot's perspective using PyBullet's camera.
        
        Returns:
            PIL.Image: Captured image from camera
        """
        # Calculate camera view matrix (looking at table from above/side)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 0, 1]
        )
        
        # Calculate camera projection matrix (wider field of view)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90,  # Wider field of view for better coverage
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Capture image
        width, height, rgb_array, depth_array, seg_array = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        # Convert to PIL Image (remove alpha channel)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        image = Image.fromarray(rgb_array, 'RGB')
        
        return image
        
    def capture_camera_image_with_segmentation(self):
        """
        Capture an image from the robot's perspective with segmentation data.
        
        Returns:
            tuple: (PIL.Image, numpy.array, numpy.array) - RGB image, depth array, segmentation array
        """
        # Calculate camera view matrix (looking at table from above/side)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 0, 1]
        )
        
        # Calculate camera projection matrix (wider field of view)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90,  # Wider field of view for better coverage
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Capture image with segmentation
        width, height, rgb_array, depth_array, seg_array = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        # Convert to PIL Image (remove alpha channel)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        image = Image.fromarray(rgb_array, 'RGB')
        
        return image, depth_array, seg_array
        
    def get_object_bounding_boxes(self, image):
        """
        Generate bounding boxes for each object based on their 3D positions.
        This is a simplified approach - in a real system you might use
        object detection or segmentation.
        
        Args:
            image (PIL.Image): Camera image
            
        Returns:
            dict: Dictionary mapping object names to bounding boxes (x1, y1, x2, y2)
        """
        bounding_boxes = {}
        
        # Get camera matrices for 3D to 2D projection
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 0, 1]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90,  # Match the wider field of view from capture_camera_image
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # For each object, project its 3D position to 2D screen coordinates
        for obj_name, obj_info in self.objects.items():
            # Get current object position
            obj_pos, _ = p.getBasePositionAndOrientation(obj_info['id'])
            
            # Project 3D position to 2D screen coordinates
            # This is a simplified approach - real systems would use object detection
            screen_coords = self.project_3d_to_2d(obj_pos, view_matrix, projection_matrix)
            
            if screen_coords is not None:
                x, y = screen_coords
                
                # Create bounding box around projected position
                # Size based on object size and distance (smaller for more precise detection)
                bbox_size = 50  # Pixels - reduced from 80 to 50 for smaller, more precise boxes
                
                x1 = max(0, int(x - bbox_size // 2))
                y1 = max(0, int(y - bbox_size // 2))
                x2 = min(self.camera_width, int(x + bbox_size // 2))
                y2 = min(self.camera_height, int(y + bbox_size // 2))
                
                bounding_boxes[obj_name] = (x1, y1, x2, y2)
        
        return bounding_boxes
        
    def project_3d_to_2d(self, world_pos, view_matrix, projection_matrix):
        """
        Project 3D world coordinates to 2D screen coordinates.
        
        Args:
            world_pos (list): 3D world position [x, y, z]
            view_matrix (list): Camera view matrix
            projection_matrix (list): Camera projection matrix
            
        Returns:
            tuple: (x, y) screen coordinates or None if behind camera
        """
        # Convert to homogeneous coordinates
        world_pos_homo = [world_pos[0], world_pos[1], world_pos[2], 1.0]
        
        # Convert matrices to numpy arrays for easier calculation
        view_mat = np.array(view_matrix).reshape(4, 4).T
        proj_mat = np.array(projection_matrix).reshape(4, 4).T
        
        # Transform world coordinates to camera coordinates
        camera_pos = view_mat @ world_pos_homo
        
        # Check if point is behind camera
        if camera_pos[2] > 0:  # Behind camera
            return None
            
        # Transform to clip coordinates
        clip_pos = proj_mat @ camera_pos
        
        # Perspective divide
        if clip_pos[3] != 0:
            ndc_x = clip_pos[0] / clip_pos[3]
            ndc_y = clip_pos[1] / clip_pos[3]
        else:
            return None
        
        # Convert to screen coordinates
        screen_x = (ndc_x + 1.0) * 0.5 * self.camera_width
        screen_y = (1.0 - ndc_y) * 0.5 * self.camera_height  # Flip Y coordinate
        
        return (screen_x, screen_y)
        
    def crop_image_regions(self, image, bounding_boxes):
        """
        Crop image regions based on bounding boxes.
        
        Args:
            image (PIL.Image): Source image
            bounding_boxes (dict): Dictionary of bounding boxes
            
        Returns:
            dict: Dictionary mapping object names to cropped images
        """
        crops = {}
        
        for obj_name, bbox in bounding_boxes.items():
            x1, y1, x2, y2 = bbox
            
            # Ensure bounding box is valid
            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crops[obj_name] = crop
            
        print(f"Created {len(crops)} image crops from {len(bounding_boxes)} bounding boxes")
        return crops
        
    def compute_object_similarity(self, crops, text_prompt):
        """
        Compute similarity scores between object crops and a text prompt.
        
        Args:
            crops (dict): Dictionary of cropped images
            text_prompt (str): Text description to match against
            
        Returns:
            dict: Dictionary mapping object names to similarity scores
        """
        print(f"Computing similarity scores for '{text_prompt}' using CLIP...")
        
        if not crops:
            print("No crops available for similarity computation")
            return {}
        
        return self._compute_clip_similarity(crops, text_prompt)
            
    def _compute_clip_similarity(self, crops, text_prompt):
        """
        Compute similarity scores using CLIP model.
        
        Args:
            crops (dict): Dictionary of cropped images
            text_prompt (str): Text description to match against
            
        Returns:
            dict: Dictionary mapping object names to similarity scores
        """
        # Prepare all images and text prompt
        crop_names = list(crops.keys())
        crop_images = list(crops.values())
        
        # Process all crops together with the text prompt for better comparison
        inputs = self.processor(
            text=[text_prompt],
            images=crop_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        similarity_scores = {}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get similarity scores for all images
            logits_per_image = outputs.logits_per_image  # Shape: (1, num_images)
            logits_per_text = outputs.logits_per_text    # Shape: (num_images, 1)
            
            # Use logits_per_text which gives similarity from text perspective
            scores = torch.softmax(logits_per_text.squeeze(), dim=0)  # Normalize across all objects
            
            # Map scores back to object names
            for i, obj_name in enumerate(crop_names):
                similarity_scores[obj_name] = scores[i].item()
        
        return similarity_scores
        
    def select_best_object(self, similarity_scores):
        """
        Select the object with the highest similarity score.
        
        Args:
            similarity_scores (dict): Dictionary of similarity scores
            
        Returns:
            tuple: (object_name, score) of best matching object
        """
        if not similarity_scores:
            return None, 0.0
            
        best_object = max(similarity_scores.items(), key=lambda x: x[1])
        return best_object
        
    def visualize_selection(self, image, bounding_boxes, similarity_scores, selected_object):
        """
        Visualize the camera image with bounding boxes and selection results.
        
        Args:
            image (PIL.Image): Camera image
            bounding_boxes (dict): Object bounding boxes
            similarity_scores (dict): Similarity scores
            selected_object (str): Name of selected object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Camera image with bounding boxes
        ax1.imshow(image)
        ax1.set_title("Robot Camera View with Object Detection")
        ax1.axis('off')
        
        # Draw bounding boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bounding_boxes)))
        for i, (obj_name, bbox) in enumerate(bounding_boxes.items()):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            
            # Highlight selected object
            linewidth = 4 if obj_name == selected_object else 2
            alpha = 1.0 if obj_name == selected_object else 0.7
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=linewidth, edgecolor=color, 
                                   facecolor='none', alpha=alpha)
            ax1.add_patch(rect)
            
            # Add label with similarity score
            score = similarity_scores.get(obj_name, 0.0)
            label = f"{obj_name}\n{score:.3f}"
            if obj_name == selected_object:
                label = f"★ {label}"
                
            ax1.text(x1, y1-5, label, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    weight='bold' if obj_name == selected_object else 'normal')
        
        # Plot 2: Similarity scores bar chart
        if similarity_scores:
            objects = list(similarity_scores.keys())
            scores = list(similarity_scores.values())
            
            bars = ax2.bar(objects, scores, color=colors[:len(objects)])
            
            # Highlight selected object
            if selected_object in similarity_scores:
                selected_idx = objects.index(selected_object)
                bars[selected_idx].set_color('red')
                bars[selected_idx].set_alpha(1.0)
            
            ax2.set_title("Object Similarity Scores")
            ax2.set_ylabel("Similarity Score")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add score values on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, "No similarity scores available", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Object Similarity Scores")
        
        plt.tight_layout()
        plt.show()
        
    # Robot movement methods (adapted from original simulation)
    def set_initial_robot_pose(self):
        """Set the robot to a safe initial pose."""
        initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i,
                                  p.POSITION_CONTROL,
                                  targetPosition=initial_joint_positions[i],
                                  force=500)
        
        for _ in range(480):
            p.stepSimulation()
            time.sleep(1./240.)
            
        print("Robot set to initial safe pose")
        
    def move_to_position(self, target_position, target_orientation=None):
        """Move the robot end-effector to a target position."""
        if target_orientation is None:
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=11,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i,
                                  p.POSITION_CONTROL,
                                  targetPosition=joint_positions[i],
                                  force=500,
                                  maxVelocity=1.0)
        
        self.wait_for_movement()
        
    def safe_move_to_position(self, target_position, target_orientation=None, min_height=0.9):
        """
        Move the robot end-effector to a target position with safe intermediate waypoint.
        This prevents the robot from going too low during horizontal movements.
        
        Args:
            target_position (list): Target [x, y, z] position
            target_orientation: Target orientation (optional)
            min_height (float): Minimum safe height during movement
        """
        if target_orientation is None:
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Get current end-effector position
        current_pos = p.getLinkState(self.robot_id, 11)[0]
        
        # If either current or target position is below safe height, use intermediate waypoint
        if current_pos[2] < min_height or target_position[2] < min_height:
            # Create intermediate waypoint at safe height
            waypoint_pos = [target_position[0], target_position[1], max(min_height, target_position[2])]
            
            print(f"   Using safe waypoint at height {waypoint_pos[2]:.2f}m")
            
            # Move to safe waypoint first
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                endEffectorLinkIndex=11,
                targetPosition=waypoint_pos,
                targetOrientation=target_orientation,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i,
                                      p.POSITION_CONTROL,
                                      targetPosition=joint_positions[i],
                                      force=500,
                                      maxVelocity=1.0)
            
            self.wait_for_movement()
        
        # Now move to final target position
        self.move_to_position(target_position, target_orientation)
        
    def control_gripper(self, position):
        """Control the gripper opening/closing."""
        # Set both gripper joints
        p.setJointMotorControl2(self.robot_id, 9,
                              p.POSITION_CONTROL,
                              targetPosition=position,
                              force=200,  # Increased force for better grip
                              maxVelocity=0.5)
        p.setJointMotorControl2(self.robot_id, 10,
                              p.POSITION_CONTROL,
                              targetPosition=position,
                              force=200,  # Increased force for better grip
                              maxVelocity=0.5)
        
        # Wait for gripper to reach position
        for _ in range(120):  # 0.5 seconds
            p.stepSimulation()
            time.sleep(1./240.)
        
    def wait_for_movement(self, timeout=5.0):
        """Wait for robot movement to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            p.stepSimulation()
            time.sleep(1./240.)
            
            if time.time() - start_time > 3.0:
                break
                
    def pick_object(self, object_name):
        """
        Pick up a specific object.
        
        Args:
            object_name (str): Name of the object to pick
            
        Returns:
            bool: True if successful, False otherwise
        """
        if object_name not in self.objects:
            print(f"Object '{object_name}' not found!")
            return False
            
        obj_info = self.objects[object_name]
        obj_pos, _ = p.getBasePositionAndOrientation(obj_info['id'])
        
        print(f"Picking up {object_name} at position {obj_pos}")
        print(f"Object height: {obj_pos[2]:.3f}m, Object size: {obj_info['config']['size']:.3f}m")
        
        # Move to home position first
        print("Moving to home position...")
        self.move_to_position(self.home_position)
        
        # Open gripper
        print("Opening gripper...")
        self.control_gripper(self.gripper_open_position)
        
        # Approach object from above (higher to avoid collisions)
        print(f"Approaching {object_name}...")
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.25]  # Increased from 0.15 to 0.25
        self.move_to_position(approach_pos)
        
        # Lower to object - go down to the actual object surface level
        print(f"Lowering to {object_name}...")
        pick_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01]  # Much closer to object surface
        self.move_to_position(pick_pos)
        
        # Close gripper
        print(f"Grasping {object_name}...")
        self.control_gripper(self.gripper_close_position)
        
        # Wait a moment to ensure good grip
        print("Ensuring secure grip...")
        for _ in range(60):  # Quarter second pause
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Lift object higher to avoid table collisions during transport
        print(f"Lifting {object_name}...")
        lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.30]  # Increased from 0.15 to 0.30
        self.move_to_position(lift_pos)
        
        return True
        
    def place_object(self):
        """Place the currently held object at the predefined place position."""
        print("Placing object...")
        
        # Move to place approach position (higher up to avoid table collision)
        place_approach = [self.place_position[0], self.place_position[1], self.place_position[2] + 0.20]  # Increased height
        self.move_to_position(place_approach)
        
        # Lower to place position
        place_target = [self.place_position[0], self.place_position[1], self.place_position[2] + 0.05]  # Higher placement
        self.move_to_position(place_target)
        
        # Release object
        print("Releasing object...")
        self.control_gripper(self.gripper_open_position)
        
        # Retreat with extra height
        retreat_pos = [self.place_position[0], self.place_position[1], self.place_position[2] + 0.20]  # Higher retreat
        self.move_to_position(retreat_pos)
        
        # Return to home
        print("Returning to home position...")
        self.move_to_position(self.home_position)
        
    def place_object_in_sorting_zone(self, object_name):
        """
        Place the currently held object in the appropriate color sorting zone.
        
        Args:
            object_name (str): Name of the object to determine target zone
        """
        print(f"Placing {object_name} in appropriate sorting zone...")
        
        # Determine target zone based on object color
        target_zone = None
        if 'red' in object_name.lower():
            target_zone = 'red_zone'
        elif 'blue' in object_name.lower():
            target_zone = 'blue_zone'
        elif 'green' in object_name.lower():
            target_zone = 'green_zone'
        elif 'yellow' in object_name.lower():
            target_zone = 'yellow_zone'
        
        if target_zone is None:
            print(f"   Warning: No matching zone found for {object_name}, using default position")
            self.place_object()
            return
        
        # Get target zone position
        zone_config = self.sorting_zones[target_zone]
        zone_position = zone_config['position']
        zone_color = target_zone.replace('_zone', '').upper()
        
        print(f"   Target zone: {zone_color} at {zone_position}")
        
        # Move safely to zone approach position (using safe waypoint movement)
        zone_approach = [zone_position[0], zone_position[1], zone_position[2] + 0.25]  # Increased from 0.20 to 0.25
        print(f"   Moving safely to approach position...")
        self.safe_move_to_position(zone_approach)
        
        # Lower to zone surface (just above the colored zone)
        zone_target = [zone_position[0], zone_position[1], zone_position[2] + 0.05]  # Higher above zone
        self.move_to_position(zone_target)
        
        # Release object
        print(f"   Releasing {object_name} in {zone_color} zone...")
        self.control_gripper(self.gripper_open_position)
        
        # Retreat with extra height using safe movement
        retreat_pos = [zone_position[0], zone_position[1], zone_position[2] + 0.25]  # Increased from 0.20 to 0.25
        print(f"   Retreating safely...")
        self.move_to_position(retreat_pos)
        
        # Return to home
        print("   Returning to home position...")
        self.move_to_position(self.home_position)
        
    def run_vision_guided_pick_and_place(self, text_prompt, visualize=True):
        """
        Execute a complete vision-guided pick and place operation.
        
        Args:
            text_prompt (str): Text description of desired object
            visualize (bool): Whether to show visualization
            
        Returns:
            dict: Results including selected object and scores
        """
        print(f"\n{'='*60}")
        print(f"VISION-GUIDED PICK AND PLACE")
        print(f"Text prompt: '{text_prompt}'")
        print(f"{'='*60}")
        
        # Step 1: Capture camera image
        print("\n1. Capturing camera image...")
        camera_image = self.capture_camera_image()
        
        # Step 2: Get object bounding boxes
        print("2. Detecting objects and generating bounding boxes...")
        bounding_boxes = self.get_object_bounding_boxes(camera_image)
        print(f"   Found {len(bounding_boxes)} objects: {list(bounding_boxes.keys())}")
        
        # Step 3: Create image crops
        print("3. Creating image crops for each object...")
        crops = self.crop_image_regions(camera_image, bounding_boxes)
        
        # Step 4: Compute similarity scores
        print("4. Computing vision-language similarity scores...")
        similarity_scores = self.compute_object_similarity(crops, text_prompt)
        
        # Print similarity results
        print("   Similarity scores:")
        for obj_name, score in similarity_scores.items():
            print(f"     {obj_name}: {score:.4f}")
        
        # Step 5: Select best matching object
        print("5. Selecting best matching object...")
        selected_object, best_score = self.select_best_object(similarity_scores)
        
        if selected_object:
            print(f"   Selected: {selected_object} (score: {best_score:.4f})")
        else:
            print("   No suitable object found!")
            return None
        
        # Step 6: Visualize results
        if visualize:
            print("6. Visualizing selection results...")
            self.visualize_selection(camera_image, bounding_boxes, similarity_scores, selected_object)
        
        # Step 7: Execute pick and place
        print("7. Executing pick and place operation...")
        success = self.pick_object(selected_object)
        
        if success:
            self.place_object_in_sorting_zone(selected_object)
            print(f"✓ Successfully completed pick and place for '{text_prompt}'!")
        else:
            print(f"✗ Failed to pick up {selected_object}")
        
        return {
            'selected_object': selected_object,
            'similarity_scores': similarity_scores,
            'success': success,
            'camera_image': camera_image,
            'bounding_boxes': bounding_boxes
        }
        
    def run_simulation_with_prompts(self, text_prompts, duration=5):
        """
        Run the simulation with multiple text prompts.
        
        Args:
            text_prompts (list): List of text prompts to try
            duration (float): Time to pause between operations
        """
        print("Setting up simulation environment...")
        self.setup_simulation()
        
        # Wait for physics to settle
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Set robot to initial pose
        self.set_initial_robot_pose()
        
        # Execute vision-guided operations for each prompt
        results = []
        for i, prompt in enumerate(text_prompts):
            print(f"\n{'='*80}")
            print(f"OPERATION {i+1}/{len(text_prompts)}")
            print(f"{'='*80}")
            
            result = self.run_vision_guided_pick_and_place(prompt, visualize=True)
            results.append(result)
            
            # Pause between operations
            if i < len(text_prompts) - 1:  # Don't pause after last operation
                print(f"\nPausing for {duration} seconds before next operation...")
                start_time = time.time()
                while time.time() - start_time < duration:
                    p.stepSimulation()
                    time.sleep(1./240.)
        
        return results
        
    def cleanup(self):
        """Clean up and disconnect from PyBullet."""
        p.disconnect()
        print("Simulation ended.")
    
    def run_color_sorting_demo(self, visualize=True):
        """
        Run a demonstration of color-based object sorting.
        The robot will pick up each colored object and place it in the matching colored zone.
        
        Args:
            visualize (bool): Whether to show visualization
            
        Returns:
            dict: Results of the sorting operation
        """
        print(f"\n{'='*60}")
        print(f"COLOR SORTING DEMONSTRATION")
        print(f"Robot will sort objects by color into matching zones")
        print(f"{'='*60}")
        
        results = []
        
        # Get all available objects
        available_objects = list(self.objects.keys())
        print(f"Available objects for sorting: {available_objects}")
        
        for i, object_name in enumerate(available_objects):
            print(f"\n--- Sorting object {i+1}/{len(available_objects)}: {object_name} ---")
            
            # Step 1: Capture current scene
            if visualize:
                print("1. Capturing scene image...")
                camera_image = self.capture_camera_image()
                bounding_boxes = self.get_object_bounding_boxes(camera_image)
                
                # Show current scene
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.imshow(camera_image)
                ax.set_title(f"Scene Before Sorting {object_name}")
                ax.axis('off')
                
                # Draw bounding boxes for remaining objects
                colors = plt.cm.Set3(np.linspace(0, 1, len(bounding_boxes)))
                for j, (obj_name, bbox) in enumerate(bounding_boxes.items()):
                    x1, y1, x2, y2 = bbox
                    color = colors[j]
                    
                    # Highlight current target object
                    linewidth = 4 if obj_name == object_name else 2
                    alpha = 1.0 if obj_name == object_name else 0.7
                    
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=linewidth, edgecolor=color, 
                                           facecolor='none', alpha=alpha)
                    ax.add_patch(rect)
                    
                    label = f"→ {obj_name}" if obj_name == object_name else obj_name
                    ax.text(x1, y1-5, label, fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                            weight='bold' if obj_name == object_name else 'normal')
                
                plt.tight_layout()
                plt.show()
            
            # Step 2: Pick up the object
            print(f"2. Picking up {object_name}...")
            success = self.pick_object(object_name)
            
            if not success:
                print(f"   Failed to pick up {object_name}")
                results.append({
                    'object_name': object_name,
                    'success': False,
                    'error': 'Failed to pick up object'
                })
                continue
            
            # Step 3: Place in appropriate sorting zone
            print(f"3. Placing {object_name} in color-matched zone...")
            self.place_object_in_sorting_zone(object_name)
            
            results.append({
                'object_name': object_name,
                'success': True,
                'target_zone': self._get_target_zone_for_object(object_name)
            })
            
            print(f"   ✓ Successfully sorted {object_name}!")
            
            # Pause between objects
            if i < len(available_objects) - 1:
                print("   Pausing before next object...")
                for _ in range(120):  # 0.5 second pause
                    p.stepSimulation()
                    time.sleep(1./240.)
        
        # Final scene visualization
        if visualize:
            print("\nFinal scene after color sorting:")
            final_image = self.capture_camera_image()
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(final_image)
            ax.set_title("Final Scene - All Objects Sorted by Color")
            ax.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Summary
        successful_sorts = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"COLOR SORTING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully sorted: {successful_sorts}/{len(available_objects)} objects")
        
        for result in results:
            if result['success']:
                target_zone = result.get('target_zone', 'unknown')
                print(f"✓ {result['object_name']} → {target_zone}")
            else:
                print(f"✗ {result['object_name']} → {result.get('error', 'Unknown error')}")
        
        return results
    
    def _get_target_zone_for_object(self, object_name):
        """
        Helper method to determine target zone for an object.
        
        Args:
            object_name (str): Name of the object
            
        Returns:
            str: Target zone name
        """
        if 'red' in object_name.lower():
            return 'red_zone'
        elif 'blue' in object_name.lower():
            return 'blue_zone'
        elif 'green' in object_name.lower():
            return 'green_zone'
        elif 'yellow' in object_name.lower():
            return 'yellow_zone'
        else:
            return 'unknown_zone'
        
    def get_object_segmentation_masks(self, segmentation_array):
        """
        Extract object segmentation masks from PyBullet's segmentation array.
        
        Args:
            segmentation_array (numpy.array): Segmentation mask from PyBullet
            
        Returns:
            dict: Dictionary mapping object names to binary masks
        """
        object_masks = {}
        # Create a mapping from object IDs to object names
        id_to_name = {}
        for obj_name, obj_info in self.objects.items():
            id_to_name[obj_info['id']] = obj_name
        # Extract masks for each object
        for obj_id, obj_name in id_to_name.items():
            # Create binary mask where segmentation matches this object ID
            mask = (segmentation_array == obj_id).astype(np.uint8)
            # Only include objects that are actually visible
            if np.any(mask):
                object_masks[obj_name] = mask
        return object_masks
    
    def get_precise_bounding_boxes_from_masks(self, object_masks):
        """
        Calculate precise bounding boxes from segmentation masks.
        
        Args:
            object_masks (dict): Dictionary of binary masks for each object
            
        Returns:
            dict: Dictionary mapping object names to tight bounding boxes (x1, y1, x2, y2)
        """
        precise_boxes = {}
        
        for obj_name, mask in object_masks.items():
            # Find all non-zero pixels (object pixels)
            nonzero_coords = np.nonzero(mask)
            
            if len(nonzero_coords[0]) > 0:  # Make sure object is visible
                # Get min/max coordinates
                y_coords, x_coords = nonzero_coords
                x1, x2 = int(x_coords.min()), int(x_coords.max())
                y1, y2 = int(y_coords.min()), int(y_coords.max())
                
                # Add smaller padding for tighter bounding boxes
                padding = 3
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(self.camera_width - 1, x2 + padding)
                y2 = min(self.camera_height - 1, y2 + padding)
                
                precise_boxes[obj_name] = (x1, y1, x2, y2)
        
        return precise_boxes
    
    def crop_image_with_masks(self, image, object_masks, bounding_boxes):
        """
        Crop image regions using segmentation masks for precise object extraction.
        This removes background and gives cleaner object crops.
        
        Args:
            image (PIL.Image): Source image
            object_masks (dict): Dictionary of binary segmentation masks
            bounding_boxes (dict): Dictionary of bounding boxes
            
        Returns:
            dict: Dictionary mapping object names to masked and cropped images
        """
        crops = {}
        
        # Convert PIL image to numpy array for processing
        img_array = np.array(image)
        
        for obj_name in object_masks.keys():
            if obj_name not in bounding_boxes:
                continue
            
            mask = object_masks[obj_name]
            x1, y1, x2, y2 = bounding_boxes[obj_name]
            
            # Crop the bounding box region
            cropped_img = img_array[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]
            
            # Apply mask to remove background
            # Create 3-channel mask for RGB image
            mask_3d = np.stack([cropped_mask, cropped_mask, cropped_mask], axis=-1)
            
            # Apply mask (keep object pixels, make background white/transparent)
            masked_img = cropped_img * mask_3d
            
            # Make background white instead of black for better CLIP performance
            background_mask = (mask_3d == 0)
            masked_img[background_mask] = 255  # White background
            
            # Convert back to PIL Image
            crop_pil = Image.fromarray(masked_img.astype(np.uint8))
            crops[obj_name] = crop_pil
        
        return crops
    
    def plot_masked_crops(self, crops):
        """
        Plot all masked crops for visual inspection.
        Args:
            crops (dict): Dictionary mapping object names to masked PIL images
        """
        import matplotlib.pyplot as plt
        n = len(crops)
        fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
        if n == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, crops.items()):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def capture_and_process_scene(self):
        """
        Capture scene image and process it to get precise object detection and cropping.
        Uses segmentation masks for accurate object boundaries.
        
        Returns:
            tuple: (PIL.Image, dict, dict) - camera image, bounding boxes, object crops
        """
        print("📸 Capturing scene with segmentation...")
        
        # Capture image with segmentation data
        camera_image, depth_array, segmentation_array = self.capture_camera_image_with_segmentation()
        
        # Extract object masks from segmentation
        print("🎯 Extracting object segmentation masks...")
        object_masks = self.get_object_segmentation_masks(segmentation_array)
        
        # Get precise bounding boxes from masks
        print("📐 Computing precise bounding boxes...")
        precise_bounding_boxes = self.get_precise_bounding_boxes_from_masks(object_masks)
        
        # Create masked crops
        print("✂️ Creating masked object crops...")
        masked_crops = self.crop_image_with_masks(camera_image, object_masks, precise_bounding_boxes)
        
        return camera_image, precise_bounding_boxes, masked_crops

# This file is intended to be imported as a module.
# Do not run as a standalone script.
