#!/usr/bin/env python3
"""
PyBullet Robotic Arm Simulation - Pick and Place Task

This script simulates a Panda robotic arm performing a simple pick and place
operation. The robot picks up a cube from one location on a table and places
it at another predefined location.

Features:
- Physics-based simulation using PyBullet
- Panda 7-DOF robotic arm
- Simple pick and place with hardcoded positions
- Modular code structure for easy understanding
"""

import pybullet as p
import pybullet_data
import numpy as np
import time


class PandaRobotSimulation:
    """
    A class to simulate Panda robot arm performing pick and place operations.
    """
    
    def __init__(self, gui_mode=True):
        """
        Initialize the PyBullet simulation environment.
        
        Args:
            gui_mode (bool): Whether to run simulation with GUI or headless
        """
        # Connect to PyBullet physics server
        if gui_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set additional search path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Initialize simulation parameters
        self.robot_id = None
        self.table_id = None
        self.cube_id = None
        self.gripper_open_position = 0.04  # Gripper fully open
        self.gripper_close_position = 0.0  # Gripper closed
        
        # Predefined positions for pick and place (based on actual table surface height ~0.65)
        self.pick_position = [0.5, 0.0, 0.6]    # Position to pick up cube (slightly above table surface)
        self.place_position = [0.3, 0.3, 0.6]   # Position to place cube (slightly above table surface) 
        self.home_position = [0.3, 0.0, 0.85]    # Home position well above table center
        
    def setup_simulation(self):
        """
        Set up the simulation environment with gravity, ground plane, table, and objects.
        """
        # Set gravity (Earth's gravity: -9.81 m/s^2 in Z direction)
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        ground_plane_id = p.loadURDF("plane.urdf")
        
        # Load and position the table at origin for better robot coordination
        table_position = [0.5, 0, 0.0]  # X, Y, Z coordinates - table at ground level
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation
        self.table_id = p.loadURDF("table/table.urdf", 
                                  table_position, 
                                  table_orientation,
                                  globalScaling=1.0)  # Use full scale
        
        # Load the Panda robot arm with base sitting ON the table surface
        # Table surface is at approximately Z=0.626, position robot base there
        robot_position = [0.0, 0.0, 0.626]  # Position robot base exactly on table surface
        robot_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                  robot_position, 
                                  robot_orientation,
                                  useFixedBase=True)
        
        # Create a simple cube to manipulate
        self.create_cube()
        
        # Set up camera for better viewing
        self.setup_camera()
        
    def create_cube(self):
        """
        Create a cube object on the table for the robot to manipulate.
        """
        # Define cube properties
        cube_size = 0.05  # 5cm cube
        cube_mass = 0.1   # 100g
        
        # Create cube collision shape
        cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, 
                                                     halfExtents=[cube_size/2] * 3)
        
        # Create cube visual shape (red color)
        cube_visual_shape = p.createVisualShape(p.GEOM_BOX, 
                                               halfExtents=[cube_size/2] * 3,
                                               rgbaColor=[1, 0, 0, 1])
        
        # Position cube on the table surface at pick location
        # Table surface is at Z ≈ 0.65 based on actual measurements
        cube_position = [self.pick_position[0], self.pick_position[1], 0.675]  # Just above table surface
        cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create the cube in simulation
        self.cube_id = p.createMultiBody(baseMass=cube_mass,
                                        baseCollisionShapeIndex=cube_collision_shape,
                                        baseVisualShapeIndex=cube_visual_shape,
                                        basePosition=cube_position,
                                        baseOrientation=cube_orientation)
        
    def setup_camera(self):
        """
        Position the camera for optimal viewing of the simulation.
        """
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                   cameraYaw=45,
                                   cameraPitch=-30,
                                   cameraTargetPosition=[0.5, 0, 0.5])
        
    def get_joint_info(self):
        """
        Get information about the robot's joints for debugging purposes.
        """
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot has {num_joints} joints:")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
            
    def set_initial_robot_pose(self):
        """
        Set the robot to a safe initial pose before starting operations.
        """
        # Define safe initial joint positions (arms up, pointing forward)
        initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Safe starting pose
        
        # Set each joint to initial position
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i,
                                  p.POSITION_CONTROL,
                                  targetPosition=initial_joint_positions[i],
                                  force=500)
        
        # Wait for robot to reach initial position
        for _ in range(480):  # 2 seconds at 240 Hz
            p.stepSimulation()
            time.sleep(1./240.)
            
        print("Robot set to initial safe pose")

    def move_to_position(self, target_position, target_orientation=None):
        """
        Move the robot end-effector to a target position using inverse kinematics.
        
        Args:
            target_position (list): [x, y, z] coordinates for end-effector
            target_orientation (list): Quaternion orientation (optional)
        """
        if target_orientation is None:
            # Default orientation pointing downward
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Use inverse kinematics to calculate joint positions
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=11,  # Panda end-effector link
            targetPosition=target_position,
            targetOrientation=target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Set joint positions (first 7 joints are arm joints) with controlled force
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i,
                                  p.POSITION_CONTROL,
                                  targetPosition=joint_positions[i],
                                  force=500,
                                  maxVelocity=1.0)  # Limit velocity for smoother motion
        
        # Wait for movement to complete
        self.wait_for_movement()
        
    def control_gripper(self, position):
        """
        Control the gripper opening/closing.
        
        Args:
            position (float): Gripper position (0.0 = closed, 0.04 = open)
        """
        # Panda gripper has two finger joints (9 and 10)
        p.setJointMotorControl2(self.robot_id, 9,
                              p.POSITION_CONTROL,
                              targetPosition=position)
        p.setJointMotorControl2(self.robot_id, 10,
                              p.POSITION_CONTROL,
                              targetPosition=position)
        
        # Wait for gripper movement with more time
        time.sleep(2.0)
        
    def wait_for_movement(self, timeout=5.0):
        """
        Wait for robot movement to complete with better settling logic.
        
        Args:
            timeout (float): Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Step the simulation
            p.stepSimulation()
            time.sleep(1./240.)  # 240 Hz simulation
            
            # Better settling time - wait for actual movement completion
            if time.time() - start_time > 3.0:  # Allow more time for movement
                break
                
    def debug_positions(self):
        """
        Print the actual positions of objects in the simulation for debugging.
        """
        # Get table position and orientation
        table_pos, table_orn = p.getBasePositionAndOrientation(self.table_id)
        print(f"Table actual position: {table_pos}")
        
        # Get robot position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        print(f"Robot actual position: {robot_pos}")
        
        # Get cube position and orientation
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
        print(f"Cube actual position: {cube_pos}")
        
        # Get table dimensions (AABB - Axis Aligned Bounding Box)
        table_aabb = p.getAABB(self.table_id)
        print(f"Table AABB (min, max): {table_aabb}")
        table_height = table_aabb[1][2]  # Maximum Z coordinate
        print(f"Table surface height: {table_height}")

    def pick_and_place_sequence(self):
        """
        Execute the complete pick and place sequence.
        """
        print("Starting pick and place sequence...")
        
        # Step 1: Move to home position
        print("1. Moving to home position...")
        self.move_to_position(self.home_position)
        
        # Step 2: Open gripper
        print("2. Opening gripper...")
        self.control_gripper(self.gripper_open_position)
        
        # Step 3: Move to pick position (above cube) - safer approach
        print("3. Moving to pick approach position...")
        pick_approach = [self.pick_position[0], self.pick_position[1], self.pick_position[2] + 0.15]
        self.move_to_position(pick_approach)
        
        # Step 4: Lower to cube slowly
        print("4. Lowering to cube...")
        pick_target = [self.pick_position[0], self.pick_position[1], self.pick_position[2] + 0.05]
        self.move_to_position(pick_target)
        
        # Step 5: Close gripper to grasp cube
        print("5. Grasping cube...")
        self.control_gripper(self.gripper_close_position)
        
        # Step 6: Lift cube carefully
        print("6. Lifting cube...")
        lift_position = [self.pick_position[0], self.pick_position[1], self.pick_position[2] + 0.15]
        self.move_to_position(lift_position)
        
        # Step 7: Move to place position (above target) - safer approach
        print("7. Moving to place approach position...")
        place_approach = [self.place_position[0], self.place_position[1], self.place_position[2] + 0.15]
        self.move_to_position(place_approach)
        
        # Step 8: Lower to place position slowly
        print("8. Lowering to place position...")
        place_target = [self.place_position[0], self.place_position[1], self.place_position[2] + 0.05]
        self.move_to_position(place_target)
        
        # Step 9: Release cube
        print("9. Releasing cube...")
        self.control_gripper(self.gripper_open_position)
        
        # Step 10: Move horizontally away from the placed cube
        print("10. Moving horizontally away from cube...")
        retreat_position = [self.place_position[0], self.place_position[1], self.place_position[2] + 0.15]
        self.move_to_position(retreat_position)
        
        # Step 11: Return to home position
        print("11. Returning to home position...")
        self.move_to_position(self.home_position)
        
        print("Pick and place sequence completed!")
        
    def run_simulation(self, duration=30):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration (float): Time to run simulation in seconds
        """
        print("Setting up simulation environment...")
        self.setup_simulation()
        
        print("Getting robot joint information...")
        self.get_joint_info()
        
        # Wait a moment for everything to settle
        for _ in range(240):  # 1 second at 240 Hz
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Debug: Print actual positions
        print("=== DEBUG: Object Positions ===")
        self.debug_positions()
        print("==============================")
        
        # Set robot to initial safe pose
        self.set_initial_robot_pose()
        
        # Execute pick and place
        self.pick_and_place_sequence()
        
        # Continue simulation to observe the result
        print(f"Continuing simulation for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            p.stepSimulation()
            time.sleep(1./240.)
            
    def cleanup(self):
        """
        Clean up and disconnect from PyBullet.
        """
        p.disconnect()
        print("Simulation ended.")


def main():
    """
    Main function to run the robotic arm simulation.
    """
    # Create and run the simulation
    sim = PandaRobotSimulation(gui_mode=True)
    
    try:
        sim.run_simulation(duration=10)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        sim.cleanup()


if __name__ == "__main__":
    main()
