"""
Custom Keyboard Device for Mink IK Controller

This device extends robosuite's Keyboard device to support the custom KINOVA3_MINK_IK controller.
It handles the conversion from keyboard input to the format expected by the Mink IK composite controller.

Usage:
    from projects.shared_devices.mink_keyboard_device import MinkKeyboardDevice
    
    device = MinkKeyboardDevice(env=env, pos_sensitivity=1.0, rot_sensitivity=1.0)
    env.viewer.add_keypress_callback(device.on_press)

Written mainly with Claude Sonnet 4 through prompting by Idris Wibowo
"""

import numpy as np
from robosuite.devices import Keyboard
from robosuite.controllers.composite.composite_controller import WholeBody


class MinkKeyboardDevice(Keyboard):
    """
    Custom Keyboard device that supports the KINOVA3_MINK_IK controller.
    
    This device extends the standard robosuite Keyboard device to properly handle
    the custom Mink IK controller by implementing appropriate action conversion.
    
    The Mink IK controller expects end-effector pose commands in the format:
    [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z] where rotation is in axis-angle representation.
    """
    
    def __init__(self, env=None, pos_sensitivity=1.0, rot_sensitivity=1.0):
        """
        Initialize the Mink Keyboard device.
        
        Args:
            env: Robosuite environment
            pos_sensitivity: Position sensitivity multiplier (default: 1.0)
            rot_sensitivity: Rotation sensitivity multiplier (default: 1.0)
        """
        super().__init__(env=env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        
        # Additional scaling factors for Mink IK
        self.mink_pos_scale = 0.2  # Smaller movements for precise control
        self.mink_rot_scale = 0.5  # Smaller rotations for precise control
    
    def get_arm_action(self, robot, arm, norm_delta=None):
        """
        Override get_arm_action to handle KINOVA3_MINK_IK controller.
        
        This method is called by input2action() and needs to return the appropriate
        action format for the specified controller type.
        
        Args:
            robot: Robot instance
            arm: Arm name ('left' or 'right')  
            norm_delta: Normalized delta action [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
            
        Returns:
            dict: Action dictionary with 'abs' and 'delta' keys containing 6DOF pose commands
        """
        if norm_delta is None:
            norm_delta = np.zeros(6)
        
        # Check if we're dealing with a composite controller
        if isinstance(robot.composite_controller, WholeBody):
            controller_name = robot.composite_controller.name
            
            if controller_name == "KINOVA3_MINK_IK":
                # realized that the composite controller does not apply to the arms 
                # we should be using "HYBRID_MOBILE_BASE" with "IK" or self defined "MINK" part controllers
                # print("Debug Case 1: Looking for controller: ", controller_name, robot.part_controllers) 
                return self._get_mink_ik_action(robot, arm, norm_delta)
            elif controller_name == "HYBRID_MOBILE_BASE":
                # print("Debug Case 1B: Looking for controller: ", controller_name, robot.part_controllers) 
                return self._get_osc_action(robot, arm, norm_delta)
            else:
                # Fall back to parent implementation for other composite controllers
                # print("Debug Case 2: Looking for controller: ", controller_name, robot.part_controllers)
                return super().get_arm_action(robot, arm, norm_delta)
        else:
            # For part controllers (non-composite), use parent implementation
            # print("Debug Case 3: Looking for controller: ", robot.part_controllers)
            # 'BASIC' uses OSC controllers. 
            return self._get_osc_action(robot, arm, norm_delta)
    
    def _get_mink_ik_action(self, robot, arm, norm_delta):
        """
        Handle action conversion for KINOVA3_MINK_IK controller.
        
        The Mink IK controller expects end-effector pose targets. We convert
        keyboard input to pose commands in the world frame.
        
        Args:
            robot: Robot instance
            arm: Arm name ('left' or 'right')
            norm_delta: Normalized delta action from keyboard
            
        Returns:
            dict: Action dictionary with 'abs' and 'delta' keys
        """
        # Scale the input delta based on sensitivity and Mink-specific scaling
        pos_delta = norm_delta[:3] * self.pos_sensitivity * self.mink_pos_scale
        rot_delta = norm_delta[3:] * self.rot_sensitivity * self.mink_rot_scale
        
        # Apply coordinate frame correction for directional inversion
        # This is necessary for keyboard to operate in the correct direction
        coord_correction = np.array([
            [-1, 0, 0],   # Invert X (right/left)
            [0, -1, 0],   # Invert Y (forward/backward)  
            [0, 0, 1]     # Keep Z (up/down)
        ])
        
        # Apply coordinate correction to position delta
        pos_delta_corrected = coord_correction @ pos_delta
        
        # For rotation, we may need similar correction depending on the robot orientation
        rot_delta_corrected = coord_correction @ rot_delta
        
        # Combine position and rotation deltas
        scaled_delta = np.concatenate([pos_delta_corrected, rot_delta_corrected])
        
        # Get current end-effector pose for absolute positioning
        try:
            # Try to get current end-effector pose
            site_name = f"gripper0_{arm}_grip_site"
            current_pose = self._get_current_eef_pose(robot, site_name)
            
            if current_pose is not None:
                current_pos = current_pose[:3]
                current_rot = current_pose[3:]
                
                # Calculate absolute target pose
                target_pos = current_pos + pos_delta_corrected
                target_rot = current_rot + rot_delta_corrected
                
                abs_action = np.concatenate([target_pos, target_rot])
            else:
                # Fallback: use delta as absolute (will be interpreted as relative by controller)
                abs_action = scaled_delta.copy()
                
        except Exception as e:
            print(f"Warning: Could not get current pose for {arm} arm: {e}")
            # Fallback to delta-only control
            abs_action = scaled_delta.copy()
        
        # Return action dictionary in the format expected by robosuite Device base class
        return {
            "delta": scaled_delta,
            "abs": abs_action
        }
    
    def _get_osc_action(self, robot, arm, norm_delta):
        """
        Handle action conversion for OSC control.
        
        Args:
            robot: Robot instance
            arm: Arm name ('left' or 'right')
            norm_delta: Normalized delta action from keyboard
            
        Returns:
            dict: Action dictionary with 'abs' and 'delta' keys
        """
        # Scale the input delta based on sensitivity
        pos_delta = norm_delta[:3] * self.pos_sensitivity
        rot_delta = norm_delta[3:] * self.rot_sensitivity
        
        # Apply coordinate frame correction for directional inversion
        # This is necessary for keyboard to operate in the correct direction
        coord_correction = np.array([
            [-1, 0, 0],   # Invert X (right/left)
            [0, -1, 0],   # Invert Y (forward/backward)  
            [0, 0, 1]     # Keep Z (up/down)
        ])
        
        # Apply coordinate correction to position delta
        pos_delta_corrected = coord_correction @ pos_delta
        
        # For rotation, we may need similar correction depending on the robot orientation
        rot_delta_corrected = coord_correction @ rot_delta
        
        # Combine position and rotation deltas
        scaled_delta = np.concatenate([pos_delta_corrected, rot_delta_corrected])
        
        # Get current end-effector pose for absolute positioning
        try:
            # Try to get current end-effector pose
            site_name = f"gripper0_{arm}_grip_site"
            current_pose = self._get_current_eef_pose(robot, site_name)
            
            if current_pose is not None:
                current_pos = current_pose[:3]
                current_rot = current_pose[3:]
                
                # Calculate absolute target pose
                target_pos = current_pos + pos_delta_corrected
                target_rot = current_rot + rot_delta_corrected
                
                abs_action = np.concatenate([target_pos, target_rot])
            else:
                # Fallback: use delta as absolute (will be interpreted as relative by controller)
                abs_action = scaled_delta.copy()
                
        except Exception as e:
            print(f"Warning: Could not get current pose for {arm} arm: {e}")
            # Fallback to delta-only control
            abs_action = scaled_delta.copy()
        
        # Return action dictionary in the format expected by robosuite Device base class
        return {
            "delta": scaled_delta,
            "abs": abs_action
        }
    
    def _get_current_eef_pose(self, robot, site_name):
        """
        Get current end-effector pose from simulation.
        
        Args:
            robot: Robot instance
            site_name: Name of the end-effector site
            
        Returns:
            np.ndarray: Current pose [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z] or None if error
        """
        try:
            # Get site ID
            site_id = robot.sim.model.site_name2id(site_name)
            
            # Get current position
            current_pos = robot.sim.data.site_xpos[site_id].copy()
            
            # Get current orientation matrix and convert to axis-angle
            current_mat = robot.sim.data.site_xmat[site_id].reshape(3, 3).copy()
            
            # Convert rotation matrix to axis-angle representation
            from scipy.spatial.transform import Rotation
            current_rot = Rotation.from_matrix(current_mat).as_rotvec()
            
            return np.concatenate([current_pos, current_rot])
            
        except Exception as e:
            print(f"Error getting current pose for {site_name}: {e}")
            return None


def create_mink_keyboard_device(env, pos_sensitivity=1.0, rot_sensitivity=1.0):
    """
    Factory function to create a MinkKeyboardDevice.
    
    This is a convenience function that can be used in place of the standard
    robosuite Keyboard device when working with Mink IK controllers.
    
    Args:
        env: Robosuite environment
        pos_sensitivity: Position sensitivity multiplier (default: 1.0)
        rot_sensitivity: Rotation sensitivity multiplier (default: 1.0)
        
    Returns:
        MinkKeyboardDevice: Configured keyboard device for Mink IK control
        
    Example:
        device = create_mink_keyboard_device(env, pos_sensitivity=0.5, rot_sensitivity=0.8)
        env.viewer.add_keypress_callback(device.on_press)
    """
    return MinkKeyboardDevice(
        env=env,
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity
    )