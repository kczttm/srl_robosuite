"""
Joint position controller for Psyonic Ability hands.
Provides direct position commands for dexterous hand control.

Author: Chuizheng Kong
Date created: 2025-08-05
"""

import numpy as np
import mujoco
from typing import Optional, Dict, Tuple
from projects.psyonic_hand_teleop.geometric_kinematics_psyonic import get_default_hand_xml_path, get_frame_transforms_from_xml
from projects.psyonic_hand_teleop.geometric_kinematics_psyonic import compute_finger_joint_angles_using_palm_box
from projects.psyonic_hand_teleop.geometric_kinematics_psyonic import compute_finger_tip_centroid, create_robot_palm_box
# from geometric_kinematics_psyonic import compute_finger_joint_angles_from_mediapipe


class AbilityHandController:
    """Position controller for Psyonic Ability hands."""
    
    def __init__(self, mujoco_model, mujoco_data, hand_side='left', 
                 kp=100, kd=0.1, debug=False):
        """
        Initialize Ability Hand controller.
        
        Args:
            mujoco_model: MuJoCo model
            mujoco_data: MuJoCo data
            hand_side: 'left' or 'right'
            kp: Proportional gain for joint control
            kd: Derivative gain for joint control (auto-computed if None)
            debug: Enable debug output
        """
        self.model = mujoco_model
        self.data = mujoco_data
        self.hand_side = hand_side
        self.debug = debug

        # Legacy gain parameters retained for API compatibility (unused)
        self.kp = kp
        self.kd = 2 * np.sqrt(kp) if kd is None else kd

        
        # Get joint indices for this hand
        self._setup_joint_indices()
        
        # Read joint limits from the model
        self._setup_joint_limits()
        
        # Goal joint angles (initialize to safe home position)
        self.q_goal = np.zeros(self.num_joints)
        
        # Initialize with current joint positions
        self._update_current_positions()
        self.q_goal = self.q_current.copy()

        print(f"Ability {hand_side.capitalize()} Hand controller initialized")
        print(f"Found {self.num_joints} joints")
        if self.debug:
            print(f"Joint names: {self.joint_names}")

        left_hand_xml_path = get_default_hand_xml_path(hand_side='left')
        right_hand_xml_path = get_default_hand_xml_path(hand_side='right')
        self.hand_frame_transforms = {
            'left': get_frame_transforms_from_xml(left_hand_xml_path),
            'right': get_frame_transforms_from_xml(right_hand_xml_path)
        }
        self.left_psyonic_palmbox = create_robot_palm_box(self.hand_frame_transforms['left'])
        self.right_psyonic_palmbox = create_robot_palm_box(self.hand_frame_transforms['right'])
        self.R_palmbox_wrist_dict = {
            'left': self.left_psyonic_palmbox['wrist_to_palmbox_R'],
            'right': self.right_psyonic_palmbox['wrist_to_palmbox_R']
        }

        self.filter_alpha = 0.1  # Low-pass filter smoothing factor
        self.filter_state = {}

    def update_data_reference(self, mujoco_data):
        """Update the MuJoCo data reference to ensure synchronization."""
        self.data = mujoco_data

    def _setup_joint_indices(self):
        """Setup joint indices and names for the hand."""
        # Define joint names based on hand side
        if self.hand_side == 'left':
            # Left hand joint names (based on ability_hands.py structure)
            self.joint_names = [
                "gripper0_left_thumb_q1", "gripper0_left_thumb_q2",
                "gripper0_left_index_q1", "gripper0_left_index_q2", 
                "gripper0_left_middle_q1", "gripper0_left_middle_q2",
                "gripper0_left_ring_q1", "gripper0_left_ring_q2",
                "gripper0_left_pinky_q1", "gripper0_left_pinky_q2"
            ]
        else:
            # Right hand joint names
            self.joint_names = [
                "gripper0_right_thumb_q1", "gripper0_right_thumb_q2",
                "gripper0_right_index_q1", "gripper0_right_index_q2",
                "gripper0_right_middle_q1", "gripper0_right_middle_q2",
                "gripper0_right_ring_q1", "gripper0_right_ring_q2",
                "gripper0_right_pinky_q1", "gripper0_right_pinky_q2"
            ]
        
        self.num_joints = len(self.joint_names)
        
        # Build joint name to ID mapping
        joint_name2id = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_name2id[joint_name] = i
        
        # Build actuator name to ID mapping
        actuator_name2id = {}
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                actuator_name2id[actuator_name] = i
        
        # Debug: Print all available joint and actuator names to understand the naming convention
        if self.debug:
            print(f"\nAll joints in MuJoCo model ({len(joint_name2id)} total):")
            for joint_name, joint_id in sorted(joint_name2id.items()):
                print(f"  {joint_id:2d}: {joint_name}")
            
            print(f"\nAll actuators in MuJoCo model ({len(actuator_name2id)} total):")
            for actuator_name, actuator_id in sorted(actuator_name2id.items()):
                print(f"  {actuator_id:2d}: {actuator_name}")
            
            print(f"\nLooking for {self.hand_side} hand joints with expected names:")
            for expected_name in self.joint_names:
                if expected_name in joint_name2id:
                    print(f"  ✓ Found joint: {expected_name}")
                else:
                    print(f"  ✗ Missing joint: {expected_name}")
                    
            print(f"\nLooking for {self.hand_side} hand actuators with expected names:")
            for expected_name in self.joint_names:
                if expected_name in actuator_name2id:
                    print(f"  ✓ Found actuator: {expected_name}")
                else:
                    print(f"  ✗ Missing actuator: {expected_name}")
        
        # Get qpos and qvel addresses and actuator IDs
        self.qpos_addrs = []
        self.qvel_addrs = []
        self.joint_ids = []
        self.actuator_ids = []
        
        for joint_name in self.joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                self.joint_ids.append(joint_id)
                
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                
                self.qpos_addrs.append(qpos_addr)
                self.qvel_addrs.append(qvel_addr)
                
                if self.debug:
                    print(f"Joint {joint_name}: ID={joint_id}, qpos_addr={qpos_addr}, qvel_addr={qvel_addr}")
                
                # Find corresponding actuator by name
                if joint_name in actuator_name2id:
                    actuator_id = actuator_name2id[joint_name]
                    self.actuator_ids.append(actuator_id)
                    if self.debug:
                        print(f"Mapped joint {joint_name} -> actuator ID {actuator_id}")
                else:
                    if self.debug:
                        print(f"Warning: Could not find actuator for joint {joint_name}")
                    self.actuator_ids.append(-1)  # Invalid actuator ID
            else:
                if self.debug:
                    print(f"Warning: Could not find hand joint {joint_name}")
        
        print(f"Found {len(self.qpos_addrs)} {self.hand_side} hand joints")
        if self.debug:
            print(f"qpos addresses: {self.qpos_addrs}")
            print(f"qvel addresses: {self.qvel_addrs}")
    
    def _setup_joint_limits(self):
        """Setup joint limits by reading from the MuJoCo model's joint ranges."""
        self.joint_limits_lower = np.zeros(self.num_joints)
        self.joint_limits_upper = np.zeros(self.num_joints)
        
        for i, joint_id in enumerate(self.joint_ids):
            if joint_id < len(self.model.jnt_range):
                # Get joint range from model (lower, upper)
                joint_range = self.model.jnt_range[joint_id]
                self.joint_limits_lower[i] = joint_range[0]
                self.joint_limits_upper[i] = joint_range[1]
                
                if self.debug:
                    joint_name = self.joint_names[i]
                    print(f"Joint {joint_name}: range=[{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
            else:
                # Fallback to conservative limits if no range is specified
                self.joint_limits_lower[i] = -1.0
                self.joint_limits_upper[i] = 1.0
                if self.debug:
                    print(f"Warning: No range found for joint {self.joint_names[i]}, using defaults")
        
        if self.debug:
            print(f"{self.hand_side.capitalize()} hand joint limits loaded from model")
    
    def _update_current_positions(self):
        """Update current joint positions from MuJoCo data."""
        # Get current joint positions and velocities
        self.q_current = np.array([self.data.qpos[addr] for addr in self.qpos_addrs])
        self.qd_current = np.array([self.data.qvel[addr] for addr in self.qvel_addrs])
        
    
    def set_joint_goals(self, joint_goals):
        """
        Set goal joint positions for the hand.
        
        Args:
            joint_goals: Array of target joint positions (radians)
        """
        if len(joint_goals) != self.num_joints:
            print(f"Warning: Expected {self.num_joints} joint goals, got {len(joint_goals)}")
            return
        
        # Clip joint goals to the limits read from the XML model
        self.q_goal = np.clip(joint_goals, self.joint_limits_lower, self.joint_limits_upper)
        
        if self.debug and not np.array_equal(joint_goals, self.q_goal):
            print(f"Joint goals clipped for {self.hand_side} hand safety")
            for i, (orig, clipped, lower, upper) in enumerate(zip(joint_goals, self.q_goal, 
                                                                 self.joint_limits_lower, 
                                                                 self.joint_limits_upper)):
                if abs(orig - clipped) > 1e-6:
                    print(f"  Joint {i} ({self.joint_names[i]}): {orig:.3f} -> {clipped:.3f} "
                          f"(limits: [{lower:.3f}, {upper:.3f}])")
    
    def compute_control_torques(self):
        """
        Return desired joint positions for hand actuators.

        Returns:
            np.array: Joint position targets clipped to joint limits
        """
        self._update_current_positions()
        return self.q_goal.copy()
    
    def compute_control_velocity(self):
        """
        Compute joint velocities using proportional control.
        
        Returns:
            np.array: Joint velocities for the hand
        """
        self._update_current_positions()
        
        # Compute position error
        q_error = self.q_goal - self.q_current
        q_error = (q_error + np.pi) % (2*np.pi) - np.pi  # Wrap to [-pi, pi]
        
        # Proportional control for velocity: vel = kp * error
        kp_vel = 2.0  # Velocity gain
        desired_velocity = kp_vel * q_error
        
        # Clamp velocity to reasonable limits
        max_vel = 3.0  # rad/s
        desired_velocity = np.clip(desired_velocity, -max_vel, max_vel)
        
        if self.debug and np.random.random() < 0.01:  # Print occasionally
            q_err_str = f"[{q_error[0]:.3f},{q_error[1]:.3f},{q_error[2]:.3f}...]"
            vel_str = f"[{desired_velocity[0]:.2f},{desired_velocity[1]:.2f},{desired_velocity[2]:.2f}...]"
            print(f"{self.hand_side.capitalize()} hand - q_err={q_err_str}, vel={vel_str}")
        
        return desired_velocity
    
    def apply_torques(self, targets):
        """Apply position targets to MuJoCo actuators."""
        for i, actuator_id in enumerate(self.actuator_ids):
            if i < len(targets) and actuator_id >= 0:
                self.data.ctrl[actuator_id] = targets[i]
                if self.debug and np.random.random() < 0.01:
                    joint_name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
                    print(f"Applied position {targets[i]:.3f} to actuator {actuator_id} for {joint_name}")
            elif actuator_id < 0 and self.debug:
                joint_name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
                print(f"Warning: Invalid actuator ID for {joint_name}, skipping position application")
    
    def update_control(self, joint_goals=None):
        """
        Complete control update: set goals, compute torques, and apply them.
        
        Args:
            joint_goals: Optional array of target joint positions
        """
        if joint_goals is not None:
            self.set_joint_goals(joint_goals)
        
        # Compute actuator targets and apply directly for position control
        targets = self.compute_control_torques()
        self.apply_torques(targets)

    def compute_hand_joint_goals_from_fingers(self, finger_positions: Dict[str, Tuple[float, float]]) -> np.array:
        """
        Compute hand joint goals from finger positions using geometric IK.

        Args:
            finger_positions (dict): Dictionary of finger positions in wrist frame

        Returns:
            np.array: Array of 10 joint goals for the hand
        """
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # Validate finger_positions data before passing to geometric IK
        if finger_positions is None:
            if self.debug:
                print(f"Warning: finger_positions is None for {self.hand_side} hand")
            finger_positions = {}
        elif not isinstance(finger_positions, dict):
            if self.debug:
                print(f"Warning: finger_positions is not a dict for {self.hand_side} hand: {type(finger_positions)}")
            finger_positions = {}

        # Compute joint goals using the geometric IK function
        try:
            # finger_angles_dict = compute_finger_joint_angles_from_mediapipe(
            #     finger_positions, hand_side=self.hand_side, filter_alpha=self.filter_alpha, filter_state=self.filter_state
            # )
            finger_angles_dict = compute_finger_joint_angles_using_palm_box(
                finger_positions_wrist_frame=finger_positions,
                transforms=self.hand_frame_transforms[self.hand_side],
                hand_side=self.hand_side,
                filter_alpha=self.filter_alpha,
                filter_state=self.filter_state
            )

        except Exception as e:
            if self.debug:
                print(f"Error in geometric IK for {self.hand_side} hand: {e}")
            finger_angles_dict = {}  # Use empty dict as fallback

        joint_goals = np.zeros(self.num_joints)
        for i, finger_name in enumerate(finger_order):
            if finger_name in finger_angles_dict.keys():
                q1, q2 = finger_angles_dict[finger_name]
                joint_goals[i * 2] = q1
                joint_goals[i * 2 + 1] = q2
            else:
                # If finger data is missing, set to zero
                if self.debug:
                    print(f"Warning: No data for {finger_name} finger, setting to zero")
                joint_goals[i * 2] = 0.0
                joint_goals[i * 2 + 1] = 0.0

        self.set_joint_goals(joint_goals)

    def reset_to_home_position(self):
        """Reset hand to home position (all joints at zero)."""
        self.q_goal = np.zeros(self.num_joints)
        print(f"{self.hand_side.capitalize()} hand reset to home position")

    
    def get_current_finger_tip_centroid(self):
        """
        Compute the centroid of the five finger tips in wrist frame using current joint angles.
        
        Returns:
            np.ndarray: 3D position of finger tip centroid in wrist frame
        """
        # Get current joint positions
        self._update_current_positions()
        
        # Convert joint angles to finger dictionary format
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_joint_dict = {}
        finger_joint_dict['R_palmbox_wrist'] = self.R_palmbox_wrist_dict[self.hand_side]
        
        for i, finger_name in enumerate(finger_order):
            # Each finger has 2 joints (q1, q2)
            q1 = self.q_current[i * 2]
            q2 = self.q_current[i * 2 + 1]
            finger_joint_dict[finger_name] = (q1, q2)
        
        # Get transforms for this hand side
        transforms = self.hand_frame_transforms[self.hand_side]
        
        # Compute and return the centroid
        centroid_pos = compute_finger_tip_centroid(transforms, finger_joint_dict)
        
        return centroid_pos
    

    def get_current_finger_mcp_centroid(self):
        """
        Compute the centroid of the five finger MCP joints in wrist frame using palmbox.
        
        Returns:
            np.ndarray: 3D position of finger MCP centroid in wrist frame
        """
        if self.hand_side == 'left':
            index_mcp = self.left_psyonic_palmbox['origin']
        else:
            index_mcp = self.right_psyonic_palmbox['origin']

        return index_mcp
    

    def get_current_joint_angles(self):
        """
        Get current joint angles for the hand.
        
        Returns:
            np.array: Current joint positions
        """
        self._update_current_positions()
        return self.q_current.copy()
    
    def get_goal_joint_angles(self):
        """
        Get goal joint angles for the hand.
        
        Returns:
            np.array: Goal joint positions
        """
        return self.q_goal.copy()
    
    def get_joint_limits(self):
        """
        Get joint limits for the hand.
        
        Returns:
            tuple: (lower_limits, upper_limits) as numpy arrays
        """
        return self.joint_limits_lower.copy(), self.joint_limits_upper.copy()