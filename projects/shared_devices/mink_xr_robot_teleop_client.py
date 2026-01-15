from projects.shared_devices.xr_robot_teleop_client import (
    XRRTCBodyPoseDevice,
    _get_body_centric_coordinates,
    _calc_gripper_state,
    Bone
)
import numpy as np
from scipy.spatial.transform import Rotation
from copy import deepcopy

class MinkXRRTCBodyPoseDevice(XRRTCBodyPoseDevice):
    """
    A device to control a robot using body pose data from XR Robot Teleop Client,
    specificaly formatted for Mink IK controller (Absolute Pose Control).
    """

    def __init__(self, env, process_bones_to_action_fn=None, **kwargs):
        # We start with no offset
        self.offsets = {
            "left": {"pos": None, "rot_offset": None},
            "right": {"pos": None, "rot_offset": None}
        }
        self.calibrated = False

        self.body_origin_pos = None

        # Additional scaling factors for Mink IK
        self.mink_pos_scale = 1.5  # increasing workspace reach
        

        # Use our mink processing function by default if none provided
        if process_bones_to_action_fn is None:
            process_bones_to_action_fn = self._mink_process_bones_to_action
            
        super().__init__(env, process_bones_to_action_fn=process_bones_to_action_fn, **kwargs)

    @staticmethod
    def _mink_process_bones_to_action(bones: list[Bone]) -> dict:
        """
        Process bones into actions compatible with Mink IK.
        Uses SEW coordinates:
        - Position: SEW["W"] (Wrist position in body frame)
        - Rotation: SEW["wrist_rot"] (Wrist rotation matrix in body frame) converted to axis-angle
        """
        action_dict = {}
        
        # Get bone positions mostly for gripper calculation
        bone_positions = {b.id: np.array(b.position) for b in bones}
        
        # Helper to get gripper state
        left_gripper_action, right_gripper_action, left_gripper_dist, right_gripper_dist = _calc_gripper_state(bone_positions)
        
        # Get Body Centric Coordinates (SEW)
        # This returns positions relative to the calculated body frame (shoulder center)
        sew_coords = _get_body_centric_coordinates(bones)

        if sew_coords is None or sew_coords["left"] is None or sew_coords["right"] is None:
            return None

        # Build actions for each arm (left/right)
        for arm in ["left", "right"]:
            # 1. Position
            # sew_coords[arm]["W"] gives the wrist position in the body frame
            pos = sew_coords[arm]["W"]
            
            # 2. Rotation
            # sew_coords[arm]["wrist_rot"] gives the wrist rotation matrix in the body frame
            rot_mat = sew_coords[arm]["wrist_rot"]
            
            if pos is None or rot_mat is None:
                continue

            # Fix shape issue: Reshape to (3,3) if it's flattened
            if rot_mat.size == 9:
                rot_mat = rot_mat.reshape(3, 3)

            # Convert rotation matrix to axis-angle as expected by Mink IK [rx, ry, rz]
            rot_vec = Rotation.from_matrix(rot_mat).as_rotvec()
            
            # 3. Store RAW absolute pose (User Frame) [x, y, z, rx, ry, rz]
            # We will correct this with offsets later
            abs_action = np.concatenate([pos, rot_vec])
            
            # Populate the dictionary with keys expected by the controller script
            # e.g. 'left_abs', 'right_abs'
            action_dict[f"{arm}_abs"] = abs_action
            
            # Also store pure position and rotation matrix for easier math in input2action
            action_dict[f"{arm}_pos_raw"] = pos
            action_dict[f"{arm}_rot_mat_raw"] = rot_mat
            
            # Gripper actions
            action_dict[f"{arm}_gripper"] = left_gripper_action if arm == "left" else right_gripper_action
            action_dict[f"{arm}_gripper_val"] = left_gripper_dist if arm == "left" else right_gripper_dist

        return action_dict
    
    def rotation_matrix_standard_wrist_to_kinova_EEF(self):

        R_W_T = Rotation.from_euler('ZYX', [np.pi/2, 0, np.pi/2]).as_matrix() # back of the hand align with +y of EE
        
        return R_W_T

    def _get_current_eef_pose(self, robot, site_name):
        """
        Get current end-effector pose from simulation.
        
        Args:
            robot: Robot instance
            site_name: Name of the end-effector site
            
        Returns:
            tuple: (pos, rot_matrix)
        """
        try:
            # Get site ID
            site_id = robot.sim.model.site_name2id(site_name)
            
            # Get current position
            current_pos = robot.sim.data.site_xpos[site_id].copy()
            
            # Get current orientation matrix
            current_mat = robot.sim.data.site_xmat[site_id].reshape(3, 3).copy()
            
            return current_pos, current_mat
            
        except Exception as e:
            print(f"Error getting current pose for {site_name}: {e}")
            return None, None

    def input2action(self, mirror_actions=False):
        """
        Returns the action dictionary with offset correction applied.
        """
        raw_input_dict = self.get_controller_state()

        if raw_input_dict is None:
            return None

        final_action_dict = deepcopy(raw_input_dict)
        
        # Apply offset logic
        # We assume self.env is available and has robots[0] as the active robot
        if self.env is not None and len(self.env.robots) > 0:
            robot = self.env.robots[0]
            
            for arm in ["left", "right"]:
                # Check if we have raw data
                if f"{arm}_pos_raw" not in raw_input_dict:
                    continue
                
                # Get body-centric origin position
                body_origin_pos = robot.base_pos + np.array([0, 0, 0.232548])
                R_W_T = self.rotation_matrix_standard_wrist_to_kinova_EEF()
                
                raw_pos = raw_input_dict[f"{arm}_pos_raw"] + body_origin_pos
                raw_rot_mat = raw_input_dict[f"{arm}_rot_mat_raw"] @ R_W_T
                
                # Reshape raw_rot_mat if needed
                if raw_rot_mat.size == 9:
                    raw_rot_mat = raw_rot_mat.reshape(3,3)

                # Check calibration
                if self.offsets[arm]["pos"] is None:
                    # Perform calibration for this arm
                     # Try to get current end-effector pose
                    site_name = f"gripper0_{arm}_grip_site"
                    curr_pos, curr_rot_mat = self._get_current_eef_pose(robot, site_name)
                    
                    if curr_pos is not None:
                        # Calculate Position Offset: Robot - User
                        self.offsets[arm]["pos"] = curr_pos - raw_pos
                        self.offsets[arm]["user_init_pos"] = raw_pos
                        
                        # Calculate Rotation Offset: T_robot = T_offset * T_user => T_offset = T_robot * T_user^-1
                        # Rotation Matrix: R_offset = R_robot @ R_user.T
                        self.offsets[arm]["rot_offset"] = curr_rot_mat @ raw_rot_mat.T
                        
                        print(f"Calibrated {arm} arm. Offset: {self.offsets[arm]['pos']}")
                
                # Apply Offset if calibrated
                if self.offsets[arm]["pos"] is not None:
                    # Position: Target = Raw + Offset
                    scaled_raw_pos = (raw_pos - self.offsets[arm]["user_init_pos"]) * self.mink_pos_scale + self.offsets[arm]["user_init_pos"]
                    target_pos = scaled_raw_pos + self.offsets[arm]["pos"]
                    
                    # Rotation: Target = Offset * Raw
                    target_rot_mat = self.offsets[arm]["rot_offset"] @ raw_rot_mat
                    target_rot_vec = Rotation.from_matrix(target_rot_mat).as_rotvec()
                    
                    # Store corrected absolute pose
                    corrected_abs = np.concatenate([target_pos, target_rot_vec])
                    final_action_dict[f"{arm}_abs"] = corrected_abs
        
        return final_action_dict
