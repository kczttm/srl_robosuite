"""
Unified MediaPipe-based teleoperation device.
Extracts human SEW (Shoulder, Elbow, Wrist) coordinates and wrist poses from camera stream.
Works standalone without robosuite framework, and can be extended for robosuite integration.

Coordinates in robot body-centric frame:
    +X - front
    +Y - left
    +Z - up

For both hands/wrists, when in zero config (arm by body side), follows the same coordinates as above.

Author: Chuizheng Kong
Date created: 2025-09-17
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import traceback
import argparse
from typing import Optional, Dict, Tuple

from scipy.spatial.transform import Rotation


class MediaPipeTeleopDevice:
    """
    Unified device for teleoperation using human pose estimation via MediaPipe.
    Extracts SEW (Shoulder, Elbow, Wrist) coordinates and wrist poses from human pose
    for controlling robot arms. Works both standalone and with robosuite integration.
    """
    
    def __init__(self, camera_id=0, debug=False, mirror_actions=False, env=None):
        """
        Initialize MediaPipe teleoperation device.
        
        Args:
            camera_id (int): Camera device ID
            debug (bool): Enable debug visualization
            mirror_actions (bool): Mirror actions (right robot arm follows left human arm)
            env: Optional robosuite environment for integration (None for standalone mode)
        """
        
        # Store environment reference for robosuite compatibility
        self.env = env
        self.standalone_mode = env is None
        
        # Initialize robosuite-specific attributes if env is provided
        if not self.standalone_mode:
            # Initialize parent class attributes that would normally come from Device
            self._initialize_robosuite_attributes()
        
        # Setup MediaPipe pose estimation
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
            
        # Test camera by reading a frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            raise RuntimeError(f"Failed to read test frame from camera {camera_id}")
        
        print(f"Camera {camera_id} initialized successfully")
        print(f"Frame shape: {test_frame.shape}")
        print(f"Frame dtype: {test_frame.dtype}")
        print(f"Frame min/max values: {test_frame.min()}/{test_frame.max()}")
        
        # Threading and state management
        self.controller_state_lock = threading.Lock()
        self.controller_state = None
        self._reset_state = 0
        self._quit_state = False
        self.stop_event = threading.Event()
        
        # Configuration
        self.mirror_actions = mirror_actions
        self.debug = debug
        self.engaged = False
        
        # Arm configuration for mirroring
        if self.mirror_actions:
            self._arm2side = {
                "right": "left",  # Right robot arm follows left human arm
                "left": "right",  # Left robot arm follows right human arm
            }
        else:
            self._arm2side = {
                "left": "left",   # Left robot arm follows left human arm
                "right": "right", # Right robot arm follows right human arm
            }
        
        # SEW pose tracking - store in body-centric coordinates
        self.human_sew_poses = {
            "left": {"S": None, "E": None, "W": None},
            "right": {"S": None, "E": None, "W": None}
        }
        
        # Wrist pose tracking - store 3x3 rotation matrices
        self.human_wrist_poses = {
            "left": None,
            "right": None
        }
        
        # Hand pose tracking for wrist orientation computation
        self.human_hand_poses = {
            "left": {"landmarks": None, "confidence": 0.0},
            "right": {"landmarks": None, "confidence": 0.0}
        }

        # Standard camera rotation transformation
        self.R_std_cam = Rotation.from_euler('ZYX', [np.pi/2, 0, -np.pi/2]).as_matrix()
        
        # Initialize gripper control for robosuite compatibility
        if not self.standalone_mode:
            self.grasp_states = [[0.0] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
        
        self._display_controls()
        self._reset_internal_state()
        
        # Start pose estimation thread
        self.pose_thread = threading.Thread(target=self._pose_estimation_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()
    
    def _initialize_robosuite_attributes(self):
        """Initialize attributes needed for robosuite compatibility."""
        # Check robot models and see if there are multiple arms
        self.robot_interface = self.env
        self.env_sim = self.env.env.sim
        self.robot_models = []
        self.bimanual = False

        for robot in self.robot_interface.robots:
            self.robot_models.append(robot.robot_model.name)
            if robot.robot_model.arm_type == 'bimanual':
                self.bimanual = True
        print("Robot models:", self.robot_models)
        
        # Initialize device attributes (mimicking robosuite Device class)
        self.active_robot = 0
        self.all_robot_arms = []
        self.num_robots = len(self.robot_interface.robots) if hasattr(self.robot_interface, 'robots') else 1
        
        # Build robot arms list
        for robot in self.robot_interface.robots:
            self.all_robot_arms.append(robot.arms)
    
    @staticmethod
    def _display_controls():
        """Display control instructions."""
        print("=" * 60)
        print("MediaPipe Teleoperation Device Controls:")
        print("- Raise both arms to shoulder height: Start pose tracking")
        print("- Lower arms: Stop pose tracking")
        print("- 'q' key in camera window: Quit")
        print("- 'r' key in camera window: Reset")
        print("=" * 60)
    
    def _reset_internal_state(self):
        """Reset internal state variables."""
        self.human_sew_poses = {
            "left": {"S": None, "E": None, "W": None},
            "right": {"S": None, "E": None, "W": None},
            "R_world_body": None
        }
        self.human_wrist_poses = {
            "left": None,
            "right": None
        }
        self.human_hand_poses = {
            "left": {"landmarks": None, "confidence": 0.0},
            "right": {"landmarks": None, "confidence": 0.0}
        }
        self.engaged = False
        
        # Reset gripper states for robosuite
        if not self.standalone_mode:
            self.grasp_states = [[0.0] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
    
    def start_control(self):
        """Start the control loop."""
        self._reset_internal_state()
        self._reset_state = 0
        self.engaged = True
    
    def _pose_estimation_loop(self):
        """Main loop for pose estimation running in separate thread."""
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                if frame is None:
                    print("Warning: Received None frame from camera")
                    time.sleep(0.1)
                    continue
                    
                # Debug frame info occasionally
                if self.debug and hasattr(self, '_debug_frame_count'):
                    self._debug_frame_count += 1
                    if self._debug_frame_count % 300 == 0:  # Every ~5 seconds at 60 FPS
                        print(f"Frame {self._debug_frame_count}: shape={frame.shape}, dtype={frame.dtype}")
                elif self.debug:
                    self._debug_frame_count = 1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                
                # Process landmarks
                self._process_pose_landmarks(pose_results, frame)
                self._process_hand_landmarks(hand_results, pose_results, frame)
                
                # Update controller state
                with self.controller_state_lock:
                    self.controller_state = {
                        'sew_poses': self.human_sew_poses.copy(),
                        'wrist_poses': self.human_wrist_poses.copy(),
                        'hand_poses': self.human_hand_poses.copy(),
                        'engaged': self.engaged
                    }
                
                # Display frame with basic info (always shown)
                self._add_basic_info_to_frame(frame)
                
                # Display additional debug info only in debug mode
                if self.debug:
                    self._add_detailed_debug_info_to_frame(frame)
                
                # Check if frame is still valid before displaying
                if frame is not None and frame.size > 0:
                    cv2.imshow('MediaPipe Pose Estimation', frame)
                else:
                    print("Warning: Frame became invalid during processing")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._quit_state = True
                    break
                elif key == ord('r'):
                    self._reset_state = 1
                
                # Check if window was closed
                if self._is_window_closed():
                    self._quit_state = True
                    break
                    
            except Exception as e:
                print(f"Error in pose estimation loop: {e}")
                traceback.print_exc()
                # Continue even if there's an error to keep the loop running
                time.sleep(0.1)
    
    def _process_pose_landmarks(self, pose_results, frame):
        """Process MediaPipe pose landmarks using body-centric coordinate system."""
        try:
            if pose_results.pose_landmarks and pose_results.pose_world_landmarks:
                # Get body-centric coordinates
                body_centric_coords = self._get_body_centric_coordinates(pose_results)
                
                if body_centric_coords:
                    # Extract SEW poses in body-centric frame
                    for side in ["left", "right"]:
                        if side in body_centric_coords:
                            self.human_sew_poses[side]["S"] = body_centric_coords[side]["S"]
                            self.human_sew_poses[side]["E"] = body_centric_coords[side]["E"]
                            self.human_sew_poses[side]["W"] = body_centric_coords[side]["W"]
                    
                    # Store body frame transformation
                    if 'body_frame' in body_centric_coords:
                        self.human_sew_poses['R_world_body'] = body_centric_coords['body_frame']['R_world_body']
                    
                    # Check engagement based on pose visibility
                    self._check_engagement_from_pose(pose_results)
                    
                    # Draw pose landmarks on frame for debugging
                    if pose_results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            pose_results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                else:
                    self._reset_pose_tracking()
                    
            else:
                # No pose landmarks - reset poses
                self._reset_pose_tracking()
        except Exception as e:
            print(f"Error processing pose landmarks: {e}")
            traceback.print_exc()
    
    def _reset_pose_tracking(self):
        """Reset pose tracking data."""
        self.human_sew_poses['left'] = {"S": None, "E": None, "W": None}
        self.human_sew_poses['right'] = {"S": None, "E": None, "W": None}
        self.human_sew_poses['R_world_body'] = None
        self._check_engagement()
    
    def _check_engagement_from_pose(self, pose_results):
        """Check engagement based on pose landmark visibility."""
        if not pose_results.pose_landmarks:
            self.engaged = False
            return
            
        landmarks = pose_results.pose_landmarks.landmark
        
        # Check visibility of key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check visibility (threshold can be adjusted)
        visibility_threshold = 0.75
        left_sew_valid = all(lm.visibility > visibility_threshold for lm in [left_shoulder, left_elbow, left_wrist])
        right_sew_valid = all(lm.visibility > visibility_threshold for lm in [right_shoulder, right_elbow, right_wrist])
        
        # Require both arms to be visible and tracked for engagement
        self.engaged = left_sew_valid and right_sew_valid
    
    def _get_body_centric_coordinates(self, pose_results):
        """
        Convert MediaPipe world landmarks to a body-centric coordinate system.
        
        Args:
            pose_results: MediaPipe pose detection results
            
        Returns:
            Dictionary containing SEW coordinates in body-centric frame
        """
        if not pose_results.pose_world_landmarks:
            return None
            
        landmarks_3d = pose_results.pose_world_landmarks.landmark
        
        # Get key body landmarks
        left_shoulder = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate body center (midpoint between shoulders and hips)
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        ])
        
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        ])
        
        # Use shoulder center as origin for upper body tracking
        body_origin = shoulder_center
        
        # Create body-centric coordinate frame
        # Y-axis: right to left (shoulder line)
        y_axis = np.array([
            left_shoulder.x - right_shoulder.x,
            left_shoulder.y - right_shoulder.y,
            left_shoulder.z - right_shoulder.z
        ])
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)  # normalize
        
        # Z-axis: up direction (shoulder to hip, inverted)
        torso_vector = hip_center - shoulder_center
        z_axis = -torso_vector / (np.linalg.norm(torso_vector) + 1e-8)  # up is positive Z

        # X-axis: forward direction (cross product)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Create transformation matrix from world to body-centric frame
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        def transform_to_body_frame(landmark):
            world_pos = np.array([landmark.x, landmark.y, landmark.z])
            relative_pos = world_pos - body_origin
            return rotation_matrix.T @ relative_pos
        
        # Extract SEW coordinates in body-centric frame
        sew_coordinates = {}
        
        for side in ['LEFT', 'RIGHT']:
            side_key = side.lower()
            
            shoulder_landmark = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_SHOULDER')]
            elbow_landmark = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_ELBOW')]
            wrist_landmark = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_WRIST')]
            
            # Transform to body-centric coordinates
            S = transform_to_body_frame(shoulder_landmark)
            E = transform_to_body_frame(elbow_landmark)
            W = transform_to_body_frame(wrist_landmark)
            
            sew_coordinates[side_key] = {
                'S': S,
                'E': E,
                'W': W
            }
        
        # Add body frame info for debugging
        sew_coordinates['body_frame'] = {
            'origin': body_origin,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'R_world_body': self.R_std_cam @ rotation_matrix
        }
        
        return sew_coordinates
    
    def _process_hand_landmarks(self, hand_results, pose_results, frame):
        """Process MediaPipe hand landmarks to compute wrist orientation."""
        # Clear stale hand data when no hands detected
        if not hand_results or not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
            self.human_wrist_poses['left'] = None
            self.human_wrist_poses['right'] = None
            self.human_hand_poses['left'] = {'landmarks': None, 'confidence': 0.0}
            self.human_hand_poses['right'] = {'landmarks': None, 'confidence': 0.0}
            return
            
        try:
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                # Get body-centric coordinates for reference
                body_centric_coords = self._get_body_centric_coordinates(pose_results)
                if not body_centric_coords:
                    return
                
                hand_centric_coords = self._get_hand_centric_coordinates(hand_results, body_centric_coords)
                
                # Compute wrist rotations from hand landmarks
                if hand_centric_coords:
                    for hand_side, hand_data in hand_centric_coords.items():
                        if hand_data['landmarks'] is not None and hand_data['confidence'] > 0.5:
                            # Compute wrist rotation matrix
                            wrist_rotation = self._compute_wrist_rotation_from_hand(hand_side, hand_data['landmarks'])
                            self.human_wrist_poses[hand_side] = wrist_rotation
                            
                            # Store hand pose data
                            self.human_hand_poses[hand_side] = hand_data
                        else:
                            self.human_wrist_poses[hand_side] = None
                            self.human_hand_poses[hand_side] = {'landmarks': None, 'confidence': 0.0}
                
                # Draw hand landmarks on frame
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        
        except Exception as e:
            print(f"Error processing hand landmarks: {e}")
            traceback.print_exc()
    
    def _get_hand_centric_coordinates(self, hand_results, body_centric_coords):
        """
        Align hand world landmarks to pose world landmarks and convert to body-centric coordinates.
        
        Args:
            hand_results: MediaPipe hand detection results
            body_centric_coords: Body-centric coordinate system from _get_body_centric_coordinates()
            
        Returns:
            Dictionary containing aligned hand landmarks in body-centric frame
        """
        if (not hand_results or not hand_results.multi_hand_world_landmarks or 
            not body_centric_coords):
            return None
            
        hand_frames = {}
        
        # Get body frame transformation matrix for converting to body-centric coordinates
        body_frame = body_centric_coords['body_frame']
        body_origin = body_frame['origin']
        body_rotation_matrix = np.column_stack([
            body_frame['x_axis'], 
            body_frame['y_axis'], 
            body_frame['z_axis']
        ])
        
        def world_to_body_frame(world_pos):
            """Convert world position to body-centric coordinates."""
            # Translate to body origin
            translated = world_pos - body_origin
            # Rotate to body frame
            body_pos = body_rotation_matrix.T @ translated
            return body_pos
        
        for hand_idx, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(
            zip(hand_results.multi_hand_landmarks, hand_results.multi_hand_world_landmarks, hand_results.multi_handedness)):
            
            # Flip MediaPipe hand labels to match body pose perspective
            mediapipe_label = handedness.classification[0].label.lower()
            actual_hand_label = 'right' if mediapipe_label == 'left' else 'left'
            
            # Get the corresponding pose wrist for alignment
            if actual_hand_label == 'left':
                pose_wrist_body = body_centric_coords['left']['W']
            else:
                pose_wrist_body = body_centric_coords['right']['W']
            
            # Get hand wrist position
            hand_wrist = hand_world_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Calculate translation offset to align hand wrist with pose wrist
            hand_wrist_world = np.array([hand_wrist.x, hand_wrist.y, hand_wrist.z])
            
            # Align all hand landmarks to pose world frame and convert to body frame
            aligned_hand_landmarks = {}
            landmark_names = [
                'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
                'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
            ]
            
            for landmark_name in landmark_names:
                landmark_id = getattr(self.mp_hands.HandLandmark, landmark_name)
                hand_landmark = hand_world_landmarks.landmark[landmark_id]
                
                # Get hand landmark in world coordinates
                hand_landmark_world = np.array([
                    hand_landmark.x,
                    hand_landmark.y, 
                    hand_landmark.z
                ])
                
                # Align hand landmark to pose world frame: 
                # Step 1: Translate hand landmark relative to hand wrist
                hand_landmark_relative = hand_landmark_world - hand_wrist_world
                # Step 2: Add pose wrist position (converts to pose world frame)
                pose_wrist_world = body_origin + body_rotation_matrix @ pose_wrist_body
                aligned_world_pos = pose_wrist_world + hand_landmark_relative
                # Step 3: Convert to body-centric coordinates
                aligned_hand_landmarks[landmark_name.lower()] = world_to_body_frame(aligned_world_pos)
            
            # Store aligned hand information
            hand_frames[actual_hand_label] = {
                'landmarks': aligned_hand_landmarks,
                'wrist_pos': pose_wrist_body,  # Wrist position in body frame
                'confidence': handedness.classification[0].score
            }
        
        return hand_frames
    
    def _compute_wrist_rotation_from_hand(self, hand_side, hand_landmarks):
        """
        Compute wrist rotation matrix from hand landmarks.
        
        Args:
            hand_side (str): 'left' or 'right'
            hand_landmarks (dict): Hand landmarks in body-centric coordinates
            
        Returns:
            np.array: 3x3 rotation matrix representing wrist orientation in body frame, or None if computation fails
        """
        try:
            # Get key landmarks for defining wrist coordinate frame
            wrist = hand_landmarks.get('wrist')
            index_mcp = hand_landmarks.get('index_finger_mcp')
            pinky_mcp = hand_landmarks.get('pinky_mcp')
            ring_mcp = hand_landmarks.get('ring_finger_mcp')
            middle_mcp = hand_landmarks.get('middle_finger_mcp')

            if wrist is None or index_mcp is None or pinky_mcp is None:
                return None
                
            # Define hand coordinate frame with wrist as origin
            # X-axis: from wrist towards palm center (average of finger MCPs)
            if middle_mcp is not None and ring_mcp is not None:
                palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4
            elif middle_mcp is not None:
                palm_center = (index_mcp + middle_mcp + pinky_mcp) / 3
            else:
                palm_center = (index_mcp + pinky_mcp) / 2
                
            x_axis = (palm_center - wrist)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            
            # Collect all available MCP points relative to wrist
            mcp_points = []
            if index_mcp is not None:
                mcp_points.append(index_mcp - wrist)
            if middle_mcp is not None:
                mcp_points.append(middle_mcp - wrist)
            if ring_mcp is not None:
                mcp_points.append(ring_mcp - wrist)
            if pinky_mcp is not None:
                mcp_points.append(pinky_mcp - wrist)
            
            if len(mcp_points) < 2:
                return None
            
            # Use least squares to find Y-axis that's perpendicular to the palm plane
            # Stack MCP vectors into matrix A
            A = np.array(mcp_points)  # shape: (n_points, 3)
            
            # Find the normal to the plane containing MCP points using SVD
            # The normal is the last column of V (smallest singular value)
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            palm_normal = Vt[-1, :]  # Last row of V^T = last column of V
            palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
            
            # Determine handedness-consistent Y-axis direction
            # Use the palm normal, but ensure correct orientation based on hand side
            wrist_to_index = index_mcp - wrist
            wrist_to_pinky = pinky_mcp - wrist
            
            # Cross product gives a reference direction
            reference_y = np.cross(wrist_to_index, wrist_to_pinky)
            reference_y = reference_y / (np.linalg.norm(reference_y) + 1e-8)
            
            # Choose palm normal direction that aligns with handedness
            if np.dot(palm_normal, reference_y) < 0:
                palm_normal = -palm_normal
            
            y_axis = palm_normal
            
             # Z-axis: Z = X × Y (ensuring right-handed coordinate system)
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # Validate that it's a proper rotation matrix
            if np.abs(np.linalg.det(rotation_matrix) - 1.0) > 0.1:
                if self.debug:
                    print(f"Warning: Invalid rotation matrix for {hand_side} hand (det={np.linalg.det(rotation_matrix)})")
                return None
                
            return rotation_matrix
            
        except Exception as e:
            if self.debug:
                print(f"Error computing wrist rotation for {hand_side} hand: {e}")
            return None
    
    def _compute_finger_positions_wrist_frame(self, hand_side, hand_data, sew_poses):
        """
        Compute finger joint positions in wrist frame coordinates.
        
        Args:
            hand_side (str): 'left' or 'right'
            hand_data (dict): Hand data containing landmarks and confidence
            sew_poses (dict): SEW poses with 'S', 'E', 'W' positions in body frame
            
        Returns:
            dict: Dictionary with finger joint positions in wrist frame, or None if invalid
        """
        try:
            # Check if hand data is valid
            if (hand_data["landmarks"] is None or 
                hand_data["confidence"] < 0.5 or
                sew_poses["W"] is None):
                return None
            
            landmarks = hand_data["landmarks"]
            wrist_pos_body = sew_poses["W"]  # Wrist position in body frame
            
            # Get wrist rotation matrix to transform from body frame to wrist frame
            wrist_rotation_matrix = self._compute_wrist_rotation_from_hand(hand_side, landmarks)
            
            if wrist_rotation_matrix is None:
                if self.debug:
                    print(f"Warning: Could not compute wrist rotation for {hand_side} hand")
                return None
            
            # Inverse of wrist rotation matrix to transform from body frame to wrist frame
            body_to_wrist_rotation = wrist_rotation_matrix.T
            
            # Define the finger joints we want to track (matching Psyonic hand structure)
            finger_joints = {
                'thumb': ['thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip'],
                'index': ['index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip'],
                'middle': ['middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip'],
                'ring': ['ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip'],
                'pinky': ['pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
            }
            
            finger_positions_wrist_frame = {}
            
            for finger_name, joint_names in finger_joints.items():
                finger_positions_wrist_frame[finger_name] = {}
                
                for joint_name in joint_names:
                    if joint_name in landmarks:
                        # Get position in body frame
                        joint_pos_body = landmarks[joint_name]
                        
                        # Transform to wrist frame (relative to wrist position)
                        joint_relative_body = joint_pos_body - wrist_pos_body
                        joint_pos_wrist = body_to_wrist_rotation @ joint_relative_body
                        
                        finger_positions_wrist_frame[finger_name][joint_name] = joint_pos_wrist
            
            return finger_positions_wrist_frame
            
        except Exception as e:
            if self.debug:
                print(f"Error computing finger positions for {hand_side} hand: {e}")
            return None
    
    def _compute_finger_tip_centroid(self, finger_positions_wrist_frame):
        """
        Compute the mean position of the five finger tips in wrist frame.
        
        Args:
            finger_positions_wrist_frame (dict): Dict with keys 'thumb', 'index', 'middle', 'ring', 'pinky', 
                                               each containing joint positions in wrist frame.
        Returns:
            np.ndarray: Mean position (3,) of the five finger tips, or None if any tip is missing.
        """
        tip_keys = {
            'thumb': 'thumb_tip',
            'index': 'index_finger_tip',
            'middle': 'middle_finger_tip',
            'ring': 'ring_finger_tip',
            'pinky': 'pinky_tip'
        }
        
        tips = []
        for finger, tip_key in tip_keys.items():
            if finger in finger_positions_wrist_frame and tip_key in finger_positions_wrist_frame[finger]:
                tips.append(finger_positions_wrist_frame[finger][tip_key])
            else:
                return None  # Missing finger tip
        
        tips = np.array(tips)
        return np.mean(tips, axis=0)
    
    def _compute_grasp_from_hand(self, hand_side, hand_data):
        """
        Compute gripper control value based on hand gesture (thumb-index finger distance).
        
        Args:
            hand_side (str): 'left' or 'right'
            hand_data (dict): Hand data containing landmarks and confidence
            
        Returns:
            float: Gripper value (0.0 = open, 1.0 = closed)
        """
        try:
            # Check if hand data is valid
            if (hand_data["landmarks"] is None or 
                hand_data["confidence"] < 0.5):
                return 0.0
            
            landmarks = hand_data["landmarks"]
            
            # Get thumb tip and index finger tip positions
            thumb_tip = landmarks.get('thumb_tip')
            index_tip = landmarks.get('index_finger_tip')
            
            if thumb_tip is None or index_tip is None:
                return 0.0
            
            # Calculate distance between thumb and index finger tips
            distance = np.linalg.norm(thumb_tip - index_tip)
            
            # Grasp threshold - close gripper when distance < 0.04 meters (4 cm)
            grasp_threshold = 0.04
            
            if distance < grasp_threshold:
                grasp_value = 1.0  # Close gripper
            else:
                grasp_value = 0.0  # Open gripper
            
            return grasp_value
            
        except Exception as e:
            if self.debug:
                print(f"Error computing grasp for {hand_side} hand: {e}")
            return 0.0  # Default to open gripper on error
    
    def _check_engagement(self):
        """Check if user is engaged based on arm positions and visibility."""
        # Check if both arms have valid SEW data
        left_sew = self.human_sew_poses["left"]
        right_sew = self.human_sew_poses["right"]
            
        left_engaged = (left_sew["S"] is not None and 
                       left_sew["E"] is not None and 
                       left_sew["W"] is not None)
        right_engaged = (right_sew["S"] is not None and 
                        right_sew["E"] is not None and 
                        right_sew["W"] is not None)

        # Require both arms to be visible and tracked for engagement
        self.engaged = left_engaged and right_engaged
    
    def _add_basic_info_to_frame(self, frame):
        """Add basic information to the display frame (always shown)."""
        y_offset = 30
        
        # Engagement status
        status_color = (0, 255, 0) if self.engaged else (0, 0, 255)
        status_text = "ENGAGED" if self.engaged else "DISENGAGED"
        cv2.putText(frame, f"Control: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        y_offset += 30
        
    
    def _add_detailed_debug_info_to_frame(self, frame):
        """Add detailed debugging information to the display frame (debug mode only)."""
        # Add detailed SEW info to frame
        self._add_sew_info_to_frame(frame)
        
        # Add hand/wrist info to frame  
        self._add_hand_info_to_frame(frame)

    
    def _add_debug_info_to_frame(self, frame):
        """Add debugging information to the display frame (legacy method for compatibility)."""
        self._add_basic_info_to_frame(frame)
        self._add_detailed_debug_info_to_frame(frame)
    
    def _add_sew_info_to_frame(self, frame):
        """Add SEW coordinate information to the display frame."""
        y_offset = 90
        cv2.putText(frame, "Body-Centric SEW Coordinates (meters):", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        y_offset += 20
        
        for side in ["left", "right"]:
            sew = self.human_sew_poses[side]
            if all(pose is not None for pose in sew.values()):
                cv2.putText(frame, f"{side.upper()} ARM:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 15
                
                for joint, pos in sew.items():
                    if pos is not None:
                        text = f"  {joint}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.35, (0, 255, 0), 1, cv2.LINE_AA)
                        y_offset += 12
                y_offset += 5
    
    def _add_hand_info_to_frame(self, frame):
        """Add hand landmark information to frame display."""
        y_offset = 250  # Start below pose info
        
        # Add header in magenta to match robosuite version
        cv2.putText(frame, "Hand-Centric Coordinates:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        y_offset += 25
        
        hand_poses = self.human_hand_poses
        if hand_poses and any(hand_poses[hand]['landmarks'] is not None for hand in ['left', 'right']):
            for hand_label in ['left', 'right']:
                hand_data = hand_poses.get(hand_label, {})
                landmarks = hand_data.get('landmarks')
                confidence = hand_data.get('confidence', 0.0)
                
                if landmarks is not None and confidence > 0:
                    # Hand label and confidence in cyan to match robosuite version
                    hand_text = f"{hand_label.upper()} HAND (conf: {confidence:.2f}):"
                    cv2.putText(frame, hand_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 20
                    
                    # Key landmarks header in light blue
                    cv2.putText(frame, "  Hand-frame landmarks:", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
                    y_offset += 15
                    
                    # Key landmarks in wrist-frame coordinates using magenta
                    if isinstance(landmarks, dict):
                        # Show thumb, index, middle tips in wrist frame
                        sew = self.human_sew_poses[hand_label]
                        finger_positions = self._compute_finger_positions_wrist_frame(hand_label, hand_data, sew)
                        if finger_positions:
                            tip_keys = [
                                ('thumb', 'thumb_tip'),
                                ('index', 'index_finger_tip'),
                                ('middle', 'middle_finger_tip')
                            ]
                            for finger, tip_key in tip_keys:
                                tip_pos = finger_positions.get(finger, {}).get(tip_key)
                                if tip_pos is not None:
                                    display_name = f"{finger.title()} Tip (wrist frame)"
                                    coord_text = f"    {display_name}: ({tip_pos[0]:+.3f}, {tip_pos[1]:+.3f}, {tip_pos[2]:+.3f})"
                                    cv2.putText(frame, coord_text, (10, y_offset), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
                                    y_offset += 12
                        
                        # Add grasp state information (still use body-centric for gesture)
                        thumb_tip = landmarks.get('thumb_tip')
                        index_tip = landmarks.get('index_finger_tip')
                        if thumb_tip is not None and index_tip is not None:
                            distance = np.linalg.norm(thumb_tip - index_tip)
                            grasp_threshold = 0.01
                            grasp_state = "GRASPING" if distance < grasp_threshold else "OPEN"
                            grasp_color = (0, 255, 0) if distance < grasp_threshold else (0, 165, 255)  # Green for grasp, orange for open
                            grasp_text = f"    Thumb-Index Dist: {distance:.4f}m - {grasp_state}"
                            cv2.putText(frame, grasp_text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, grasp_color, 1, cv2.LINE_AA)
                            y_offset += 12
                    y_offset += 10  # Extra space between hands
                else:
                    # Show hand not detected in gray
                    hand_text = f"{hand_label.upper()} HAND: Not detected"
                    cv2.putText(frame, hand_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
                    y_offset += 25
        else:
            cv2.putText(frame, "No hands detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
    
    def get_controller_state(self):
        """
        Get current controller state with comprehensive output supporting both standalone and robosuite formats.
        
        Returns:
            Dict: Controller state containing:
                - Standalone format: sew_poses, wrist_poses, hand_poses, engaged
                - Robosuite format: left_sew, right_sew, left_grasp, etc.
        """
        # Use timeout to prevent hanging
        if self.controller_state_lock.acquire(timeout=0.1):
            try:
                if not self.engaged:
                    # Return None for robosuite compatibility when not engaged
                    if not self.standalone_mode:
                        return None
                    else:
                        # For standalone mode, return structure with engaged=False
                        return {
                            'sew_poses': self.human_sew_poses.copy(),
                            'wrist_poses': self.human_wrist_poses.copy(),
                            'hand_poses': self.human_hand_poses.copy(),
                            'engaged': self.engaged
                        }
                
                # Create comprehensive controller state dict with both formats
                controller_state = {
                    # Standalone format (for non-robosuite demos like G1, etc.)
                    'sew_poses': self.human_sew_poses.copy(),
                    'wrist_poses': self.human_wrist_poses.copy(), 
                    'hand_poses': self.human_hand_poses.copy(),
                    'engaged': self.engaged
                }
                
                # Add robosuite format (for robosuite demos)
                for arm_side in ["left", "right"]:
                    sew = self.human_sew_poses[arm_side]
                    hand_data = self.human_hand_poses[arm_side]
                    
                    # Check if all SEW poses are available
                    if all(pose is not None for pose in sew.values()):
                        # Start with basic SEW coordinates
                        sew_action = np.concatenate([sew["S"], sew["E"], sew["W"]])
                        
                        # Always extend to 18 elements for consistent controller interface
                        # Check if hand pose data is available to compute wrist rotation matrix
                        if (hand_data["landmarks"] is not None and 
                            hand_data["confidence"] > 0.5):  # Minimum confidence threshold
                            
                            # Compute wrist rotation matrix from hand landmarks
                            wrist_rotation_matrix = self._compute_wrist_rotation_from_hand(arm_side, hand_data["landmarks"])
                            
                            if wrist_rotation_matrix is not None:
                                # Use computed rotation matrix (flatten row-wise)
                                rotation_flat = wrist_rotation_matrix.flatten()
                            else:
                                # Hand detection failed - use identity matrix
                                rotation_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                        else:
                            # No hand pose or low confidence - use identity matrix  
                            rotation_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                        
                        # Always create 18-element action: SEW (9) + rotation matrix (9)
                        sew_action = np.concatenate([sew_action, rotation_flat])
                        
                        controller_state[f"{arm_side}_sew"] = sew_action
                        controller_state[f"{arm_side}_valid"] = True
                    else:
                        # Send invalid/empty action to hold current pose - use 18 elements with NaN
                        controller_state[f"{arm_side}_sew"] = np.full(18, np.nan)
                        controller_state[f"{arm_side}_valid"] = False
                    
                    # Hand gesture-based gripper control
                    grasp_value = self._compute_grasp_from_hand(arm_side, hand_data)
                    controller_state[f"{arm_side}_grasp"] = grasp_value
                    
                    # Add finger joint positions in wrist frame
                    finger_positions = self._compute_finger_positions_wrist_frame(arm_side, hand_data, sew)
                    controller_state[f"{arm_side}_fingers"] = finger_positions

                    # Add finger tip centroid
                    if finger_positions is not None:
                        finger_tip_centroid = self._compute_finger_tip_centroid(finger_positions)
                        controller_state[f"{arm_side}_finger_tip_centroid"] = finger_tip_centroid
                        controller_state[f"{arm_side}_finger_mcp_centroid"] = finger_positions['index']['index_finger_mcp']
                    else:
                        controller_state[f"{arm_side}_finger_tip_centroid"] = None
                        controller_state[f"{arm_side}_finger_mcp_centroid"] = None

                    # No reset triggered
                    controller_state[f"{arm_side}_reset"] = False
                
                return controller_state
            finally:
                self.controller_state_lock.release()
        else:
            # Timeout occurred, return None
            if self.debug:
                print("Warning: Controller state lock timeout")
            return None
                
    def get_sew_and_wrist_poses(self):
        """Get current SEW poses and wrist orientations."""
        return {
            'sew_poses': self.human_sew_poses.copy(),
            'wrist_poses': self.human_wrist_poses.copy()
        }
    
    def should_quit(self):
        """Check if quit was requested."""
        return self._quit_state
    
    def should_reset(self):
        """Check if reset was requested."""
        if self._reset_state == 1:
            self._reset_state = 0
            return True
        return False
    
    def _is_window_closed(self):
        """Check if OpenCV window was closed."""
        try:
            # Try to get window property - will throw exception if window is closed
            prop = cv2.getWindowProperty('MediaPipe Pose Estimation', cv2.WND_PROP_VISIBLE)
            return prop < 1
        except cv2.error:
            return True
        except Exception as e:
            if self.debug:
                print(f"Window check error: {e}")
            return True
    
    # Robosuite compatibility methods
    def input2action(self, mirror_actions=False):
        """
        Converts pose input into valid action sequence for env.step().
        Only available when integrated with robosuite environment.
        
        Args:
            mirror_actions (bool): Whether to mirror actions for different viewpoint
            
        Returns:
            Optional[Dict]: Dictionary of actions for env.step() or None if reset
        """
        if self.standalone_mode:
            raise RuntimeError("input2action() is only available in robosuite integration mode")
            
        robot = self.env.robots[self.active_robot]
        state = self.get_controller_state()
        
        ac_dict = {}
        
        # Process each robot arm
        for arm in robot.arms:
            human_side = self._arm2side[arm]
            
            if state is not None:
                # We have a valid state (engaged)
                sew_valid = state[f"{human_side}_valid"]
                sew_action = state[f"{human_side}_sew"]
                grasp = state[f"{human_side}_grasp"]
                reset = state[f"{human_side}_reset"]
                fingers = state[f"{human_side}_fingers"]
                finger_tip_centroid = state[f"{human_side}_finger_tip_centroid"]
                finger_mcp_centroid = state[f"{human_side}_finger_mcp_centroid"]

                # If reset is triggered, return None
                if reset:
                    return None
            else:
                # Not engaged - use NaN to signal "hold current pose" with 18 elements
                sew_valid = False
                sew_action = np.full(18, np.nan)
                grasp = 0.0
                reset = False
                fingers = None  # No finger positions available
                finger_tip_centroid = None
                finger_mcp_centroid = None

            # Create action dict entries
            ac_dict[f"{arm}_sew"] = sew_action
            ac_dict[f"{arm}_gripper"] = np.array([grasp * 1.6 - 0.8])
            ac_dict[f"{arm}_fingers"] = fingers
            ac_dict[f"{arm}_finger_tip_centroid"] = finger_tip_centroid
            ac_dict[f"{arm}_finger_mcp_centroid"] = finger_mcp_centroid

            # For compatibility with existing action structure
            ac_dict[f"{arm}_abs"] = sew_action  # SEW positions (and rotation) as absolute coordinates
            # Delta should be 18 elements to match SEW action size
            ac_dict[f"{arm}_delta"] = np.zeros(18)

        return ac_dict
    
    def stop(self):
        """Stop the device and cleanup resources."""
        print("Stopping MediaPipe device...")
        self.stop_event.set()
        if hasattr(self, 'pose_thread') and self.pose_thread.is_alive():
            self.pose_thread.join(timeout=2.0)
        
        if self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("MediaPipe device stopped.")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except:
            pass


# Standalone mode execution
if __name__ == "__main__":
    """
    Standalone demonstration of unified MediaPipe teleoperation device.
    This allows testing the pose estimation and hand tracking functionality
    without requiring a full robosuite environment.
    """
    
    def print_pose_data(device):
        """Print current pose and hand data in a formatted way."""
        state = device.get_controller_state()
        if state and state['engaged']:
            print("\n" + "="*50)
            print("ENGAGED - Current Pose Data:")
            print("="*50)
            
            # Print SEW poses
            sew_poses = state['sew_poses']
            for side in ['left', 'right']:
                print(f"\n{side.upper()} ARM:")
                sew = sew_poses[side]
                for joint in ['S', 'E', 'W']:
                    if sew[joint] is not None:
                        pos = sew[joint]
                        print(f"  {joint}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
                    else:
                        print(f"  {joint}: None")
            
            # Print hand/wrist data
            wrist_poses = state['wrist_poses']
            hand_poses = state['hand_poses']
            
            print(f"\nHAND DATA:")
            for side in ['left', 'right']:
                wrist_rot = wrist_poses[side]
                hand_conf = hand_poses[side]['confidence']
                
                if wrist_rot is not None:
                    # Show rotation matrix determinant and trace as health indicators
                    det = np.linalg.det(wrist_rot)
                    trace = np.trace(wrist_rot)
                    print(f"  {side.upper()}: conf={hand_conf:.2f}, det={det:.3f}, trace={trace:.3f}")
                else:
                    print(f"  {side.upper()}: conf={hand_conf:.2f}, wrist_rot=None")
        else:
            print("DISENGAGED - Raise both arms to shoulder height to start tracking")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified MediaPipe Teleoperation Device - Standalone Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')
    parser.add_argument('--mirror', action='store_true', help='Mirror actions')
    parser.add_argument('--print-interval', type=float, default=2.0, help='Print data interval in seconds (default: 2.0)')
    parser.add_argument('--no-print', action='store_true', help='Disable periodic data printing')
    
    args = parser.parse_args()
    
    print("Unified MediaPipe Teleoperation Device - Standalone Mode")
    print("=" * 60)
    print("Controls:")
    print("  - Raise both arms to shoulder height to start tracking")
    print("  - Lower arms to stop tracking")
    print("  - Press 'q' in camera window to quit")
    print("  - Press 'r' in camera window to reset")
    print("  - Close camera window to quit")
    print("")
    
    try:
        # Initialize the device in standalone mode (env=None)
        device = MediaPipeTeleopDevice(
            camera_id=args.camera,
            debug=args.debug,
            mirror_actions=args.mirror,
            env=None  # Standalone mode
        )
        
        print(f"Camera {args.camera} opened successfully!")
        print("Starting pose estimation...")
        
        # Start the device
        device.start_control()
        
        last_print_time = time.time()
        
        # Main loop
        while True:
            if device.should_quit():
                print("Quit requested")
                break
                
            if device.should_reset():
                print("Reset requested")
                device.start_control()
                
            # Print pose data periodically
            current_time = time.time()
            if not args.no_print and (current_time - last_print_time) >= args.print_interval:
                print_pose_data(device)
                last_print_time = current_time
            
            # Small delay to prevent busy waiting
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
    finally:
        # Clean up
        try:
            device.stop()
        except:
            pass
        print("Cleanup complete.")