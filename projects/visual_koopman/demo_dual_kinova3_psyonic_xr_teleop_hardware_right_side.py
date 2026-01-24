"""
Streamlined demo script for dual Kinova3 robots with Psyonic Ability hands using XR WebRTC teleoperation.

This streamlined version leverages the helper functions from sew_mimic_to_joint_position.py
to minimize code duplication and improve maintainability.

Usage:
    python demo_dual_kinova3_psyonic_xr_teleop_streamlined.py --environment Lift

Author: Chuizheng Kong
Date created: 2025-08-16
"""

import argparse
import os
import sys
import time
import threading
import queue

import mujoco
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

import robosuite as suite

# Add path for XR WebRTC teleoperation device
from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice

# Import the Arm Class
from projects.psyonic_hand_teleop.ability_hand.dual_kinova3_robot_psyonic_gripper import DualKinova3PsyonicHand

# Import custom environments to register them with robosuite
import projects.experiment_envs

from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.camera_utils import CameraMover
import robosuite.utils.transform_utils as t_utils

# Import the hand controller
# from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller import AbilityHandController
from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller_teensy import AbilityHandControllerTeensy

# Import SEW converter helper functions - this handles most of the heavy lifting!
from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
    initialize_sew_converters_for_robot,
    update_sew_converter_origins_from_robot,
    convert_sew_actions_to_joint_positions,
    create_sew_visualization_wrapper
)

# Import Kortex API for hardware control
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2
import projects.impact_control.tool_box_no_ros as tb
import projects.impact_control.kortex_utilities as ku

from projects.shared_scripts.mujoco_camera_utils import set_viewer_camera, list_available_cameras, R_std_mjcam
import cv2

# Get robosuite path for config
repo_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)


def find_least_numbered_subfolder(root_dir: str):
    """
    Find the smallest integer appearing in the names of immediate subfolders.

    Args:
        root_dir: path to the parent folder

    Returns:
        The smallest integer found, or None if no numbers exist.
    """
    nums = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            try:
                nums.append(int(name))
            except ValueError:
                pass

    return max(nums) + 1 if nums else 0

def stream_webcam(cam_id=0, width=640, height=480):
    """
    Detects a webcam and streams images.

    Args:
        cam_id (int): camera index (0 is default webcam)
        width (int): frame width
        height (int): frame height
    """
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with id {cam_id}")

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Webcam streaming started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam Stream", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_act_imgs(actions, observations, images, save_path):
    images_root = os.path.join(save_path, "images")
    observations_root = os.path.join(save_path, "observations")
    actions_root = os.path.join(save_path, "actions")
    compare_img_root = os.path.join(save_path, "compare_imgs")

    os.makedirs(images_root, exist_ok=True)
    os.makedirs(observations_root, exist_ok=True)
    os.makedirs(actions_root, exist_ok=True)
    os.makedirs(compare_img_root, exist_ok=True)
    
    demo_index_image = find_least_numbered_subfolder(images_root)
    demo_index_obs = find_least_numbered_subfolder(observations_root)
    demo_index_act = find_least_numbered_subfolder(actions_root)
    print(demo_index_image, demo_index_obs, demo_index_act)

    if not (demo_index_image == demo_index_obs == demo_index_act):
        print("Warning: image, obs, and act folders are not synchronized.")
        print(f"Index image: {demo_index_image}")
        print(f"Index obs: {demo_index_obs}")
        print(f"Index act: {demo_index_act}")
        
    demo_index = demo_index_image

    # Create subfolders for this trajectory, deleting existing ones if needed
    traj_img_dir = os.path.join(images_root, f"{demo_index_image}")
    traj_obs_dir = os.path.join(observations_root, f"{demo_index}")
    traj_act_dir = os.path.join(actions_root, f"{demo_index}")

    for d in [traj_img_dir, traj_obs_dir, traj_act_dir]:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)
        os.makedirs(d)

    # Save images (N images)
    for step_idx, img in enumerate(images):
        im = Image.fromarray(img)
        im.save(os.path.join(traj_img_dir, f"{step_idx}.png"))
    print("Number of images:", len(images))
        
    # observation: list of arrays, each [N]
    obs_arm_joints_array = np.stack(observations['obs_arm_joints'], axis=0)   # shape (T, 7)
    obs_hand_joints_array = np.stack(observations['obs_hand_joints'], axis=0)   # shape (T, 6)
    observation_array = np.concatenate(
        [obs_arm_joints_array, obs_hand_joints_array],
        axis=1
    ) 

    np.save(os.path.join(traj_obs_dir, "observation.npy"), observation_array)
    print("observation_array shape:", observation_array.shape)
    
    # observation: list of arrays, each [N + 1]
    target_arm_joints_array = np.stack(actions['target_arm_joints'], axis=0)   # shape (T, 7)
    target_hand_joints_array = np.stack(actions['target_hand_joints'], axis=0)   # shape (T, 6)
    action_array = np.concatenate(
        [target_arm_joints_array, target_hand_joints_array],
        axis=1
    ) 

    print("action_array shape:", action_array.shape)
    np.save(os.path.join(traj_act_dir, "action.npy"), action_array)
    
    # plot the observation and action figures
    T, N = observation_array.shape
    T1, N_act = action_array.shape
    assert N_act == N, f"Expected action dim {N}, got {N_act}"

    time_obs = np.arange(T)      # 0..T-1
    time_act = np.arange(T1)      # 0..T-1
    
    n_cols = 4
    n_rows = math.ceil(N / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 2.5 * n_rows),
        sharex=True
    ) 

    # Make axes always 2D
    axes = np.atleast_2d(axes)
    
    # Assuming left arm first (0-6) then right (7-13)? Or just one arm?
    # The original code concatenate obs_arm_joints and obs_hand_joints
    # We need to know the structure. Based on collection loop later, we will adapt.
    
    # Simple labels for now
    state_names = [f'State {i}' for i in range(N)]
    
    for i in range(N):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        # Plot Observation
        ax.plot(
            time_obs,
            observation_array[:, i],
            label='Observation',
            color='blue',
            linewidth=1.5
        )

        # Plot Action
        ax.plot(
            time_act,
            action_array[:, i],
            label='Action',
            color='red',
            linestyle='--',
            linewidth=1.5
        )

        # Formatting
        ax.set_title(state_names[i] if i < len(state_names) else f"Dim {i}", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Turn off unused subplots
    for j in range(N, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis('off')

    # Common labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Step", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Value", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,
        handlelength=3,
        handletextpad=1.0
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
    fig.savefig(os.path.join(compare_img_root, f"{demo_index}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("stop recording")

# Optionally import the Ostrich neck hardware interface for physical neck actuation
try:
    from ostrich_neck.hardware import OstrichNeckHardwareInterface
    from ostrich_neck.controller import DynamixelConnectionError
except ImportError:
    ostrich_neck_src = os.path.abspath(
        os.path.join(repo_path, "ostrich-neck", "src")
    )
    if os.path.isdir(ostrich_neck_src) and ostrich_neck_src not in sys.path:
        sys.path.insert(0, ostrich_neck_src)
    try:
        from ostrich_neck.hardware import OstrichNeckHardwareInterface
        from ostrich_neck.controller import DynamixelConnectionError
    except ImportError:
        OstrichNeckHardwareInterface = None  # type: ignore[misc]
        class DynamixelConnectionError(RuntimeError):
            """Placeholder exception when the hardware package is unavailable."""
            pass
        print("Warning: ostrich_neck package not found. Neck hardware control disabled.")

dual_kinova3_sew_config_path = os.path.join(
    repo_path, "SEW-Geometric-Teleop", "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_sew_mimic_joint_position.json"
)


def create_environment(args):
    """Create robosuite environment with dual Kinova3 and Psyonic hands."""
    controller_config = load_composite_controller_config(
        controller=dual_kinova3_sew_config_path,
        robot="DualKinova3PsyonicHand",
    )
    
    env = suite.make(
        env_name=args.environment,
        robots="DualKinova3PsyonicHand",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=30,
    )
    env.table_offset = np.array((0.5, 0, 0.6))
    return env


def sync_joint_pos_with_kortex(action_dict, left_base, left_base_cyclic, right_base, right_base_cyclic):
    """
    Sync simulation joint positions with hardware robots using joint velocity control.
    
    Args:
        action_dict: Dictionary containing joint position targets from simulation
        left_base: Left arm BaseClient
        left_base_cyclic: Left arm BaseCyclicClient  
        right_base: Right arm BaseClient
        right_base_cyclic: Right arm BaseCyclicClient
    """
    kp = 5.0  # Position control gain
    kd = 0.5  # Velocity damping gain

    #### Get simulation joint positions for left arm ####
    # Extract left arm joint positions from action_dict
    # sim_left_q = action_dict.get("left", np.zeros(7))

    ## Retrieve the hardware robot states
    if left_base_cyclic is not None:
        left_base_feedback = left_base_cyclic.RefreshFeedback()
        real_left_q, real_left_qd = tb.get_realtime_q_qdot(left_base_feedback)
    else:
        real_left_q, real_left_qd = np.zeros(7), np.zeros(7)

    ## Compute joint velocity control command using PD control
    # position_error_left = sim_left_q - real_left_q
    # position_error_left = (position_error_left + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    # left_qd_cmd = np.degrees(kp * position_error_left - kd * real_left_qd)  # Kinova uses degrees/sec
    # left_joint_speeds = Base_pb2.JointSpeeds()

    #### Get simulation joint positions for right arm ####
    # Extract right arm joint positions from action_dict
    sim_right_q = action_dict.get("right", np.zeros(7))

    ## Retrieve the hardware robot states
    right_base_feedback = right_base_cyclic.RefreshFeedback()
    real_right_q, real_right_qd = tb.get_realtime_q_qdot(right_base_feedback)

    ## Compute joint velocity control command using PD control
    position_error_right = sim_right_q - real_right_q
    position_error_right = (position_error_right + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    right_qd_cmd = np.degrees(kp * position_error_right - kd * real_right_qd)
    right_joint_speeds = Base_pb2.JointSpeeds()

    #### Populate and send joint speed command ####
    for joint_id in range(7):
        # left_js = left_joint_speeds.joint_speeds.add()
        # left_js.joint_identifier = joint_id 
        # left_js.value = left_qd_cmd[joint_id]
        # left_js.duration = 0

        right_js = right_joint_speeds.joint_speeds.add()
        right_js.joint_identifier = joint_id 
        right_js.value = right_qd_cmd[joint_id]
        right_js.duration = 0

    try:
        # left_base.SendJointSpeedsCommand(left_joint_speeds)
        right_base.SendJointSpeedsCommand(right_joint_speeds)
    except Exception as e:
        print(f"Error sending joint speed commands: {e}")
    
    return real_left_q, real_right_q


def kortex_sync_worker(action_queue, stop_event, left_base, left_base_cyclic, right_base, right_base_cyclic, robot_state):
    """
    Background thread worker for syncing with Kortex hardware.
    This runs independently to avoid blocking the main control loop.
    """
    print("Kortex sync worker thread started")
    current_action = None
    
    while not stop_event.is_set():
        try:
            # Get the latest action dict, drop old ones
            try:
                while True:
                    current_action = action_queue.get_nowait()
            except queue.Empty:
                pass
            
            if current_action is not None:
                real_left_q, real_right_q = sync_joint_pos_with_kortex(current_action, left_base, left_base_cyclic, right_base, right_base_cyclic)
                robot_state['left'] = real_left_q
                robot_state['right'] = real_right_q
            else:
                time.sleep(0.001)  # Small sleep to prevent busy-waiting
                
        except Exception as e:
            print(f"Error in Kortex sync worker: {e}")
            import traceback
            traceback.print_exc()
    
    print("Kortex sync worker thread stopped")



def main():
    """Main demo function - streamlined using helper functions."""
    parser = argparse.ArgumentParser(description="Streamlined Dual Kinova3 + Psyonic XR Demo")
    parser.add_argument("--environment", type=str, default="Lift", help="Environment to use")
    parser.add_argument("--max_fr", default=30, type=int, help="Maximum frame rate")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--collision_filtering", action="store_true", default=True, 
                       help="Enable collision filtering (default: enabled)")
    parser.add_argument("--record_data", action="store_true", help="Enable CSV recording of body pose and action data")
    parser.add_argument("--output_dir", type=str, default="./xr_recordings", help="Output directory for CSV recordings")
    parser.add_argument("--ndigits", type=int, default=3, help="Number of decimal places for recorded values")
    parser.add_argument("--left_arm_ip", type=str, default="192.168.2.12", help="IP address of left arm")
    parser.add_argument("--right_arm_ip", type=str, default="192.168.1.10", help="IP address of right arm")
    parser.add_argument("--save_path", type=str, default="./cloth_recordings", help="Output directory for image/action recordings")
    #TODO @kczttm - figure out VisualizationWrapper and CameraMover compatibility in future
    args = parser.parse_args()

    # Handle collision filtering arguments
    enable_collision_filtering = args.collision_filtering

    # Create environment
    print("Creating robosuite environment...")
    env = create_environment(args)
    env = create_sew_visualization_wrapper(env)

    left_arm_args = tb.TCPArguments(ip=args.left_arm_ip)
    right_arm_args = tb.TCPArguments(ip=args.right_arm_ip)
    
    # Camera parameters
    teleop_camera_sn = "337122071053"
    teleop_camera_intric = [385.383, 385.383, 317.368, 243.951]
    image_fps = 30

    cam_id=2
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with id {cam_id}")

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Use context managers for proper connection handling
    # with ku.DeviceConnection.createTcpConnection(left_arm_args) as left_arm_conn, \
    with ku.DeviceConnection.createTcpConnection(right_arm_args) as right_arm_conn:
        
        # Create clients
        # left_base = BaseClient(left_arm_conn)
        left_base = None
        right_base = BaseClient(right_arm_conn)
        # left_base_cyclic = BaseCyclicClient(left_arm_conn)
        left_base_cyclic = None
        right_base_cyclic = BaseCyclicClient(right_arm_conn)
        
        print("Hardware connections established!")
        
        # --- Control Config Check ---
        # left_control_config = ControlConfigClient(left_arm_conn)
        left_control_config = None
        right_control_config = ControlConfigClient(right_arm_conn)
        
        print("\n--- Left Arm Control Config ---")
        try:
            # print(f"Control Mode: {left_control_config.GetControlMode()}")
            # left_soft, _ = tb.get_kinematic_limits(left_control_config)
            # print(f"Soft Speed Limits: {left_soft.joint_speed_limits}")
            print("Left arm not connected.")
        except Exception as e:
            print(f"Error getting left arm config: {e}")
            
        print("\n--- Right Arm Control Config ---")
        try:
            print(f"Control Mode: {right_control_config.GetControlMode()}")
            right_soft, _ = tb.get_kinematic_limits(right_control_config)
            print(f"Soft Speed Limits: {right_soft.joint_speed_limits}")
        except Exception as e:
            print(f"Error getting right arm config: {e}")
        print("-" * 30 + "\n")
        # ----------------------------

        # Set both arm joint limit to max
        # tb.set_joint_speed_soft_limits(left_control_config)
        tb.set_joint_speed_soft_limits(right_control_config)

        # check speed again after setting
        print("\n--- Speed Limits (After Setting) ---")
        try:
            # left_soft, _ = tb.get_kinematic_limits(left_control_config)
            # print(f"Left Soft Speed Limits: {left_soft.joint_speed_limits}")
            right_soft, _ = tb.get_kinematic_limits(right_control_config)
            print(f"Right Soft Speed Limits: {right_soft.joint_speed_limits}")
        except Exception as e:
            print(f"Error getting config: {e}")
        print("-" * 30 + "\n")

        # Home both arms
        print("Homing both arms...")
        # tb.home_both_arms(left_base, right_base, "Home")
        tb.move_to_home_position(right_base, action_name="R_Home")
    
        # Initialize XR device
        print("Initializing XR WebRTC device...")
        xr_device = XRRTCBodyPoseDevice(
            env=env,
            record_data=args.record_data,
            output_dir=args.output_dir,
            ndigits=args.ndigits
        )

        # Initialize neck interface if available
        neck_interface = None
        if OstrichNeckHardwareInterface is None:
            print("Warning: Ostrich neck hardware interface unavailable. Running simulation only.")
        else:
            try:
                neck_interface = OstrichNeckHardwareInterface()
                neck_interface.connect(move_home=True)
                print("Ostrich neck hardware connected and homed.")
            except DynamixelConnectionError as exc:
                print(f"Warning: Failed to initialize Ostrich neck hardware: {exc}")
                neck_interface = None
            except Exception as exc:  # pragma: no cover - defensive for unexpected hardware errors
                print(f"Warning: Unexpected error initializing Ostrich neck hardware: {exc}")
                neck_interface = None

        # Wait for connection
        print("Waiting for VR client connection...")
        while not xr_device.is_connected:
            time.sleep(0.5)
        print("Client connected!")
        
        # Reset environment and get robot
        obs = env.reset()
        robot = env.robots[0]
        
        # Initialize SEW converters using helper function with collision filtering option
        collision_status = "enabled" if enable_collision_filtering else "disabled"
        print(f"Initializing SEW converters with collision avoidance {collision_status}...")
        sew_converters = initialize_sew_converters_for_robot(robot, env.sim, enable_collision_filtering=enable_collision_filtering)
        # Update SEW converter origins using helper function
        update_sew_converter_origins_from_robot(sew_converters, robot)
        
        model = env.sim.model._model
        data = env.sim.data._data
        print("Initializing hand controllers...")
        # left_hand_controller = AbilityHandController(model, data, hand_side='left', debug=False)
        # right_hand_controller = AbilityHandController(model, data, hand_side='right', debug=False)

        # left_hand_controller = AbilityHandControllerTeensy(model, data, hand_side='left', hardware=True, port="/dev/ttyACM0", debug=False)
        # left_hand_controller = None
        right_hand_controller = AbilityHandControllerTeensy(model, data, hand_side='right', hardware=True, port="/dev/ttyACM0", debug=False)
        
        # Start Kortex sync worker thread
        robot_state = {}
        kortex_action_queue = queue.Queue(maxsize=2)
        kortex_stop_event = threading.Event()
        kortex_thread = threading.Thread(
            target=kortex_sync_worker,
            args=(kortex_action_queue, kortex_stop_event, left_base, left_base_cyclic, right_base, right_base_cyclic, robot_state),
            daemon=True
        )
        kortex_thread.start()
        print("Kortex sync worker started in background thread")
        
        print("Starting teleoperation...")
        print("=" * 60)
        print("XR Teleoperation: Move your arms and hands to control the robots!")
        print("=" * 60)
        
        # Data recording buffers
        images = []
        observations = {'obs_arm_joints': [], 'obs_hand_joints': []}
        actions = {'target_arm_joints': [], 'target_hand_joints': []}
        recording_active = False
        recording_finished = False

        try:
            with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
                viewer.opt.geomgroup[0] = 0  # Hide collision meshes
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -15
                viewer.cam.lookat[:] = [0, 0, 1.0]
                
                step_count = 0
                
                while viewer.is_running():
                    start_time = time.time()

                    try:
                        # Get XR input
                        input_ac_dict = xr_device.get_controller_state()

                        # Check if recording is active/inactive
                        # is_recording = input_ac_dict.get("record_data", False) if input_ac_dict else False
                        
                        # Handle recording state transitions
                        # if is_recording and not recording_active:
                        if input_ac_dict is not None and not recording_active and not recording_finished:
                            # Start recording
                            print("Recording STARTED...")
                            recording_active = True
                            images = []
                            observations = {'obs_arm_joints': [], 'obs_hand_joints': []}
                            actions = {'target_arm_joints': [], 'target_hand_joints': []}
                            prev_actions = np.zeros(6)

                            right_arm_q = env.sim.data.qpos[robot.part_controllers['right'].qpos_index]
                            current_arm_joints = right_arm_q # np.concatenate([left_arm_q, right_arm_q])
                            current_hand_joints = prev_actions # np.concatenate([left_hand_q, right_hand_q])
                            
                            target_right_q = current_hand_joints
                            target_arm_joints = current_arm_joints # np.concatenate([target_left_q, target_right_q])
                            actions['target_arm_joints'].append(target_arm_joints)
                            actions['target_hand_joints'].append(target_right_q)

                        # elif not is_recording and recording_active:
                        elif not xr_device.is_connected and recording_active:
                            # Stop recording and save
                            print("Recording STOPPED. Saving data...")

                            recording_active = False
                            recording_finished = True
                            if len(images) > 0:
                                save_act_imgs(actions, observations, images, args.save_path)
                            else:
                                print("No data recorded.")
                            cap.release()
                            cv2.destroyAllWindows()
                        
                        if input_ac_dict is not None:
                            
                            # Convert SEW actions to joint positions using helper function
                            # Get the human finger mcp centroid positions dict
                            human_finger_mcp_centroid = {
                                'left': input_ac_dict.get('left_finger_mcp_centroid', None),
                                'right': input_ac_dict.get('right_finger_mcp_centroid', None)
                            }

                            # get the robot tcp centroid position dict
                            robot_tcp_centroid = {
                                'left': None,
                                'right': right_hand_controller.get_current_finger_mcp_centroid()
                            }
                            # robot_tcp_centroid =  {'left': None, 'right': None}  # Disable for now

                            # Convert SEW positions to joint positions using the arm-specific converter
                            action_dict = convert_sew_actions_to_joint_positions(sew_converters, input_ac_dict, robot, env.sim, 
                                                                                    safety_layer=enable_collision_filtering, debug=args.debug,
                                                                                    human_hands_centroid=human_finger_mcp_centroid,
                                                                                    robot_tcp_centroid=robot_tcp_centroid)
                            
                            neck_input = action_dict.get("head") if isinstance(action_dict, dict) else None
                            if neck_interface is not None and neck_input is not None:
                                try:
                                    neck_interface.move_from_angles(neck_input)
                                except DynamixelConnectionError as neck_exc:
                                    print(f"Warning: Ostrich neck command failed: {neck_exc}")
                                    neck_interface.shutdown()
                                    neck_interface = None
                                except Exception as neck_exc:  # pragma: no cover - hardware communication errors
                                    print(f"Warning: Unexpected Ostrich neck error: {neck_exc}")
                                    neck_interface.shutdown()
                                    neck_interface = None

                            # Override gripper actions with Psyonic hand control
                            for arm in robot.arms:
                                if arm == 'left': continue # Only control right arm/hand

                                hand_controller = right_hand_controller
                                # Use finger data if available
                                if f"{arm}_fingers" in input_ac_dict and input_ac_dict[f"{arm}_fingers"]:
                                    finger_positions = input_ac_dict[f"{arm}_fingers"]
                                    hand_controller.compute_hand_joint_goals_from_fingers(finger_positions)
                                else:
                                    hand_controller.set_joint_goals(np.zeros(10))  # Home position
                                
                                # Manually update simulation state for hand joints (since simulation is bypassed)
                                if env.sim.data is not None:
                                     for i, addr in enumerate(hand_controller.qpos_addrs):
                                         env.sim.data.qpos[addr] = hand_controller.q_goal[i]

                                # Compute control torques for the hand
                                action_dict[f"{arm}_gripper"] = hand_controller.compute_control_torques()

                                # If connected to hardware, update the hand state from the real hand
                                hand_controller.send_joint_positions_hw()
                            
                            # Execute action
                            # env_action = robot.create_action_vector(action_dict)
                            # env.step(env_action)
                            
                            # Update simulation state from hardware feedback
                            sim_q = env.sim.data.qpos.copy()
                            updated = False
                            for arm in robot.arms:
                                if arm == 'left': continue
                                if arm in robot_state:
                                    sim_q[robot.part_controllers[arm].qpos_index] = robot_state[arm]
                                    updated = True
                            
                            if updated:
                                env.sim.data.qpos[:] = sim_q
                                env.sim.forward()

                            # Queue action for hardware sync (non-blocking)
                            try:
                                kortex_action_queue.put_nowait(action_dict)
                            except queue.Full:
                                # Drop oldest and try again
                                try:
                                    _ = kortex_action_queue.get_nowait()
                                    kortex_action_queue.put_nowait(action_dict)
                                except (queue.Empty, queue.Full):
                                    pass  # Skip this update if queue management fails
                            
                            # Record data frame if recording is active
                            if recording_active:
                                # Capture image from viewer (requires offscreen renderer setup, but let's try grabbing from viewer)
                                # Since we launch passive viewer, we might not get images easily via viewer.read_pixels()
                                # robosuite's offscreen renderer is available via env.sim.renderer if configured.
                                # But we're running passive viewer. We can try using env.sim.renderer to render offscreen.
                                # The env creation has has_offscreen_renderer=True now.
                                
                                # Capture offscreen image
                                # We need to set camera first. 
                                # Assuming 'agentview' or 'frontview' exists. 
                                # For visual koopman, usually need specific camera view. Let's assume 'agentview' for now.
                                obs = env._get_observations(force_update=False) # Get obs to ensure cameras are updated? No, expensive.
                                # Just render directly
                                # To get proper image, we need to ensure the camera is looking at right place.
                                # robosuite render:
                                ret, frame = cap.read()
                        
                                if not ret:
                                    print("Failed to grab frame")
                                    break

                                cv2.imshow("Webcam Stream", frame)
                                images.append(frame[:, :, ::-1])  # Convert BGR to RGB

                                # Press 'q' to exit
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                
                                # Record observations (robot state)
                                # Arm joints: left (7) + right (7) = 14
                                # Hand joints: left (6 mapped/10 real?) + right (6/10)
                                # The oscmink script records:
                                # obs_arm_joints: 14 dim (concat left, right?)
                                # obs_hand_joints: ? dim
                                
                                # Get current robot state from simulation (which is updated from hardware)
                                # left_arm_q = env.sim.data.qpos[robot.part_controllers['left'].qpos_index]
                                right_arm_q = env.sim.data.qpos[robot.part_controllers['right'].qpos_index]
                                current_arm_joints = right_arm_q # np.concatenate([left_arm_q, right_arm_q])
                                observations['obs_arm_joints'].append(current_arm_joints)
                                
                                # Get current hand state - we can use q_goal or mapped positions?
                                # Ideally actual state, but we only have q_goal updated in sim for hands.
                                # Psyonic hand controller keeps q_current.
                                # left_hand_q = left_hand_controller.mapped_positions if hasattr(left_hand_controller, 'mapped_positions') and left_hand_controller.mapped_positions else np.zeros(6)
                                right_hand_q = prev_actions                                # mapped_positions is 6-dim. 
                                current_hand_joints = right_hand_q # np.concatenate([left_hand_q, right_hand_q])
                                observations['obs_hand_joints'].append(current_hand_joints)
                                
                                # Record actions (targets)
                                # action_dict has targets.
                                # Arm targets
                                # target_left_q = action_dict.get("left", np.zeros(7))
                                target_right_q = action_dict.get("right", np.zeros(7))
                                target_arm_joints = target_right_q # np.concatenate([target_left_q, target_right_q])
                                actions['target_arm_joints'].append(target_arm_joints)
                                
                                # Hand targets
                                # We used hand_controller.mapped_positions above for obs which comes from q_current?
                                # For action, we should use the goal.
                                # Since we force update simulation qpos to q_goal, and then update mapped_positions from q_current (which is q_goal),
                                # right_hand_controller.mapped_positions represents the target command.
                                
                                target_right_hand = right_hand_controller.mapped_positions if hasattr(right_hand_controller, 'mapped_positions') and right_hand_controller.mapped_positions else np.zeros(6)
                                target_hand_joints = target_right_hand
                                actions['target_hand_joints'].append(target_hand_joints)
                                prev_actions = target_hand_joints.copy()
                        else:
                            # No XR input - hold position
                            # env.step(np.zeros(env.action_dim))
                            if step_count % 300 == 0:
                                print("Waiting for XR data...")
                                
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        try:
                            # left_base.Stop()
                            right_base.Stop()
                        except:
                            pass                    
                    # Sync viewer and maintain framerate
                    viewer.sync()
                    
                    # Update hand controller data references periodically
                    # left_hand_controller.update_data_reference(data)
                    right_hand_controller.update_data_reference(data)
                    
                    # Maintain target framerate
                    elapsed = time.time() - start_time
                    if elapsed < 1 / args.max_fr:
                        time.sleep(1 / args.max_fr - elapsed)
                    
                    step_count += 1
                    
                    if step_count % 300 == 0 and args.debug:
                        print(f"Step {step_count} - Running smoothly")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping hardware...")
        
        # Stop Kortex sync thread
        print("Stopping Kortex sync worker...")
        kortex_stop_event.set()
        kortex_thread.join(timeout=2)
        if kortex_thread.is_alive():
            print("Warning: Kortex sync thread did not stop cleanly")
        
        # Stop arms
        try:
            # left_base.Stop()
            right_base.Stop()
        except:
            print("Error stopping arms")
        
        # Cleanup hand controllers
        print("Closing hand controllers...")
        # left_hand_controller.close()
        right_hand_controller.close()
        
        # Cleanup
        print("Cleaning up...")
        if args.record_data:
            xr_device.cleanup_recording()

        if hasattr(xr_device, 'stop'):
            xr_device.stop()
        env.close()
        if neck_interface is not None:
            neck_interface.shutdown()

        # 
        if recording_active and len(images) > 0:
            print("Recording STOPPED. Saving data...")
            print(len(images))
            if len(images) > 0:
                save_act_imgs(actions, observations, images, args.save_path)
            else:
                print("No data recorded.")
        print("Demo completed.")


if __name__ == "__main__":
    main()