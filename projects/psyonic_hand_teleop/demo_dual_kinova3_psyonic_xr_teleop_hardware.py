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
import time
import threading
import queue

import mujoco
import numpy as np

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
from kortex_api.autogen.messages import Base_pb2
import projects.impact_control.tool_box_no_ros as tb
import projects.impact_control.kortex_utilities as ku


from projects.shared_scripts.mujoco_camera_utils import set_viewer_camera, list_available_cameras, R_std_mjcam

# Get robosuite path for config
repo_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
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
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
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
    kp = 2.0  # Position control gain
    kd = 0.5  # Velocity damping gain

    #### Get simulation joint positions for left arm ####
    # Extract left arm joint positions from action_dict
    sim_left_q = action_dict.get("left", np.zeros(7))

    ## Retrieve the hardware robot states
    left_base_feedback = left_base_cyclic.RefreshFeedback()
    real_left_q, real_left_qd = tb.get_realtime_q_qdot(left_base_feedback)

    ## Compute joint velocity control command using PD control
    position_error_left = sim_left_q - real_left_q
    position_error_left = (position_error_left + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    left_qd_cmd = np.degrees(kp * position_error_left - kd * real_left_qd)  # Kinova uses degrees/sec
    left_joint_speeds = Base_pb2.JointSpeeds()

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
        left_js = left_joint_speeds.joint_speeds.add()
        left_js.joint_identifier = joint_id 
        left_js.value = left_qd_cmd[joint_id]
        left_js.duration = 0

        right_js = right_joint_speeds.joint_speeds.add()
        right_js.joint_identifier = joint_id 
        right_js.value = right_qd_cmd[joint_id]
        right_js.duration = 0

    try:
        left_base.SendJointSpeedsCommand(left_joint_speeds)
        right_base.SendJointSpeedsCommand(right_joint_speeds)
    except Exception as e:
        print(f"Error sending joint speed commands: {e}")


def kortex_sync_worker(action_queue, stop_event, left_base, left_base_cyclic, right_base, right_base_cyclic):
    """
    Background thread worker for syncing with Kortex hardware.
    This runs independently to avoid blocking the main control loop.
    """
    print("Kortex sync worker thread started")
    while not stop_event.is_set():
        try:
            # Get the latest action dict, drop old ones
            action_dict = None
            try:
                while True:
                    action_dict = action_queue.get_nowait()
            except queue.Empty:
                pass
            
            if action_dict is not None:
                sync_joint_pos_with_kortex(action_dict, left_base, left_base_cyclic, right_base, right_base_cyclic)
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
    parser.add_argument("--fpv", action="store_true", default=False, help="Enable first-person view camera control")
    parser.add_argument("--record_data", action="store_true", help="Enable CSV recording of body pose and action data")
    parser.add_argument("--output_dir", type=str, default="./xr_recordings", help="Output directory for CSV recordings")
    parser.add_argument("--ndigits", type=int, default=3, help="Number of decimal places for recorded values")
    parser.add_argument("--left_arm_ip", type=str, default="192.168.0.10", help="IP address of left arm")
    parser.add_argument("--right_arm_ip", type=str, default="192.168.1.10", help="IP address of right arm")
    #TODO @kczttm - figure out VisualizationWrapper and CameraMover compatibility in future
    args = parser.parse_args()
    
    # Handle collision filtering arguments
    enable_collision_filtering = args.collision_filtering

    # Create environment
    print("Creating robosuite environment...")
    env = create_environment(args)
    
    # Choose visualization wrapper based on FPV setting
    if args.fpv:
        print("FPV enabled: Using basic VisualizationWrapper (CameraMover compatible)")
        env = VisualizationWrapper(env)
    else:
        print("FPV disabled: Using SEW visualization wrapper with collision indicators")
        env = create_sew_visualization_wrapper(env)

    left_arm_args = tb.TCPArguments(ip=args.left_arm_ip)
    right_arm_args = tb.TCPArguments(ip=args.right_arm_ip)
    
    # Use context managers for proper connection handling
    with ku.DeviceConnection.createTcpConnection(left_arm_args) as left_arm_conn, \
            ku.DeviceConnection.createTcpConnection(right_arm_args) as right_arm_conn:
        
        # Create clients
        left_base = BaseClient(left_arm_conn)
        right_base = BaseClient(right_arm_conn)
        left_base_cyclic = BaseCyclicClient(left_arm_conn)
        right_base_cyclic = BaseCyclicClient(right_arm_conn)
        
        print("Hardware connections established!")

        # Home both arms
        print("Homing both arms...")
        tb.home_both_arms(left_base, right_base, "Home")
    
        # Initialize XR device
        print("Initializing XR WebRTC device...")
        xr_device = XRRTCBodyPoseDevice(
            env=env,
            record_data=args.record_data,
            output_dir=args.output_dir,
            ndigits=args.ndigits
        )

        # Wait for connection
        print("Waiting for VR client connection...")
        while not xr_device.is_connected:
            time.sleep(0.5)
        print("Client connected!")
        
        # Reset environment and get robot
        obs = env.reset()
        camera_mover = None
        if args.fpv:
            camera_mover = CameraMover(env, camera="agentview")  # NOTE: CameraMover monkey-patches
            # underlying model XML to mutate the specified camera by adding a mocap body to be its
            # immediate parent, and subsequently moving the mocap body to control the camera pose.
            # This must be called *after* any environment reset, otherwise the mocap body disappears.
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

        left_hand_controller = AbilityHandControllerTeensy(model, data, hand_side='left', hardware=True, port="/dev/ttyACM0", debug=False)
        right_hand_controller = AbilityHandControllerTeensy(model, data, hand_side='right', hardware=True, port="/dev/ttyACM1", debug=False)
        
        # Start Kortex sync worker thread
        kortex_action_queue = queue.Queue(maxsize=2)
        kortex_stop_event = threading.Event()
        kortex_thread = threading.Thread(
            target=kortex_sync_worker,
            args=(kortex_action_queue, kortex_stop_event, left_base, left_base_cyclic, right_base, right_base_cyclic),
            daemon=True
        )
        kortex_thread.start()
        print("Kortex sync worker started in background thread")
        
        print("Starting teleoperation...")
        print("=" * 60)
        print("XR Teleoperation: Move your arms and hands to control the robots!")
        print("=" * 60)
        try:
            with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
                viewer.opt.geomgroup[0] = 0  # Hide collision meshes
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -15
                viewer.cam.lookat[:] = [0, 0, 1.0]
                
                step_count = 0
                
                # print(list_available_cameras(model)) # DEBUG
                if args.fpv:
                    set_viewer_camera(viewer, model, "agentview")
                
                while viewer.is_running():
                    start_time = time.time()
                    
                    try:
                        # Get XR input
                        input_ac_dict = xr_device.get_controller_state()
                        
                        if input_ac_dict is not None:
                            
                            # Convert SEW actions to joint positions using helper function
                            # Get the human finger mcp centroid positions dict
                            human_finger_mcp_centroid = {
                                'left': input_ac_dict.get('left_finger_mcp_centroid', None),
                                'right': input_ac_dict.get('right_finger_mcp_centroid', None)
                            }

                            # get the robot tcp centroid position dict
                            robot_tcp_centroid = {
                                'left': left_hand_controller.get_current_finger_mcp_centroid(),
                                'right': right_hand_controller.get_current_finger_mcp_centroid()
                            }
                            # robot_tcp_centroid =  {'left': None, 'right': None}  # Disable for now

                            # Convert SEW positions to joint positions using the arm-specific converter
                            action_dict = convert_sew_actions_to_joint_positions(sew_converters, input_ac_dict, robot, env.sim, 
                                                                                    safety_layer=enable_collision_filtering, debug=args.debug,
                                                                                    human_hands_centroid=human_finger_mcp_centroid,
                                                                                    robot_tcp_centroid=robot_tcp_centroid)
                            
                            # Override gripper actions with Psyonic hand control
                            for arm in robot.arms:
                                hand_controller = left_hand_controller if arm == 'left' else right_hand_controller
                                # Use finger data if available
                                if f"{arm}_fingers" in input_ac_dict and input_ac_dict[f"{arm}_fingers"]:
                                    finger_positions = input_ac_dict[f"{arm}_fingers"]
                                    hand_controller.compute_hand_joint_goals_from_fingers(finger_positions)
                                else:
                                    hand_controller.set_joint_goals(np.zeros(10))  # Home position
                                
                                # Compute control torques for the hand
                                action_dict[f"{arm}_gripper"] = hand_controller.compute_control_torques()

                                # If connected to hardware, update the hand state from the real hand
                                hand_controller.send_joint_positions_hw()
                            
                            # Update FPV camera only if enabled
                            if args.fpv and camera_mover and "head_rotation" in input_ac_dict:
                                # print(f"{input_ac_dict['head_rotation']=}")  # DEBUG
                                R_body_head = input_ac_dict["head_rotation"]
                                base_body_name = f"{robot.robot_model.naming_prefix}base"
                                base_body_id = env.sim.model.body_name2id(base_body_name)
                                base_body_pose = env.sim.data.body_xpos[base_body_id].copy()
                                base_body_pose += np.array([0.1, 0, 0.55])  # Slightly above base
                                camera_mover.set_camera_pose(pos=base_body_pose,
                                    quat=t_utils.mat2quat(R_body_head @ R_std_mjcam))
            

                            # Execute action
                            env_action = robot.create_action_vector(action_dict)
                            env.step(env_action)

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
                            
                        else:
                            # No XR input - hold position
                            # env.step(np.zeros(env.action_dim))
                            if step_count % 300 == 0:
                                print("Waiting for XR data...")
                                
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        try:
                            left_base.Stop()
                            right_base.Stop()
                        except:
                            pass                    
                    # Sync viewer and maintain framerate
                    viewer.sync()
                    
                    # Update hand controller data references periodically
                    left_hand_controller.update_data_reference(data)
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
            left_base.Stop()
            right_base.Stop()
        except:
            pass
        
        # Cleanup hand controllers
        print("Closing hand controllers...")
        left_hand_controller.close()
        right_hand_controller.close()
        
        # Cleanup
        print("Cleaning up...")
        if args.record_data:
            xr_device.cleanup_recording()

        if hasattr(xr_device, 'stop'):
            xr_device.stop()
        env.close()
        print("Demo completed.")


if __name__ == "__main__":
    main()