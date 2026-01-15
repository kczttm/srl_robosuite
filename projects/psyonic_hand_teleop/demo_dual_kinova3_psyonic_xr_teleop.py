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
from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller import AbilityHandController
# from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller_teensy import AbilityHandControllerTeensy

# Import SEW converter helper functions - this handles most of the heavy lifting!
from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
    initialize_sew_converters_for_robot,
    update_sew_converter_origins_from_robot,
    convert_sew_actions_to_joint_positions,
    create_sew_visualization_wrapper
)

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
    env.table_offset = np.array((0.5, 0, 0.4))
    return env


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

        # Set FOV
        # NOTE: CameraMover creates a new camera without preserving original fovy; only mode, name, and pos are copied.
        cam_id = env.sim.model.camera_name2id("agentview")
        env.sim.model.cam_fovy[cam_id] = 75
    robot = env.robots[0]
    
    # Initialize SEW converters using helper function with collision filtering option
    collision_status = "enabled" if enable_collision_filtering else "disabled"
    print(f"Initializing SEW converters with collision avoidance {collision_status}...")
    sew_converters = initialize_sew_converters_for_robot(robot, env.sim, enable_collision_filtering=enable_collision_filtering)
    # Update SEW converter origins using helper function
    update_sew_converter_origins_from_robot(sew_converters, robot)
    
    model = env.sim.model._model
    data = env.sim.data._data

    # simulation time stepsize
    time_step = 0.005  # 10ms
    model.opt.timestep = time_step

    print("Initializing hand controllers...")
    left_hand_controller = AbilityHandController(model, data, hand_side='left', debug=False)
    right_hand_controller = AbilityHandController(model, data, hand_side='right', debug=False)

    # left_hand_controller = AbilityHandControllerTeensy(model, data, hand_side='left', hardware=False, port="/dev/ttyACM0", debug=False)
    # right_hand_controller = AbilityHandControllerTeensy(model, data, hand_side='right', hardware=False, port="/dev/ttyACM1", debug=False)

    print("Starting teleoperation...")
    print("=" * 60)
    print("XR Teleoperation: Move your arms and hands to control the robots!")
    print("=" * 60)
    
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) as viewer:
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
                    # print("Human Finger MCP Centroid:", human_finger_mcp_centroid)

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
                        # hand_controller.send_joint_positions_hw()
                    
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
                    
                else:
                    # No XR input - hold position
                    # env.step(np.zeros(env.action_dim))
                    if step_count % 300 == 0:
                        print("Waiting for XR data...")
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                # env.step(np.zeros(env.action_dim))
            
            # Sync viewer and maintain framerate
            viewer.sync()
            
            # Update hand controller data references periodically
            if step_count % 100 == 0:
                left_hand_controller.update_data_reference(data)
                right_hand_controller.update_data_reference(data)
            
            # Maintain target framerate
            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)
            
            step_count += 1
            
            if step_count % 300 == 0 and args.debug:
                print(f"Step {step_count} - Running smoothly")
    
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