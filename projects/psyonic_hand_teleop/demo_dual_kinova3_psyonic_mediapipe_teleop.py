"""
Demo script for dual Kinova3 robots with Psyonic Ability hands using MediaPipe teleoperation.

This script demonstrates:
1. Dual Kinova3 arm control using SEW converter with joint position controller
2. Psyonic Ability hand control using joint torque controller
3. MediaPipe integration for human pose estimation via MediaPipeTeleopDevice
4. Body-centric coordinate transformation for robust teleoperation

Usage:
    python demo_dual_kinova3_psyonic_mediapipe_teleop.py --environment Lift

Note: Uses MediaPipeTeleopDevice from the rby1_teleop folder for pose estimation.

Author: Chuizheng Kong
Date created: 2025-08-05
"""

import argparse
import os
import time
from copy import deepcopy

import mujoco
import numpy as np

import robosuite as suite
from scipy.spatial.transform import Rotation

# Import custom environments to register them with robosuite
import projects.experiment_envs

# Add path for MediaPipe teleoperation device
from projects.shared_devices.mediapipe_teleop_device_robosuite_ver import MediaPipeTeleop 

# from projects.shared_devices.mediapipe_teleop_robosuite_wrapper import MediaPipeTeleopRobosuiteWrapper as MediaPipeTeleop

# Import the Arm Class
from projects.psyonic_hand_teleop.ability_hand.dual_kinova3_robot_psyonic_gripper import DualKinova3PsyonicHand

from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# Import the hand controller
# from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller import AbilityHandController
from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller_teensy import AbilityHandController

# Import SEW converter for joint position control
from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
    initialize_sew_converters_for_robot,
    update_sew_converter_origins_from_robot,
    convert_sew_actions_to_joint_positions,
    get_current_joint_positions,
    create_sew_visualization_wrapper
)

# Get the path of robosuite
repo_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
dual_kinova3_sew_config_path = os.path.join(
    repo_path, "SEW-Geometric-Teleop", "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_sew_mimic_joint_position.json"
)



def create_dual_kinova3_with_ability_hands(args):
    """
    Create environment with dual Kinova3 and Ability hands.
    
    Returns:
        robosuite environment
    """
    # Load controller config
    controller_config = load_composite_controller_config(
        controller=dual_kinova3_sew_config_path,
        robot="DualKinova3PsyonicHand",
    )
    
    # Create environment with Ability hands
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
    env.table_offset = np.array((0,0,0.6))

    # logic for disabling logging if available & requested
    if "Exp" in args.environment:
        if args.no_log:
            env.log_data = False
            
    return env


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Dual Kinova3 with Psyonic hands MediaPipe teleoperation demo"
    )
    parser.add_argument("--environment", type=str, default="ExpTwoArmLift", 
                       help="Environment to use")
    parser.add_argument("--max_fr", default=30, type=int, 
                       help="Maximum frame rate")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    parser.add_argument("--no_camera", action="store_true", 
                       help="Disable camera input (for testing)")
    parser.add_argument("--no_log", action="store_true",
                        help="Disable logging")
    args = parser.parse_args()
    
    # Create environment
    print("Creating robosuite environment...")
    env = create_dual_kinova3_with_ability_hands(args)
    # Create centralized SEW visualization wrapper
    env = create_sew_visualization_wrapper(env)
    
    # Print robot information
    robot = env.robots[0]
    if hasattr(robot, 'init_qpos'):
        print(f"Init qpos shape: {robot.init_qpos.shape}")
        print(f"Init qpos values: {robot.init_qpos}")
    
    # Initialize MediaPipe teleoperation device
    print("Initializing MediaPipe teleoperation device...")
    if not args.no_camera:
        mediapipe_device = MediaPipeTeleop(
            env=env,
            camera_id=0, 
            debug=args.debug,
            mirror_actions=False
        )
        mediapipe_device.start_control()
    else:
        mediapipe_device = None
        print("Camera disabled - running without MediaPipe teleoperation")
    
    # Reset environment
    obs = env.reset()
    
    # Initialize robot to its defined init_qpos
    robot = env.robots[0]
    
    # Initialize SEW converters with actual robot joint positions and origin information
    sew_converters = initialize_sew_converters_for_robot(robot, env.sim, enable_collision_filtering=True)

    update_sew_converter_origins_from_robot(sew_converters, robot)
    
    model = env.sim.model._model
    data = env.sim.data._data

    left_hand_controller = AbilityHandController(model, data, hand_side='left', debug=False)
    right_hand_controller = AbilityHandController(model, data, hand_side='right', debug=False)
    
    # Initialize hand controllers AFTER getting the model/data references
    print("Initializing hand controllers...")

    # TRUE ==> CONNECT TO PSYONIC HAND HARDWARE / FALSE ==> SIMULATION ONLY
    hardware = False
    
    left_hand_controller = AbilityHandController(
        model,
        data, 
        hand_side='left', 
        debug=True  # Enable debug to see joint limits
    )
    right_hand_controller = AbilityHandController(
        model, 
        data, 
        hand_side='right', 
        debug=False  # Enable debug to see joint limits
    )
    
    print("Starting simulation...")
    
    # Print teleoperation instructions
    if mediapipe_device is not None:
        print("=" * 80)
        print("Human Pose Teleoperation Demo for Dual Kinova3 with Psyonic Hands")
        print("=" * 80)
        print("This demo uses MediaPipe to track your arm movements and control the robots.")
        print("\nSetup Instructions:")
        print("1. Position yourself in front of the camera")
        print("2. Make sure your full upper body is visible")
        print("3. Raise both arms to shoulder height to start control")
        print("4. Lower arms to stop control")
        print("\nCamera Controls:")
        print("- 'q' key: Quit")
        print("=" * 80)
    
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Disable geom group 0 (collision meshes) at start
        viewer.opt.geomgroup[0] = 0
        
        # Set camera for good view of dual arms
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]
        
        step_count = 0
        while viewer.is_running():
            start_time = time.time()
            
            # Get pose data from MediaPipe device
            if mediapipe_device is not None:
                # Check for device quit signal
                if hasattr(mediapipe_device, 'should_quit') and mediapipe_device.should_quit():
                    print("MediaPipe device quit signal received")
                    break
                
                # Get the newest action from human pose
                try:
                    input_ac_dict = mediapipe_device.input2action()
                    
                    if input_ac_dict is not None:
                        # Create action for arm control using MediaPipeTeleop output

                        # Get the human finger tip centroid positions dict
                        human_finger_mcp_centroid = {
                            'left': input_ac_dict.get('left_finger_mcp_centroid', None),
                            'right': input_ac_dict.get('right_finger_mcp_centroid', None)
                        }

                        # get the robot tcp centroid position dict
                        robot_tcp_centroid = {
                            'left': left_hand_controller.get_current_finger_mcp_centroid(),
                            'right': right_hand_controller.get_current_finger_mcp_centroid()
                        }
                        # print("Robot TCP Centroid:", robot_tcp_centroid)

                        # Convert SEW positions to joint positions using the arm-specific converter
                        action_dict = convert_sew_actions_to_joint_positions(sew_converters, input_ac_dict, robot, env.sim, 
                                                                             safety_layer=True, debug=args.debug,
                                                                             human_hands_centroid=human_finger_mcp_centroid,
                                                                             robot_tcp_centroid=robot_tcp_centroid)

                        for arm in robot.arms:
                            # Override gripper actions with Psyonic hand control
                            hand_controller = left_hand_controller if arm == 'left' else right_hand_controller

                            # Use finger data if available
                            if f"{arm}_fingers" in input_ac_dict and input_ac_dict[f"{arm}_fingers"]:
                                finger_positions = input_ac_dict[f"{arm}_fingers"]
                                hand_controller.compute_hand_joint_goals_from_fingers(finger_positions)
                            else:
                                hand_controller.set_joint_goals(np.zeros(10))  # Home position

                            
                            # hand_controller.send_joint_positions_hw()
                            
                            # Compute control torques for the hand
                            action_dict[f"{arm}_gripper"] = hand_controller.compute_control_torques()
                            
                        # Create action vector from action_dict (now includes gripper actions)
                        env_action = robot.create_action_vector(action_dict)
                        env.step(env_action)
                        # env.sim.step() # null action for debugging
                        
                    else:
                        # MediaPipe device returned None - not engaged or waiting
                        action = np.zeros(env.action_dim)
                        env.step(action)
                        
                        if step_count % 300 == 0:
                            print("Waiting for human engagement (raise both arms to shoulder height)")
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    # On error - just hold position with zero action
                    # action = np.zeros(env.action_dim)
                    # env.step(action)
            else:
                # No camera mode - just hold position with zero action
                action = np.zeros(env.action_dim)
                env.step(action)
            
            
            # Sync viewer
            viewer.sync()
            
            # Ensure hand controllers have the latest data reference (every 100 steps)
            if step_count % 100 == 0:
                left_hand_controller.update_data_reference(data)
                right_hand_controller.update_data_reference(data)
            
            # Maintain target frame rate
            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)
            
            step_count += 1
            
            # Print status occasionally
            if step_count % 300 == 0:  # Every 10 seconds at 30 FPS
                print(f"Running demo - Step {step_count}")
    
    # Cleanup
    print("Cleaning up...")
    if mediapipe_device is not None:
        mediapipe_device.stop()
    env.close()
    left_hand_controller.close()
    right_hand_controller.close()  
    print("Demo completed.")


if __name__ == "__main__":
    main()
