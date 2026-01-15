"""
Offline XR teleoperation demo for Dual Kinova3 robots with Psyonic Ability hands.

This demo replays recorded XR body pose data from a CSV file to control the dual Kinova3
robots with Psyonic hands in the robosuite environment.

Usage:
    python demo_dual_kinova3_psyonic_xr_offline.py --csv_file <path_to_csv> --environment Lift

Author: Chuizheng Kong
Date created: 2025-10-02
"""

import argparse
import os
import time

import mujoco
import numpy as np

import robosuite as suite

# Get the path of robosuite
repo_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
dual_kinova3_sew_config_path = os.path.join(
    repo_path,
    "SEW-Geometric-Teleop",
    "projects",
    "dual_kinova3_teleop",
    "controllers",
    "config",
    "robots",
    "dualkinova3_sew_mimic_joint_position.json",
)

# Import the Arm Class
from projects.psyonic_hand_teleop.ability_hand.dual_kinova3_robot_psyonic_gripper import DualKinova3PsyonicHand
import projects.experiment_envs

# Import the hand controller
from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller import AbilityHandController

from robosuite import load_composite_controller_config
from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
    convert_sew_actions_to_joint_positions,
    create_sew_visualization_wrapper,
    initialize_sew_converters_for_robot,
    update_sew_converter_origins_from_robot,
)
from projects.shared_devices.offline_openxr_pose_reader import CSVDataReader
from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
from robosuite.wrappers import VisualizationWrapper

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
    env.table_offset = np.array((0, 0, 0.6))
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline XR teleoperation demo for Dual Kinova3 with Psyonic hands using CSV playback"
    )
    parser.add_argument("--environment", type=str, default="ExpNarrowGap", help="Name of the robosuite environment to load")
    parser.add_argument(
        "--csv_file",
        type=str,
        # default="xr_recordings/body_pose_data_hand_arm_penetration.csv",
        default="xr_recordings/super_good_pot_3.csv",
        # default="xr_recordings/shoulder_singularity_hard.csv",
        required=False,
        help="Path to CSV file containing body pose data",
    )
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = normal speed)",
    )
    parser.add_argument("--max_fr", default=30, type=int)
    parser.add_argument(
        "--safety_layer",
        default=True,
        action="store_true",
        help="Enable safety layer checks and SEW collision filtering",
    )
    parser.add_argument(
        "--max_displacement",
        type=float,
        default=2.0,
        help="Maximum allowed displacement in meters for SEW safety (default: 2.0)",
    )
    args = parser.parse_args()

    env = create_environment(args)
    
    env = create_sew_visualization_wrapper(env)
    # env.env = env

    csv_reader = CSVDataReader(args.csv_file, args.playback_speed)
    print("CSV data loaded! Starting robosuite simulation.")

    obs = env.reset()
    active_robot = env.robots[0]  # Assuming single robot for this demo
    sew_converters = initialize_sew_converters_for_robot(active_robot, env.sim, enable_collision_filtering=args.safety_layer)

    # Update SEW converter origins from robot controllers if available
    update_sew_converter_origins_from_robot(sew_converters, active_robot)

    model = env.sim.model._model
    data = env.sim.data._data

    # Initialize hand controllers AFTER getting the model/data references
    print("Initializing hand controllers...")
    left_hand_controller = AbilityHandController(
        model,
        data, 
        hand_side='left', 
        debug=False
    )
    right_hand_controller = AbilityHandController(
        model, 
        data, 
        hand_side='right', 
        debug=False
    )

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,
    ) as viewer:
        # Set initial camera parameters for good view of dual arms
        # viewer.cam.distance = 3.0
        # viewer.cam.azimuth = 0
        # viewer.cam.elevation = -95
        # viewer.cam.lookat[:] = np.array([-0.5, 0.0, 0.0])
        viewer.opt.geomgroup[0] = 0  # Hide collision meshes
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]

        # Initialize timing for CSV playback
        sim_start_time = time.time()

        while viewer.is_running():
            loop_start_time = time.time()

            # Calculate elapsed time since simulation start
            elapsed_time = loop_start_time - sim_start_time

            try:
                bones = csv_reader.get_bones_at_time(elapsed_time)

                if bones is not None:
                    # Process bones to get action using the same function as real-time version
                    input_ac_dict = (
                        XRRTCBodyPoseDevice._default_process_bones_to_action(bones)
                    )

                    if input_ac_dict is not None:
                        # Create action for arm control using CSV data
                        active_robot = env.robots[0]

                        # Get the human finger mcp centroid positions dict
                        human_finger_mcp_centroid = {
                            'left': input_ac_dict.get('left_finger_mcp_centroid', None),
                            'right': input_ac_dict.get('right_finger_mcp_centroid', None)
                        }

                        # Get the robot tcp centroid position dict
                        robot_tcp_centroid = {
                            'left': left_hand_controller.get_current_finger_mcp_centroid(),
                            'right': right_hand_controller.get_current_finger_mcp_centroid()
                        }

                        # Convert SEW coordinates to joint positions using helper function
                        action_dict = convert_sew_actions_to_joint_positions(
                            sew_converters,
                            input_ac_dict,
                            active_robot,
                            env.sim,
                            safety_layer=args.safety_layer,
                            debug=False,
                            human_hands_centroid=human_finger_mcp_centroid,
                            robot_tcp_centroid=robot_tcp_centroid,
                            # max_displacement=args.max_displacement,
                        )

                        # Override gripper actions with Psyonic hand control
                        for arm in active_robot.arms:
                            hand_controller = left_hand_controller if arm == 'left' else right_hand_controller
                            # Use finger data if available
                            if f"{arm}_fingers" in input_ac_dict and input_ac_dict[f"{arm}_fingers"]:
                                finger_positions = input_ac_dict[f"{arm}_fingers"]
                                hand_controller.compute_hand_joint_goals_from_fingers(finger_positions)
                            else:
                                hand_controller.set_joint_goals(np.zeros(10))  # Home position
                            
                            # Compute control torques for the hand
                            action_dict[f"{arm}_gripper"] = hand_controller.compute_control_torques()

                        # Create action vector from action_dict
                        env_action = active_robot.create_action_vector(action_dict)
                        env.step(env_action)
                    else:
                        # Could not process bones - hold position
                        action = np.zeros(env.action_dim)
                        env.step(action)
                else:
                    # No bone data available - hold position
                    action = np.zeros(env.action_dim)
                    env.step(action)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing CSV data: {e}")
                # On error - just hold position with zero action
                action = np.zeros(env.action_dim)
                env.step(action)

            # Sync viewer
            viewer.sync()

            # Update hand controller data references periodically
            step_count = int((time.time() - sim_start_time) * args.max_fr)
            if step_count % 100 == 0:
                left_hand_controller.update_data_reference(data)
                right_hand_controller.update_data_reference(data)

            # Maintain target frame rate
            elapsed = time.time() - loop_start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    # Cleanup
    print("\nSimulation finished. Closing environment...")
    env.close()
    print("Demo completed.")
