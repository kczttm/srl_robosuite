"""
Exp Script for Dual Kinova 3 with XR Teleoperation.
Supports 'mink' (Mink IK) and 'sew_mimic' (OSC/SEW) controllers via XR/WebRTC.

Usage:
    python projects/experiment_envs/exp_bikino_xr.py --environment ExpCabinet --controller mink
"""

import argparse
import time
import numpy as np
import robosuite as suite
import mujoco
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

import projects.experiment_envs
from projects.dual_kinova3_teleop.controllers import WholeBodyMinkIK

import os
import sys
from copy import deepcopy

# Import SEW converter for joint position control
# needed for "sew_mimic" controller option
from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
    initialize_sew_converters_for_robot,
    update_sew_converter_origins_from_robot,
    convert_sew_actions_to_joint_positions,
    create_sew_visualization_wrapper
)

# Camera adjustments (applied to the named camera)
CAM_POS_OFFSET = [0.0, 0.0, 0.3]
CAM_EULER_OFFSET = [0.0, 0.0, 0.0]  # negative pitch tilts camera up
INTEROCULAR_DISTANCE = 0.063  # 63mm
CAMERA_NAME = "robot0_robotview"

def compute_env_action(env, device, controller_arg, sew_converters, all_prev_gripper_actions):
    active_robot = env.robots[0]
    
    # Get raw input from device first
    input_ac_dict = None
    if controller_arg == "sew_mimic":
            # For SEW mimic, we want the raw dictionary (SEW coords keys) to pass to converter
        input_ac_dict = device.get_controller_state()
    else:
        # For Mink and others, use input2action to map keys (if device implements it uniquely)
        input_ac_dict = device.input2action()
    
    if input_ac_dict is None:
        return None

    action_dict = deepcopy(input_ac_dict)

    # Map inputs to arms
    for arm in active_robot.arms:
        # If using Mink (Absolute control)
        if controller_arg == "mink":
            if f"{arm}_abs" in input_ac_dict:
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
        
        # If using SEW Mimic - handled in next block
    
    # Construct Action Vector
    env_action = None

    # 1. Mink Control
    if controller_arg == "mink":
        for i, robot in enumerate(env.robots):
            # We are only controlling the active robot (index 0 usually for DualKinova3)
            if i == 0:
                # Update gripper history
                for arm in robot.arms:
                    grip_key = f"{arm}_gripper"
                    if grip_key in action_dict:
                        all_prev_gripper_actions[i][grip_key] = action_dict[grip_key]
                
                # Create full action vector
                try:
                    robot_action = robot.create_action_vector(action_dict)
                    env_action_parts = [robot_action]
                except Exception as e:
                    # Occasional dimension mismatches during startup/switching
                    print(f"Error creating action vector: {e}")
                    env_action_parts = [np.zeros(robot.dof)]
            else:
                # Passive robots
                env_action_parts.append(robot.create_action_vector(all_prev_gripper_actions[i]))
        
        env_action = np.concatenate(env_action_parts)

    # 2. SEW Mimic Control
    elif controller_arg == "sew_mimic":
            try:
                # input_ac_dict is the raw dictionary from XR device (SEW coords)
                # convert_sew_actions_to_joint_positions handles the computation
                # It returns a dictionary compatible with the robot's action creator
                joint_action_dict = convert_sew_actions_to_joint_positions(
                    sew_converters, 
                    input_ac_dict, 
                    active_robot, 
                    env.sim, 
                    debug=False,
                    safety_layer=True
                )
                env_action = active_robot.create_action_vector(joint_action_dict)
            except Exception as e:
                print(f"Error creating action vector (SEW): {e}")
                env_action = np.zeros(env.action_dim)

    # 3. Default fallback
    else: 
            # Create action for all robots (usually just one)
        env_action_parts = []
        for i, robot in enumerate(env.robots):
            # We are only controlling the active robot (index 0 usually for DualKinova3)
            if i == 0:
                # Update gripper history
                for arm in robot.arms:
                    grip_key = f"{arm}_gripper"
                    if grip_key in action_dict:
                        all_prev_gripper_actions[i][grip_key] = action_dict[grip_key]
                
                # Create full action vector
                try:
                    robot_action = robot.create_action_vector(action_dict)
                    env_action_parts.append(robot_action)
                except Exception as e:
                    # Occasional dimension mismatches during startup/switching
                    print(f"Error creating action vector: {e}")
                    env_action_parts.append(np.zeros(robot.dof))
            else:
                # Passive robots
                env_action_parts.append(robot.create_action_vector(all_prev_gripper_actions[i]))
        env_action = np.concatenate(env_action_parts)
    
    return env_action

if __name__ == "__main__":
    # Get the repo path
    # Assuming this script is running from workspace root or is correctly located
    # We will derive paths relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # Define Controller Paths
    mink_controller_path = os.path.join(
        repo_path, "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_mink_ik.json"
    )
    # SEW Mimic Joint Position path (used for sew_mimic)
    sew_mimic_controller_path = os.path.join(
        repo_path, "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_sew_mimic_joint_position.json"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="ExpCabinet")
    parser.add_argument("--robots", nargs="+", type=str, default="DualKinova3", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument(
        "--controller",
        type=str,
        default="mink",
        help="Choice of controller. Options: 'mink', 'sew_mimic', or path to a json config file",
    )
    parser.add_argument("--device_ip", type=str, default="0.0.0.0", help="IP for WebRTC")
    
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument("--no_log", action="store_true", help="Disable logging")
    
    args = parser.parse_args()

    # Resolve controller
    controller_arg = args.controller
    controller_file = None
    
    if controller_arg == "mink":
        controller_file = mink_controller_path
    elif controller_arg == "sew_mimic":
        controller_file = sew_mimic_controller_path
    else:
        controller_file = controller_arg

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=controller_file,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment
    if "TwoArm" in args.environment or args.robots[0] == "DualKinova3":
         if "TwoArm" in args.environment:
            config["env_configuration"] = args.config
    else:
        args.config = None

    env = suite.make(
            **config,
            has_renderer=False,
            has_offscreen_renderer=False, # Disable offscreen renderer
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
            log_data=not args.no_log,
        )
    env.table_offset = np.array((0.0,0,0.6))

    # If using sew_mimic, we need to correct the table offset and use SEW Vis Wrapper
    if controller_arg == "sew_mimic":
        # Add SEW Vis Wrapper
        env = create_sew_visualization_wrapper(env)
    else:
        # Wrap in standard visualization
        env = VisualizationWrapper(env, indicator_configs=None)

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Initialize Device
    device = None
    if controller_arg == "mink":
        print("Initializing Mink XR Device...")
        from projects.shared_devices.mink_xr_robot_teleop_client import MinkXRRTCBodyPoseDevice
        device = MinkXRRTCBodyPoseDevice(env=env)
    elif controller_arg == "sew_mimic":
        print("Initializing Standard XR Device (SEW/Mimic)...")
        from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
        device = XRRTCBodyPoseDevice(env=env)
    else:
        # Fallback for generic controller path, try to guess or default to XRRTC
        print(f"Unknown controller shortcut '{controller_arg}'. Defaulting to standard XR Device.")
        from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
        device = XRRTCBodyPoseDevice(env=env)

    # Get camera ID
    cam_id = env.sim.model.camera_name2id(CAMERA_NAME)

    # Main Loop
    while True:
        obs = env.reset()
        
        # device.start_control() # Some devices need this, XR device starts thread in __init__ usually but good to keep pattern if method exists
        if hasattr(device, "start_control"):
            device.start_control()

        # Initialize gripper state tracking
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        active_robot = env.robots[0] # Assuming single robot (DualKinova3 is one robot object)

        # Initialize SEW converters if needed (only for sew_mimic)
        sew_converters = None
        if controller_arg == "sew_mimic":
            print("Initializing SEW converters...")
            sew_converters = initialize_sew_converters_for_robot(active_robot, env.sim, enable_collision_filtering=True)
            update_sew_converter_origins_from_robot(sew_converters, active_robot)

        model = env.sim.model._model
        data = env.sim.data._data

        # Configure passive viewer
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=True,
        ) as viewer:
            viewer.opt.geomgroup[0] = 0 # Hide collision geoms
            
            # Camera view
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -15
            viewer.cam.lookat[:] = [0, 0, 1.0]
            
            step_count = 0
            
            while viewer.is_running():
                start = time.time()
                active_robot = env.robots[0] # Assuming single robot (DualKinova3 is one robot object)

                # Get raw input from device first
                input_ac_dict = None
                if controller_arg == "sew_mimic":
                     # For SEW mimic, we want the raw dictionary (SEW coords keys) to pass to converter
                    input_ac_dict = device.get_controller_state()
                else:
                    # For Mink and others, use input2action to map keys (if device implements it uniquely)
                    input_ac_dict = device.input2action()
                
                if input_ac_dict is None:
                    # Depending on device implementation, None might mean no data yet or reset
                    # For XR device, if not connected, it returns None usually or waits
                    
                    # Do not step physics if no input (prevents drift/motion before connection)
                    # Just render static scene
                    viewer.sync()
                    
                    if args.max_fr is not None:
                        elapsed = time.time() - start
                        diff = 1 / args.max_fr - elapsed
                        if diff > 0:
                            time.sleep(diff)
                    continue

                action_dict = deepcopy(input_ac_dict)

                # Map inputs to arms
                for arm in active_robot.arms:
                    # Determine input type expected by controller
                    if isinstance(active_robot.composite_controller, WholeBody):
                         controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                         controller_input_type = active_robot.part_controllers[arm].input_type
                    
                    # If using Mink (Absolute control)
                    if controller_arg == "mink":
                        if f"{arm}_abs" in input_ac_dict:
                            action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    
                    # If using SEW Mimic - handled in next block
                
                # Construct Action Vector
                env_action = None

                # 1. Mink Control
                if controller_arg == "mink":
                    for i, robot in enumerate(env.robots):
                        # We are only controlling the active robot (index 0 usually for DualKinova3)
                        if i == 0:
                            # Update gripper history
                            for arm in robot.arms:
                                grip_key = f"{arm}_gripper"
                                if grip_key in action_dict:
                                    all_prev_gripper_actions[i][grip_key] = action_dict[grip_key]
                            
                            # Create full action vector
                            try:
                                robot_action = robot.create_action_vector(action_dict)
                                env_action_parts = [robot_action]
                            except Exception as e:
                                # Occasional dimension mismatches during startup/switching
                                print(f"Error creating action vector: {e}")
                                env_action_parts = [np.zeros(robot.dof)]
                        else:
                            # Passive robots
                            env_action_parts.append(robot.create_action_vector(all_prev_gripper_actions[i]))
                    
                    env_action = np.concatenate(env_action_parts)

                # 2. SEW Mimic Control
                elif controller_arg == "sew_mimic":
                     try:
                        # input_ac_dict is the raw dictionary from XR device (SEW coords)
                        # convert_sew_actions_to_joint_positions handles the computation
                        # It returns a dictionary compatible with the robot's action creator
                        joint_action_dict = convert_sew_actions_to_joint_positions(
                            sew_converters, 
                            input_ac_dict, 
                            active_robot, 
                            env.sim, 
                            debug=False,
                            safety_layer=True
                        )
                        env_action = active_robot.create_action_vector(joint_action_dict)
                     except Exception as e:
                        print(f"Error creating action vector (SEW): {e}")
                        env_action = np.zeros(env.action_dim)

                # 3. Default fallback
                else: 
                     # Create action for all robots (usually just one)
                    env_action_parts = []
                    for i, robot in enumerate(env.robots):
                        # We are only controlling the active robot (index 0 usually for DualKinova3)
                        if i == 0:
                            # Update gripper history
                            for arm in robot.arms:
                                grip_key = f"{arm}_gripper"
                                if grip_key in action_dict:
                                    all_prev_gripper_actions[i][grip_key] = action_dict[grip_key]
                            
                            # Create full action vector
                            try:
                                robot_action = robot.create_action_vector(action_dict)
                                env_action_parts.append(robot_action)
                            except Exception as e:
                                # Occasional dimension mismatches during startup/switching
                                print(f"Error creating action vector: {e}")
                                env_action_parts.append(np.zeros(robot.dof))
                        else:
                            # Passive robots
                            env_action_parts.append(robot.create_action_vector(all_prev_gripper_actions[i]))
                    env_action = np.concatenate(env_action_parts)
                
                # Step
                env.step(env_action)
                viewer.sync()
                
                # Frame rate control
                if args.max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1 / args.max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)
                
                step_count += 1
                if step_count % 300 == 0:
                    print(f"Step {step_count}")

