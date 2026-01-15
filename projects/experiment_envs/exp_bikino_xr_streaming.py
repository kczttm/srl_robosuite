"""
Reproduction script based on the working demo_dual_kinova3_xr_robot_teleop_mediamtx.py.
This script is adapted to reside in projects/experiment_envs/ and run the experiment environment
logic, but using the EXACT streaming loop structure from the working demo.
"""

import argparse
import os
import subprocess
import time
from copy import deepcopy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# Adjust path to find modules properly given new location
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import projects.experiment_envs

from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
    initialize_sew_converters_for_robot,
    update_sew_converter_origins_from_robot,
    convert_sew_actions_to_joint_positions,
    create_sew_visualization_wrapper
)

from projects.shared_devices.xr_media_mtx_streamer import MediaMTXStreamer, StereoCameraRig
from robosuite.utils.camera_utils import CameraMover
import robosuite.utils.transform_utils as t_utils
from projects.shared_scripts.mujoco_camera_utils import R_std_mjcam

# Camera adjustments
# CAM_POS_OFFSET = [-0.18, 0.0, 0.3]  # Original demo offset
CAM_POS_OFFSET = [-0.12, 0.0, 0.25] # Adjusted for experiment envs
CAM_EULER_OFFSET = [0.0, 0.0, 0.0]  # negative pitch tilts camera up
# INTEROCULAR_DISTANCE = 0.12  # 120mm zed cam destance
INTEROCULAR_DISTANCE = 0.065  # 63mm average human IPD
CAMERA_NAME = "robot0_robotview"



def create_env(args):
    # Paths from original script logic
    # projects/dual_kinova3_teleop/controllers/config/robots/dualkinova3_sew_mimic_joint_position.json
    controller_path = os.path.join(
        repo_path, "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_sew_mimic_joint_position.json"
    )
    
    if args.controller == 'mink':
        controller_path = os.path.join(
             repo_path, "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_mink_ik.json"
        )
    elif args.controller != 'sew_mimic':
         # Assume it's a path or default to sew_mimic if not specified
         pass

    # Load controller config
    controller_config = load_composite_controller_config(
        controller=controller_path,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    
    # Handle environment specific config
    if "TwoArm" in args.environment or args.robots[0] == "DualKinova3":
         if "TwoArm" in args.environment:
            config["env_configuration"] = args.config

    # Create environment with offscreen renderer enabled
    env = suite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,  # FORCE Enable offscreen rendering for streaming
        ignore_done=True,
        use_camera_obs=False,
        control_freq=30, # Match demo freq
    )
    
    env.table_offset = np.array((0.0, 0, 0.6))
        
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduction of working demo logic for experiment envs")
    parser.add_argument("--environment", type=str, default="ExpCabinet")
    parser.add_argument("--robots", nargs="+", type=str, default="DualKinova3")
    # Simplify controller arg to match exp_bikino but keep logic simple
    parser.add_argument("--controller", type=str, default="sew_mimic", help="sew_mimic or mink") 
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--device", type=str, default="keyboard") # unused really if using XR
    parser.add_argument("--max_fr", default=30, type=int)
    parser.add_argument("--record_data", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./xr_recordings")
    parser.add_argument("--ndigits", type=int, default=3)
    parser.add_argument("--fpv", action="store_true", default=True, help="Enable first-person view camera control (Head Tracking)")
    
    # MediaMTX streaming options
    parser.add_argument("--rtsp_url", type=str, default="rtsp://localhost:8554/mujoco")
    parser.add_argument("--stream_width", type=int, default=1920)
    parser.add_argument("--stream_height", type=int, default=1080)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--sbs", action="store_true")

    args = parser.parse_args()

    # Interactive selection for Environment
    print("\nSelect Environment:")
    print("1. ExpCabinet")
    print("2. ExpGlassGap")
    print("3. ExpPickPlace")
    
    while True:
        try:
            choice = input("Enter choice (1-3) [default: 1]: ").strip()
            if not choice:
                choice = "1"
            
            if choice == "1":
                args.environment = "ExpCabinet"
                break
            elif choice == "2":
                args.environment = "ExpGlassGap"
                break
            elif choice == "3":
                args.environment = "ExpPickPlace"
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input.")

    # Interactive selection for Controller
    print("\nSelect Controller:")
    print("1. sew_mimic")
    print("2. mink")

    while True:
        try:
            choice = input("Enter choice (1-2) [default: 1]: ").strip()
            if not choice:
                choice = "1"
            
            if choice == "1":
                args.controller = "sew_mimic"
                break
            elif choice == "2":
                args.controller = "mink"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input.")
            
    print(f"\nStarting with Environment: {args.environment}, Controller: {args.controller}\n")

    # Create centralized SEW visualization wrapper
    # Checks if wrapper exists (it does in this workspace)
    # env = create_sew_visualization_wrapper(env)  # capsule visualization
    env = create_env(args)
    
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Initialize Device
    # We will use the generic XR device logic or specific based on controller,
    # but the working demo uses XRRTCBodyPoseDevice directly.
    # exp_bikino uses MinkXRRTCBodyPoseDevice for mink.
    if args.controller == "mink":
        print("Initializing Mink XR Device...")
        from projects.shared_devices.mink_xr_robot_teleop_client import MinkXRRTCBodyPoseDevice
        device = MinkXRRTCBodyPoseDevice(env=env)
    else:
        print("Initializing Standard XR Device...")
        device = XRRTCBodyPoseDevice(
            env=env,
            record_data=args.record_data,
            output_dir=args.output_dir,
            ndigits=args.ndigits
        )

    print("== SKIPPING WAITING FOR CLIENT TO CONNECT (DEBUG REMOVE)")

    # Run the simulation loop
    obs = env.reset()

    # Initialize CameraMover if enabled (Must be after reset)
    camera_mover = None
    pos_offset = CAM_POS_OFFSET
    euler_offset = CAM_EULER_OFFSET
    
    # We will compute transforms manually if FPV is on, to avoid CameraMover crash
    # on 'robotview' camera.
    cam_id = env.sim.model.camera_name2id(CAMERA_NAME)
    
    if args.fpv:
        # Override offsets for StereoCameraRig when using FPV
        # We want the camera to be centered on the FPV 'Head'
        pos_offset = [0.0, 0.0, 0.0]
        euler_offset = [0.0, 0.0, 0.0]
        
        # Adjust FOV for VR
        env.sim.model.cam_fovy[cam_id] = 75 

    # Initialize Stereo Camera Rig (Refactored)
    rig = StereoCameraRig(
        env.sim, 
        CAMERA_NAME, 
        pos_offset=pos_offset, 
        euler_offset=euler_offset, 
        ipd=INTEROCULAR_DISTANCE
    )
    rig.initialize()

    # Initialize SEW converters
    active_robot = env.robots[0]
    sew_converters = None
    if args.controller == 'sew_mimic':
        print("Initializing SEW converters...")
        sew_converters = initialize_sew_converters_for_robot(
            active_robot,
            env.sim,
            enable_collision_filtering=True,
            use_robotiq_gripper=True
        )
        update_sew_converter_origins_from_robot(sew_converters, active_robot)

    # Set up MediaMTX streamer(s)
    use_gpu = not getattr(args, 'no_gpu', False)
    use_sbs = args.sbs
    print(f"Using {'GPU (NVENC)' if use_gpu else 'CPU (libx264)'} encoding")

    if use_sbs:
        streamer_stereo = MediaMTXStreamer(
            rtsp_url=args.rtsp_url if args.rtsp_url != "rtsp://localhost:8554/mujoco" else "rtsp://localhost:8554/cam/stereo",
            # Note: demo used separate URLs, exp uses replace logic. We trust demo logic here but ensure port is correct.
            width=args.stream_width * 2,
            height=args.stream_height,
            fps=args.max_fr,
            use_gpu=use_gpu
        )
        streamer_stereo.start()
    else:
        streamer_left = MediaMTXStreamer(
            rtsp_url=args.rtsp_url.replace("mujoco","cam/left") if "mujoco" in args.rtsp_url else "rtsp://localhost:8554/cam/left",
            width=args.stream_width,
            height=args.stream_height,
            fps=args.max_fr,
            use_gpu=use_gpu
        )
        streamer_right = MediaMTXStreamer(
            rtsp_url=args.rtsp_url.replace("mujoco","cam/right") if "mujoco" in args.rtsp_url else "rtsp://localhost:8554/cam/right",
            width=args.stream_width,
            height=args.stream_height,
            fps=args.max_fr,
            use_gpu=use_gpu
        )
        streamer_left.start()
        streamer_right.start()

    # Precompute stereo offsets - Handled by Rig now
    # left_cam_offset = cam_right_vector * (INTEROCULAR_DISTANCE / 2)
    # right_cam_offset = cam_right_vector * (INTEROCULAR_DISTANCE / 2)

    # Preallocate side-by-side stereo frame buffer
    if use_sbs:
        stereo_frame = np.zeros((args.stream_height, args.stream_width * 2, 3), dtype=np.uint8)

    running = True
    try:
        while running:
            start_time = time.time()

            # Get pose data from XR WebRTC device
            try:
                # Use exp_bikino's input logic style to support mink/sew selection
                # But keep the loop structure clean
                input_ac_dict = None
                if args.controller == "sew_mimic":
                    input_ac_dict = device.get_controller_state()
                else:
                    input_ac_dict = device.input2action() # Mink usually uses this

                if input_ac_dict is not None:
                    # Update FPV Camera (Manual Override logic replacing CameraMover)
                    if args.fpv and "head_rotation" in input_ac_dict:
                        R_user_head = input_ac_dict["head_rotation"] # Rotation Matrix

                        # Remove rotation about x-axis in local frame 
                        r_obj = R.from_matrix(R_user_head)
                        local_euler = r_obj.as_euler('ZYX', degrees=False)
                        local_euler[2] = 0.0  # Zero out roll
                        R_user_head = R.from_euler('ZYX', local_euler, degrees=False).as_matrix()

                        # Use Rig's set_world_pose directly for manual CameraMover-like behavior
                        
                        # Desired World Pose:
                        base_body_name = f"{active_robot.robot_model.naming_prefix}base"
                        base_body_id = env.sim.model.body_name2id(base_body_name)
                        base_pos = env.sim.data.body_xpos[base_body_id]
                        
                        # Match user head rotation with standard camera orientation offset
                        target_cam_rot = R.from_matrix(R_user_head @ R_std_mjcam)
                        target_cam_pos = base_pos + np.array([0.1, 0, 0.55]) # Standard head position
                        
                        rig.set_world_pose(target_cam_pos, target_cam_rot)

                    # Logic from exp_bikino for action creation
                    action_dict = deepcopy(input_ac_dict)
                    
                    # 1. Mink Control
                    if args.controller == "mink":
                        # Simplification: assume single robot for now as per demo
                        # Map absolute inputs
                        for arm in active_robot.arms:
                             if f"{arm}_abs" in input_ac_dict:
                                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                        
                        try:
                            # Note: we skip the complex multi-robot loop of exp_bikino for this repro
                            # because demo works with single robot. 
                            env_action = active_robot.create_action_vector(action_dict)
                        except Exception as e:
                            env_action = np.zeros(env.action_dim)
                            
                    # 2. SEW Mimic Control
                    elif args.controller == "sew_mimic":
                        try:
                            joint_action_dict = convert_sew_actions_to_joint_positions(
                                sew_converters, input_ac_dict, active_robot, env.sim,
                                debug=False, safety_layer=True
                            )
                            env_action = active_robot.create_action_vector(joint_action_dict)
                        except Exception as e:
                            env_action = np.zeros(env.action_dim)
                    else:
                        env_action = np.zeros(env.action_dim)
                        
                    env.step(env_action)
                else:
                    # Do not step physics if no input (prevents drift/motion before connection)
                    # Just render static scene
                    pass

            except Exception as e:
                # print(f"Error getting XR state: {e}")
                pass

            # Render stereo frames via Rig
            # Note: Rig returns them already flipped and contiguous if flip=True
            frame_left, frame_right = rig.get_stereo_frames(
                args.stream_width, 
                args.stream_height, 
                flip=True
            )

            if use_sbs:
                # Frames are already flipped by Rig, so we just copy them into the buffer
                stereo_frame[:, :args.stream_width, :] = frame_left
                stereo_frame[:, args.stream_width:, :] = frame_right
                streamer_stereo.send_frame(stereo_frame)
            else:
                # Frames are already contiguous and flipped from the rig
                streamer_left.send_frame(frame_left)
                streamer_right.send_frame(frame_right)

            # Maintain target frame rate
            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nSimulation finished. Closing environment...")
        if use_sbs:
            streamer_stereo.stop()
        else:
            streamer_left.stop()
            streamer_right.stop()
        if args.record_data:
            device.cleanup_recording()
        env.close()
        print("Demo completed.")
