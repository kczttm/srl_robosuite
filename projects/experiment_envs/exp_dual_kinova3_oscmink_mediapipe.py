"""Modified version of keyboard teleoperation for Dual Kinova 3.

***Choose user input option with the --device argument***

Mediapipe:
    We use mediapipe to control the end-effector of the robot.

    Note:
    - Make sure your full upper body is visible to the camera
    - Raise both arms to shoulder height to start control
    - Lower arms to stop control

    *** Control Mediapipe Parameters ***
        * --camera_id: Camera ID (int) to capture images from.
        * --mirror_actions: If enabled, will mirror the user inputs for controlling the robot.
        * --debug: If enabled, will show debug information for the mediapipe device.



***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"parallel" and "opposed"}

            -"parallel": Sets up the environment such that two robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of robot names to be specified
                in the --robots argument.

            -"opposed": Sets up the environment such that two robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config parallel --controller osc


"""

import argparse
import time

import numpy as np

import robosuite as suite
import mujoco
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

# Import custom environments to register them with robosuite
import projects.experiment_envs
# mink-specific import
from projects.dual_kinova3_teleop.controllers import WholeBodyMinkIK

import os

if __name__ == "__main__":
    # Get the path of robosuite
    repo_path = os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir)
    )
    # Mink path
    mink_controller_path = os.path.join(
        repo_path, "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_mink_ik.json"
    )
    # OSC path
    osc_controller_path = os.path.join(
        repo_path, "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_osc.json"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="ExpCabinet")
    parser.add_argument("--robots", nargs="+", type=str, default="DualKinova3", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=osc_controller_path,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="Mediapipe", help="Device to use for teleoperation: 'mediapipe'")
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument("--no_log", action="store_true",
        help="Disable logging")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID (int) to capture images from.")
    parser.add_argument("--mirror_actions", action="store_true", help="Mirror actions for the robot.")
    parser.add_argument("--debug", action="store_false", help="Enable debug mode.")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    env = suite.make(
            **config,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
            log_data=not args.no_log,
        )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "Mediapipe":
        from projects.shared_devices.mediapipe_teleop_robosuite_wrapper import MediaPipeTeleop

        device = MediaPipeTeleop(env=env, 
                                 camera_id=args.camera_id, 
                                 mirror_actions=args.mirror_actions, 
                                 debug=args.debug
        )
    else:
        raise Exception("Invalid device choice: choose 'Mediapipe'.")

    # Initialize device control
    device.start_control()

    print("=" * 80)
    print("Human Pose Teleoperation Demo for Dual Kinova3 Robots")
    print("=" * 80)
    print("This demo uses MediaPipe to track your arm movements and control the robots.")
    print("\nSetup Instructions:")
    print("1. Position yourself in front of the camera")
    print("2. Make sure your full upper body is visible")
    print("3. Raise both arms to shoulder height to start control")
    print("4. Lower arms to stop control")
    print(f"Mirror actions: {'Enabled' if args.mirror_actions else 'Disabled'}")
    print(f"Camera ID: {args.camera_id}")
    print("=" * 80)

    while True:
        if hasattr(device, 'should_quit') and device.should_quit():
            print("Quit signal detected before environment reset. Exiting...")
            device.stop()
            break

        # Reset the environment
        obs = env.reset()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        print("\nEnvironment reset complete. Starting teleoperation...")
        print("Waiting for pose tracking to engage...")
        print("Raise both arms to shoulder height to begin control.")

        model = env.sim.model._model
        data = env.sim.data._data

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=True,
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
                start = time.time()

                # Set active robot
                active_robot = env.robots[device.active_robot]

                if hasattr(device, 'should_quit') and device.should_quit():
                    print("Quit signal detected. Stopping simulation...")
                    break 

                # Get the newest action
                input_ac_dict = device.input2action()

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    break

                from copy import deepcopy

                action_dict = deepcopy(input_ac_dict)  # {}
                # set arm actions
                for arm in active_robot.arms:
                    if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                        controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = active_robot.part_controllers[arm].input_type

                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    else:
                        raise ValueError

                # Maintain gripper state for each robot but only update the active robot with action
                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
                env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

                env.step(env_action)
                # env.render()

                # Sync viewer
                viewer.sync()

                # limit frame rate if necessary
                if args.max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1 / args.max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)
                
                step_count += 1
                
                # Print status occasionally
                if step_count % 300 == 0:  # Every 10 seconds at 30 FPS
                    print(f"Running demo - Step {step_count}")
        if hasattr(device, 'should_quit') and device.should_quit():
            print("Quit signal detected after viewer loop. Exiting...")
            break
    print("\nCleaning up...")
    env.close() 
    device.stop()  
    print("Demo completed successfully!")

    

