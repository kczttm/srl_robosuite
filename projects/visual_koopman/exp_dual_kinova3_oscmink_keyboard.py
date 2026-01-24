"""Modified version of keyboard teleoperation for Dual Kinova 3.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


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

import matplotlib.pyplot as plt
from PIL import Image
import re

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
            match = re.search(r"\d+", name)
            if match:
                nums.append(int(match.group()))

    return min(nums) + 1 if nums else 0

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

    if not (demo_index_image == demo_index_obs == demo_index_act):
        raise ValueError(
            f"Mismatch in demo indices: "
            f"images={demo_index_image}, "
            f"observations={demo_index_obs}, "
            f"actions={demo_index_act}"
        )

    demo_index = demo_index_image

    # Create subfolders for this trajectory, deleting existing ones if needed
    traj_img_dir = os.path.join(images_root, f"{demo_index_image}")
    traj_obs_dir = os.path.join(observations_root, f"{demo_index}")
    traj_act_dir = os.path.join(actions_root, f"{demo_index}")

    for d in [traj_img_dir, traj_obs_dir, traj_act_dir]:
        os.makedirs(d, exist_ok=True)

    # Save images (N images)
    for step_idx, img in enumerate(images):
        img_path = os.path.join(traj_img_dir, f"frame_{step_idx:04d}.png")
        Image.fromarray(img.astype("uint8")).save(img_path)
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
    assert T1 == T + 1, f"Expected action time {T+1}, got {T1}"

    time_obs = np.arange(T)      # 0..T-1
    time_act = np.arange(T1)      # 0..T-1
    
    import math
    n_cols = 4
    n_rows = math.ceil(N / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 2.5 * n_rows),
        sharex=True
    ) 

    # Make axes always 2D
    axes = np.atleast_2d(axes)

    state_names = (
        [f'Arm j_pos {i}' for i in range(1,7)]
        + [f'finger j_pos {i}' for i in range(1,17)]
        + [f'Arm j_vel {i}' for i in range(1,7)]
    ) # this is the state name for the Dexart task
    
    for i in range(N):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]

        ax.plot(time_obs, observation_array[:, i], label="obs")
        ax.plot(time_act, action_array[:, i], label="act")
        ax.set_title(state_names[i])
        ax.grid(True, alpha=0.3)

    # Turn off unused subplots
    for j in range(N, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    # Common labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Time step")

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,          # 👈 increase text size
        handlelength=3,       # 👈 longer line samples
        handletextpad=1.0     # 👈 space between line & text
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
    fig.savefig(os.path.join(compare_img_root, f"{demo_index}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument(
        "--controller",
        type=str,
        default=osc_controller_path,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument("--no_log", action="store_true",
        help="Disable logging")
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
            has_offscreen_renderer=True,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=True,
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
    if args.device == "keyboard":
        from projects.shared_devices.mink_keyboard_device import MinkKeyboardDevice
        
        device = MinkKeyboardDevice(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        # env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # set True when the teleoperation begins (using the feedback from the Quest)
    is_collecting = True

    # task name
    task_name = "grasping"

    # Test data is saved under the task repo
    save_path = os.path.join(
        "/home/yhan389/Desktop/srl_robosuite/Training_set",   # or test set
        task_name
    )

    images = []
    observations = {
        "obs_arm_joints": [],
        "obs_hand_joints": []
    }
    actions = {
        "target_arm_joints": [],
        "target_hand_joints": []
    }

    while True:
        # Reset the environment
        obs = env.reset()   
        
        # to be replaced with the realsense streaming
        frame = obs.get('agentview_image')
        if frame is None:
            raise ValueError("agentview_image is missing (obs['agentview_image'] is None).")
        
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        if is_collecting:
            # collecting the current image
            images.append(frame)

            # collecting the current joint angles
            observations['obs_arm_joints'].append(obs['robot0_joint_pos'][:7])
            observations['obs_hand_joints'].append(obs['robot0_right_gripper_qpos'])

            # collecting the action commands
            actions['target_arm_joints'].append(obs['robot0_joint_pos'][:7])
            actions['target_hand_joints'].append(obs['robot0_right_gripper_qpos'])

        # Setup rendering
        # cam_id = 0
        # num_cam = len(env.sim.model.camera_names)
        # env.render()

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
                
                # 15 dim, the first 6 are the pos and euler angles for the right arm, and next 6 are left arms, and then right gripper's open, and then left gripper's open
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]
                
                if is_collecting:
                    # collecting the action commands
                    actions['target_arm_joints'].append(obs['robot0_joint_pos'][:7])
                    actions['target_hand_joints'].append(obs['robot0_right_gripper_qpos'])

                obs, r, d, info = env.step(env_action)
                
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
                
                # test
                if step_count == 50:
                    save_act_imgs(actions, observations, images, save_path)
                    break

                # to be replaced with the realsense streaming
                frame = obs.get('agentview_image')
                if frame is None:
                    raise ValueError("agentview_image is missing (obs['agentview_image'] is None).")
                
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                if is_collecting:
                    # collecting the current image
                    images.append(frame)

                    # collecting the current joint angles
                    observations['obs_arm_joints'].append(obs['robot0_joint_pos'][:7])
                    observations['obs_hand_joints'].append(obs['robot0_right_gripper_qpos'])
