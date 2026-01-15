from collections import OrderedDict

import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject
from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from projects.custom_objs import CenteredDoorObject, TableShadowArena

import mujoco
import time
import h5py
import os
from collections import OrderedDict

class ExpTwoArmLift(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch = True,
        table_full_size=(1.6, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        log_data=True, # for creating HDF5 log files
        log_filename=None, 
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.4))  # made changes

        # settings for door
        self.use_latch = use_latch

        # Omron LD-60 Mobile Base setting
        self.init_torso_height = 0.342

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # HDF5 Log File initializer
        self.log_data = log_data
        self.data_buffer = {
            'time': [],
            'table_contacts': [],
            'contact_details': []
        }

        if self.log_data:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            os.makedirs(data_dir, exist_ok=True)

            if log_filename is None:
                import datetime
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"explog_twoarmlift{current_time}.h5"
            
            self.log_filepath = os.path.join(data_dir, log_filename)
            self.step_count = 0

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableShadowArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.door = CenteredDoorObject(
            name="Door",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
        )

        self.pot = PotWithHandlesObject(name="pot")

        self.objs = [self.door, self.pot]
        self.objs_cls = (CenteredDoorObject, PotWithHandlesObject)
        self.objs_pos = ([-0.3, -0.3], [0, 0]) # range of x positions for each object (in this case just an exact position for each)

         # Create placement initializer
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

            for obj, x_val in zip(self.objs, self.objs_pos):
                self.placement_initializer.append_sampler(sampler=UniformRandomSampler(
                    name=f"{obj.name}Sampler",
                    x_range=x_val,
                    y_range=[0, 0],
                    rotation=(0,0),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement= True, # you can turn off for closeness but should be on if object ranges overlap
                    reference_pos=self.table_offset,
                    z_offset=0.22, 
                ))

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (obj_cls, obj_name) in enumerate(
            zip(
                self.objs_cls,
                [obj.name for obj in self.objs]
            )
        ):
            obj = self.objs[i]
            # Add this object to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add objs to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{obj_name}Sampler", mujoco_objects=obj)
            else:
                # This is assumed to be a flat sampler, so we just add all objs to this sampler
                self.placement_initializer.add_objects(obj)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objs,
        )

    def _setup_references(self):
        """
        Sets up references to important components
        """
        super()._setup_references()

        # Additional object references from this env

        # two handle pot
        self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)
        self.handle0_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle0"])
        self.handle1_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle1"])
        self.pot_center_id = self.sim.model.site_name2id(self.pot.important_sites["center"])

        # door
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.door.latch_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[1])

        # table
        self.table_top_id = self.sim.model.site_name2id("table_top")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name == "Door": # need to make a more general way to handle this
                    door_body_id = self.sim.model.body_name2id(self.door.root_body)
                    self.sim.model.body_pos[door_body_id] = obj_pos
                    self.sim.model.body_quat[door_body_id] = obj_quat
                else:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            # for obj_pos, obj_quat, obj in object_placements.values():
            #     self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _collect_step_data(self, action):
        """
        Collect time and contact data for logging
        """
        current_time = self.step_count * (1.0 / self.control_freq)

        contact_details = []
        robot_geom_ids = []
        robot_geom_names = []
        table_contact = False

        table_geom_id = self.sim.model.geom_name2id('table_collision')

        # get all robot geoms
        for robot in self.robots:
            for geom_name in robot.robot_model.contact_geoms:
                try:
                    geom_id = self.sim.model.geom_name2id(geom_name)
                    robot_geom_ids.append(geom_id)
                    robot_geom_names.append(geom_name)
                except:
                    # if nonexistent skip
                    continue

        # check all active contacts
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            # any table or robot part?
            robot_geom = None
            if contact.geom1 == table_geom_id and contact.geom2 in robot_geom_ids:
                table_contact = True
                robot_geom_idx = robot_geom_ids.index(contact.geom2)
                robot_geom = robot_geom_names[robot_geom_idx]
            elif contact.geom2 == table_geom_id and contact.geom1 in robot_geom_ids:
                table_contact = True
                robot_geom_idx = robot_geom_ids.index(contact.geom1)
                robot_geom = robot_geom_names[robot_geom_idx]

            if robot_geom:
                # get force data
                force_vector = np.zeros(6)
                mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, i, force_vector)
                contact_force_magnitude = np.linalg.norm(force_vector[:3])

                contact_details.append({
                    'robot_geom': robot_geom,
                    'contact_pos': contact.pos.copy(),
                    'contact_force_magnitude': contact_force_magnitude
                })

        # Append data to buffer
        self.data_buffer['time'].append(current_time)
        self.data_buffer['table_contacts'].append(table_contact)
        self.data_buffer['contact_details'].append(contact_details)

        self.step_count += 1

        if len(self.data_buffer['time']) % 1000 == 0:
            self._write_buffer_to_file()

    def _write_buffer_to_file(self):
        """
        Writes the collected data buffer to a file.
        """
        if not self.log_data or len(self.data_buffer['time']) == 0:
            return
        
        with h5py.File(self.log_filepath, 'a') as f:
            if 'time' not in f:
                f.create_dataset('time', data=np.array(self.data_buffer['time']), maxshape=(None,), chunks=True)

                f.create_dataset('table_contacts', data=np.array(self.data_buffer['table_contacts']), maxshape=(None,), chunks=True, dtype='bool')

                contact_summary = [str(len(details)) for details in self.data_buffer['contact_details']]
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('contact_details', data=np.array(contact_summary), maxshape=(None,), chunks=True, dtype=dt)

            else:
                # Append to existing datasets
                datasets = ['time', 'table_contacts']
                data_keys = ['time', 'table_contacts']
                
                for dataset_name, data_key in zip(datasets, data_keys):
                    dset = f[dataset_name]
                    old_size = dset.shape[0]
                    new_size = old_size + len(self.data_buffer[data_key])
                    dset.resize((new_size,))
                    dset[old_size:] = self.data_buffer[data_key]
                
                # Handle contact count
                contact_summary = [str(len(details)) for details in self.data_buffer['contact_details']]
                if 'contact_count' in f:
                    dset = f['contact_count']
                    old_size = dset.shape[0]
                    new_size = old_size + len(contact_summary)
                    dset.resize((new_size,))
                    dset[old_size:] = contact_summary
        
        # clear buffer
        for key in self.data_buffer:
            self.data_buffer[key] = []

    def step(self, action):
        """
        Step the environment forward by one timestep, with added logging.
        Args:
            action (np.array): Action to execute within the environment
        """
        if self.log_data:
            self._collect_step_data(action)

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def reward(self, action):
        """
        Placeholder reward function
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            float: Reward from environment
        """
        # For now, return a constant reward of 0 since we're just observing
        return 0.0
    
    def close(self):
        if self.log_data:
            self._write_buffer_to_file()
        super().close()

class TimeKeeper:
    def __init__(self, desired_freq=60):
        self.period = 1.0 / desired_freq
        self.last_time = time.perf_counter()
        self.time_accumulator = 0
        self.frame_count = 0
        self.start_time = self.last_time

    def should_step(self):
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        self.time_accumulator += frame_time
        return self.time_accumulator >= self.period

    def consume_step(self):
        self.time_accumulator -= self.period
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Two Arm Lift Environment with XR or MediaPipe Teleoperation")
    parser.add_argument("--device", type=str, default="mediapipe", choices=["XR", "mediapipe"], 
                       help="Teleoperation device to use (default: mediapipe)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID for MediaPipe (default: 0)")
    parser.add_argument("--mirror_actions", action="store_true", help="Mirror actions for MediaPipe")
    parser.add_argument("--max_fr", default=30, type=int, help="Maximum frame rate")
    parser.add_argument("--collision_filtering", action="store_true", default=True, 
                       help="Enable collision filtering (default: enabled)")
    parser.add_argument("--no_log", action="store_true",
                        help="Disable logging")
    args = parser.parse_args()

    simulation_time = 10.0 # seconds
    env_step_size = 0.0001 # seconds
    horizon = int(simulation_time / env_step_size)

    # Import common modules
    from robosuite import load_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper
    from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller import AbilityHandController
    from projects.dual_kinova3_teleop.controllers.sew_mimic_to_joint_position import (
        initialize_sew_converters_for_robot,
        update_sew_converter_origins_from_robot,
        convert_sew_actions_to_joint_positions
    )
    from projects.psyonic_hand_teleop.ability_hand.dual_kinova3_robot_psyonic_gripper import DualKinova3PsyonicHand

    # Import device-specific modules
    if args.device == "XR":
        from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
    elif args.device == "mediapipe":
        from projects.shared_devices.mediapipe_teleop_device_robosuite_ver import MediaPipeTeleop

    # Get robosuite path for config
    repo_path = os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    )
    dual_kinova3_sew_config_path = os.path.join(
        repo_path, "SEW-Geometric-Teleop", "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_sew_mimic_joint_position.json"
    )

    # Create controller configuration
    controller_config = load_composite_controller_config(
        controller=dual_kinova3_sew_config_path,
        robot="DualKinova3PsyonicHand",
    )
    
    env = suite.make(
        env_name="ExpTwoArmLift",
        robots="DualKinova3PsyonicHand",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=30,
        horizon=horizon,
        log_data=not args.no_log,
    )
    env = VisualizationWrapper(env)

    # Initialize device based on selection
    print(f"Initializing {args.device} device...")
    if args.device == "XR":
        device = XRRTCBodyPoseDevice(env=env)
        print("Waiting for VR client connection...")
        while not device.is_connected:
            time.sleep(0.5)
        print("Client connected!")
    elif args.device == "mediapipe":
        device = MediaPipeTeleop(
            env=env,
            debug=args.debug,
            camera_id=args.camera_id,
            mirror_actions=args.mirror_actions
        )
        device.start_control()
        print("MediaPipe device initialized!")
        print("Setup Instructions:")
        print("1. Position yourself in front of the camera")
        print("2. Make sure your full upper body is visible")
        print("3. Raise both arms to shoulder height to start control")
        print("4. Lower arms to stop control")
        print("Camera Controls:")
        print("- 'q' key: Quit")

    # Reset the environment and get robot
    obs = env.reset()
    robot = env.robots[0]
    
    # Initialize SEW converters using helper function with collision filtering
    print("Initializing SEW converters with collision avoidance enabled...")
    sew_converters = initialize_sew_converters_for_robot(robot, env.sim, enable_collision_filtering=True)
    # Update SEW converter origins using helper function
    update_sew_converter_origins_from_robot(sew_converters, robot)

    # Get model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    print("Initializing hand controllers...")
    left_hand_controller = AbilityHandController(model, data, hand_side='left', debug=False)
    right_hand_controller = AbilityHandController(model, data, hand_side='right', debug=False)
    
    print("Starting teleoperation...")
    print("=" * 60)
    if args.device == "XR":
        print("XR Teleoperation: Move your arms and hands to control the robots!")
    elif args.device == "mediapipe":
        print("MediaPipe Teleoperation: Use your body movements to control the robots!")
    print("=" * 60)
    
    # Set smaller timestep for more accurate physics simulation
    # model.opt.timestep = env_step_size  # commented out to use the default timestep

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) as viewer:
        # Set initial camera parameters
        viewer.opt.geomgroup[0] = 0  # Hide collision meshes
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        step_count = 0

        while viewer.is_running():
            start_time = time.time()
            
            # Check for quit signal first (common for both devices)
            if hasattr(device, 'should_quit') and device.should_quit():
                print("Quit signal detected. Stopping simulation...")
                break
            
            try:
                # Get input from device (unified approach)
                if args.device == "XR":
                    input_ac_dict = device.get_controller_state()
                elif args.device == "mediapipe":
                    try:
                        input_ac_dict = device.input2action()
                    except Exception as e:
                        if args.debug:
                            print(f"Error getting MediaPipe input: {e}")
                        input_ac_dict = None
                
                # Process input if available (common logic for both devices)
                if input_ac_dict is not None:
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

                    # Convert SEW actions to joint positions (common for both devices)
                    action_dict = convert_sew_actions_to_joint_positions(
                        sew_converters, input_ac_dict, robot, env.sim, 
                        safety_layer=args.collision_filtering, debug=args.debug,
                        human_hands_centroid=human_finger_mcp_centroid,
                        robot_tcp_centroid=robot_tcp_centroid,
                        max_displacement=0.8
                    )
                    
                    # Handle gripper/hand control (common logic)
                    for arm in robot.arms:
                        hand_controller = left_hand_controller if arm == 'left' else right_hand_controller
                        
                        # Use finger data if available (works for both XR and MediaPipe)
                        if f"{arm}_fingers" in input_ac_dict and input_ac_dict[f"{arm}_fingers"]:
                            finger_positions = input_ac_dict[f"{arm}_fingers"]
                            hand_controller.compute_hand_joint_goals_from_fingers(finger_positions)
                        else:
                            hand_controller.set_joint_goals(np.zeros(10))  # Home position
                        
                        # Compute control torques for the hand
                        action_dict[f"{arm}_gripper"] = hand_controller.compute_control_torques()

                    # Execute action
                    env_action = robot.create_action_vector(action_dict)
                    env.step(env_action)
                    
                else:
                    # No input available - hold position
                    env.step(np.zeros(env.action_dim))
                    if step_count % 300 == 0:
                        if args.device == "XR":
                            print("Waiting for XR data...")
                        elif args.device == "mediapipe":
                            print("Waiting for human engagement (raise both arms to shoulder height)")
                    
            except Exception as e:
                if args.debug:
                    print(f"Error: {e}")
                env.step(np.zeros(env.action_dim))
            
            # Sync viewer and maintain framerate
            viewer.sync()
            
            # Update hand controller data references periodically
            if step_count % 100 == 0:
                left_hand_controller.update_data_reference(data)
                right_hand_controller.update_data_reference(data)
            
            # Maintain target framerate (30 FPS)
            elapsed = time.time() - start_time
            if elapsed < 1 / 30:
                time.sleep(1 / 30 - elapsed)
            
            step_count += 1
            
            if step_count % 300 == 0 and args.debug:
                print(f"Step {step_count} - Running smoothly")
                
    # Cleanup
    print("Cleaning up...")
    if hasattr(device, 'stop'):
        device.stop()
    env.close()
    print("Demo completed.")