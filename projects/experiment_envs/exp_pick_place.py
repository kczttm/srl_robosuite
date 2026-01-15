import random
from collections import OrderedDict

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from projects.custom_objs.custom_arenas import BinsArena
from robosuite.models.objects import (
    BreadObject,
    BreadVisualObject,
    CanObject,
    CanVisualObject,
    CerealObject,
    CerealVisualObject,
    MilkObject,
    MilkVisualObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

import mujoco
import time
import h5py
import os
from projects.data import ContactLogger, setup_collision_tracking

class ExpPickPlace(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.6, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        
        bin1_pos=(0.3, -0.75, 0.8), # modified from original positions
        bin2_pos=(0.3, 0.5, 0.8),
        z_offset=0.0,
        z_rotation=None,
        single_object_mode=0,
        object_type=None,

        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
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
        log_data=True,
        log_filename=None,
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.object_id_to_sensors = {}  # Maps object id to sensor names for that object
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        if object_type is not None:
            assert object_type in self.object_to_id.keys(), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[object_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)
        self.z_offset = z_offset  # z offset for initializing items in bin
        self.z_rotation = z_rotation  # z rotation for initializing items in bin

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # HDF5 Log File initializer
        # Generate filename if not provided
        if log_filename is None:
            import datetime
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = f"explog_pickplace{current_time}.h5"

        self.log_data = log_data
        self.contact_logger = ContactLogger(
            log_filename=self.log_filename,
            control_freq=control_freq
        )

        # goal achieved check related variables (different from other exp environments)
        self.goal_achieved = False
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

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.2 # not entirely sure why these offsets are needed, but is necessary to spawn them within bounds - (Idris)
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.2

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=self.z_offset,
            )
        )

        # each visual object should just be at the center of each target bin
        index = 0
        for vis_obj in self.visual_objects:

            # get center of target bin
            bin_x_low = self.bin2_pos[0] - 0.1 # not entirely sure why this offset was needed, but is necessary to spawn them within bounds - (Idris)
            bin_y_low = self.bin2_pos[1]
            if index == 0 or index == 2:
                bin_x_low -= self.bin_size[0] / 2 - 0.2 # not entirely sure why this offset was needed, but is necessary to spawn them within bounds - (Idris)
            if index < 2:
                bin_y_low -= self.bin_size[1] / 2
            bin_x_high = bin_x_low + self.bin_size[0] / 2
            bin_y_high = bin_y_low + self.bin_size[1] / 2
            bin_center = np.array(
                [
                    (bin_x_low + bin_x_high) / 2.0,
                    (bin_y_low + bin_y_high) / 2.0,
                ]
            )

            # placement is relative to object bin, so compute difference and send to placement initializer
            rel_center = bin_center - self.bin1_pos[:2]

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{vis_obj.name}ObjectSampler",
                    mujoco_objects=vis_obj,
                    x_range=[rel_center[0], rel_center[0]],
                    y_range=[rel_center[1], rel_center[1]],
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.bin1_pos,
                    z_offset=self.bin2_pos[2] - self.bin1_pos[2],
                )
            )
            index += 1

    def _construct_visual_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
            (MilkVisualObject, BreadVisualObject, CerealVisualObject, CanVisualObject),
            self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        for obj_cls, obj_name in zip(
            (MilkObject, BreadObject, CerealObject, CanObject),
            self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos, table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size

        # make objects
        self._construct_visual_objects()
        self._construct_objects()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects,
        )

        # Generate placement initializer
        self._get_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in self.visual_objects + self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.0
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.0
            bin_x_low += self.bin_size[0] / 4.0
            bin_y_low += self.bin_size[1] / 4.0
            self.target_bin_placements[i, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]

        # contact logging debug - get geom names
        # print(self.sim.model.geom_names)

        environment_geoms = []

        for name in self.sim.model.geom_names:
            if "bin" in name:
                environment_geoms.append(name)

        self.collision_to_track_ids, self.robot_geom_ids, self.robot_geom_names = setup_collision_tracking(
            self.sim, self.robots, environment_geoms
        )


    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # Reset obj sensor mappings
            self.object_id_to_sensors = {}

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [
                self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)
            enableds = [True] * len(sensors)

            for i, obj in enumerate(self.objects):
                # Create object sensors
                using_obj = self.single_object_mode == 0 or self.object_id == i
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                enableds += [using_obj] * len(obj_sensor_names)
                actives += [using_obj] * len(obj_sensor_names)
                self.object_id_to_sensors[i] = obj_sensor_names

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def obj_id(obs_cache):
                    return self.object_id

                sensors.append(obj_id)
                names.append("obj_id")
                enableds.append(True)
                actives.append(True)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        sensors = [
            self._get_rel_obj_eef_sensor(arm_pf, obj_name, f"{obj_name}_to_{full_pf}eef_pos", full_pf, modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        sensors += [
            self._get_obj_eef_rel_quat_sensor(full_pf, obj_name, f"{obj_name}_to_{full_pf}eef_quat", modality)
            for full_pf in full_prefixes
        ]
        names = [fn.__name__ for fn in sensors]
        sensors += [obj_pos, obj_quat]
        names += [f"{obj_name}_pos", f"{obj_name}_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objects}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(obj_names))
            for obj_type, i in self.object_to_id.items():
                if obj_type.lower() in self.obj_to_use.lower():
                    self.object_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.object_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active object, else False
                    self._observables[name].set_enabled(i == self.object_id)
                    self._observables[name].set_active(i == self.object_id)

    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if all objects are placed correctly
        """
        # remember objects that are in the correct bins
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = min(
                [
                    np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos)
                    for arm in self.robots[0].arms
                ]
            )
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int((not self.not_in_bin(obj_pos, i)) and r_reach < 0.6)

        # returns True if a single object is in the correct bin
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # if the robot has multiple arms color each arm independently based on its closest object
            for arm in self.robots[0].arms:
                # find closest object
                dists = [
                    self._gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=obj.root_body,
                        target_type="body",
                        return_distance=True,
                    )
                    for obj in self.objects
                ]
                closest_obj_id = np.argmin(dists)
                # Visualize the distance to this target
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper[arm],
                    target=self.objects[closest_obj_id].root_body,
                    target_type="body",
                )

    def step(self, action):
        """
        Step the environment forward by one timestep, with logging
        """
        # different from other goal checking, use the built in function.
        current_time = self.step_count * (1.0 / self.control_freq)
        if self._check_success() and not self.goal_achieved:
            self.goal_achieved = True
            print(f"Goal achieved at time {current_time:.2f}!")
        if not self._check_success() and self.goal_achieved:
            self.goal_achieved = False
            print(f"Objects moved from goal position at time {current_time:.2f}!")

        if self.log_data:
            self.contact_logger.goal_set(self.goal_achieved, current_time)
            self.contact_logger.collect_step_data(
                self.sim,
                self.collision_to_track_ids,
                self.robot_geom_ids,
                self.robot_geom_names
            )

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

          - a discrete reward of 1.0 per object if it is placed in its correct bin

        Un-normalized components if using reward shaping, where the maximum is returned if not solved:

          - Reaching: in [0, 0.1], proportional to the distance between the gripper and the closest object
          - Grasping: in {0, 0.35}, nonzero if the gripper is grasping an object
          - Lifting: in {0, [0.35, 0.5]}, nonzero only if object is grasped; proportional to lifting height
          - Hovering: in {0, [0.5, 0.7]}, nonzero only if object is lifted; proportional to distance from object to bin

        Note that a successfully completed task (object in bin) will return 1.0 per object irregardless of whether the
        environment is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 4.0 (or 1.0 if only a single object is
        being used) as well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 4.0
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct bins
        active_objs = []
        for i, obj in enumerate(self.objects):
            if self.objects_in_bins[i]:
                continue
            active_objs.append(obj)

        # reaching reward governed by distance to closest object
        r_reach = 0.0
        if active_objs:
            # get reaching reward via minimum distance to a target object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_obj.root_body,
                    target_type="body",
                    return_distance=True,
                )
                for active_obj in active_objs
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[g for active_obj in active_objs for g in active_obj.contact_geoms],
                )
            )
            * grasp_mult
        )

        # lifting reward for picking up an object
        r_lift = 0.0
        if active_objs and r_grasp > 0.0:
            z_target = self.bin2_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, 2
            ]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover reward for getting object above bin
        r_hover = 0.0
        if active_objs:
            target_bin_ids = [self.object_to_id[active_obj.name.lower()] for active_obj in active_objs]
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, :2
            ]
            y_check = (
                np.abs(object_xy_locs[:, 1] - self.target_bin_placements[target_bin_ids, 1]) < self.bin_size[1] / 4.0
            )
            x_check = (
                np.abs(object_xy_locs[:, 0] - self.target_bin_placements[target_bin_ids, 0]) < self.bin_size[0] / 4.0
            )
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(self.target_bin_placements[target_bin_ids, :2] - object_xy_locs, axis=1)
            # objects to the left get r_lift added to hover reward,
            # those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(active_objs))
            r_hover_all[objects_above_bins] = lift_mult + (1 - np.tanh(10.0 * dists[objects_above_bins])) * (
                hover_mult - lift_mult
            )
            r_hover_all[objects_not_above_bins] = r_lift + (1 - np.tanh(10.0 * dists[objects_not_above_bins])) * (
                hover_mult - lift_mult
            )
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
        return res

    def close(self):
        if self.log_data:
            self.contact_logger.finalize()
        super().close()

class ExpPickPlaceSingle(ExpPickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class ExpPickPlaceMilk(ExpPickPlace):
    """
    Easier version of task - place one milk into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="milk", **kwargs)


class ExpPickPlaceBread(ExpPickPlace):
    """
    Easier version of task - place one bread into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="bread", **kwargs)


class ExpPickPlaceCereal(ExpPickPlace):
    """
    Easier version of task - place one cereal into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="cereal", **kwargs)


class ExpPickPlaceCan(ExpPickPlace):
    """
    Easier version of task - place one can into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="can", **kwargs)

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

# Main execution for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pick and Place Environment with XR or MediaPipe Teleoperation")
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
        env_name="ExpPickPlace",
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

            # Code to visualize contact points
            with viewer.lock(): # need to see if we can visualize only robot and gripper contacts
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            
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