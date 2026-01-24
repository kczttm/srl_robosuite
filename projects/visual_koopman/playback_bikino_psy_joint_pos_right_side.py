"""
Playback script for dual Kinova3 robot with Psyonic Ability hands using recorded joint positions.
Reads from a .npy file and replays the motion on the Right Arm and Hand in simulation.

Structure based on demo_dual_kinova3_psyonic_xr_teleop.py.
"""

import argparse
import os
import time
import numpy as np
import mujoco
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# Import the reader device
from projects.shared_devices.koopman_bikino_psy_right_side_joint_pos_reader_device import KoopmanBiKinoPsyRightSideJointPosReaderDevice

# Import Hand Controller
# from projects.psyonic_hand_teleop.ability_hand.ability_hand_controller import AbilityHandController

# Import robot class (needed for registration usually, but here just importing to ensure accessible if needed)
from projects.psyonic_hand_teleop.ability_hand.dual_kinova3_robot_psyonic_gripper import DualKinova3PsyonicHand

# Register custom environments
import projects.experiment_envs

def get_repo_path():
    return os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir)
    )

def create_environment(args):
    repo_path = get_repo_path()
    
    # Specific config path as requested
    dual_kinova3_config_path = os.path.join(
        repo_path, 
        "projects", "dual_kinova3_teleop", "controllers", "config", "robots", "dualkinova3_sew_mimic_joint_position.json"
    )
    
    controller_config = load_composite_controller_config(
        controller=dual_kinova3_config_path,
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
    env.table_offset = np.array((0.0, 0, 0.4))
    return env

def get_intersection_circles(o0, r0, o1, r1):
    """
    Helper function for the above function. 
    Solves for the intersection of two circles.
    
    Behavior of this function for circles that intersect at only one point, or 
    circles that do not intersect, is not defined.
    
    INPUTS: 
        o0: origin of circle 0
        r0: radius of circle 0
        o1: origin of circle 1
        r1: origin of circle 1
    OUTPUS:
        sol0: 2d position of the first intersection point
        sol1: 2d position of the second intersection point
    """
    d = np.sqrt(np.sum((o0 - o1) ** 2))

    sol0 = np.zeros(2)
    sol1 = np.zeros(2)

    r0_sq = r0 * r0
    r1_sq = r1 * r1
    d_sq = d * d

    # solve for a
    a = (r0_sq - r1_sq + d_sq) / (2 * d)

    # solve for h
    h_sq = r0_sq - a * a
    
    # Safety check: ensure h_sq is non-negative before taking sqrt
    if h_sq < 0:
        # Circles don't intersect - use approximation by moving them closer
        # This handles edge cases in the 4-bar linkage calculation
        h_sq = 0.0  # Set to zero for minimal perturbation
    
    h = np.sqrt(h_sq)

    # find p2
    p2 = o0 + a * (o1 - o0) / d

    t1 = h * (o1[1] - o0[1]) / d
    t2 = h * (o1[0] - o0[0]) / d

    sol0[0] = p2[0] + t1
    sol0[1] = p2[1] - t2

    sol1[0] = p2[0] - t1
    sol1[1] = p2[1] + t2

    return sol0, sol1


def get_abh_4bar_driven_angle(q1):
    q1 = (
        q1 + 0.084474
    )  # factor in offset imposed by our choice of link frame attachments

    # L0 = 9.5
    L1 = 38.6104
    L2 = 36.875
    L3 = 9.1241
    p3 = np.array(
        [9.47966, -0.62133, 0]
    )  # if X of the base frame was coincident with L3, p3 = [9.5, 0 0]. However, our frame choices are different to make the 0 references for the fingers nice, so this location is a little less convenient.

    cq1 = np.cos(q1)
    sq1 = np.sin(q1)
    p1 = np.array([L1 * cq1, L1 * sq1, 0])

    sol0, sol1 = get_intersection_circles(p3, L2, p1, L3)

    # copy_vect3(&p2, &sols[1])
    p2 = sol1

    # calculate the linkage intermediate angle!
    q2pq1 = np.arctan2(p2[1] - L1 * sq1, p2[0] - L1 * cq1)
    q2 = q2pq1 - q1
    q2 = np.mod(q2 + np.pi, 2 * np.pi) - np.pi
    return q2


def map_6dof_hand_to_10dof_sim(hand_pos_6dof):
    """
    Map 6 DOF hardware hand positions to 10 DOF simulation joints.
    Assumption:
    Input: [Index, Middle, Ring, Pinky, ThumbFlex, ThumbRot] (Typical Psyonic mapping)
    Output: [Th_q1, Th_q2, In_q1, In_q2, Mi_q1, Mi_q2, Ri_q1, Ri_q2, Pi_q1, Pi_q2]
    
    Mapping strategy:
    - Finger Flexion (Index, Middle, Ring, Pinky) maps to q1. q2 is computed via 4-bar linkage kinematics.
    - Thumb Flex maps to q1.
    - Thumb Rot maps to q2.
    """
    if len(hand_pos_6dof) != 6:
        return np.zeros(10)

    # Unpack (Assuming standard order)
    # Convert degrees to radians (Hardware reports 0-100 deg approx)
    idx_flex_deg = hand_pos_6dof[0]
    mid_flex_deg = hand_pos_6dof[1]
    rng_flex_deg = hand_pos_6dof[2]
    pnk_flex_deg = hand_pos_6dof[3]
    # Based on AbilityHandControllerTeensy mapping:
    # HW[4] comes from Sim[1] (thumb_q2)
    # HW[5] comes from Sim[0] (thumb_q1)
    thb_q2_deg = hand_pos_6dof[4]
    thb_q1_deg = hand_pos_6dof[5]

    idx_flex = np.radians(idx_flex_deg)
    mid_flex = np.radians(mid_flex_deg)
    rng_flex = np.radians(rng_flex_deg)
    pnk_flex = np.radians(pnk_flex_deg)
    thb_q1 = np.radians(thb_q1_deg)
    thb_q2 = np.radians(thb_q2_deg)

    joints = np.zeros(10)
    
    # Thumb (0, 1) - Corrected mapping based on controller
    joints[0] = thb_q1
    joints[1] = thb_q2

    # Index (2, 3)
    joints[2] = idx_flex
    joints[3] = get_abh_4bar_driven_angle(idx_flex)

    # Middle (4, 5)
    joints[4] = mid_flex
    joints[5] = get_abh_4bar_driven_angle(mid_flex)

    # Ring (6, 7)
    joints[6] = rng_flex
    joints[7] = get_abh_4bar_driven_angle(rng_flex)

    # Pinky (8, 9)
    joints[8] = pnk_flex
    joints[9] = get_abh_4bar_driven_angle(pnk_flex)
    
    return joints

def main():
    parser = argparse.ArgumentParser(description="Playback Dual Kinova3 + Psyonic Right Side Joint Positions")
    parser.add_argument("--environment", type=str, default="Lift", help="Environment to use")
    parser.add_argument("--npy_file", type=str, default="projects/data/action.npy", help="Path to .npy file with joint positions")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    print("Creating environment...")
    env = create_environment(args)
    env.reset()
    
    # Setup visualization wrapper
    # Using Sew visualization wrapper not requested, just plain env or VisualizationWrapper
    # The prompt says "similar to demo_dual_kinova3_psyonic_xr_teleop.py"
    # But says "doesn't use sew_mimic". 
    # I'll adding basic wrapper if needed, but env.render() works if has_renderer=True.
    
    robot = env.robots[0]
    
    # Initialize Hand Controllers
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Initialize Reader Device
    print(f"Initializing playback device with {args.npy_file}...")
    playback_device = KoopmanBiKinoPsyRightSideJointPosReaderDevice(
        npy_path=args.npy_file, 
        frequency=30, 
        loop=args.loop
    )
    
    step_count = 0
    t_last = time.time()
    
    print("Starting playback...")
    
    # We need to maintain left arm position. 
    # Let's grab initial position.
    # The robot composite action structure:
    # It takes a flat vector constructed from separate parts.
    # We can use robot.create_action_vector(action_dict).
    
    # Initial left arm config
    # We can get it from observing the robot state
    # But getting just the joint positions might be tricky without hardcoded indices.
    # However, if we just send zeros to a POSITION controller, it will move to 0.
    # We want it to stay put. 
    # We'll read the current qpos for left arm joints.
    
    # Helper to find current joint pos
    # We assume standard naming convention for DualKinova3 in Robosuite
    # Prefix is usually 'robot0_' + 'left_' + 'joint_' + i
    # Let's try to detect prefix dynamically or fallback
    bikino_psy = DualKinova3PsyonicHand()
    current_left_q = bikino_psy.init_qpos[9:16]  # Left arm joints are indices 9-15 in init_qpos
    
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) as viewer:
        viewer.opt.geomgroup[0] = 0  # Hide collision meshes
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]

        while viewer.is_running():
            t_now = time.time()
            
            # Get next frame
            state = playback_device.get_controller_state()
            
            if state is None:
                print("Playback finished.")
                break
                
            right_arm_q = state['right_arm_positions']
            right_hand_q_6dof = state['right_hand_positions']
            
            # Map hand
            right_hand_goals = map_6dof_hand_to_10dof_sim(right_hand_q_6dof)
            
            # Build action dict
            action_dict = {}
            
            # Right Arm
            action_dict['right'] = right_arm_q
            
            # Left Arm (Stationary)
            # Ensure we send the position we started at, so it stays there.
            action_dict['left'] = current_left_q
            
            # Hands
            action_dict['right_gripper'] = right_hand_goals
            action_dict['left_gripper'] = np.zeros(10)
            
            # Step environment
            action_vector = robot.create_action_vector(action_dict)
            env.step(action_vector)
            
            # Sync viewer and maintain framerate
            viewer.sync()

            # Sync to 30Hz
            elapsed = time.time() - t_now
            if elapsed < 1.0/30.0:
                time.sleep(1.0/30.0 - elapsed)
                
            step_count += 1
            
            # Update hand controller references periodically
            # if step_count % 100 == 0:
            #     right_hand_controller.update_data_reference(env.sim.data._data)
            #     left_hand_controller.update_data_reference(env.sim.data._data)

    env.close()

if __name__ == "__main__":
    main()
