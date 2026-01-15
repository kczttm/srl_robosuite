import csv
import os
import threading
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation
from xr_robot_teleop_server import configure_logging
from xr_robot_teleop_server.schemas.body_pose import (
    Bone,
    deserialize_pose_data,
)
from xr_robot_teleop_server.schemas.openxr_skeletons import FullBodyBoneId
from xr_robot_teleop_server.streaming import WebRTCServer

from robosuite.devices import Device


# Finger bones for hand tracking
simplified_finger_bones = [
    FullBodyBoneId.FullBody_LeftHandThumbMetacarpal,
    FullBodyBoneId.FullBody_LeftHandThumbProximal,
    FullBodyBoneId.FullBody_LeftHandThumbDistal,
    FullBodyBoneId.FullBody_LeftHandThumbTip,
    FullBodyBoneId.FullBody_LeftHandIndexProximal,
    FullBodyBoneId.FullBody_LeftHandIndexIntermediate,
    FullBodyBoneId.FullBody_LeftHandIndexTip,
    FullBodyBoneId.FullBody_LeftHandMiddleProximal,
    FullBodyBoneId.FullBody_LeftHandMiddleIntermediate,
    FullBodyBoneId.FullBody_LeftHandMiddleTip,
    FullBodyBoneId.FullBody_LeftHandRingProximal,
    FullBodyBoneId.FullBody_LeftHandRingIntermediate,
    FullBodyBoneId.FullBody_LeftHandRingTip,
    FullBodyBoneId.FullBody_LeftHandLittleProximal,
    FullBodyBoneId.FullBody_LeftHandLittleIntermediate,
    FullBodyBoneId.FullBody_LeftHandLittleTip,
    FullBodyBoneId.FullBody_RightHandThumbMetacarpal,
    FullBodyBoneId.FullBody_RightHandThumbProximal,
    FullBodyBoneId.FullBody_RightHandThumbDistal,
    FullBodyBoneId.FullBody_RightHandThumbTip,
    FullBodyBoneId.FullBody_RightHandIndexProximal,
    FullBodyBoneId.FullBody_RightHandIndexIntermediate,
    FullBodyBoneId.FullBody_RightHandIndexTip,
    FullBodyBoneId.FullBody_RightHandMiddleProximal,
    FullBodyBoneId.FullBody_RightHandMiddleIntermediate,
    FullBodyBoneId.FullBody_RightHandMiddleTip,
    FullBodyBoneId.FullBody_RightHandRingProximal,
    FullBodyBoneId.FullBody_RightHandRingIntermediate,
    FullBodyBoneId.FullBody_RightHandRingTip,
    FullBodyBoneId.FullBody_RightHandLittleProximal,
    FullBodyBoneId.FullBody_RightHandLittleIntermediate,
    FullBodyBoneId.FullBody_RightHandLittleTip,
]


def _get_body_frame(
    bone_positions: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates the body-centric coordinate frame (origin and rotation matrix).

    Args:
        bone_positions (dict): A dictionary mapping bone IDs to their positions.

    Returns:
        A tuple containing:
        - np.ndarray: The origin of the body frame (shoulder_center).
        - np.ndarray: The rotation matrix from world to body frame (R_world_body).
        Returns (None, None) if essential bones are missing.
    """
    # Get key body landmarks
    left_shoulder = bone_positions.get(FullBodyBoneId.FullBody_LeftArmUpper)
    right_shoulder = bone_positions.get(FullBodyBoneId.FullBody_RightArmUpper)
    # spine_upper = bone_positions.get(FullBodyBoneId.FullBody_SpineUpper) # not the best
    left_hip = bone_positions.get(FullBodyBoneId.FullBody_LeftUpperLeg)
    right_hip = bone_positions.get(FullBodyBoneId.FullBody_RightUpperLeg)

    if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
        return None, None

    # Calculate body center (origin of the body frame)
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    # Create body-centric coordinate frame
    # Y-axis: right to left (shoulder line)
    y_axis = left_shoulder - right_shoulder
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    torso_vector = shoulder_center - hip_center

    # X-axis: forward direction (cross product)
    x_axis = np.cross(y_axis, torso_vector)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Z-axis: up direction (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Create transformation matrix from world to body-centric frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    # normalize the rotation matrix
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_body = U_rot @ V_rot_T

    return shoulder_center, R_world_body


def _get_lower_body_frame(
    bone_positions: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates the lower body-centric coordinate frame (origin and rotation matrix).

    Args:
        bone_positions (dict): A dictionary mapping bone IDs to their positions.

    Returns:
        A tuple containing:
        - np.ndarray: The origin of the lower body frame (hip_center).
        - np.ndarray: The rotation matrix from world to lower body frame (R_world_lower_body).
        Returns (None, None) if essential bones are missing.
    """
    # Get key lower body landmarks
    left_hip = bone_positions.get(FullBodyBoneId.FullBody_LeftUpperLeg)
    right_hip = bone_positions.get(FullBodyBoneId.FullBody_RightUpperLeg)
    spine_middle = bone_positions.get(FullBodyBoneId.FullBody_SpineMiddle)
    spine_lower = bone_positions.get(FullBodyBoneId.FullBody_SpineLower)

    if left_hip is None or right_hip is None or spine_lower is None:
        return None, None

    # Calculate lower body center (origin of the lower body frame)
    hip_center = (left_hip + right_hip) / 2

    # Create lower body-centric coordinate frame
    # Y-axis: right to left (hip line)
    y_axis = left_hip - right_hip
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    torso_vector = spine_middle - spine_lower

    # X-axis: forward direction (cross product)
    x_axis = np.cross(y_axis, torso_vector)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Z-axis: up direction (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Create transformation matrix from world to lower body-centric frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    # normalize the rotation matrix
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_lower_body = U_rot @ V_rot_T

    return hip_center, R_world_lower_body


def _get_body_centric_coordinates(bones: list[Bone], get_wrist_rot_from_hand=True) -> dict:
    """
    Convert bone positions to a body-centric coordinate system for dual Kinova3 robot.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    shoulder_center, R_world_body = _get_body_frame(bone_positions)

    if shoulder_center is None or R_world_body is None:
        return None

    def transform_to_body_frame(world_pos):
        """Transform a world position to body-centric coordinates."""
        translated = world_pos - shoulder_center
        body_pos = R_world_body.T @ translated
        return body_pos

    # Extract SEW coordinates in body-centric frame
    sew_coordinates = {}

    for side in ["left", "right"]:
        side_key_pascal = side.capitalize()

        shoulder_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}ArmUpper")
        elbow_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}ArmLower")
        wrist_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}HandWrist")

        shoulder_pos = bone_positions.get(shoulder_id)
        elbow_pos = bone_positions.get(elbow_id)
        wrist_pos = bone_positions.get(wrist_id)

        if shoulder_pos is None or elbow_pos is None or wrist_pos is None:
            sew_coordinates[side] = None
            continue

        # Transform to body-centric coordinates
        S_body = transform_to_body_frame(shoulder_pos)
        E_body = transform_to_body_frame(elbow_pos)
        W_body = transform_to_body_frame(wrist_pos)

        # Get wrist rotation
        wrist_rot = (
            q2R(bone_rotations.get(wrist_id)) if wrist_id in bone_rotations else None
        )
        if not get_wrist_rot_from_hand and wrist_rot is not None:
            # Convert wrist rotation to body frame
            wrist_rot = R_world_body.T @ wrist_rot
            if side == "left":  # Z in palm, -X in thumb, Y in fingers pointing
                wrist_rot = (
                    wrist_rot
                    @ Rotation.from_euler("zyx", [np.pi / 2, -np.pi / 2, 0]).as_matrix()
                )
            else:  # right arm: -Z in palm, X in thumb, -Y in fingers pointing
                wrist_rot = (
                    wrist_rot
                    # @ Rotation.from_euler("zyx", [-np.pi / 2, np.pi / 2, 0]).as_matrix()  # ORIG
                    @ Rotation.from_euler("zyx", [np.pi / 2, -np.pi / 2, 0]).as_matrix()  # TEMP NOTE: Only for FBX-based data; This breaks normal usage
                )
        else:
            wrist_rot = R_world_body.T @ _get_hand_frame(side=side, bone_positions=bone_positions)
        body_frame_wrist_rot = wrist_rot

        sew_coordinates[side] = {
            "S": S_body,
            "E": E_body,
            "W": W_body,
            "wrist_rot": body_frame_wrist_rot.flatten(),
        }

        sew_coordinates['R_world_body'] = R_world_body

    return sew_coordinates


def _get_lower_body_centric_coordinates(bones: list[Bone]) -> dict:
    """
    Convert bone positions to a lower body-centric coordinate system.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    hip_center, R_world_lower_body = _get_lower_body_frame(bone_positions)

    if hip_center is None or R_world_lower_body is None:
        return None

    def transform_to_lower_body_frame(world_pos):
        """Transform a world position to lower body-centric coordinates."""
        translated = world_pos - hip_center
        lower_body_pos = R_world_lower_body.T @ translated
        return lower_body_pos

    # Extract key lower body coordinates in lower body-centric frame
    lower_body_coordinates = {}

    for side in ["left", "right"]:
        side_key_pascal = side.capitalize()

        hip_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}UpperLeg")
        knee_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}LowerLeg")
        ankle_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootAnkle")

        hip_pos = bone_positions.get(hip_id)
        knee_pos = bone_positions.get(knee_id)
        ankle_pos = bone_positions.get(ankle_id)

        if hip_pos is None or knee_pos is None or ankle_pos is None:
            lower_body_coordinates[side] = None
            continue

        # Transform to lower body-centric coordinates
        H_lower_body = transform_to_lower_body_frame(hip_pos)
        K_lower_body = transform_to_lower_body_frame(knee_pos)
        A_lower_body = transform_to_lower_body_frame(ankle_pos)

        # Get Foot Rotation from bone poses
        foot_ball_wd = bone_positions.get(getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootBall"))
        foot_trans_wd = bone_positions.get(getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootTransverse"))
        foot_heel_wd = bone_positions.get(getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootSubtalar"))

        if foot_ball_wd is None or foot_trans_wd is None or foot_heel_wd is None:
            R_body_foot = None
        else:
            foot_ball_pos = transform_to_lower_body_frame(foot_ball_wd)
            foot_trans_pos = transform_to_lower_body_frame(foot_trans_wd)
            foot_heel_pos = transform_to_lower_body_frame(foot_heel_wd)

            foot_x_axis = foot_ball_pos - foot_heel_pos
            foot_x_axis /= np.linalg.norm(foot_x_axis) + 1e-8

            trans_heel_vec = foot_trans_pos - foot_heel_pos
            foot_y_axis = np.cross(trans_heel_vec, foot_x_axis)
            foot_y_axis /= np.linalg.norm(foot_y_axis) + 1e-8

            foot_z_axis = np.cross(foot_x_axis, foot_y_axis)
            foot_z_axis /= np.linalg.norm(foot_z_axis) + 1e-8

            rotation = np.column_stack([foot_x_axis, foot_y_axis, foot_z_axis])
            U_rot, _, V_rot_T = np.linalg.svd(rotation)
            R_body_foot = U_rot @ V_rot_T

        lower_body_coordinates[side] = {
            "H": H_lower_body,
            "K": K_lower_body,
            "A": A_lower_body,
            "ankle_rot": R_body_foot,
        }

    lower_body_coordinates['R_world_lower_body'] = R_world_lower_body

    return lower_body_coordinates


def _get_finger_coordinates(bones: list[Bone], coord_type: str, bones_to_return: list[int]) -> dict:
    """
    Get finger bone coordinates in either absolute, body-centric, or hand-centric frame.

    Args:
        bones (list[Bone]): A list of bone objects from the XR device.
        coord_type (str): The coordinate system to use ("absolute", "body_centric", or "hand_centric").
        bones_to_return (list[int]): A list of FullBodyBoneId enums for which to return coordinates.

    Returns:
        dict: A dictionary mapping bone IDs to their coordinates.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    returned_coords = {}

    if coord_type == "absolute":
        for bone_id in bones_to_return:
            if bone_id in bone_positions:
                returned_coords[bone_id] = bone_positions[bone_id]
            else:
                returned_coords[bone_id] = None
        return returned_coords

    elif coord_type == "body_centric":
        shoulder_center, R_world_body = _get_body_frame(bone_positions)

        if shoulder_center is None or R_world_body is None:
            # Cannot compute body frame, return None for all requested bones
            for bone_id in bones_to_return:
                returned_coords[bone_id] = None
            return returned_coords

        def transform_to_body_frame(world_pos):
            """Transform a world position to body-centric coordinates."""
            translated = world_pos - shoulder_center
            body_pos = R_world_body.T @ translated
            return body_pos

        for bone_id in bones_to_return:
            if bone_id in bone_positions:
                world_pos = bone_positions[bone_id]
                returned_coords[bone_id] = transform_to_body_frame(world_pos)
            else:
                returned_coords[bone_id] = None
        return returned_coords

    elif coord_type == "hand_centric":
        # First transform to body-centric coordinates (like other coordinate types)
        shoulder_center, R_world_body = _get_body_frame(bone_positions)

        if shoulder_center is None or R_world_body is None:
            # Cannot compute body frame, return None for all requested bones
            for bone_id in bones_to_return:
                returned_coords[bone_id] = None
            return returned_coords

        def transform_to_body_frame(world_pos):
            """Transform a world position to body-centric coordinates."""
            translated = world_pos - shoulder_center
            body_pos = R_world_body.T @ translated
            return body_pos

        # Get SEW coordinates which contain wrist rotation information in body frame
        sew_coords = _get_body_centric_coordinates(bones)

        if sew_coords is None or sew_coords["left"] is None or sew_coords["right"] is None:
            # Cannot compute hand frame, return None for all requested bones
            for bone_id in bones_to_return:
                returned_coords[bone_id] = None
            return returned_coords

        # Get wrist positions in body frame (from SEW coordinates)
        left_wrist_pos_body = sew_coords["left"]["W"]  # W is wrist position in body frame
        right_wrist_pos_body = sew_coords["right"]["W"]

        # Extract wrist rotation matrices (reshape from flattened form) - already in body frame
        left_wrist_rot = sew_coords["left"]["wrist_rot"].reshape(3, 3) if sew_coords["left"]["wrist_rot"] is not None else None
        right_wrist_rot = sew_coords["right"]["wrist_rot"].reshape(3, 3) if sew_coords["right"]["wrist_rot"] is not None else None

        for bone_id in bones_to_return:
            bone_name = FullBodyBoneId(bone_id).name
            world_pos = bone_positions.get(bone_id)

            if world_pos is None:
                returned_coords[bone_id] = None
                continue

            # First transform world position to body frame
            body_pos = transform_to_body_frame(world_pos)

            if "LeftHand" in bone_name and left_wrist_pos_body is not None and left_wrist_rot is not None:
                # Transform from body-centric to hand-centric coordinates
                translated = body_pos - left_wrist_pos_body
                returned_coords[bone_id] = left_wrist_rot.T @ translated
            elif "RightHand" in bone_name and right_wrist_pos_body is not None and right_wrist_rot is not None:
                # Transform from body-centric to hand-centric coordinates
                translated = body_pos - right_wrist_pos_body
                returned_coords[bone_id] = right_wrist_rot.T @ translated
            else:
                returned_coords[bone_id] = None

        return returned_coords

    else:
        raise ValueError("coord_type must be 'absolute', 'body_centric', or 'hand_centric'")


def _calc_gripper_state(bone_positions: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Heuristically determine gripper state based on thumb and index finger distance.
    """
    # Gripper state based on thumb and index finger distance
    left_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandThumbTip)
    left_index_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandIndexTip)
    right_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandThumbTip)
    right_index_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandIndexTip)

    left_gripper_dist = (
        np.linalg.norm(left_thumb_tip - left_index_tip)
        if left_thumb_tip is not None and left_index_tip is not None
        else 0.1
    )
    right_gripper_dist = (
        np.linalg.norm(right_thumb_tip - right_index_tip)
        if right_thumb_tip is not None and right_index_tip is not None
        else 0.1
    )

    left_gripper_action = np.array([-1]) if left_gripper_dist > 0.05 else np.array([1])
    right_gripper_action = (
        np.array([-1]) if right_gripper_dist > 0.05 else np.array([1])
    )
    return left_gripper_action, right_gripper_action, left_gripper_dist, right_gripper_dist


def _get_hand_frame(
    side: str, bone_positions: dict
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates the hand-centric coordinate frame for a given side.

    Args:
        side (str): "left" or "right".
        bone_positions (dict): A dictionary mapping bone IDs to their positions.

    Returns:
        A tuple containing:
        - np.ndarray: The origin of the hand frame (wrist position).
        - np.ndarray: The rotation matrix from world to hand frame.
        Returns (None, None) if essential bones are missing.
    """
    # print(f"{bone_positions=}")
    side_pascal = side.capitalize()
    wrist_pos = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandWrist")
    )

    # index_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexMetacarpal")
    # )
    # middle_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleMetacarpal")
    # )
    # ring_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingMetacarpal")
    # )
    # little_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleMetacarpal")
    # )
    index_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexProximal")
    )
    middle_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleProximal")
    )
    ring_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingProximal")
    )
    little_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleProximal")
    )

    # X-axis: from wrist towards palm center (average of finger MCPs)
    if middle_mcp is not None and ring_mcp is not None:
        palm_center = (index_mcp + middle_mcp + ring_mcp + little_mcp) / 4
    elif middle_mcp is not None:
        palm_center = (index_mcp + middle_mcp + little_mcp) / 3
    else:
        palm_center = (index_mcp + little_mcp) / 2

    x_axis = (palm_center - wrist_pos)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Collect all available MCP points relative to wrist
    mcp_points = []
    if index_mcp is not None:
        mcp_points.append(index_mcp - wrist_pos)
    if middle_mcp is not None:
        mcp_points.append(middle_mcp - wrist_pos)
    if ring_mcp is not None:
        mcp_points.append(ring_mcp - wrist_pos)
    if little_mcp is not None:
        mcp_points.append(little_mcp - wrist_pos)

    if len(mcp_points) < 2:
        return None  # Need at least 2 MCP points for reliable computation

    # Use least squares to find Y-axis that's perpendicular to the palm plane
    # Stack MCP vectors into matrix A
    A = np.array(mcp_points)  # shape: (n_points, 3)

    # Find the normal to the plane containing MCP points using SVD
    # The normal is the last column of V (smallest singular value)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    palm_normal = Vt[-1, :]  # Last row of V^T = last column of V
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)

    # Determine handedness-consistent Y-axis direction
    # Use the palm normal, but ensure correct orientation based on hand side
    wrist_to_index = index_mcp - wrist_pos
    wrist_to_pinky = little_mcp - wrist_pos

    # Cross product gives a reference direction
    reference_y = np.cross(wrist_to_index, wrist_to_pinky)
    reference_y = reference_y / (np.linalg.norm(reference_y) + 1e-8)

    # Choose palm normal direction that aligns with handedness
    if np.dot(palm_normal, reference_y) < 0:
        palm_normal = -palm_normal

    y_axis = palm_normal

        # Z-axis: Z = X × Y (ensuring right-handed coordinate system)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Create rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_hand = U_rot @ V_rot_T

    return R_world_hand


def _convert_bones_to_finger_dict(finger_coords: dict, which_side: str) -> dict:
    """
    Convert bone coordinate dictionary to finger position format expected by hand controller.
    If no valid finger data is available, returns empty dict (hand controller will use zeros).

    Args:
        finger_coords: Dictionary mapping bone IDs to 3D coordinates
        which_side: "left" or "right"

    Returns:
        Dictionary in format expected by compute_finger_joint_angles_from_mediapipe:
        {
            'thumb': {'thumb_mcp': pos, 'thumb_pip': pos},
            'index': {'index_finger_mcp': pos, 'index_finger_pip': pos},
            'middle': {'middle_finger_mcp': pos, 'middle_finger_pip': pos},
            'ring': {'ring_finger_mcp': pos, 'ring_finger_pip': pos},
            'pinky': {'pinky_mcp': pos, 'pinky_pip': pos}
        }
        Or empty dict if no valid data (controller will use zero joint angles)
    """
    # Check if we have any valid finger coordinate data
    has_any_valid_data = False
    if finger_coords:
        for bone_id, coord in finger_coords.items():
            if coord is not None and np.isfinite(coord).all():
                has_any_valid_data = True
                break

    # If no valid finger data, return empty dict - hand controller will use zeros
    if not has_any_valid_data:
        return {}

    side_pascal = which_side.capitalize()
    finger_dict = {}
    # Define the mapping from bone IDs to finger positions
    thumb_distal = finger_coords[getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbDistal", None).value]
    if thumb_distal is None:
        # it's probably vmd data set use the vmd mapping
        finger_mapping = {
            'thumb': {
                'thumb_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbMetacarpal", None),
                'thumb_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbProximal", None),
                'thumb_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbTip", None)
                },
            'index': {
                'index_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexIntermediate", None),
                'index_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexTip", None),
                'index_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexTip", None)
            },
            'middle': {
                'middle_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleIntermediate", None),
                'middle_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleTip", None),
                'middle_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleTip", None)
            },
            'ring': {
                'ring_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingIntermediate", None),
                'ring_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingTip", None),
                'ring_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingTip", None)
            },
            'pinky': {
                'pinky_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleIntermediate", None),
                'pinky_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleTip", None),
                'pinky_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleTip", None)
            }
        }
    else:
        finger_mapping = {
        'thumb': {
            'thumb_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbProximal", None),
            'thumb_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbDistal", None),
            'thumb_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbTip", None)
        },
        'index': {
            'index_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexProximal", None),
            'index_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexIntermediate", None),
            'index_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexTip", None)
        },
        'middle': {
            'middle_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleProximal", None),
            'middle_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleIntermediate", None),
            'middle_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleTip", None)
        },
        'ring': {
            'ring_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingProximal", None),
            'ring_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingIntermediate", None),
            'ring_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingTip", None)
        },
        'pinky': {
            'pinky_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleProximal", None),
            'pinky_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleIntermediate", None),
            'pinky_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleTip", None)
        }
    }



    for finger_name, positions in finger_mapping.items():
        finger_positions = {}
        has_valid_finger_data = False

        for position_name, bone_id in positions.items():
            if bone_id is not None and bone_id in finger_coords:
                coord = finger_coords[bone_id]
                # Check for NaN or Inf values
                if coord is not None and np.isfinite(coord).all():
                    finger_positions[position_name] = np.array(coord)
                    has_valid_finger_data = True

        # Only add finger if we have valid data for it
        if has_valid_finger_data:
            # Fill in missing positions with zeros for this finger
            for position_name, bone_id in positions.items():
                if position_name not in finger_positions:
                    finger_positions[position_name] = np.array([0.0, 0.0, 0.0])
            finger_dict[finger_name] = finger_positions

    return finger_dict


def _compute_finger_tip_centroids(finger_dict: dict) -> np.ndarray | None:
    """
    Compute the centroid (mean position) of the five finger tips from the finger positions.

    Args:
        finger_dict (dict): Dictionary with keys 'thumb', 'index', 'middle', 'ring', 'pinky',
                           each containing joint positions (e.g., 'thumb_tip', 'index_finger_tip', etc.).

    Returns:
        np.ndarray: Mean position (3,) of the five finger tips in hand frame, or None if any tip is missing.
    """
    if not finger_dict:
        return None

    tip_keys = {
        # 'thumb': 'thumb_tip',
        'index': 'index_finger_tip',
        'middle': 'middle_finger_tip',
        'ring': 'ring_finger_tip',
        'pinky': 'pinky_tip'
    }

    tips = []
    for finger, tip_key in tip_keys.items():
        finger_data = finger_dict.get(finger, {})
        tip_pos = finger_data.get(tip_key)
        if tip_pos is None:
            return None  # If any tip is missing, return None
        tips.append(tip_pos)

    tips = np.array(tips)
    return np.mean(tips, axis=0)


def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.

    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [qv;q0] or [x, y, z, w]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix
    """

    I = np.identity(3)
    qhat = hat(q[0:3])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[-1] * qhat + 2 * qhat2


def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat


# --- Robosuite and WebRTC Integration ---


class RobosuiteTeleopState:
    """
    Thread-safe state to share data from the WebRTC server thread
    to the main robosuite simulation thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.pose_action = None
        self.is_connected = False

    def update_pose(self, action_dict):
        with self._lock:
            self.pose_action = action_dict

    def get_pose(self):
        with self._lock:
            return self.pose_action.copy() if self.pose_action else None


class StateFactory:
    """A factory to create and hold a reference to the state object."""

    def __init__(self):
        self.instance = None

    def __call__(self):
        # Called by WebRTCServer to create a state for a new peer.
        # We store the instance so the main thread can access it.
        if self.instance is None:
            self.instance = RobosuiteTeleopState()
        return self.instance


class XRRTCBodyPoseDevice(Device):
    """
    A device to control a robot using body pose data from XR Robot Teleop Client.
    """

    def __init__(self, env, process_bones_to_action_fn=None, record_data=False, output_dir="./recordings", ndigits=3, **kwargs):
        if env is not None:
            super().__init__(env)
        else:
            self.env = None
            self.all_robot_arms = []
            self.all_robot_grippers = []
            self.num_robot_arms = 0
            self.num_robot_grippers = 0
        self.state_factory = StateFactory()

        if process_bones_to_action_fn is None:
            self.process_bones_to_action_fn = self._default_process_bones_to_action
        else:
            self.process_bones_to_action_fn = process_bones_to_action_fn

        # CSV recording setup
        self.record_data = record_data
        if self.record_data:
            self.csv_file = None
            self.csv_writer = None
            self.started = False
            self.start_time = None
            self.ndigits = ndigits

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"body_pose_data_{timestamp}.csv"
            self.output_file = os.path.join(output_dir, csv_filename)

            print(f"CSV recording enabled: {self.output_file}")

        datachannel_handlers = {"body_pose": self.on_body_pose_message}

        configure_logging(level="INFO")  # set xr_robot_teleop's verbosity
        self.server = WebRTCServer(
            datachannel_handlers=datachannel_handlers,
            state_factory=self.state_factory,
            video_track_factory=None,
        )

        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        print("=" * 80)
        print("Integrated Robosuite WebRTC Teleoperation Server")
        print("The WebRTC server is running in the background.")
        print("Connect your VR client to this machine on port 8080.")
        print("=" * 80)

    @property
    def is_connected(self):
        """
        Returns true if the WebRTC client is connected.
        """
        return (
            self.state_factory.instance is not None
            and self.state_factory.instance.is_connected
        )

    @staticmethod
    def _default_process_bones_to_action(bones: list[Bone]) -> dict:
        """
        Custom function to process bones into actions for dual Kinova3 robot.
        """
        action_dict = {}

        # Get bone positions and rotations
        bone_positions = {b.id: np.array(b.position) for b in bones}
        bone_rotations = {b.id: np.array(b.rotation) for b in bones}

        # Hand Control
        left_gripper_action, right_gripper_action, left_gripper_dist, right_gripper_dist = _calc_gripper_state(bone_positions)


        finger_coords = _get_finger_coordinates(
            bones=bones,
            coord_type="hand_centric",
            bones_to_return=simplified_finger_bones,
        )

        # Arm Control (absolute SEW)
        sew_coords = _get_body_centric_coordinates(bones)

        if sew_coords is None or sew_coords["left"] is None or sew_coords["right"] is None:
            print("Warning: Could not calculate SEW coordinates. Skipping action.")
            return None

        # Head rotation for camera
        R_world_upper_body = sew_coords['R_world_body']
        head_rot_q = bone_rotations.get(FullBodyBoneId.FullBody_Head)
        R_world_head = q2R(head_rot_q) @ Rotation.from_euler("ZYX",
                                                             [-np.pi / 2, -np.pi / 2, 0]
                                                             ).as_matrix() if head_rot_q is not None else np.eye(3)
        action_dict["head_rotation"] = R_world_upper_body.T @ R_world_head
        # Get wrist rotations
        left_rot_matrix = sew_coords["left"]["wrist_rot"]
        right_rot_matrix = sew_coords["right"]["wrist_rot"]

        if left_rot_matrix is None or right_rot_matrix is None:
            print("Warning: Could not get wrist rotation. Skipping action.")
            return None

        # Assemble final action
        left_sew_pos = np.concatenate(
            [sew_coords["left"]["S"], sew_coords["left"]["E"], sew_coords["left"]["W"]]
        )
        right_sew_pos = np.concatenate(
            [sew_coords["right"]["S"], sew_coords["right"]["E"], sew_coords["right"]["W"]]
        )

        left_sew = np.concatenate([left_sew_pos, left_rot_matrix])
        right_sew = np.concatenate([right_sew_pos, right_rot_matrix])

        # lower body coordinates
        hka_coords = _get_lower_body_centric_coordinates(bones)

        # compute torso rotation
        R_world_lower_body = hka_coords['R_world_lower_body']
        R_lower_upper = R_world_lower_body.T @ R_world_upper_body

        # Body center for mocap control
        body_center_in_world = bone_positions.get(FullBodyBoneId.FullBody_SpineMiddle)

        action_dict["left_sew"] = left_sew
        action_dict["right_sew"] = right_sew
        action_dict["left_gripper"] = left_gripper_action
        action_dict["right_gripper"] = right_gripper_action
        action_dict["left_gripper_val"] = left_gripper_dist
        action_dict["right_gripper_val"] = right_gripper_dist
        action_dict["right_fingers"] = _convert_bones_to_finger_dict(finger_coords, "right")
        action_dict["left_fingers"] = _convert_bones_to_finger_dict(finger_coords, "left")
        action_dict["left_finger_tip_centroid"] = _compute_finger_tip_centroids(action_dict["left_fingers"])
        action_dict["right_finger_tip_centroid"] = _compute_finger_tip_centroids(action_dict["right_fingers"])
        action_dict["left_finger_mcp_centroid"] = action_dict["left_fingers"].get('index', {}).get('index_finger_mcp', None)
        action_dict["right_finger_mcp_centroid"] = action_dict["right_fingers"].get('index', {}).get('index_finger_mcp', None)

        # Add lower body HKA coordinates to action_dict
        action_dict["left_hka"] = hka_coords.get("left")
        action_dict["right_hka"] = hka_coords.get("right")
        action_dict["body_center"] = body_center_in_world
        action_dict["R_world_lower_body"] = hka_coords.get("R_world_lower_body")
        action_dict["R_lower_upper"] = R_lower_upper

        return action_dict

    def _record_to_csv(self, bones: list[Bone], action_dict: dict):
        """Record bones and action data to CSV file."""
        if not self.started:
            self.started = True
            self.start_time = time.time()

            # Initialize CSV file and writer
            self.csv_file = open(self.output_file, "w", newline="")
            fieldnames = [
                "time_elapsed",
                "data_type",  # "bone" or "action"
                "id",  # bone_id for bones, action_key for actions
                "pos_x",
                "pos_y",
                "pos_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "rot_w",
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            print(f"Started recording data to {self.output_file}")

        timestamp = time.time()
        time_elapsed = timestamp - self.start_time

        # Record raw bones data
        for bone in bones:
            row = {
                "time_elapsed": round(time_elapsed, self.ndigits),
                "data_type": "bone",
                "id": bone.id,
                "pos_x": round(float(bone.position[0]), self.ndigits),
                "pos_y": round(float(bone.position[1]), self.ndigits),
                "pos_z": round(float(bone.position[2]), self.ndigits),
                "rot_x": round(float(bone.rotation[0]), self.ndigits),
                "rot_y": round(float(bone.rotation[1]), self.ndigits),
                "rot_z": round(float(bone.rotation[2]), self.ndigits),
                "rot_w": round(float(bone.rotation[3]), self.ndigits),
            }
            self.csv_writer.writerow(row)

        # Record processed actions data
        if action_dict:
            for action_key, action_value in action_dict.items():
                if isinstance(action_value, np.ndarray):
                    # Handle numpy arrays - flatten and record as position/rotation
                    flat_value = action_value.flatten()
                    if len(flat_value) >= 3:
                        pos_x = round(float(flat_value[0]), self.ndigits)
                        pos_y = round(float(flat_value[1]), self.ndigits)
                        pos_z = round(float(flat_value[2]), self.ndigits)
                    else:
                        pos_x = pos_y = pos_z = 0.0

                    if len(flat_value) >= 7:  # SEW format with rotation
                        rot_x = round(float(flat_value[9]), self.ndigits) if len(flat_value) > 9 else 0.0
                        rot_y = round(float(flat_value[10]), self.ndigits) if len(flat_value) > 10 else 0.0
                        rot_z = round(float(flat_value[11]), self.ndigits) if len(flat_value) > 11 else 0.0
                        rot_w = 1.0  # Default quaternion w component
                    else:
                        rot_x = rot_y = rot_z = 0.0
                        rot_w = 1.0

                    row = {
                        "time_elapsed": round(time_elapsed, self.ndigits),
                        "data_type": "action",
                        "id": action_key,
                        "pos_x": pos_x,
                        "pos_y": pos_y,
                        "pos_z": pos_z,
                        "rot_x": rot_x,
                        "rot_y": rot_y,
                        "rot_z": rot_z,
                        "rot_w": rot_w,
                    }
                    self.csv_writer.writerow(row)
                elif action_key in ["left_gripper_val", "right_gripper_val"]:
                    # Handle gripper distance values
                    row = {
                        "time_elapsed": round(time_elapsed, self.ndigits),
                        "data_type": "action",
                        "id": action_key,
                        "pos_x": round(float(action_value), self.ndigits),
                        "pos_y": 0.0,
                        "pos_z": 0.0,
                        "rot_x": 0.0,
                        "rot_y": 0.0,
                        "rot_z": 0.0,
                        "rot_w": 1.0,
                    }
                    self.csv_writer.writerow(row)

        # Flush the file to ensure data is written
        self.csv_file.flush()

    def cleanup_recording(self):
        """Clean up CSV recording resources."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"CSV recording stopped and saved to: {self.output_file}")

    def on_body_pose_message(self, message: bytes, state: RobosuiteTeleopState):
        """
        Callback for the 'body_pose' data channel. This is called by the
        WebRTCServer whenever a message is received.
        """
        if not state.is_connected:
            state.is_connected = True
            print("\n[WebRTC] Body pose data channel connected and receiving data.")

        try:
            bones = deserialize_pose_data(message)
            if bones:
                action_dict = self.process_bones_to_action_fn(bones)
                state.update_pose(action_dict)

                # CSV recording
                if self.record_data:
                    self._record_to_csv(bones, action_dict)
        except Exception as e:
            import traceback
            print(f"Error processing body pose message: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        # No-op for this device, as the server is already running.
        pass

    def get_controller_state(self):
        """
        Returns the current state of the device, a dictionary of pos, orn, grasp, and reset.
        """
        shared_state = self.state_factory.instance
        if not self.is_connected:
            return None

        pose_action = shared_state.get_pose()
        if pose_action is None:
            return None
        return pose_action

    def input2action(self, mirror_actions=False):
        """
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call
        If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action
        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.
        Returns:
            Optional[Dict]: Dictionary of actions to be fed into env.step()
                            if reset is triggered, returns None
        """

        input_ac_dict = self.get_controller_state()

        if input_ac_dict is None:
            return None

        # Create the action vector for robosuite
        if self.env is None:
            return input_ac_dict

        # Create the action vector for robosuite environments
        active_robot = self.env.robots[0]
        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            action_dict[arm] = input_ac_dict.get(f"{arm}_sew")
            action_dict[f"{arm}_gripper"] = input_ac_dict.get(f"{arm}_gripper")

        return action_dict
