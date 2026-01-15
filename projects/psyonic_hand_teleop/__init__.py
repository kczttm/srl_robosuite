"""
Psyonic Hand Teleoperation Project

This project contains components for teleoperation of dual Kinova3 robots
with Psyonic Ability hands using MediaPipe pose estimation.

Components:
- AbilityHandController: Joint torque controller for Psyonic hands
- MediaPipe integration for hand pose estimation
- SEW (Shoulder-Elbow-Wrist) teleoperation for arms

Author: Chuizheng Kong
Date created: 2025-08-05
"""
from .ability_hand.dual_kinova3_robot_psyonic_gripper import DualKinova3PsyonicHand
from .ability_hand.ability_hands_corrected import AbilityLeftHandCorrected, AbilityRightHandCorrected

from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

# Register DualKinova3PsyonicHand as FixedBaseRobot in the mapping (to match the robot definition)
REGISTERED_ROBOTS["DualKinova3PsyonicHand"] = DualKinova3PsyonicHand
ROBOT_CLASS_MAPPING["DualKinova3PsyonicHand"] = FixedBaseRobot