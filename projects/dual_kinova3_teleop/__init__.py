"""
Dual Kinova3 Teleoperation Project

This project contains components for teleoperation of dual Kinova3 robots
with Psyonic Ability hands using MediaPipe pose estimation.

Components:
- AbilityHandController: Joint torque controller for Psyonic hands
- MediaPipe integration for hand pose estimation
- SEW (Shoulder-Elbow-Wrist) teleoperation for arms

Author: Chuizheng Kong
Date created: 2025-08-05
"""
import os
import sys

# Add the current directory to Python path to enable absolute imports
# current_dir = os.path.dirname(__file__)
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)


from .dual_kinova3.dual_kinova3_robot import DualKinova3

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

# Register DualKinova3 as FixedBaseRobot in the mapping
REGISTERED_ROBOTS["DualKinova3"] = DualKinova3
ROBOT_CLASS_MAPPING["DualKinova3"] = FixedBaseRobot

# Make DualKinova3 available when importing from this package
# __all__ = ["DualKinova3"]

from .grippers.robotiq_85_gripper import Robotiq85GripperCorrected