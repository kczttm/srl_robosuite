"""
Dexterous hands for GR1 robot.
"""
import os
import numpy as np
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.gripper_model import GripperModel

_abh_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir)
)
_left_hand_XML_path = os.path.join(
    _abh_path, "ability_left_hand_corrected.xml"
)
_right_hand_XML_path = os.path.join(
    _abh_path, "ability_right_hand_corrected.xml"
)

@register_gripper
class AbilityLeftHandCorrected(GripperModel):
    """
    Dexterous left Ability hand.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(_left_hand_XML_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0] * 10)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 10


@register_gripper
class AbilityRightHandCorrected(GripperModel):
    """
    Dexterous right Ability hand.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(_right_hand_XML_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0] * 10)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 10
