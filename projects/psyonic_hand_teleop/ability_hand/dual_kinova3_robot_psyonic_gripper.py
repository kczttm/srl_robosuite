import numpy as np

from projects.dual_kinova3_teleop.dual_kinova3.dual_kinova3_robot import DualKinova3


class DualKinova3PsyonicHand(DualKinova3):
    """
    The Gen3 robot is the sparkly newest addition to the Kinova line

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """
    def __init__(self, idn=0):
        super().__init__(idn=idn)

    @property
    def default_base(self):
        # return "OmronMobileBaseSRL"
        # return "OmronMobileBase"
        return "RethinkMount"

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "AbilityRightHandCorrected", "left": "AbilityLeftHandCorrected"}

