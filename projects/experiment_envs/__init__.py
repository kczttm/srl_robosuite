"""
Custom experimental environments for robosuite.

This module contains experimental environments that are not part of the core robosuite package.
They are designed for specific research and development purposes.

Environments:
- Bounce: Environment for bounce ball manipulation tasks
- DualKinova3SRLEnv: Dual Kinova3 environment with SRL capabilities  
- ExpTwoArmLift: Experimental two-arm lifting environment

Author: Chuizheng Kong
Date: 2025-08-17
"""

from robosuite.environments.base import register_env

# Import the custom environments
# from .bounce import Bounce
from .dualkinova3_srl_env import DualKinova3SRLEnv
from .exp_two_arm_lift import ExpTwoArmLift
from .exp_narrow_gap import ExpNarrowGap
from .exp_pick_place import ExpPickPlace   
from .exp_cabinet import ExpCabinet
from .exp_glass_gap import ExpGlassGap

# Register the environments with robosuite
# register_env(Bounce)
register_env(DualKinova3SRLEnv)
register_env(ExpTwoArmLift)
register_env(ExpNarrowGap)
register_env(ExpPickPlace)
register_env(ExpCabinet)
register_env(ExpGlassGap)

# Make environments available for direct import
__all__ = [
    "Bounce",
    "DualKinova3SRLEnv", 
    "ExpTwoArmLift",
    "ExpNarrowGap",
    "ExpPickPlace",
    "ExpCabinet",
    "ExpGlassGap",
]
