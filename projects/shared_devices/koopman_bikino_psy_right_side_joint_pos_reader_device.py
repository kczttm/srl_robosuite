
import numpy as np
import time

class KoopmanBiKinoPsyRightSideJointPosReaderDevice:
    """
    Device to read recorded joint positions (Right Arm + Right Hand) from an .npy file
    and serve them for playback.
    
    The .npy file is expected to have shape (T, 13) where:
    - Columns 0-6: Right Arm Joint Positions (7 DOF)
    - Columns 7-12: Right Hand Joint Positions (6 DOF) - [Index, Middle, Ring, Pinky, ThumbFlex, ThumbRot] (Assumption)
    """
    def __init__(self, npy_path, frequency=30.0, loop=False):
        """
        Args:
            npy_path (str): Path to the .npy file containing joint data.
            frequency (float): Expected playback frequency in Hz.
            loop (bool): Whether to loop the playback when finished.
        """
        try:
            self.data = np.load(npy_path)
        except Exception as e:
            raise ValueError(f"Could not load npy file from {npy_path}: {e}")
            
        if self.data.ndim != 2 or self.data.shape[1] != 13:
            # Try to handle if shape is different, but warn
            print(f"Warning: Expected shape (T, 13), got {self.data.shape}")
        
        self.num_samples = self.data.shape[0]
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.loop = loop
        
        self.current_idx = 0
        self.start_time = None
        self.last_call_time = 0
        
        print(f"KoopmanBiKinoPsyRightSideJointPosReaderDevice: Loaded {self.num_samples} samples from {npy_path}")

    def get_controller_state(self):
        """
        Returns the next joint configuration.
        
        Returns:
            dict or None: 
                {
                    'right_arm_positions': np.array(7),
                    'right_hand_positions': np.array(6)
                }
                Returns None if end of trajectory and not looping.
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Simple frame stepping
        if self.current_idx >= self.num_samples:
            if self.loop:
                self.current_idx = 0
                print("Looping playback...")
            else:
                return None
        
        sample = self.data[self.current_idx]
        self.current_idx += 1
        
        # Parse sample
        # Assuming layout: [Arm(7), Hand(6)]
        arm_positions = sample[:7]
        hand_positions = sample[7:]
        
        return {
            'right_arm_positions': arm_positions,
            'right_hand_positions': hand_positions
        }
    
    def reset(self):
        self.current_idx = 0
        self.start_time = None
