"""
Robosuite wrapper for the unified MediaPipe teleoperation device.
This wrapper makes the unified MediaPipe device compatible with robosuite's Device interface.

Author: Chuizheng Kong  
Date created: 2025-09-17
"""

import robosuite.utils.transform_utils as T
from robosuite.devices import Device
from projects.shared_devices.mediapipe_teleop_device import MediaPipeTeleopDevice


class MediaPipeTeleopRobosuiteWrapper(Device):
    """
    Robosuite-compatible wrapper for the unified MediaPipe teleoperation device.
    This inherits from robosuite's Device class and wraps the unified MediaPipeTeleopDevice
    to provide seamless integration with robosuite environments.
    """
    
    def __init__(self, env=None, debug=False, camera_id=0, mirror_actions=True):
        """
        Initialize the robosuite wrapper.
        
        Args:
            env: Robosuite environment
            debug (bool): Enable debug visualization
            camera_id (int): Camera device ID
            mirror_actions (bool): Mirror actions (right robot arm follows left human arm)
        """
        # Initialize the parent Device class
        super().__init__(env)
        
        # Create the unified MediaPipe device with robosuite environment
        self.mediapipe_device = MediaPipeTeleopDevice(
            camera_id=camera_id,
            debug=debug,
            mirror_actions=mirror_actions,
            env=env  # Pass the environment for robosuite integration
        )
        
        print(f"MediaPipe Robosuite Wrapper initialized with camera {camera_id}")
    
    def start_control(self):
        """Start the control loop."""
        # Don't call super() since it's abstract and raises NotImplementedError
        self.mediapipe_device.start_control()
    
    def get_controller_state(self):
        """Get current controller state."""
        return self.mediapipe_device.get_controller_state()
    
    def input2action(self, mirror_actions=False):
        """
        Convert pose input into valid action sequence for env.step().
        
        Args:
            mirror_actions (bool): Whether to mirror actions for different viewpoint
            
        Returns:
            Optional[Dict]: Dictionary of actions for env.step() or None if reset
        """
        return self.mediapipe_device.input2action(mirror_actions=mirror_actions)
    
    def get_sew_and_wrist_poses(self):
        """Get current SEW poses and wrist orientations."""
        return self.mediapipe_device.get_sew_and_wrist_poses()
    
    def should_quit(self):
        """Check if quit was requested."""
        return self.mediapipe_device.should_quit()
    
    def should_reset(self):
        """Check if reset was requested."""
        return self.mediapipe_device.should_reset()
    
    @property
    def engaged(self):
        """Get engagement status from the underlying device."""
        return self.mediapipe_device.engaged
    
    def quit(self):
        """Request the device to quit."""
        self.mediapipe_device._quit_state = True
    
    def _reset_internal_state(self):
        """Reset internal state variables."""
        # Call parent implementation only if it exists and is not abstract
        if hasattr(super(), '_reset_internal_state'):
            try:
                super()._reset_internal_state()
            except NotImplementedError:
                pass  # Parent method is abstract, skip it
        self.mediapipe_device._reset_internal_state()
    
    def stop(self):
        """Stop the device and cleanup resources."""
        self.mediapipe_device.stop()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except:
            pass


# For backward compatibility, create an alias with the original class name
MediaPipeTeleop = MediaPipeTeleopRobosuiteWrapper


if __name__ == "__main__":
    """
    Test the robosuite wrapper with a mock environment.
    """
    print("Testing MediaPipe Robosuite Wrapper...")
    
    # Create a simple mock environment for testing
    class MockRobot:
        def __init__(self):
            self.arms = ['left', 'right']
            self.robot_model = MockRobotModel()
    
    class MockRobotModel:
        def __init__(self):
            self.name = 'mock_robot'
            self.arm_type = 'bimanual'
    
    class MockSim:
        pass
    
    class MockInnerEnv:
        def __init__(self):
            self.sim = MockSim()
    
    class MockEnvironment:
        def __init__(self):
            self.robots = [MockRobot()]
            self.env = MockInnerEnv()
    
    try:
        # Test wrapper creation with camera window display
        mock_env = MockEnvironment()
        wrapper = MediaPipeTeleopRobosuiteWrapper(
            env=mock_env,
            debug=True,
            camera_id=0,
            mirror_actions=False
        )
        
        print("Wrapper created successfully!")
        print("Starting control...")
        wrapper.start_control()
        
        # Test getting controller state
        state = wrapper.get_controller_state()
        print(f"Initial state: {state is not None}")
        
        # Test action conversion
        try:
            actions = wrapper.input2action()
            print(f"Action conversion successful: {actions is not None}")
        except Exception as e:
            print(f"Action conversion test failed (expected in mock env): {e}")
        
        # Test running for a few seconds in headless mode
        print("Testing headless mode for 3 seconds...")
        import time
        start_time = time.time()
        while time.time() - start_time < 300.0:
            if wrapper.should_quit():
                print("Device quit unexpectedly")
                break
            time.sleep(0.1)
        else:
            print("Headless mode test completed successfully!")
        
        # Manually quit the device
        wrapper.quit()
        time.sleep(0.5)  # Give it time to quit
        
        print("Stopping wrapper...")
        wrapper.stop()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()