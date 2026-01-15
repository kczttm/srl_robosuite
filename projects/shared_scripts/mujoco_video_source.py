import time
from typing import Optional

import mujoco
import numpy as np

from xr_robot_teleop_server.sources.base import VideoSource


class MuJoCoVideoSource(VideoSource):
    """
    A video source that captures frames from a MuJoCo simulation.
    
    This class implements the VideoSource interface and provides real-time
    video streaming from a MuJoCo simulation environment.
    
    NOTE: On macOS, MuJoCo renderer must be created and used on the main thread
    to avoid NSWindow threading issues. This implementation is synchronous.
    """
    
    def __init__(self, model, data, camera_name: str = "frontview", 
                 width: int = 640, height: int = 480, fps: float = 30.0,
                 env: Optional = None):
        """
        Initialize the MuJoCo video source.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            camera_name: Name of the camera to use for rendering
            width: Video frame width
            height: Video frame height
            fps: Target frames per second
            env: Optional robosuite environment (for using existing renderer)
        """
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self._width = width
        self._height = height
        self._fps = fps
        self.env = env
        
        # Find camera ID
        try:
            self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        except Exception:
            # If named camera doesn't exist, use camera index 0
            self.camera_id = 0
            print(f"Warning: Camera '{camera_name}' not found, using camera index 0")
        
        # Use environment's renderer if available, otherwise create our own
        # NOTE: Creating renderer on main thread to avoid macOS threading issues
        if env and hasattr(env, 'sim') and hasattr(env.sim, 'render'):
            self.renderer = None  # Will use env.sim.render()
            print("Using environment's built-in renderer")
        else:
            try:
                self.renderer = mujoco.Renderer(model, height, width)
                print("Created new MuJoCo renderer")
            except Exception as e:
                print(f"Failed to create MuJoCo renderer: {e}")
                self.renderer = None
        
        self._running = True
        self._last_render_time = 0
    
    def _render_frame(self) -> np.ndarray:
        """Render a single frame from the simulation."""
        try:
            if self.env and hasattr(self.env, 'sim'):
                # Use environment's built-in rendering (recommended approach)
                frame = self.env.sim.render(
                    width=self._width, 
                    height=self._height,
                    camera_name=self.camera_name
                )
                # robosuite returns RGB, but we need to flip vertically for consistency
                frame = np.flipud(frame)
                
            elif self.renderer is not None:
                # Use our own MuJoCo renderer
                self.renderer.update_scene(self.data, camera=self.camera_id)
                frame = self.renderer.render()
                
            else:
                # Fallback: return black frame
                frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                
            return frame
            
        except Exception as e:
            print(f"Error rendering frame: {e}")
            # Return black frame on error
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)
    
    @property
    def width(self) -> int:
        """Width of the video frames."""
        return self._width
    
    @property
    def height(self) -> int:
        """Height of the video frames."""
        return self._height
    
    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps
    
    def __iter__(self):
        """Allows the class to be used as an iterator."""
        return self
    
    def __next__(self) -> np.ndarray:
        """Returns the next frame as a NumPy array."""
        if not self._running:
            raise StopIteration
        
        # Throttle frame rate (optional - WebRTC will handle this too)
        current_time = time.time()
        elapsed = current_time - self._last_render_time
        target_frame_time = 1.0 / self._fps
        
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)
        
        # Render frame synchronously on calling thread (main thread)
        frame = self._render_frame()
        self._last_render_time = time.time()
        
        return frame
    
    def release(self):
        """Releases the video source and cleans up resources."""
        self._running = False
        
        if hasattr(self, 'renderer') and self.renderer is not None:
            self.renderer.close()