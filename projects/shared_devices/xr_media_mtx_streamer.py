import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

class StereoCameraRig:
    """
    Helper class to manage stereo camera offsets and rendering updates in MuJoCo.
    """
    def __init__(self, sim, camera_name, pos_offset=[0.0, 0.0, 0.0], euler_offset=[0.0, 0.0, 0.0], ipd=0.063): 
        self.sim = sim
        self.camera_name = camera_name
        self.pos_offset = np.array(pos_offset)
        self.euler_offset = np.array(euler_offset)
        self.ipd = ipd
        self.cam_id = sim.model.camera_name2id(camera_name)
        
        # Cache parent body ID for dynamic world pose updates
        self.parent_body_id = sim.model.cam_bodyid[self.cam_id] if self.cam_id >= 0 else -1
        
        self.base_cam_pos = None
        self.left_cam_offset = None
        self.right_cam_offset = None

    def initialize(self):
        """
        Call this to apply offsets and prepare for stereo rendering.
        Captures the current camera state as the 'base' and calculates stereo vectors.
        Start from the CURRENT sim model state (assuming it is the base state).
        To be safe, we should probably not use += if called multiple times, 
        but for now we assume this is called once per 'setup'.
        """
        if self.cam_id < 0:
            print(f"Warning: Camera '{self.camera_name}' not found.")
            return

        # 1. Apply static position offset to the model
        # Use [:] to update in place
        self.sim.model.cam_pos[self.cam_id][:] += self.pos_offset
        
        # 2. Apply static rotation offset
        # Convert MuJoCo quaternion [w,x,y,z] to scipy [x,y,z,w]
        current_quat = self.sim.model.cam_quat[self.cam_id].copy() 
        current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        
        offset_rot = R.from_euler('xyz', self.euler_offset, degrees=True)
        new_rot = offset_rot * current_rot
        new_quat_scipy = new_rot.as_quat() # [x, y, z, w]
        
        # Convert back to MuJoCo [w,x,y,z]
        self.sim.model.cam_quat[self.cam_id][:] = [new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]]
        
        # 3. Store the adjusted base position (we will pivot around this for stereo)
        self.base_cam_pos = self.sim.model.cam_pos[self.cam_id].copy()
        
        # 4. Calculate Stereo Vectors
        # Re-read rotation from model (it now includes offset)
        cam_quat = self.sim.model.cam_quat[self.cam_id]
        cam_rot = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])
        cam_right_vector = cam_rot.apply([1, 0, 0])
        
        self.left_cam_offset = cam_right_vector * (self.ipd / 2)
        self.right_cam_offset = cam_right_vector * (self.ipd / 2)
        
        print(f"Stereo Rig Initialized: '{self.camera_name}' IPD={self.ipd}m")

    def set_world_pose(self, world_pos, world_rot):
        """
        Sets the camera's base pose based on a desired world pose.
        Automatically handles the parent body's transformation.
        
        world_pos: [x, y, z] Desired global position of the camera (centered between eyes)
        world_rot: scipy.spatial.transform.Rotation object representing global orientation
        """
        if self.parent_body_id < 0:
            return

        # 1. Get Parent Body Pose (World Frame)
        parent_pos = self.sim.data.body_xpos[self.parent_body_id]
        parent_quat = self.sim.data.body_xquat[self.parent_body_id] # [w,x,y,z]
        R_parent = R.from_quat([parent_quat[1], parent_quat[2], parent_quat[3], parent_quat[0]])
        R_parent_mat = R_parent.as_matrix()
        
        # 2. Calculate Local Position: R_parent^T * (P_world - P_parent)
        local_pos = R_parent_mat.T @ (np.array(world_pos) - parent_pos)
        
        # 3. Calculate Local Orientation: R_parent^T * R_world
        # R_local = R_parent.inv() * R_world
        R_local = R_parent.inv() * world_rot
        
        local_quat_scipy = R_local.as_quat() # [x, y, z, w]
        local_quat_mj = [local_quat_scipy[3], local_quat_scipy[0], local_quat_scipy[1], local_quat_scipy[2]]
        
        # 4. Update the Rig with calculated local pose
        self.update_pose(local_pos, local_quat_mj)

    def update_pose(self, pos, quat):
        """
        Update the base camera pose (local to parent body) for dynamic tracking.
        pos: [x, y, z]
        quat: [w, x, y, z] (MuJoCo format)
        """
        # Update base position and orientation in model
        self.sim.model.cam_pos[self.cam_id][:] = pos
        self.sim.model.cam_quat[self.cam_id][:] = quat
        
        # Update internal base reference
        self.base_cam_pos = np.array(pos)
        
        # Recalculate stereo vectors based on new orientation
        # quat is [w, x, y, z]
        cam_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        cam_right_vector = cam_rot.apply([1, 0, 0])
        
        self.left_cam_offset = cam_right_vector * (self.ipd / 2)
        self.right_cam_offset = cam_right_vector * (self.ipd / 2)

    def get_stereo_frames(self, width, height, flip=True):
        """
        Render left and right frames.
        Returns (frame_left, frame_right).
        """
        if self.cam_id < 0 or self.base_cam_pos is None:
            # Return black frames if not initialized
            empty = np.zeros((height, width, 3), dtype=np.uint8)
            return empty, empty

        # Render Left
        self.sim.model.cam_pos[self.cam_id][:] = self.base_cam_pos - self.left_cam_offset
        self.sim.forward()
        frame_left = self.sim.render(width=width, height=height, camera_name=self.camera_name)
        
        # Render Right
        self.sim.model.cam_pos[self.cam_id][:] = self.base_cam_pos + self.right_cam_offset
        self.sim.forward()
        frame_right = self.sim.render(width=width, height=height, camera_name=self.camera_name)
        
        if flip:
            # Note: The array returned by render is usually contiguous.
            # Slicing [::-1] makes it non-contiguous.
            # We enforce contiguity here to avoid issues downstream.
            return np.ascontiguousarray(frame_left[::-1]), np.ascontiguousarray(frame_right[::-1])
        
        return frame_left, frame_right

class MediaMTXStreamer:
    """Streams video frames to MediaMTX via FFmpeg RTSP."""

    def __init__(self, rtsp_url: str = "rtsp://localhost:8554/mujoco",
                 width: int = 640, height: int = 480, fps: int = 30,
                 use_gpu: bool = True):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.use_gpu = use_gpu
        self.process = None

    def start(self):
        """Start the FFmpeg process for RTSP streaming."""
        if self.use_gpu:
            # NVIDIA GPU encoding (h264_nvenc) - ultra low latency settings
            command = [
                'ffmpeg',
                '-y',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'h264_nvenc',
                '-preset', 'p2',     # p2 is faster/better compromise than p1
                '-tune', 'll',       # Low latency tuning
                '-rc', 'cbr',        # Constant bitrate
                '-b:v', '6M',        # Increased to 6 Mbps for quality
                '-minrate', '6M',
                '-maxrate', '6M',
                '-bufsize', '3M',    # 0.5s buffer
                '-bf', '0',          # No B-frames (critical for low latency)
                '-rc-lookahead', '0',# No lookahead
                '-g', str(self.fps), # Keyframe every 1 second
                '-pix_fmt', 'yuv420p',
                '-fflags', 'nobuffer',
                '-f', 'rtsp',
                # '-rtsp_transport', 'tcp', # Force TCP if needed for stability
                self.rtsp_url
            ]
            # command = [
            #     'ffmpeg',
            #     '-y',
            #     '-fflags', 'nobuffer',
            #     '-flags', 'low_delay',
            #     '-f', 'rawvideo',
            #     '-pix_fmt', 'rgb24',
            #     '-s', f'{self.width}x{self.height}',
            #     '-r', str(self.fps),
            #     '-i', '-',
            #     '-c:v', 'h264_nvenc',
            #     '-preset', 'p1',  # Fastest NVENC preset
            #     '-tune', 'll',   # Low latency tuning
            #     '-zerolatency', '1',
            #     '-rc', 'cbr',    # Constant bitrate for consistent streaming
            #     '-b:v', '4M',    # 4 Mbps bitrate
            #     '-pix_fmt', 'yuv420p',
            #     '-fflags', 'nobuffer',
            #     '-f', 'rtsp',
            #     self.rtsp_url
            # ]
        else:
            # CPU encoding fallback (libx264) - ultra low latency settings
            command = [
                'ffmpeg',
                '-y',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-b:v', '4M',        # Set bitrate for CPU
                '-bf', '0',          # Ensure no B-frames
                '-g', str(self.fps), # Keyframe every 1 second
                '-pix_fmt', 'yuv420p',
                '-fflags', 'nobuffer',
                '-f', 'rtsp',
                self.rtsp_url
            ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"Started streaming to {self.rtsp_url}")

    def send_frame(self, frame: np.ndarray):
        """
        Send a frame to the stream.
        frame: numpy array of shape (height, width, 3) usually RGB.
        """
        if self.process is not None and self.process.poll() is None:
            if not hasattr(self, "_debug_printed"):
                print(f"[MediaMTXStreamer] Frame shape: {frame.shape}, dtype: {frame.dtype}, contiguous: {frame.flags['C_CONTIGUOUS']}")
                self._debug_printed = True

            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Warning: FFmpeg pipe broken, stream may have stopped")

    def stop(self):
        """Stop the FFmpeg process."""
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
            print("Stopped streaming")
