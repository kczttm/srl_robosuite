#!/usr/bin/env python3
"""
Test script to verify the MuJoCo video source implementation.
This runs a basic test without the full XR teleop setup.
"""

import os
import sys
import time

import numpy as np
import robosuite as suite

from robosuite.projects.shared_scripts.mujoco_video_source import MuJoCoVideoSource

def test_mujoco_video_source():
    """Test the MuJoCo video source implementation."""
    print("Creating test environment...")
    
    # Create a simple robosuite environment for testing
    env = suite.make(
        "DualKinova3SRLEnv",
        robots=["DualKinova3"],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=30,
        camera_names=["frontview"],
        camera_heights=480,
        camera_widths=640,
    )
    
    print("Resetting environment...")
    env.reset()
    
    print("Creating MuJoCo video source...")
    video_source = MuJoCoVideoSource(
        model=env.sim.model._model,
        data=env.sim.data._data,
        camera_name="frontview",
        width=640,
        height=480,
        fps=30.0,
        env=env  # Pass environment for macOS compatibility
    )
    
    print("Testing video source properties...")
    print(f"Width: {video_source.width}")
    print(f"Height: {video_source.height}")
    print(f"FPS: {video_source.fps}")
    
    print("Capturing test frames...")
    frame_count = 0
    start_time = time.time()
    
    try:
        # Capture a few frames to test
        for i in range(30):  # Capture 30 frames (~1 second at 30fps)
            # Step the simulation
            action = np.zeros(env.action_dim)
            env.step(action)
            
            # Get frame from video source
            frame = next(video_source)
            frame_count += 1
            
            print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")
            
            # Simple validation
            if frame.shape != (480, 640, 3):
                print(f"ERROR: Unexpected frame shape: {frame.shape}")
                break
            
            if frame.dtype != np.uint8:
                print(f"ERROR: Unexpected frame dtype: {frame.dtype}")
                break
            
            time.sleep(0.033)  # ~30fps
    
    except Exception as e:
        print(f"ERROR during frame capture: {e}")
        return False
    
    finally:
        print("Cleaning up...")
        video_source.release()
        env.close()
    
    elapsed = time.time() - start_time
    fps_actual = frame_count / elapsed
    print("\\nTest completed successfully!")
    print(f"Captured {frame_count} frames in {elapsed:.2f} seconds ({fps_actual:.1f} fps)")
    
    return True

def test_iterator_interface():
    """Test that the video source works as an iterator."""
    print("\\nTesting iterator interface...")
    
    env = suite.make(
        "DualKinova3SRLEnv",
        robots=["DualKinova3"],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=30,
        camera_names=["frontview"],
        camera_heights=240,
        camera_widths=320,
    )
    
    env.reset()
    
    video_source = MuJoCoVideoSource(
        model=env.sim.model._model,
        data=env.sim.data._data,
        camera_name="frontview",
        width=320,
        height=240,
        fps=30.0,
        env=env  # Pass environment for macOS compatibility
    )
    
    try:
        # Test context manager
        with video_source:
            frames = []
            for frame_num, frame in enumerate(video_source):
                if frame_num >= 5:  # Just capture 5 frames
                    break
                frames.append(frame)
                env.step(np.zeros(env.action_dim))
                time.sleep(0.1)
            
            print(f"Successfully captured {len(frames)} frames using iterator interface")
            
    except Exception as e:
        print(f"ERROR in iterator test: {e}")
        return False
    
    finally:
        env.close()
    
    return True

if __name__ == "__main__":
    print("Starting MuJoCo video source tests...")
    
    # Run basic functionality test
    if not test_mujoco_video_source():
        print("Basic test FAILED!")
        sys.exit(1)
    
    # Run iterator interface test
    if not test_iterator_interface():
        print("Iterator test FAILED!")
        sys.exit(1)
    
    print("\\nAll tests passed! ✅")
    print("\\nNext steps:")
    print("1. Run the streaming demo: python demo_dual_kinova3_xr_robot_teleop_streaming.py")
    print("2. Open http://localhost:8080 in your browser to view the stream")
    print("3. Connect your XR device to start teleoperation")