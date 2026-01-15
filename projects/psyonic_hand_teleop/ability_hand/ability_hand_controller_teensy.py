"""
Joint torque controller for Psyonic Ability hands.
Provides position control with gravity compensation for dexterous hand control.

Author: Chuizheng Kong
Date created: 2025-08-05
"""
from .ability_hand_controller import AbilityHandController
import numpy as np
import struct
import serial
import threading
import queue
import time


class AbilityHandControllerTeensy(AbilityHandController):
    """
    Joint torque controller for Psyonic Ability hands using position control
    with gravity compensation.
    """
    
    def __init__(self, mujoco_model, mujoco_data, hand_side='left', 
                 kp=100, kd=0.1, hardware=False, port=None, debug=False):
        """
        Initialize Ability Hand controller.
        
        Args:
            mujoco_model: MuJoCo model
            mujoco_data: MuJoCo data
            hand_side: 'left' or 'right'
            kp: Proportional gain for joint control
            kd: Derivative gain for joint control (auto-computed if None)
            hardware: Whether to connect to real hardware
            debug: Enable debug output
        """

        super().__init__(mujoco_model=mujoco_model, mujoco_data=mujoco_data, hand_side=hand_side, kp=kp, kd=kd, debug=debug)
        
        # Serial Client for Ability Hand Communication
        self.hardware = hardware
        self.client = None
        self.port = port
        self.serial_thread = None
        self.send_queue = None
        self._serial_stop_event = threading.Event()
        self._serial_baudrate = 460800
        self._serial_timeout = 1


        if self.hardware:
            if self.port is None:
                raise ValueError("Port must be specified when hardware=True")
            
            self.send_queue = queue.Queue(maxsize=2)
            self.serial_thread = threading.Thread(
                target=self._serial_worker,
                daemon=True)
            self.serial_thread.start()
            print(f"Serial worker thread started for {self.hand_side} hand on port {self.port}")

        else: 
            self.client = None

    def send_joint_positions_hw(self):
        """
        Update mapped positions and queue them for sending to hardware.
        """
        if self.hardware and self.send_queue is not None:
            self.update_mapped_positions()
            if self.mapped_positions is not None:
                try:
                    self.send_queue.put_nowait(self.mapped_positions)
                except queue.Full:
                    # Drop the oldest item to make room for the latest command
                    try:
                        _ = self.send_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.send_queue.put_nowait(self.mapped_positions)
                    except queue.Full:
                        if self.debug:
                            print(f"[{self.hand_side}] send queue full; dropping packet")

    def update_mapped_positions(self):
        """
        Update the mapped positions for the Ability Hand hardware.
        """
        joint_positions = self.q_current.tolist()
        joint_indexes = [2, 4, 6, 8, 1, 0]

        joint_positions_mapped = np.zeros(6)
        for i, idx in enumerate(joint_indexes):
            if (i < 5):
                joint_positions_mapped[i] = float(np.clip((self.map_pos(joint_positions[idx])), 0, 100))
            else:
                joint_positions_mapped[i] = float(np.clip((self.map_pos(joint_positions[idx])), -100, 0))

        self.mapped_positions = joint_positions_mapped.tolist()
        

    def map_pos(self, val):
        return 180*val/np.pi
    
    def _serial_worker(self):
        """
        This function runs in a separate thread.
        It handles all serial communication to avoid blocking the main loop.
        """
        print(f"[{self.hand_side}] Serial thread starting (background worker) for {self.port}...")

        # Exponential backoff parameters for reconnect attempts (keeps worker non-disruptive)
        backoff = 0.1
        max_backoff = 5.0

        while not self._serial_stop_event.is_set():
            data_to_send = None
            try:
                try:
                    data_to_send = self.send_queue.get(block=True, timeout=0.05)
                    # print(data_to_send)
                    while True:
                        try:
                            data_to_send = self.send_queue.get_nowait()
                        except queue.Empty:
                            break
                except queue.Empty:
                    data_to_send = None

                if data_to_send is None:
                    continue

                if self.client is None:
                    if self.debug:
                        print(f"[{self.hand_side}] Attempting serial connection to {self.port}...")
                    try:
                        self.client = serial.Serial(
                            port=self.port,
                            baudrate=self._serial_baudrate,
                            timeout=self._serial_timeout,
                            write_timeout=0.05,
                        )
                        if self.debug:
                            print(f"[{self.hand_side}] Hand hardware CONNECTED on {self.port}")
                        backoff = 0.1
                    except serial.SerialException as e:
                        if self.debug:
                            print(f"[{self.hand_side}] Serial open failed: {e}; retrying in {backoff:.1f}s")
                        time.sleep(backoff)
                        backoff = min(max_backoff, backoff * 2)
                        continue
                try:
                    packed_data = struct.pack('<6f', *data_to_send)
                    self.client.write(packed_data)
                    if self.debug:
                        pos_str = ", ".join([f"{pos:.1f}" for pos in data_to_send])
                        print(f"[{self.hand_side}] HW_SEND: [{pos_str}]")
                    backoff = 0.1
                except serial.SerialTimeoutException as e:
                    if self.debug:
                        print(f"[{self.hand_side}] Write timed out (drop packet): {e}")
                    continue
                except serial.SerialException as e:
                    print(f"[{self.hand_side}] Serial exception during write: {e}")
                    try:
                        self.client.close()
                    except Exception:
                        pass
                    self.client = None
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 2)
                    continue

            except Exception as e:
                if self.debug:
                    print(f"[{self.hand_side}] Unhandled error in serial thread: {e}")
                time.sleep(0.05)
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
        self.client = None
        if self.debug:
            print(f"[{self.hand_side}] Serial thread stopped.")

    def close(self):
        """Close the Ability Hand hardware client connection if it exists."""
        if self.hardware and self.serial_thread is not None:
            self._serial_stop_event.set()
            self.serial_thread.join(timeout=2)
            if self.serial_thread.is_alive():
                print(f"Warning: {self.hand_side} serial thread did not shut down cleanly.")

        # Close the client if open
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
        