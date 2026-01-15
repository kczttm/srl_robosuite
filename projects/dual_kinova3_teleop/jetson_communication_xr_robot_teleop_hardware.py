import serial
import time
import struct
import threading
import queue


class JestonCommunication():
    def __init__(self, port=None, baudrate=9600, timeout=0.1, debug=False):
        self.port = port
        self.client = None
        self.debug = debug 

        # Serial communication parameters
        self._serial_baudrate = baudrate
        self._serial_timeout = timeout  # seconds
        self.send_queue = queue.Queue()
        self._serial_stop_event = threading.Event()
        self.serial_thread = threading.Thread(target=self._serial_worker, daemon=True)
        self.serial_thread.start()
        
        
    def _serial_worker(self):
            """
            This function runs in a separate thread.
            It handles all serial communication to avoid blocking the main loop.
            """

            # Exponential backoff parameters for reconnect attempts (keeps worker non-disruptive)
            backoff = 0.1
            max_backoff = 5.0

            while not self._serial_stop_event.is_set():
                data_to_send = None
                try:
                    try:
                        # Wait up to 0.5s for data so we can notice stop requests quickly
                        data_to_send = self.send_queue.get(block=True, timeout=0.5)
                        print(data_to_send)
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
                            print(f"Attempting serial connection to {self.port}...")
                        try:
                            self.client = serial.Serial(
                                port=self.port,
                                baudrate=self._serial_baudrate,
                                timeout=self._serial_timeout,
                                write_timeout=0.05,
                            )
                            if self.debug:
                                print(f"Jetson CONNECTED on {self.port}")
                            backoff = 0.1
                        except serial.SerialException as e:
                            if self.debug:
                                print(f"Serial open failed: {e}; retrying in {backoff:.1f}s")
                            time.sleep(backoff)
                            backoff = min(max_backoff, backoff * 2)
                            continue
                    try:
                        packed_data = struct.pack('<2f', *data_to_send)
                        self.client.write(packed_data)
                        if self.debug:
                            pos_str = ", ".join([f"{pos:.1f}" for pos in data_to_send])
                            print(f"HW_SEND: [{pos_str}]")
                        backoff = 0.1
                    except serial.SerialTimeoutException as e:
                        if self.debug:
                            print(f"Write timed out (drop packet): {e}")
                        continue
                    except serial.SerialException as e:
                        print(f"Serial exception during write: {e}")
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
                        print(f"Unhandled error in serial thread: {e}")
                    time.sleep(0.05)
            try:
                if self.client:
                    self.client.close()
            except Exception:
                pass
            self.client = None
            if self.debug:
                print(f"Serial thread stopped.")

    def close(self):
        """Close the Jetson client connection if it exists."""
        if self.hardware and self.serial_thread is not None:
            self._serial_stop_event.set()
            self.serial_thread.join(timeout=2)
            if self.serial_thread.is_alive():
                print(f"Warning: Jetson serial thread did not shut down cleanly.")

        # Close the client if open
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
