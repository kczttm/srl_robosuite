class TeleopKeyCallback:
    """Key callback for teleoperation control"""

    def __init__(self):
        self.reset_requested = False
        self.teleop_enabled = True  # Enabled by default for offline playback
        self.home_requested = False
        self.pause_requested = False

    def __call__(self, key: int) -> None:
        if key == ord("r") or key == ord("R"):
            self.reset_requested = True
            print("Reset requested")
        elif key == ord("q") or key == ord("Q"):
            print("Quit requested")
        elif key == ord("t") or key == ord("T"):
            self.teleop_enabled = not self.teleop_enabled
            status = "ENABLED" if self.teleop_enabled else "DISABLED"
            print(f"Teleoperation {status}")
        elif key == ord("h") or key == ord("H"):
            self.home_requested = True
            print("Home position requested")
        elif key == ord("p") or key == ord("P"):
            self.pause_requested = True
            print("Pause/unpause requested")

