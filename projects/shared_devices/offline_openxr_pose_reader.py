import numpy as np
import pandas as pd

from xr_robot_teleop_server.schemas.body_pose import Bone


class CSVDataReader:
    """Class to read body pose data from CSV file and simulate real-time playback."""

    def __init__(self, csv_file_path, playback_speed=1.0):
        self.csv_file_path = csv_file_path
        self.playback_speed = playback_speed
        self.data = None
        self.current_index = 0
        self.start_time = None
        self.csv_start_timestamp = None
        self.time_column = None
        self.is_relative_time = False
        self.load_data()

    def load_data(self):
        """Load CSV data and prepare for playback."""
        print(f"Loading CSV data from {self.csv_file_path}...")
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Loaded {len(self.data)} rows of body pose data")

        # Auto-detect time column format
        if 'time_elapsed' in self.data.columns:
            self.time_column = 'time_elapsed'
            self.is_relative_time = True
            print("Detected relative time format (time_elapsed)")
        elif 'timestamp' in self.data.columns:
            self.time_column = 'timestamp'
            self.is_relative_time = False
            print("Detected absolute timestamp format")
        else:
            raise ValueError("CSV must contain either 'time_elapsed' or 'timestamp' column")

        # Get unique time values to understand data structure
        unique_times = self.data[self.time_column].unique()
        print(f"Data contains {len(unique_times)} unique time values")

        # Store the first time value for timing calculations
        self.csv_start_timestamp = unique_times[0]

    def get_bones_at_time(self, elapsed_time):
        """Get bone data at a specific elapsed time from start."""
        if self.data is None or len(self.data) == 0:
            return None

        # Calculate target time based on elapsed time and playback speed
        if self.is_relative_time:
            # For relative time (time_elapsed), directly use scaled elapsed time
            target_time = elapsed_time * self.playback_speed
        else:
            # For absolute timestamps, add to start timestamp
            target_time = self.csv_start_timestamp + (elapsed_time * self.playback_speed)

        # Find closest time in data
        unique_times = self.data[self.time_column].unique()
        if self.is_relative_time:
            # For relative time, loop within the duration
            max_time = unique_times[-1]
            if target_time >= max_time:
                target_time = target_time % max_time
        else:
            # For absolute timestamps, loop back to start if we've reached the end
            if target_time >= unique_times[-1]:
                target_time = self.csv_start_timestamp + ((elapsed_time * self.playback_speed) % (unique_times[-1] - self.csv_start_timestamp))

        closest_time_idx = np.argmin(np.abs(unique_times - target_time))
        closest_time = unique_times[closest_time_idx]

        # Get all bone data for this time
        frame_data = self.data[self.data[self.time_column] == closest_time]

        # Convert to Bone objects
        bones = []
        for _, row in frame_data.iterrows():
            id = find_either_or(row, 'bone_id', 'id')
            if id[0].isalpha():  # row is not a bone (probably an action)
                continue
            bone = Bone(
                # id=int(row['bone_id'] or row['id']),
                id=int(id),
                position=[row['pos_x'], row['pos_y'], row['pos_z']],
                rotation=[row['rot_x'], row['rot_y'], row['rot_z'], row['rot_w']]
            )
            bones.append(bone)

        return bones

def find_either_or(d, a, b):
    if a in d:
        return d[a]
    elif b in d:
        return d[b]
