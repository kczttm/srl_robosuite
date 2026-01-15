import h5py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def select_log_file_gui():
    """
    Open file dialog to select log file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    filename = filedialog.askopenfilename(
        title="Select Log File",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    
    root.destroy()
    return filename

def analyze_log_file(filename):
    """Analyze and plot data from log file"""
    print(f"Analyzing file: {os.path.basename(filename)}")
    
    with h5py.File(filename, "r") as f:
        times = f['time'][:]
        contacts = f['contacts'][:]

        # contact_details = f['contact_details'][:]
        contact_geoms1 = f['contact_geom1'][:]
        contact_geoms2 = f['contact_geom2'][:]
        contact_positions = f['contact_position'][:]
        contact_forces = f['contact_force'][:]
        goal_achieved = False
        if (f['goal_achieved']):
            goal_achieved = f['goal_achieved'][0]
            goal_time = f['goal_time'][0]

        # print(np.shape(contacts), np.shape(times), np.shape(contact_geoms1))

        # Find contact events
        # contact_times = times[contacts]
        
        print(f"Total simulation time: {times[-1]:.3f} seconds")
        print(f"Number of contact events: {np.sum(contacts)}")
        print(f"Contact time percentage: {100*np.sum(contacts)/len(contacts):.1f}%")
        if goal_achieved:
            print(f"Successfully completed goal at time {goal_time}")
        else:
            print("Goal not achieved or completed incorrectly.")

        user_in = input("Print all contact details? (y/n): ")

        if user_in.lower() == 'y':
            for i, contact in enumerate(contacts):
                if contact:
                    print(f"Time: {times[i]:.3f}: {contact_geoms1[i]} contact with {contact_geoms2[i]} at position {contact_positions[i]} with force {contact_forces[i]}")

        user_in = input("Show graph of contact events over time? (y/n): ")

        if user_in.lower() == 'y':
            # Plot contact timeline
            plt.figure(figsize=(16, 24))
            plt.subplot(2,1,1)
            plt.plot(times, contacts.astype(int), 'r.', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Robot-Obstacle Contact')
            plt.title(f'Robot-Obstacle Contact Events - {os.path.basename(filename)}')
            plt.grid(True)

            plt.subplot(2,1,2)
            plt.plot(times, contact_forces, 'r.', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Robot-Obstacle Contact Forces (N)')
            plt.title(f'Robot-Obstacle Contact Forces - {os.path.basename(filename)}')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    selected_file = select_log_file_gui()
    if selected_file:
        analyze_log_file(selected_file)
    else:
        print("No file selected.")