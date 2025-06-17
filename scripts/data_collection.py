import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

PATH = os.getcwd()

vicon_csv = f'{PATH}/bag/aggressive_driving/vicon_car_5.csv'
drive_csv = f'{PATH}/bag/aggressive_driving/drive.csv'

vicon_csv = pd.read_csv(vicon_csv)
drive_csv = pd.read_csv(drive_csv)

x = vicon_csv['pos_x'].to_numpy()
y = vicon_csv['pos_y'].to_numpy()
ts = vicon_csv['timestamp'].to_numpy()

vx = np.diff(x) / (np.diff(ts) * 1e-9)
vy = np.diff(y) / (np.diff(ts) * 1e-9)

qx = vicon_csv['quat_x'].to_numpy()
qy = vicon_csv['quat_y'].to_numpy()
qz = vicon_csv['quat_z'].to_numpy()
qw = vicon_csv['quat_w'].to_numpy()

siny_cosp = 2 * (qw * qz + qx * qy)
cosy_cosp = 1 - 2 * (qy**2 + qz**2)
yaw = np.arctan2(siny_cosp, cosy_cosp)

# extrapolated velocities and yaw
vx = savgol_filter(vx, 51, 3)
vy = savgol_filter(vy, 51, 3)
yaw = savgol_filter(yaw, 51, 3)
yaw_rate = savgol_filter(np.diff(yaw) / (np.diff(ts) * 1e-9), 51, 3)

# drive data
throttle = drive_csv['speed'].to_numpy()
steering = drive_csv['steering_angle'].to_numpy()

# Let's examine the timestamps from both datasets
print("Vicon timestamps:")
print(f"Start: {ts[0]}, End: {ts[-1]}")
print(f"Duration: {(ts[-1] - ts[0])/1e9:.2f} seconds")
print(f"Number of samples: {len(ts)}")
print(f"Average sampling rate: {len(ts)/((ts[-1] - ts[0])/1e9):.2f} Hz")

drive_ts = drive_csv['timestamp'].to_numpy()
print("\nDrive timestamps:")
print(f"Start: {drive_ts[0]}, End: {drive_ts[-1]}")
print(f"Duration: {(drive_ts[-1] - drive_ts[0])/1e9:.2f} seconds")
print(f"Number of samples: {len(drive_ts)}")
print(f"Average sampling rate: {len(drive_ts)/((drive_ts[-1] - drive_ts[0])/1e9):.2f} Hz")

# Check if there's overlap between the timestamps
vicon_start, vicon_end = ts[0], ts[-1]
drive_start, drive_end = drive_ts[0], drive_ts[-1]

overlap_start = max(vicon_start, drive_start)
overlap_end = min(vicon_end, drive_end)

if overlap_start < overlap_end:
    print(f"\nTimestamp overlap: {(overlap_end - overlap_start)/1e9:.2f} seconds")
    print(f"Overlap percentage of Vicon data: {(overlap_end - overlap_start)/(vicon_end - vicon_start)*100:.2f}%")
    print(f"Overlap percentage of Drive data: {(overlap_end - overlap_start)/(drive_end - drive_start)*100:.2f}%")
else:
    print("\nNo timestamp overlap between datasets!")

# Convert timestamps to seconds from start for easier interpretation
vicon_time_sec = (ts - ts[0]) / 1e9
drive_time_sec = (drive_ts - drive_ts[0]) / 1e9

# Plot timestamps to visualize the alignment
plt.figure(figsize=(12, 6))
plt.plot(vicon_time_sec, np.ones_like(vicon_time_sec), 'b|', label='Vicon data')
plt.plot(drive_time_sec, np.ones_like(drive_time_sec)*0.9, 'r|', label='Drive data')
plt.xlabel('Time (seconds from start)')
plt.title('Timestamp Distribution')
plt.legend()
plt.grid(True)
plt.show()

from scipy.interpolate import interp1d

def align_datasets(source_time, source_data, target_time, kind='linear'):
    """
    Interpolate source_data from source_time to target_time
    
    Parameters:
    -----------
    source_time: array of timestamps for source data
    source_data: array of data values to be interpolated
    target_time: array of timestamps to interpolate to
    kind: interpolation method ('linear', 'cubic', etc.)
    
    Returns:
    --------
    Interpolated data at target_time timestamps
    """
    # Convert timestamps to seconds from epoch for interpolation
    source_time_sec = (source_time - source_time[0]) / 1e9
    target_time_sec = (target_time - source_time[0]) / 1e9
    
    # Create interpolation function
    f = interp1d(source_time_sec, source_data, kind=kind, 
                bounds_error=False, fill_value=np.nan)
    
    # Apply interpolation
    return f(target_time_sec)

# Decide on a common timeline - let's use the Vicon timeline as reference
common_time = ts

# Interpolate drive data to Vicon timeline
aligned_throttle = align_datasets(drive_ts, throttle, common_time)
aligned_steering = align_datasets(drive_ts, steering, common_time)

# Check for NaN values (out-of-bounds interpolation)
nan_count_throttle = np.sum(np.isnan(aligned_throttle))
nan_count_steering = np.sum(np.isnan(aligned_steering))

print(f"NaN values in aligned throttle: {nan_count_throttle}/{len(aligned_throttle)} ({nan_count_throttle/len(aligned_throttle)*100:.2f}%)")
print(f"NaN values in aligned steering: {nan_count_steering}/{len(aligned_steering)} ({nan_count_steering/len(aligned_steering)*100:.2f}%)")

# Plot original vs aligned data
plt.figure(figsize=(12, 8))

# Plot throttle
plt.subplot(2, 1, 1)
plt.plot(drive_time_sec, throttle, 'r-', label='Original throttle')
plt.plot(vicon_time_sec, aligned_throttle, 'b-', label='Aligned throttle')
plt.title('Throttle Data Alignment')
plt.ylabel('Speed')
plt.legend()
plt.grid(True)

# Plot steering
plt.subplot(2, 1, 2)
plt.plot(drive_time_sec, steering, 'r-', label='Original steering')
plt.plot(vicon_time_sec, aligned_steering, 'b-', label='Aligned steering')
plt.title('Steering Data Alignment')
plt.xlabel('Time (seconds)')
plt.ylabel('Steering Angle')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Create a synchronized dataset
valid_indices = ~np.isnan(aligned_throttle) & ~np.isnan(aligned_steering)
print(f"Valid synchronized data points: {np.sum(valid_indices)}/{len(ts)} ({np.sum(valid_indices)/len(ts)*100:.2f}%)")

# Create a new dataframe with synchronized data
sync_data = pd.DataFrame({
    'timestamp': ts[valid_indices],
    'pos_x': x[valid_indices],
    'pos_y': y[valid_indices],
    'quat_x': qx[valid_indices],
    'quat_y': qy[valid_indices],
    'quat_z': qz[valid_indices],
    'quat_w': qw[valid_indices],
    'speed': aligned_throttle[valid_indices],
    'steering_angle': aligned_steering[valid_indices],
    'yaw': yaw[valid_indices]
})

# Display the first few rows of the synchronized dataset
print("\nSynchronized dataset (first 5 rows):")
print(sync_data.head())

# Save the synchronized dataset to CSV
output_path = f'{PATH}/bag/aggressive_driving/synchronized_data.csv'
sync_data.to_csv(output_path, index=False)
print(f"Synchronized data saved to: {output_path}")

# Let's also visualize the synchronized data
plt.figure(figsize=(12, 10))

# Plot trajectory with color-coded steering angle
plt.subplot(2, 2, 1)
scatter = plt.scatter(sync_data['pos_x'], sync_data['pos_y'], 
                     c=sync_data['steering_angle'], cmap='coolwarm', s=5)
plt.colorbar(scatter, label='Steering Angle')
plt.title('Trajectory with Steering Angle')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.grid(True)

# Plot trajectory with color-coded speed
plt.subplot(2, 2, 2)
scatter = plt.scatter(sync_data['pos_x'], sync_data['pos_y'], 
                     c=sync_data['speed'], cmap='viridis', s=5)
plt.colorbar(scatter, label='Speed')
plt.title('Trajectory with Speed')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.grid(True)

# Plot time series of steering and yaw
time_sec = (sync_data['timestamp'] - sync_data['timestamp'].iloc[0]) / 1e9
plt.subplot(2, 2, 3)
plt.plot(time_sec.to_numpy(), sync_data['steering_angle'].to_numpy(), 'b-', label='Steering Angle')
plt.plot(time_sec.to_numpy(), sync_data['yaw'].to_numpy(), 'r-', label='Yaw')
plt.title('Steering Angle and Yaw Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (radians)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()