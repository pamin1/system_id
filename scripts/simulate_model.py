import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bicycle_model import DynamicBicycleModel
import os
import sys

sys.path.append(os.path.dirname(__file__))

def main():
    """
    Simulates the dynamic bicycle model with a given set of parameters and
    compares the resulting trajectory to actual data from a CSV file.
    """
    # --- Configuration ---
    # Path to the data file
    csv_path = os.path.join(os.getcwd(), 'bag/aggressive_driving/synchronized_data.csv')

    # Define how many seconds of data to simulate
    simulation_duration = 10.0  # seconds

    # Vehicle parameters (use the values found from optimization)
    vehicle_params = {
        'm': 3.74,
        'lf': 0.183,
        'lr': 0.148,
        'h_cg': 0.12,
        'Iy': 0.02,
        'Iz': 5.05168963e-02,
        'Cf': 4.29062283e+02,
        'Cr': 5.57480744e+02,
        'alpha_sat': np.deg2rad(7.0),
    }
    
    # Load and Process Data
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Convert timestamp to seconds from start and filter data
    time_all = ((data['timestamp'] - data['timestamp'].iloc[0]) / 1e9).to_numpy()
    
    # Filter data to the desired simulation duration
    mask = time_all <= simulation_duration
    time_actual = time_all[mask]

    if len(time_actual) < 2:  # Need at least 2 points for interpolation
        print(f"Error: Not enough data points for a {simulation_duration}s simulation.")
        return

    x_actual = data['pos_x'].to_numpy()[mask]
    y_actual = data['pos_y'].to_numpy()[mask]
    yaw_actual = data['yaw'].to_numpy()[mask]
    speed_actual = data['speed'].to_numpy()[mask]
    steering_actual = data['steering_angle'].to_numpy()[mask]

    # Create interpolation functions for control inputs
    speed_interp = interp1d(time_actual, speed_actual, kind='linear', fill_value="extrapolate")
    steering_interp = interp1d(time_actual, steering_actual, kind='linear', fill_value="extrapolate")

    # Simulation
    # Instantiate the model with the specified parameters
    print("Instantiating model with the following parameters:")
    for key, value in vehicle_params.items():
        print(f"  {key}: {value}")
    model = DynamicBicycleModel(**vehicle_params)

    # Set the initial state from the first data point
    model.set_initial_state(
        x_actual[0], y_actual[0],
        yaw_actual[0],      # psi
        speed_actual[0],    # initial u
        0.0,                # v lateral
        0.0,                # r yaw-rate
        0.0,                # theta pitch
        0.0                 # q pitch rate
    )
    
    # Set up the simulation time
    dt = 0.01
    t_span = np.arange(0, simulation_duration, dt)

    print("\nRunning simulation...")
    
    # Data logging for simulation results
    history = {
        't': [], 'x': [], 'y': [], 'psi': [], 'v_lat': [], 'yaw_rate': [],
        'speed_input': [], 'steering_input': []
    }
    
    for t in t_span:
        current_speed = speed_interp(t)
        current_steering = steering_interp(t)
        
        # Step the model (returns full state)
        state = model.step(current_speed, current_steering, dt)
        x, y, psi, u, v_lat, yaw_rate, theta, q = state
        
        history['t'].append(t)
        history['x'].append(x)
        history['y'].append(y)
        history['psi'].append(psi)
        history['v_lat'].append(v_lat)
        history['yaw_rate'].append(yaw_rate)
        history['speed_input'].append(current_speed)
        history['steering_input'].append(current_steering)
        
    print("Simulation complete.")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dynamic Bicycle Model Simulation vs. Actual Data', fontsize=16)

    # Trajectory plot
    axes[0, 0].plot(history['x'], history['y'], 'b-', label='Simulated Trajectory (Optimized)')
    axes[0, 0].plot(x_actual, y_actual, 'r--', label='Actual Trajectory (CSV)')
    axes[0, 0].plot(x_actual[0], y_actual[0], 'go', markersize=10, label='Start')
    axes[0, 0].plot(x_actual[-1], y_actual[-1], 'ro', markersize=10, label='End')
    axes[0, 0].set_xlabel('X Position [m]')
    axes[0, 0].set_ylabel('Y Position [m]')
    axes[0, 0].set_title('Vehicle Trajectory Comparison')
    axes[0, 0].axis('equal')
    axes[0, 0].legend()

    # Inputs plot
    ax_inputs1 = axes[0, 1]
    ax_inputs2 = ax_inputs1.twinx()
    ax_inputs1.plot(history['t'], history['speed_input'], 'b-', label='Speed Command [m/s]')
    ax_inputs2.plot(history['t'], np.rad2deg(history['steering_input']), 'r-', label='Steering [deg]')
    ax_inputs1.set_xlabel('Time [s]')
    ax_inputs1.set_ylabel('Speed [m/s]', color='b')
    ax_inputs2.set_ylabel('Steering Angle [deg]', color='r')
    ax_inputs1.set_title('Control Inputs')
    ax_inputs1.legend(loc='upper left')
    ax_inputs2.legend(loc='upper right')
    
    # Yaw rate plot
    axes[1, 0].plot(history['t'], np.rad2deg(history['yaw_rate']), label='Simulated Yaw Rate')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Yaw Rate [deg/s]')
    axes[1, 0].set_title('Yaw Rate')
    axes[1, 0].legend()

    # Lateral velocity plot
    axes[1, 1].plot(history['t'], history['v_lat'], label='Simulated Lateral Velocity')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Lateral Velocity [m/s]')
    axes[1, 1].set_title('Lateral Velocity')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Error Calculation
    # Interpolate actual data to simulation time points for a fair comparison
    x_actual_interp = np.interp(history['t'], time_actual, x_actual)
    y_actual_interp = np.interp(history['t'], time_actual, y_actual)
    
    # Calculate position error
    position_error = np.sqrt((np.array(history['x']) - x_actual_interp)**2 + (np.array(history['y']) - y_actual_interp)**2)

    print("\nError Metrics:")
    print(f"Mean position error: {np.mean(position_error):.3f} m")
    print(f"Max position error: {np.max(position_error):.3f} m")
    print(f"RMS position error: {np.sqrt(np.mean(position_error**2)):.3f} m")

if __name__ == '__main__':
    main()