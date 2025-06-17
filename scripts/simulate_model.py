import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from bicycle_model import DynamicBicycleModel
import os

def main():
    """
    Simulates the dynamic bicycle model with a given set of parameters and
    compares the resulting trajectory to actual data from a CSV file.
    """
    # --- Configuration ---
    # Path to the data file
    csv_path = os.path.join(os.getcwd(), 'bag/aggressive_driving/synchronized_data.csv')

    # Define how many seconds of data to simulate
    simulation_duration = 15.0  # seconds

    # Vehicle parameters (use the values found from optimization)
    vehicle_params = {
        'm': 3.74,
        'lf': 0.183,
        'lr': 0.148,
        'Iz': 0.04763,
        'Cf': 4.9636,  # Optimized value
        'Cr': 5.7522,  # Optimized value
        'wheelbase': 0.183 + 0.148,
        'h_cg': 0.20,  # Center of gravity height
        'mu': 0.8     # Coefficient of friction
    }
    
    # --- Load and Process Data ---
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

    # --- Simulation ---
    # Instantiate the model with the specified parameters
    print("Instantiating model with the following parameters:")
    for key, value in vehicle_params.items():
        print(f"  {key}: {value}")
    model = DynamicBicycleModel(**vehicle_params)

    # Set the initial state from the first data point
    # Assume vehicle starts with its initial speed, but zero lateral velocity and yaw rate.
    initial_vx = speed_actual[0] if len(speed_actual) > 0 else 0
    model.set_initial_state(x_actual[0], y_actual[0], yaw_actual[0], initial_vx, 0.0, 0.0)
    
    # Set up the simulation time
    dt = 0.01
    t_span = np.arange(0, simulation_duration, dt)

    # Data logging for simulation results
    history = {
        't': [], 'x': [], 'y': [], 'vx': [], 'vy': [], 'yaw_rate': [],
        'target_speed': [], 'steering_input': []
    }

    # Simple P-controller for speed
    Kp_speed = 1.0

    print("\nRunning simulation...")
    for t in t_span:
        target_speed = speed_interp(t)
        current_steering = steering_interp(t)
        
        # Calculate force command from P-controller
        speed_error = target_speed - model.vx
        Fx_request = Kp_speed * speed_error
        
        x, y, _, vx, vy, yaw_rate = model.step(Fx_request, current_steering, dt)
        
        history['t'].append(t)
        history['x'].append(x)
        history['y'].append(y)
        history['vx'].append(vx)
        history['vy'].append(vy)
        history['yaw_rate'].append(yaw_rate)
        history['target_speed'].append(target_speed)
        history['steering_input'].append(current_steering)
    print("Simulation complete.")

    # --- Plotting ---
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
    ax_inputs1.plot(history['t'], history['target_speed'], 'b-', label='Target Speed [m/s]')
    ax_inputs1.plot(history['t'], history['vx'], 'b--', label='Actual Speed [m/s]')
    ax_inputs2.plot(history['t'], np.rad2deg(history['steering_input']), 'r-', label='Steering [deg]')
    ax_inputs1.set_xlabel('Time [s]')
    ax_inputs1.set_ylabel('Speed [m/s]', color='b')
    ax_inputs2.set_ylabel('Steering Angle [deg]', color='r')
    ax_inputs1.set_title('Control Inputs & Actual Speed')
    ax_inputs1.legend(loc='upper left')
    ax_inputs2.legend(loc='upper right')
    
    # Yaw rate plot
    axes[1, 0].plot(history['t'], np.rad2deg(history['yaw_rate']), label='Simulated Yaw Rate')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Yaw Rate [deg/s]')
    axes[1, 0].set_title('Yaw Rate')
    axes[1, 0].legend()

    # Lateral velocity plot
    axes[1, 1].plot(history['t'], history['vy'], label='Simulated Lateral Velocity')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Lateral Velocity [m/s]')
    axes[1, 1].set_title('Lateral Velocity')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- Error Calculation ---
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