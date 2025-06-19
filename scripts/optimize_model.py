import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from bicycle_model import DynamicBicycleModel
import os
import sys

sys.path.append(os.path.dirname(__file__))

# --- Objective Function ---
def objective_function(params, static_params, time_actual, x_actual, y_actual, yaw_actual, speed_actual, speed_interp, steering_interp, sim_duration):
    """
    Calculates the RMS position error for a given set of vehicle parameters.
    """
    Cf, Cr, Iz = params
    
    # Combine static and optimized parameters
    vehicle_params = {
        'm': static_params['m'],
        'lf': static_params['lf'],
        'lr': static_params['lr'],
        'h_cg': static_params['h_cg'],
        'Iy': static_params['Iy'],
        'Iz': Iz,
        'Cf': Cf,
        'Cr': Cr,
        'alpha_sat': static_params['alpha_sat'],
    }

    # --- Simulation ---
    model = DynamicBicycleModel(**vehicle_params)
    model.set_initial_state(
        x_actual[0], y_actual[0],
        yaw_actual[0],
        speed_actual[0],
        0.0, 0.0, 0.0, 0.0
    )

    dt = 0.01
    t_span = np.arange(0, sim_duration, dt)
    history = {'t': [], 'x': [], 'y': []}

    for t in t_span:
        current_speed = speed_interp(t)
        current_steering = steering_interp(t)
        
        state = model.step(current_speed, current_steering, dt)
        
        history['t'].append(t)
        history['x'].append(state[0])
        history['y'].append(state[1])

    # --- Error Calculation ---
    x_actual_interp = np.interp(history['t'], time_actual, x_actual)
    y_actual_interp = np.interp(history['t'], time_actual, y_actual)
    
    position_error = np.sqrt((np.array(history['x']) - x_actual_interp)**2 + (np.array(history['y']) - y_actual_interp)**2)
    rms_error = np.sqrt(np.mean(position_error**2))

    # Print progress
    print(f"Params: [Cf: {Cf:.2f}, Cr: {Cr:.2f}, Iz: {Iz:.4f}] -> RMS Error: {rms_error:.4f}")

    return rms_error

def main():
    """
    Uses optimization to find the best vehicle parameters to minimize
    trajectory error against actual data.
    """
    # --- Configuration ---
    csv_path = os.path.join(os.getcwd(), 'bag/aggressive_driving/synchronized_data.csv')
    simulation_duration = 20.0  # Use a longer duration for better tuning

    # --- Load and Process Data ---
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    time_all = ((data['timestamp'] - data['timestamp'].iloc[0]) / 1e9).to_numpy()
    mask = time_all <= simulation_duration
    time_actual = time_all[mask]

    x_actual = data['pos_x'].to_numpy()[mask]
    y_actual = data['pos_y'].to_numpy()[mask]
    yaw_actual = data['yaw'].to_numpy()[mask]
    speed_actual = data['speed'].to_numpy()[mask]
    steering_actual = data['steering_angle'].to_numpy()[mask]

    speed_interp = interp1d(time_actual, speed_actual, kind='linear', fill_value="extrapolate")
    steering_interp = interp1d(time_actual, steering_actual, kind='linear', fill_value="extrapolate")
    
    # --- Optimization ---
    # Define static (non-optimized) parameters
    static_params = {
        'm': 3.74, 'lf': 0.183, 'lr': 0.148,
        'h_cg': 0.12, 'Iy': 0.02,
        'alpha_sat': np.deg2rad(7.0),
    }

    # Initial guesses for the parameters to be optimized [Cf, Cr, Iz]
    initial_params = [470.0, 540.0, 0.04763]

    # Bounds for the parameters [ (Cf_min, Cf_max), (Cr_min, Cr_max), (Iz_min, Iz_max) ]
    param_bounds = [(100, 2000), (100, 2000), (0.01, 0.1)]

    # Arguments to pass to the objective function
    args = (static_params, time_actual, x_actual, y_actual, yaw_actual, speed_actual, speed_interp, steering_interp, simulation_duration)

    print("--- Starting Parameter Optimization ---")
    result = minimize(
        objective_function,
        initial_params,
        args=args,
        method='Nelder-Mead',  # A good gradient-free method
        bounds=param_bounds,
        options={'disp': True, 'maxiter': 100, 'fatol': 1e-4}
    )

    print("\n--- Optimization Finished ---")
    optimized_params = result.x
    print(f"Final RMS Error: {result.fun:.4f}")
    print(f"Optimized Parameters [Cf, Cr, Iz]: {optimized_params}")

    # --- Plotting with Optimized Parameters ---
    Cf_opt, Cr_opt, Iz_opt = optimized_params
    vehicle_params_opt = static_params.copy()
    vehicle_params_opt.update({'Cf': Cf_opt, 'Cr': Cr_opt, 'Iz': Iz_opt})
    
    model = DynamicBicycleModel(**vehicle_params_opt)
    model.set_initial_state(
        x_actual[0], y_actual[0], yaw_actual[0],
        speed_actual[0], 0.0, 0.0, 0.0, 0.0
    )

    dt = 0.01
    t_span = np.arange(0, simulation_duration, dt)
    history = {
        't': [], 'x': [], 'y': []
    }

    for t in t_span:
        state = model.step(speed_interp(t), steering_interp(t), dt)
        x, y, psi, u, v_lat, yaw_rate, theta, q = state
        history['t'].append(t)
        history['x'].append(x)
        history['y'].append(y)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(history['x'], history['y'], 'b-', label='Simulated Trajectory (Optimized)')
    ax.plot(x_actual, y_actual, 'r--', label='Actual Trajectory (CSV)')
    ax.plot(x_actual[0], y_actual[0], 'go', markersize=10, label='Start')
    ax.plot(x_actual[-1], y_actual[-1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_title('Optimized Vehicle Trajectory Comparison')
    ax.axis('equal')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main() 