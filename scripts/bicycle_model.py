import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import copy

class DynamicBicycleModel:
    """
    Python implementation of a dynamic bicycle model, including longitudinal
    load transfer and friction ellipse constraints.
    """
    def __init__(self, m, lf, lr, Iz, Cf, Cr, wheelbase, h_cg, mu=1.0, g=9.81):
        # Vehicle parameters
        self.m = m          # mass [kg]
        self.lf = lf        # distance from CG to front axle [m]
        self.lr = lr        # distance from CG to rear axle [m]
        self.Iz = Iz        # yaw moment of inertia [kg·m²]
        self.Cf = Cf        # front cornering stiffness [N/rad]
        self.Cr = Cr        # rear cornering stiffness [N/rad]
        self.wheelbase = wheelbase # wheelbase [m]
        self.h_cg = h_cg    # height of center of gravity [m]
        self.mu = mu        # coefficient of friction
        self.g = g          # acceleration due to gravity [m/s^2]

        # State variables [x, y, yaw, vx, vy, yaw_rate]
        self.x = 0.0        # global x position [m]
        self.y = 0.0        # global y position [m]
        self.yaw = 0.0      # yaw angle [rad]
        self.vx = 0.0       # longitudinal velocity (body frame) [m/s]
        self.vy = 0.0       # lateral velocity (body frame) [m/s]
        self.yaw_rate = 0.0 # yaw rate [rad/s]

    def set_initial_state(self, x, y, yaw, vx, vy, yaw_rate):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.yaw_rate = yaw_rate

    def _calculate_tire_forces(self, Fx_request, delta):
        # Distribute longitudinal force (front-wheel drive assumption)
        Fx_f = Fx_request
        Fx_r = 0.0

        # Calculate longitudinal acceleration for load transfer (using requested force)
        ax = Fx_request / self.m

        # Calculate vertical tire forces with load transfer
        Fz_f = self.m * self.g * self.lr / self.wheelbase - self.m * self.h_cg * ax / self.wheelbase
        Fz_r = self.m * self.g * self.lf / self.wheelbase + self.m * self.h_cg * ax / self.wheelbase
        Fz_f = max(0, Fz_f) # Prevent negative normal force
        Fz_r = max(0, Fz_r)

        # Calculate slip angles
        if abs(self.vx) < 0.1:
            alpha_f = 0.0
            alpha_r = 0.0
        else:
            alpha_f = np.arctan2(self.vy + self.lf * self.yaw_rate, self.vx) - delta
            alpha_r = np.arctan2(self.vy - self.lr * self.yaw_rate, self.vx)

        # Calculate potential lateral forces (linear tire model)
        Fy_f_potential = -self.Cf * alpha_f
        Fy_r_potential = -self.Cr * alpha_r

        # Apply friction ellipse constraints
        # Front tire
        F_friction_f_max = self.mu * Fz_f
        Fx_f = np.clip(Fx_f, -F_friction_f_max, F_friction_f_max)
        Fy_f_max_available = np.sqrt(max(0, F_friction_f_max**2 - Fx_f**2))
        Fy_f = np.clip(Fy_f_potential, -Fy_f_max_available, Fy_f_max_available)

        # Rear tire
        F_friction_r_max = self.mu * Fz_r
        Fy_r = np.clip(Fy_r_potential, -F_friction_r_max, F_friction_r_max)

        return Fx_f, Fx_r, Fy_f, Fy_r

    def step(self, Fx_request, delta, dt):
        """
        Updates the vehicle state for one time step dt.
        This model uses a full dynamic state and includes load transfer.

        :param Fx_request: Requested total longitudinal force [N]
        :param delta: steering angle [rad]
        :param dt: time step [s]
        """
        # Get tire forces with load transfer and friction constraints
        Fx_f, Fx_r, Fy_f, Fy_r = self._calculate_tire_forces(Fx_request, delta)

        # Equations of Motion (body frame)
        vx_dot = (Fx_f * np.cos(delta) - Fy_f * np.sin(delta) + Fx_r) / self.m + self.vy * self.yaw_rate
        vy_dot = (Fx_f * np.sin(delta) + Fy_f * np.cos(delta) + Fy_r) / self.m - self.vx * self.yaw_rate
        yaw_rate_dot = (self.lf * (Fx_f * np.sin(delta) + Fy_f * np.cos(delta)) - self.lr * Fy_r) / self.Iz

        # Integrate vehicle state
        self.vx += vx_dot * dt
        self.vy += vy_dot * dt
        self.yaw_rate += yaw_rate_dot * dt

        # Global frame kinematics
        x_dot = self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)
        y_dot = self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)
        
        # Integrate global pose
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.yaw += self.yaw_rate * dt

        return self.x, self.y, self.yaw, self.vx, self.vy, self.yaw_rate

def run_simulation(model, t_span, dt, speed_interp, steering_interp):
    """
    Runs the simulation for a given model and returns the trajectory history.
    Uses a P-controller to track a target speed.
    """
    history = {
        't': [], 'x': [], 'y': [], 'yaw': [], 'vx': [], 'vy': [], 'yaw_rate': [], 
        'speed_input': [], 'steering_input': []
    }
    
    # Simple P-controller for speed
    Kp_speed = 2.0

    for t in t_span:
        target_speed = speed_interp(t)
        current_steering = steering_interp(t)
        
        # Calculate force command from P-controller
        speed_error = target_speed - model.vx
        Fx_request = Kp_speed * speed_error
        
        # Step the model
        x, y, yaw, vx, vy, yaw_rate = model.step(Fx_request, current_steering, dt)
        
        history['t'].append(t)
        history['x'].append(x)
        history['y'].append(y)
        history['yaw'].append(yaw)
        history['vx'].append(vx)
        history['vy'].append(vy)
        history['yaw_rate'].append(yaw_rate)
        history['speed_input'].append(target_speed)
        history['steering_input'].append(current_steering)
        
    return history

def objective_function(params, vehicle_params, initial_state, t_span, dt, speed_interp, steering_interp, x_actual, y_actual, time_actual):
    """
    Cost function for optimization. Calculates the root-mean-square error
    between simulated and actual trajectory.
    """
    # Unpack parameters to be optimized
    Cf, Cr = params
    
    # Create a copy of the vehicle params and update it
    opt_vehicle_params = copy.deepcopy(vehicle_params)
    opt_vehicle_params['Cf'] = Cf
    opt_vehicle_params['Cr'] = Cr
    
    # Create model with new params
    model = DynamicBicycleModel(**opt_vehicle_params)
    model.set_initial_state(*initial_state)

    # Run simulation
    sim_history = run_simulation(model, t_span, dt, speed_interp, steering_interp)

    # Check for invalid simulation results (NaN, Inf)
    sim_x = np.array(sim_history['x'])
    if np.any(np.isnan(sim_x)) or np.any(np.isinf(sim_x)):
        return 1e12  # Large penalty for instability

    # Interpolate actual data to simulation time points for comparison
    x_actual_interp = np.interp(sim_history['t'], time_actual, x_actual)
    y_actual_interp = np.interp(sim_history['t'], time_actual, y_actual)

    # Calculate Root Mean Square Error for position
    error_x = sim_x - x_actual_interp
    error_y = np.array(sim_history['y']) - y_actual_interp
    rms_error = np.sqrt(np.mean(error_x**2 + error_y**2))
    
    # Add a penalty for very large errors, which also indicates instability
    if rms_error > 500:  # If trajectory is off by >500m, it's unstable
        return 1e12
        
    return rms_error

def main():
    """
    Main function to load data, run optimization to fit model parameters,
    and plot the final comparison.
    """
    # Load data from CSV
    try:
        csv_path = 'bag/aggressive_driving/synchronized_data.csv'
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Process data
    time_csv = ((data['timestamp'] - data['timestamp'].iloc[0]) / 1e9).to_numpy()
    speed_csv = data['speed'].to_numpy()
    steering_csv = data['steering_angle'].to_numpy()
    x_actual = data['pos_x'].to_numpy()
    y_actual = data['pos_y'].to_numpy()
    yaw_actual = data['yaw'].to_numpy()

    # Create interpolation functions for control inputs
    speed_interp = interp1d(time_csv, speed_csv, kind='linear', fill_value="extrapolate")
    steering_interp = interp1d(time_csv, steering_csv, kind='linear', fill_value="extrapolate")

    # Initial vehicle parameters (base for optimization)
    vehicle_params = {
        'm': 3.74, 'lf': 0.183, 'lr': 0.148, 'Iz': 0.04763,
        'Cf': 4.718, 'Cr': 5.4562, 'wheelbase': 0.183 + 0.148,
        'h_cg': 0.20, 'mu': 0.8
    }

    # Set initial state from data
    # Assume vehicle starts with zero lateral velocity and yaw rate
    initial_vx = speed_csv[0] if len(speed_csv) > 0 else 0
    initial_state = (x_actual[0], y_actual[0], yaw_actual[0], initial_vx, 0.0, 0.0)
    
    # --- Parameter Optimization ---
    initial_params = [vehicle_params['Cf'], vehicle_params['Cr']]
    param_bounds = [(0.1, 20.0), (0.1, 20.0)] 

    # Simulation time parameters
    dt = 0.01
    sim_time = time_csv[-1]
    t_span = np.arange(0, sim_time, dt)

    print("Starting parameter optimization...")
    result = minimize(
        objective_function,
        initial_params,
        args=(vehicle_params, initial_state, t_span, dt, speed_interp, steering_interp, x_actual, y_actual, time_csv),
        method='Nelder-Mead',
        options={'disp': True, 'xatol': 1e-4, 'fatol': 1e-4}
    )

    if result.success:
        optimized_params = result.x
        vehicle_params['Cf'] = optimized_params[0]
        vehicle_params['Cr'] = optimized_params[1]
        print("\nOptimization successful.")
        print(f"Final cost (RMSE): {result.fun:.4f} m")
        print(f"Optimized Cf: {vehicle_params['Cf']:.4f} N/rad")
        print(f"Optimized Cr: {vehicle_params['Cr']:.4f} N/rad")
    else:
        print("\nOptimization failed. Using initial parameters.")
        print(result.message)
    # --- End of Optimization ---

    # Run final simulation with the best parameters
    print("\nRunning final simulation with optimized parameters...")
    final_model = DynamicBicycleModel(**vehicle_params)
    final_model.set_initial_state(*initial_state)
    history = run_simulation(final_model, t_span, dt, speed_interp, steering_interp)

    # Plotting results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimized Dynamic Bicycle Model vs. Actual Data', fontsize=16)

    # Trajectory plot
    axes[0, 0].plot(history['x'], history['y'], 'b-', label='Simulated Trajectory (Optimized)')
    axes[0, 0].plot(x_actual, y_actual, 'r--', label='Actual Trajectory (CSV)')
    axes[0, 0].set_xlabel('X Position [m]')
    axes[0, 0].set_ylabel('Y Position [m]')
    axes[0, 0].set_title('Vehicle Trajectory Comparison')
    axes[0, 0].axis('equal')
    axes[0, 0].legend()

    # Inputs plot
    ax_inputs1 = axes[0, 1]
    ax_inputs2 = ax_inputs1.twinx()
    ax_inputs1.plot(history['t'], history['speed_input'], 'b-', label='Target Speed [m/s]')
    ax_inputs1.plot(history['t'], history['vx'], 'b--', label='Actual Speed [m/s]')
    ax_inputs2.plot(history['t'], np.rad2deg(history['steering_input']), 'r-', label='Steering [deg]')
    ax_inputs1.set_xlabel('Time [s]')
    ax_inputs1.set_ylabel('Speed [m/s]', color='b')
    ax_inputs2.set_ylabel('Steering Angle [deg]', color='r')
    ax_inputs1.set_title('Control Inputs (from CSV)')
    ax_inputs1.legend(loc='upper left')
    ax_inputs2.legend(loc='upper right')
    
    # Yaw rate plot
    axes[1, 0].plot(history['t'], np.rad2deg(history['yaw_rate']), label='Simulated yaw_rate')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Yaw Rate [deg/s]')
    axes[1, 0].set_title('Yaw Rate')
    axes[1, 0].legend()

    # Lateral velocity plot
    axes[1, 1].plot(history['t'], history['vy'], label='Simulated v_y')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Lateral Velocity [m/s]')
    axes[1, 1].set_title('Lateral Velocity')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

if __name__ == '__main__':
    main()
