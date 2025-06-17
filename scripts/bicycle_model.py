# # Dynamic Bicycle Model with Longitudinal Load Transfer
# 
# This notebook implements a dynamic bicycle model that includes:
# - Longitudinal and lateral dynamics
# - Yaw dynamics
# - Longitudinal load transfer effects
# - Tire forces using simplified Pacejka magic formula
# 
# ## Model Parameters and Variables
# 
# ### State Variables
# - X, Y: Global position coordinates
# - ψ (psi): Yaw angle
# - V_x: Longitudinal velocity
# - V_y: Lateral velocity
# - ψ_dot: Yaw rate
# 
# ### Input Variables
# - δ (delta): Steering angle
# - T: Drive/brake torque
# 
# ### Vehicle Parameters
# - m: Total mass
# - I_z: Yaw moment of inertia
# - l_f: Distance from CG to front axle
# - l_r: Distance from CG to rear axle
# - h_cg: Height of center of gravity
# - C_αf, C_αr: Cornering stiffness (front, rear)
# 

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

PATH = os.getcwd()
vicon_csv = f'{PATH}/bag/aggressive_driving/vicon_car_5.csv'
drive_csv = f'{PATH}/bag/aggressive_driving/drive.csv'

# Vehicle parameters (example values for a passenger car)
class VehicleParams:
    def __init__(self):
        # Mass and inertia
        self.m = 3.74  # Total mass [kg]
        self.I_z = 0.04763  # Yaw moment of inertia [kg⋅m²]
        
        # Geometric parameters
        self.l_f = 0.183  # Distance from CG to front axle [m]
        self.l_r = 0.148  # Distance from CG to rear axle [m]
        self.h_cg = 0.20 # CG height [m]
        self.wheelbase = self.l_f + self.l_r
        
        # Tire parameters (simplified)
        self.C_af = 4.7180  # Front cornering stiffness [N/rad]
        self.C_ar = 5.4562  # Rear cornering stiffness [N/rad]
        
        # Tire parameters
        self.mu = 1.0  # Friction coefficient
        self.g = 9.81  # Gravity [m/s²]

vehicle = VehicleParams()

def calculate_tire_forces(state, delta, Fx_request, vehicle):
    """
    Returns longitudinal & lateral forces at each axle.
    Fx_request: desired total longitudinal force (+drive, -brake) [N]
                Positive = acceleration.
    """
    _, _, _, Vx, Vy, r = state
    m, g = vehicle.m, vehicle.g
    lf, lr = vehicle.l_f, vehicle.l_r
    L = lf + lr

    # slip angles
    if abs(Vx) > 0.1:
        alpha_f = np.arctan2(Vy + lf * r, Vx) - delta
        alpha_r = np.arctan2(Vy - lr * r, Vx)
    else:
        alpha_f = alpha_r = 0.0

    # split longitudinal force
    Fx_f = Fx_request   # all drive/brake on front axle
    Fx_r = 0.0

    # load transfer
    ax = (Fx_f + Fx_r) / m
    dFz =  m * vehicle.h_cg * ax / L

    Fz_f = m * g * (lr / L) - dFz
    Fz_r = m * g * (lf / L) + dFz

    # lateral tire forces
    Fy_f = -vehicle.C_af * alpha_f
    Fy_r = -vehicle.C_ar * alpha_r

    # friction saturation
    mu = vehicle.mu
    Fy_f = np.clip(Fy_f, -mu * Fz_f,  mu * Fz_f)
    Fy_r = np.clip(Fy_r, -mu * Fz_r,  mu * Fz_r)

    return Fx_f, Fx_r, Fy_f, Fy_r, Fz_f, Fz_r


def bicycle_model_dynamics(t, state, delta, Fx_request, vehicle):
    """
    EoM for a load-transfer bicycle (body-frame).
    state = [X, Y, psi, Vx, Vy, r]
    """
    X, Y, psi, Vx, Vy, r = state
    m, Iz = vehicle.m, vehicle.I_z
    lf, lr = vehicle.l_f, vehicle.l_r

    # get tire forces
    Fx_f, Fx_r, Fy_f, Fy_r, *_ = calculate_tire_forces(
        state, delta, Fx_request, vehicle
    )

    # get translational dynamics
    Vx_dot = (Fx_f * np.cos(delta) - Fy_f * np.sin(delta) + Fx_r) / m + Vy * r
    Vy_dot = (Fx_f * np.sin(delta) + Fy_f * np.cos(delta) + Fy_r) / m - Vx * r

    # yaw
    r_dot = (lf * (Fx_f * np.sin(delta) + Fy_f * np.cos(delta)) - lr * Fy_r) / Iz

    # inertial kinematics
    X_dot = Vx * np.cos(psi) - Vy * np.sin(psi)
    Y_dot = Vx * np.sin(psi) + Vy * np.cos(psi)
    psi_dot = r

    return np.array([X_dot, Y_dot, psi_dot, Vx_dot, Vy_dot, r_dot])

def simulate_vehicle(t_span, initial_state, steering_interp, speed_interp, vehicle):
    """
    Simulate the vehicle dynamics
    
    Parameters:
    -----------
    t_span: tuple (t_start, t_end)
    initial_state: list [X, Y, psi, Vx, Vy, psi_dot]
    steering_interp: function that returns steering angle at time t
    speed_interp: function that returns target speed at time t
    vehicle: VehicleParams object
    """
    def dynamics_wrapper(t, state):
        delta = steering_interp(t)
        
        # P-controller for longitudinal speed
        Vx = state[3]
        target_Vx = speed_interp(t)
        Kp = 1.0  # Reduced proportional gain to prevent system stiffness
        Fx_request = Kp * (target_Vx - Vx)
        
        return bicycle_model_dynamics(t, state, delta, Fx_request, vehicle)
    
    # Solve ODE
    solution = solve_ivp(
        dynamics_wrapper,
        t_span,
        initial_state,
        method='BDF',
        t_eval=np.linspace(t_span[0], t_span[1], 1000),
        rtol=1e-6,
        atol=1e-6
    )
    
    return solution

def interpolate_actual_data(t_sim, actual_time, actual_data):
    return np.interp(t_sim, actual_time, actual_data)

# Load and process data
drive_data = pd.read_csv(drive_csv)
vicon_data = pd.read_csv(vicon_csv)

# Convert timestamps to seconds from start
drive_data['time'] = (drive_data['timestamp'] - drive_data['timestamp'].iloc[0]) / 1e9
vicon_data['time'] = (vicon_data['timestamp'] - vicon_data['timestamp'].iloc[0]) / 1e9

# Convert DataFrame columns to numpy arrays
time_array = drive_data['time'].to_numpy()
steering_array = drive_data['steering_angle'].to_numpy()
speed_array = drive_data['speed'].to_numpy()

vicon_time = vicon_data['time'].to_numpy()
vicon_x = vicon_data['pos_x'].to_numpy()
vicon_y = vicon_data['pos_y'].to_numpy()
vicon_quat_z = vicon_data['quat_z'].to_numpy()
vicon_quat_w = vicon_data['quat_w'].to_numpy()
vicon_vel_x = vicon_data['lin_vel_x'].to_numpy()
vicon_vel_y = vicon_data['lin_vel_y'].to_numpy()
vicon_ang_vel_z = vicon_data['ang_vel_z'].to_numpy()

# Create interpolation functions for control inputs
steering_interp = interp1d(time_array, steering_array, 
                          kind='linear', fill_value='extrapolate')
speed_interp = interp1d(time_array, speed_array,
                       kind='linear', fill_value='extrapolate')

# Get simulation time span - use the shorter of the two datasets
t_span = (0, min(time_array[-1], vicon_time[-1]))

# Set initial state from Vicon data
# Convert quaternion to yaw angle
yaw = 2 * np.arctan2(vicon_quat_z[0], vicon_quat_w[0])

initial_state = [
    vicon_x[0],          # X
    vicon_y[0],          # Y
    yaw,                 # psi
    vicon_vel_x[0],      # Vx
    vicon_vel_y[0],      # Vy
    vicon_ang_vel_z[0]   # psi_dot
]

# Run simulation
print("Running simulation...")
solution = simulate_vehicle(t_span, initial_state, steering_interp, speed_interp, vehicle)
print("Simulation complete")

# Create comparison plots
plt.figure(figsize=(15, 10))

# Plot 1: Trajectory Comparison
plt.subplot(2, 2, 1)
plt.plot(solution.y[0], solution.y[1], 'b-', label='Model')
plt.plot(vicon_x, vicon_y, 'r--', label='Vicon')
plt.plot(initial_state[0], initial_state[1], 'go', label='Start')
plt.plot(vicon_x[-1], vicon_y[-1], 'ro', label='End')
plt.xlabel('X Position [m]')
plt.ylabel('Y Position [m]')
plt.title('Vehicle Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Plot 2: Velocity Comparison
plt.subplot(2, 2, 2)
plt.plot(solution.t, solution.y[3], 'b-', label='Model v_x')
plt.plot(solution.t, solution.y[4], 'g-', label='Model v_y')
plt.plot(vicon_time, vicon_vel_x, 'r--', label='Vicon v_x')
plt.plot(vicon_time, vicon_vel_y, 'm--', label='Vicon v_y')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity Components')
plt.grid(True)
plt.legend()

# Plot 3: Yaw Angle Comparison
plt.subplot(2, 2, 3)
vicon_yaw = 2 * np.arctan2(vicon_quat_z, vicon_quat_w)
plt.plot(solution.t, np.rad2deg(solution.y[2]), 'b-', label='Model')
plt.plot(vicon_time, np.rad2deg(vicon_yaw), 'r--', label='Vicon')
plt.xlabel('Time [s]')
plt.ylabel('Yaw Angle [deg]')
plt.title('Yaw Angle')
plt.grid(True)
plt.legend()

# Plot 4: Control Inputs
plt.subplot(2, 2, 4)
plt.plot(time_array, np.rad2deg(steering_array), 'b-', label='Steering')
plt.plot(time_array, speed_array, 'r-', label='Speed')
plt.xlabel('Time [s]')
plt.ylabel('Steering [deg] / Speed [m/s]')
plt.title('Control Inputs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Interpolate Vicon data to simulation time points
x_actual = interpolate_actual_data(solution.t, vicon_time, vicon_x)
y_actual = interpolate_actual_data(solution.t, vicon_time, vicon_y)
vx_actual = interpolate_actual_data(solution.t, vicon_time, vicon_vel_x)
vy_actual = interpolate_actual_data(solution.t, vicon_time, vicon_vel_y)

# Calculate RMS errors
position_error = np.sqrt((solution.y[0] - x_actual)**2 + (solution.y[1] - y_actual)**2)
velocity_error = np.sqrt((solution.y[3] - vx_actual)**2 + (solution.y[4] - vy_actual)**2)

print("\nError Metrics:")
print(f"Mean position error: {np.mean(position_error):.3f} m")
print(f"Max position error: {np.max(position_error):.3f} m")
print(f"RMS position error: {np.sqrt(np.mean(position_error**2)):.3f} m")
print(f"\nMean velocity error: {np.mean(velocity_error):.3f} m/s")
print(f"Max velocity error: {np.max(velocity_error):.3f} m/s")
print(f"RMS velocity error: {np.sqrt(np.mean(velocity_error**2)):.3f} m/s")