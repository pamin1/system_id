import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import copy

class DynamicBicycleModel:
    """
    Implementation of the dynamic bicycle model with piece wise tire model and load transfer.
    """
    def __init__(self,
                 m: float,                     # mass                [kg]
                 lf: float, lr: float,         # CG → front / rear   [m]
                 h_cg: float,                  # CG height           [m]
                 Iy: float, Iz: float,         # pitch & yaw inertia [kg·m²]
                 Cf: float, Cr: float,         # cornering stiff.    [N/rad]
                 alpha_sat: float,             # break-slip (rad)
                 c_drag: float = 0.0,          # quadratic drag coeff (N·s²/m²)
                 Fx_split: float = 0.45):       # fraction of Fx on the front axle
        # geometry + inertias
        self.m, self.lf, self.lr = m, lf, lr
        self.h_cg, self.Iy, self.Iz = h_cg, Iy, Iz
        self.Cf, self.Cr, self.alpha_sat = Cf, Cr, alpha_sat
        self.c_drag, self.Fx_split = c_drag, Fx_split

        # state initialisation
        self.x = self.y = self.psi = 0.0
        self.u = self.v = 0.0
        self.r = 0.0
        self.theta = self.q = 0.0
        # gravitational constant
        self.g = 9.81  # [m/s^2]

    def set_initial_state(self, x, y, psi, u, v, r, theta, q):
        self.x = x
        self.y = y
        self.psi = psi
        self.u = u
        self.v = v
        self.r = r
        self.theta = theta
        self.q = q
    
    def set_state(self, x, y, psi, u, v, r, theta, q):
        self.x, self.y, self.psi = x, y, psi
        self.u, self.v, self.r = u, v, r
        self.theta, self.q = theta, q
    
    def get_state(self):
        return np.array([self.x, self.y, self.psi,
                         self.u, self.v, self.r,
                         self.theta, self.q])
        
    def step(self, speed, delta, dt):
        """
        Updates the vehicle state for one time step dt using either a
        kinematic or dynamic model based on speed.

        :param u_lon: longitudinal velocity [m/s]
        :param delta: steering angle [rad]
        :param dt: time step [s]
        """
        
        # longitudinal force
        # approximate longitudinal acceleration commanded by the speed input
        max_lon_acc = 6.0          # m/s²
        u_dot_cmd = np.clip((speed - self.u) / dt, 0, max_lon_acc)

        # include centripetal correction
        Fx_total = self.m * (u_dot_cmd - self.v * self.r)

        # split longitudinal force front/rear
        Fx_f = self.Fx_split * Fx_total
        Fx_r = Fx_total - Fx_f
        u_safe = max(self.u, 0.1)

        # tyre slip angles
        alpha_f = np.arctan2(self.v + self.lf * self.r, u_safe) - delta
        alpha_r = np.arctan2(self.v - self.lr * self.r, u_safe)

        # lateral tyre forces (piece-wise)
        Fy_f = self._fy_piecewise(self.Cf, alpha_f, self.alpha_sat)
        Fy_r = self._fy_piecewise(self.Cr, alpha_r, self.alpha_sat)

        # longitudinal dynamics
        u_dot = ((Fx_f * np.cos(delta) - Fy_f * np.sin(delta) + Fx_r)
                / self.m + self.v * self.r)

        # lateral dynamics
        v_dot = ((Fy_f * np.cos(delta) + Fx_f * np.sin(delta) + Fy_r)
                / self.m - self.u * self.r)

        # yaw dynamics
        r_dot = (self.lf * (Fy_f * np.cos(delta) + Fx_f * np.sin(delta))
                - self.lr * Fy_r) / self.Iz

        # vertical loads (quasi-static)
        Fz_f = (self.m * self.g * self.lr - self.m * self.h_cg * u_dot) / (self.lf + self.lr)
        Fz_r = self.m * self.g - Fz_f

        # pitch dynamics
        q_dot = (self.lf * Fz_f - self.lr * Fz_r) / self.Iy
        theta_dot = self.q

        # integrate vehicle-frame velocities
        self.u += u_dot * dt
        self.v += v_dot * dt
        self.r += r_dot * dt
        self.q += q_dot * dt
        self.theta += theta_dot * dt

        # global kinematics
        cos_psi, sin_psi = np.cos(self.psi), np.sin(self.psi)
        x_dot = self.u * cos_psi - self.v * sin_psi
        y_dot = self.u * sin_psi + self.v * cos_psi

        # Integrate pose using Euler's method
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.psi += self.r * dt
        return self.get_state()

    # helpers
    @staticmethod
    def _fy_piecewise(C, alpha, alpha_sat):
        """Piece-wise linear lateral tyre force."""
        if abs(alpha) <= alpha_sat:
            return -C * alpha
        return -C * alpha_sat * np.sign(alpha)
