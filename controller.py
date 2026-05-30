import numpy as np
import math

class Controller:
    def __init__(self, K1, K2, K3, wheelbase, max_steer_deg):
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.integral_error = 0.0
        self.previous_error = 0.0

        # Car-like
        self.wheelbase = wheelbase
        self.max_steer = np.radians(max_steer_deg)


    def wrap_to_pi(self, a):
        return np.arctan2(np.sin(a), np.cos(a))
    
    def to_steering_angle(self, v, omega):
        if abs(v) < 1e-6:
            return 0.0
        delta = np.arctan2(omega * self.wheelbase, v)
        return float(np.clip(delta, -self.max_steer, self.max_steer))

    def error_coordinates(self, q_est, q_d):
        dphi = q_est[2]-q_d[2]
        #print(dphi)
        phi_e = self.wrap_to_pi(dphi)
        dx = q_est[0] - q_d[0]
        dy = q_est[1] - q_d[1]


        x_e = math.cos(q_d[2])*dx + math.sin(q_d[2])*dy
        y_e = -math.sin(q_d[2])*dx + math.cos(q_d[2])*dy
        return x_e, y_e, phi_e
    
    # Choose reference point P, not on axis of two driving wheels
    # Desired trajectory: q_d(t)
    # Reference point: P = (x_Pd(t), y_Pd(t))
    
    # [v, w]^T = J^-1 * [control_signal_x, control_signal_y]^T
    # J^-1 = 1/x_r * [[x_r*cos(phi)-y_r*sin(phi), x_r*sin(phi)+y_r*cos(phi)], [-sin(phi), cos(phi)]]


    # Non lin FB controller + fb control law:
    def non_lin_fb_controller(self, q_est, q_d, v_d, w_d):
        x_e, y_e, phi_e = self.error_coordinates(q_est, q_d)

        distance = np.hypot(q_est[0] - q_d[0], q_est[1] - q_d[1])

        # Rotation to alighn with goal heading:
        if v_d == 0.0 and distance < 0.3:
            v = 0.0
            w = np.clip(-self.K3 * phi_e, -2.5, 2.5)
            return v, w

        # If reference has stopped but robot is far away, use go-to-goal fallback
        if v_d == 0.0 and distance > 0.3:
            dx, dy = q_d[0] - q_est[0], q_d[1] - q_est[1]
            heading_error = self.wrap_to_pi(np.arctan2(dy, dx) - q_est[2])
            v = self.K1 * distance * max(0.0, np.cos(heading_error))
            w = self.K3 * heading_error
            return np.clip(v, 0.0, 1.0), np.clip(w, -2.5, 2.5)

        # Guard against singularity at phi_e → +-pi/2 (book assumes |phi_e| < pi/2)
        if abs(phi_e) >= np.pi / 2:
            # Align heading first before attempting trajectory tracking
            v = 0.0
            w = np.clip(-self.K3 * phi_e, -2.5, 2.5)
            return v, w

        cos_phi = np.cos(phi_e)
        tan_phi = np.tan(phi_e)

        # Eq. 13.31 from Modern Robotics
        v = (v_d - self.K1 * abs(v_d) * (x_e + y_e * tan_phi)) / cos_phi
        w = w_d - (self.K2 * v_d * y_e + self.K3 * abs(v_d) * tan_phi) * (cos_phi ** 2)

        turn_penalty = np.cos(phi_e)**2
        v *= turn_penalty

        v = np.clip(v, -1.0, 1.0)
        w = np.clip(w, -2.5, 2.5)

        return v, w
    
    def car_controller(self, q_est, q_d, v_d, w_d):
        v, omega = self.non_lin_fb_controller(q_est, q_d, v_d, w_d)

        v = np.clip(v, 0.0, 1.0) # No reverse for now, planner does not account for it

        delta = self.to_steering_angle(v, omega)
        return v, delta

