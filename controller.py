import numpy as np
import math

class Controller:
    def __init__(self, K1, K2, K3):
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.integral_error = 0.0
        self.previous_error = 0.0

    def wrap_to_pi(self, a):
        return np.arctan2(np.sin(a), np.cos(a))

    def error_coordinates(self, q, q_d):
        dphi = q[2]-q_d[2]
        #print(dphi)
        phi_e = self.wrap_to_pi(dphi)
        dx = q[0] - q_d[0]
        dy = q[1] - q_d[1]


        x_e = math.cos(q_d[2])*dx + math.sin(q_d[2])*dy
        y_e = -math.sin(q_d[2])*dx + math.cos(q_d[2])*dy
        return x_e, y_e, phi_e
    
    # Choose reference point P, not on axis of two driving wheels
    # Desired trajectory: q_d(t)
    # Reference point: P = (x_Pd(t), y_Pd(t))
    
    # [v, w]^T = J^-1 * [control_signal_x, control_signal_y]^T
    # J^-1 = 1/x_r * [[x_r*cos(phi)-y_r*sin(phi), x_r*sin(phi)+y_r*cos(phi)], [-sin(phi), cos(phi)]]


    # Non lin FB controller + fb control law:
    # def non_lin_fb_controller(self, q, q_d, v_d, w_d):
    #     # Compute the control signals for the linear and angular velocities
    #     x_e, y_e, phi_e = self.error_coordinates(q, q_d)

    #     if abs(phi_e) > np.pi / 3:
    #         v = 0.0
    #         w = -2.0 * phi_e
    #         return v, w

    #     #v = (v_d - self.Kp*np.abs(v_d)*(x_e+y_e*math.tan(phi_e)))/math.cos(phi_e)
    #     v = v_d - self.K1 * abs(v_d) * (x_e + y_e * np.tan(phi_e)) / np.cos(phi_e)
    #     w = w_d - (self.K2*v_d*y_e + self.K3*np.abs(v_d)*math.tan(phi_e))*(math.cos(phi_e))**2

    #     return v, w
    
    # def non_lin_fb_controller(self, q, q_d, v_d, w_d):
    #     x_e, y_e, phi_e = self.error_coordinates(q, q_d)

    #     # Softer heading correction — don't zero out v entirely
    #     if abs(phi_e) >= np.pi / 2:
    #         # Pure rotation to align heading, with a small creep forward
    #         v = 0.0
    #         w = -self.K3 * phi_e  # proportional, not hardcoded gain
    #         return v, w

    #     cos_phi = np.cos(phi_e)
    #     tan_phi = np.tan(phi_e)

    #     # Eq 13.31 from Modern Robotics
    #     v = (v_d - self.K1 * abs(v_d) * (x_e + y_e * tan_phi)) / cos_phi
    #     w = w_d - (self.K2 * v_d * y_e + self.K3 * abs(v_d) * tan_phi) * (cos_phi ** 2)

    #     # Clamp to prevent runaway near phi_e → ±π/2
    #     v = np.clip(v, -1.5, 1.5)
    #     w = np.clip(w, -3.0, 3.0)

    #     return v, w

    # def non_lin_fb_controller(self, q, q_d, v_d, w_d):
    #     x_e, y_e, phi_e = self.error_coordinates(q, q_d)

    #     # If heading error is too large, spin in place to align first
    #     # Use a threshold well below π/2 to avoid the singularity
    #     HEADING_THRESHOLD = np.pi / 4  # 45 degrees

    #     if abs(phi_e) > HEADING_THRESHOLD:
    #         v = 0.0
    #         # Spin proportionally but cap it
    #         w = np.clip(-self.K3 * phi_e, -2.0, 2.0)
    #         return v, w

    #     # Normal operation: phi_e is small enough that cos(phi_e) is safe
    #     cos_phi = np.cos(phi_e)
    #     tan_phi = np.tan(phi_e)

    #     v = (v_d - self.K1 * abs(v_d) * (x_e + y_e * tan_phi)) / cos_phi
    #     w = w_d - (self.K2 * v_d * y_e + self.K3 * abs(v_d) * tan_phi) * (cos_phi ** 2)

    #     v = np.clip(v, -1.5, 1.5)
    #     w = np.clip(w, -3.0, 3.0)

    #     return v, w

# Works with this, however wrong version
    # def non_lin_fb_controller(self, q, q_d, v_d, w_d):
    #     dx = q_d[0] - q[0]
    #     dy = q_d[1] - q[1]

    #     desired_heading = np.arctan2(dy, dx)
    #     heading_error = self.wrap_to_pi(desired_heading - q[2])

    #     distance = np.hypot(dx, dy)

    #     v = self.K1 * distance * max(0.0, np.cos(heading_error))
    #     w = self.K3 * heading_error

    #     v = np.clip(v, 0.0, 0.8)
    #     w = np.clip(w, -2.5, 2.5)

    #     return v, w
    
    def non_lin_fb_controller(self, q, q_d, v_d, w_d):
        x_e, y_e, phi_e = self.error_coordinates(q, q_d)

        distance = np.hypot(q[0] - q_d[0], q[1] - q_d[1])

            # If reference has stopped but robot is far away, use go-to-goal fallback
        if v_d == 0.0 and distance > 0.3:
            dx, dy = q_d[0] - q[0], q_d[1] - q[1]
            heading_error = self.wrap_to_pi(np.arctan2(dy, dx) - q[2])
            v = self.K1 * distance * max(0.0, np.cos(heading_error))
            w = self.K3 * heading_error
            return np.clip(v, 0.0, 1.0), np.clip(w, -2.5, 2.5)

        # Guard against singularity at phi_e → ±π/2 (book assumes |phi_e| < π/2)
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

