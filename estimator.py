import numpy as np

class EKF:
    def __init__(self, x0, y0, theta0, model='diff-drive'):
        self.model = model
        if self.model == 'car_like':
            # x = [x, y, theta, delta]
            self.x = np.array([x0, y0, theta0, 0.0])   # start delta=0
            self.P = np.diag([1e-3, 1e-3, 1e-2, 1e-3])  # 4x4
            self.H = np.array([[1,0,0,0],
                            [0,1,0,0]])
            self.R = np.diag([0.0009, 0.0009])
        else:
            # diff-drive 3-state behavior

            # initial state estimate
            self.x = np.array([x0, y0, theta0])
            
            # initial covariance - high uncertainty in theta since GPS doesn't measure it
            self.P = np.diag([0.1, 0.1, 1.0])
            
            # process noise covariance Q - how much we trust the dynamics model
            # higher = less trust in model, more trust in measurements
            # self.Q = np.diag([0.1, 0.1, 0.1])
            # manual tuning:
            # self.Q = np.diag([0.5, 0.5, 0.1])   # large process noise due to model mismatch
            # removing self.Q, going to use Jacobian Q
            
            # measurement noise covariance R - from simulation: noise std = 0.03
            self.R = np.diag([0.0009, 0.0009])
            
            # measurement matrix H - GPS measures x and y only
            self.H = np.array([[1, 0, 0],
                            [0, 1, 0]])

    def predict(self, u, w, dt, wheelbase=0.5, Q_control=None):
        if self.model != 'car_like':
            x, y, theta = self.x
        
            # propagate state using differential drive dynamics
            self.x = np.array([
                x + u * np.cos(theta) * dt,
                y + u * np.sin(theta) * dt,
                np.arctan2(np.sin(theta + w * dt), np.cos(theta + w * dt))  # normalize
            ])
            
            # jacobian of dynamics w.r.t. state (linearization)
            A = np.array([
                [1, 0, -u * np.sin(theta) * dt],
                [0, 1,  u * np.cos(theta) * dt],
                [0, 0,  1]
            ])

            #------------------------------------------------------
            # jacobian of dynamics w.r.t. control inputs [u, w]
            B = np.array([
                [np.cos(theta) * dt,  0],
                [np.sin(theta) * dt,  0],
                [0,                   dt]
            ])

            # control noise covariance, std=1.0 on both u and w from model mismatch
            Q_control = np.diag([0.05**2, 0.05**2])   # np.diag([1.0**2, 1.0**2])
        
            # propagate control noise through dynamics
            Q = B @ Q_control @ B.T

            # to fix NEES, add small additional uncertainty
            # Q += np.diag([0.000001, 0.000001, 0.000005])
            # rather continue with pure Jacobian Q, can be tunned manually if needed later
            #-------------------------------------------------------
            
            # propagate covariance
            # self.P = A @ self.P @ A.T + self.Q  # with manual self.Q
            self.P = A @ self.P @ A.T + Q         # with Jacobian Q  
            
            return
        
        v = u
        delta_dot = w
        x, y, theta, delta = self.x

        # optional steering limits
        delta_max = np.radians(35.0)
        delta = np.clip(delta, -delta_max, delta_max)

        # state propagation (discrete Euler)
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v * np.tan(delta) / wheelbase) * dt
        delta_new = delta + delta_dot * dt

        # normalize theta
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        self.x = np.array([x_new, y_new, theta_new, delta_new])

        # Linearize: build A = df/dx, B = df/du for u=[v, delta_dot]
        A = np.eye(4)
        A[0,2] = -v * np.sin(theta) * dt
        A[1,2] =  v * np.cos(theta) * dt
        # ∂theta/∂delta = v * dt / (L * cos^2(delta))
        A[2,3] = v * dt / (wheelbase * (np.cos(delta)**2))

        B = np.zeros((4,2))
        B[0,0] = np.cos(theta) * dt                 # ∂x/∂v
        B[1,0] = np.sin(theta) * dt                 # ∂y/∂v
        B[2,0] = np.tan(delta) / wheelbase * dt     # ∂theta/∂v
        B[3,1] = dt                                 # ∂delta/∂delta_dot

        # control noise covariance (tunable)
        if Q_control is None:
            var_v = 0.02**2
            var_delta_dot = (np.radians(2.0))**2
            Qc = np.diag([var_v, var_delta_dot])
        else:
            Qc = Q_control

        Q = B @ Qc @ B.T

        # propagate covariance
        self.P = A @ self.P @ A.T + Q

        # store innovation covariance placeholder (for NEES/NIS)
        # self.S = None

    
    def update(self, z):
        # innovation: difference between measurement and prediction
        nu = z - self.H @ self.x
        
        # innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.S)
        
        # update state and covariance
        self.x = self.x + K @ nu
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))  # normalize theta
        #self.P = (np.eye(3) - K @ self.H) @ self.P

        # corrected identity matrix size (works for both 3-state and 4-state)
        I = np.eye(self.x.size)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        # return self.x
        return self.x[:3].copy()