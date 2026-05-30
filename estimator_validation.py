"""
EKF Standalone Validation — NEES/NIS consistency check
Run this script independently to validate the EKF estimator.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from estimator import EKF

# -- Simulation parameters -----------------------------------------------
dt = 0.01
simulation_time = 20
sensor_noise_stddev = 0.03

# -- Simple circular trajectory dynamics ---------------------------------
def discrete_dynamics(state, control_input, dt, model_mismatch=True):
    x, y, theta = state
    u, w = control_input
    if model_mismatch:
        u += np.random.normal(0, 0.05)
        w += np.random.normal(0, 0.05)
    x += u * np.cos(theta) * dt
    y += u * np.sin(theta) * dt
    theta += w * dt
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return np.array([x, y, theta])

# -- Run simulation ----------------------------------------------------------
def run_ekf_validation():
    ekf = EKF(x0=0.0, y0=0.0, theta0=0.0, model='diff-drive')

    history = {
        'gps_x': [], 'gps_y': [],
        'ekf_x': [], 'ekf_y': [], 'ekf_theta': [],
        'true_x': [], 'true_y': [], 'true_theta': [],
        'P': [], 'S': []
    }

    # fixed circular control inputs
    u, w = 2.0, -2.0
    q = np.array([0.0, 0.0, 0.0])

    for t in np.arange(0, simulation_time, dt):
        z = np.array([q[0], q[1]]) + np.random.normal(0, sensor_noise_stddev, 2)

        ekf.predict(u, w, dt)
        ekf.update(z)
        x_est, y_est, theta_est = ekf.get_state()

        history['gps_x'].append(z[0])
        history['gps_y'].append(z[1])
        history['ekf_x'].append(x_est)
        history['ekf_y'].append(y_est)
        history['ekf_theta'].append(theta_est)
        history['true_x'].append(q[0])
        history['true_y'].append(q[1])
        history['true_theta'].append(q[2])
        history['P'].append(ekf.P.copy())
        history['S'].append(ekf.S.copy())

        q = discrete_dynamics(q, [u, w], dt, model_mismatch=True)

    return history

# -- Plot trajectory and error ------------------------------------------------
def plot_trajectory(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['gps_x'], history['gps_y'], 'r.', alpha=0.3,
                 markersize=2, label='GPS raw')
    axes[0].plot(history['ekf_x'], history['ekf_y'], 'b-',
                 linewidth=1.5, label='EKF estimate')
    axes[0].plot(history['true_x'], history['true_y'], 'g-',
                 linewidth=1.5, label='True')
    axes[0].set_title('XY trajectory')
    axes[0].legend()
    axes[0].set_aspect('equal')

    axes[1].plot(history['true_theta'], 'g-', label='True theta')
    axes[1].plot(history['ekf_theta'], 'b-', label='EKF theta')
    axes[1].set_title('Theta over time')
    axes[1].legend()

    ekf_err = np.sqrt((np.array(history['ekf_x']) - np.array(history['true_x']))**2 +
                      (np.array(history['ekf_y']) - np.array(history['true_y']))**2)
    gps_err = np.sqrt((np.array(history['gps_x']) - np.array(history['true_x']))**2 +
                      (np.array(history['gps_y']) - np.array(history['true_y']))**2)

    axes[2].plot(gps_err, 'r-', alpha=0.5, label='GPS error')
    axes[2].plot(ekf_err, 'b-', label='EKF error')
    axes[2].set_title('Position error over time')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# -- NEES / NIS consistency check -------------------------------------------
def plot_nees_nis(history):
    nees_values, nis_values = [], []

    for i in range(len(history['true_x'])):
        x_true = np.array([history['true_x'][i],
                           history['true_y'][i],
                           history['true_theta'][i]])
        x_est  = np.array([history['ekf_x'][i],
                           history['ekf_y'][i],
                           history['ekf_theta'][i]])
        P = history['P'][i]

        x_tilde = x_true - x_est
        x_tilde[2] = np.arctan2(np.sin(x_tilde[2]), np.cos(x_tilde[2]))
        nees_values.append(x_tilde @ np.linalg.inv(P) @ x_tilde)

        z    = np.array([history['gps_x'][i], history['gps_y'][i]])
        x_bar = np.array([history['ekf_x'][i], history['ekf_y'][i]])
        nu   = z - x_bar
        S    = history['S'][i]
        nis_values.append(nu @ np.linalg.inv(S) @ nu)

    nees_values = np.array(nees_values)
    nis_values  = np.array(nis_values)

    n, m = 3, 2
    nees_lower, nees_upper = chi2.ppf(0.025, df=n), chi2.ppf(0.975, df=n)
    nis_lower,  nis_upper  = chi2.ppf(0.025, df=m), chi2.ppf(0.975, df=m)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(nees_values, 'b-', alpha=0.6, linewidth=0.8, label='NEES')
    axes[0].axhline(nees_upper, color='r', linestyle='--',
                    label=f'95% upper ({nees_upper:.2f})')
    axes[0].axhline(nees_lower, color='r', linestyle='--',
                    label=f'95% lower ({nees_lower:.2f})')
    axes[0].axhline(n, color='g', linestyle='-', alpha=0.5,
                    label=f'Expected mean ({n})')
    axes[0].set_title('NEES — should stay within red bounds 95% of the time')
    axes[0].set_ylabel('NEES')
    axes[0].set_xlabel('timestep')
    axes[0].legend()
    axes[0].set_ylim(0, 20)

    axes[1].plot(nis_values, 'b-', alpha=0.6, linewidth=0.8, label='NIS')
    axes[1].axhline(nis_upper, color='r', linestyle='--',
                    label=f'95% upper ({nis_upper:.2f})')
    axes[1].axhline(nis_lower, color='r', linestyle='--',
                    label=f'95% lower ({nis_lower:.2f})')
    axes[1].axhline(m, color='g', linestyle='-', alpha=0.5,
                    label=f'Expected mean ({m})')
    axes[1].set_title('NIS — should stay within red bounds 95% of the time')
    axes[1].set_ylabel('NIS')
    axes[1].set_xlabel('timestep')
    axes[1].legend()
    axes[1].set_ylim(0, 15)

    plt.tight_layout()
    plt.show()

    pct_nees = np.mean((nees_values >= nees_lower) &
                       (nees_values <= nees_upper)) * 100
    pct_nis  = np.mean((nis_values  >= nis_lower)  &
                       (nis_values  <= nis_upper))  * 100

    print(f"NEES: {pct_nees:.1f}% within 95% bounds (expect ~95%)")
    print(f"NIS:  {pct_nis:.1f}% within 95% bounds (expect ~95%)")
    print(f"Mean NEES: {nees_values.mean():.2f} (expect ~{n})")
    print(f"Mean NIS:  {nis_values.mean():.2f}  (expect ~{m})")

# -- Entry point ------------------------------
if __name__ == '__main__':
    history = run_ekf_validation()
    plot_trajectory(history)
    plot_nees_nis(history)