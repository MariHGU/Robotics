import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
 
# --- Robot parameters ---
L = [0.8, 0.6, 0.5]   # link lengths l1, l2, l3
 
# --- Forward kinematics ---
def fk(t1, t2, t3):
    """Joint positions: base → j1 → j2 → end-effector."""
    j0 = np.array([0.0, 0.0])
    j1 = j0 + L[0] * np.array([np.cos(t1), np.sin(t1)])
    j2 = j1 + L[1] * np.array([np.cos(t1+t2), np.sin(t1+t2)])
    j3 = j2 + L[2] * np.array([np.cos(t1+t2+t3), np.sin(t1+t2+t3)])
    return j0, j1, j2, j3
 
# --- Inverse kinematics ---
def ik_2R(wx, wy, elbow_up=True):
    """2R sub-problem: find t1, t2 to reach wrist (wx, wy)."""
    r2 = wx**2 + wy**2
    cos_t2 = (r2 - L[0]**2 - L[1]**2) / (2*L[0]*L[1])
    cos_t2 = np.clip(cos_t2, -1, 1)
    t2 = np.arccos(cos_t2) * (1 if elbow_up else -1)
    alpha = np.arctan2(wy, wx)
    beta  = np.arctan2(L[1]*np.sin(t2), L[0] + L[1]*np.cos(t2))
    t1 = alpha - beta
    return t1, t2
 
def ik_full(x, y, phi, elbow_up=True):
    """Full 3R IK: pose (x, y, phi) → (t1, t2, t3)."""
    wx = x - L[2]*np.cos(phi)
    wy = y - L[2]*np.sin(phi)
    t1, t2 = ik_2R(wx, wy, elbow_up)
    t3 = phi - t1 - t2
    return t1, t2, t3
 
def ik_pos_only(x, y, t3_free):
    """
    IK for position only (phi unconstrained).
    t3_free is the absolute world-frame angle of link 3.
    """
    wx = x - L[2]*np.cos(t3_free)
    wy = y - L[2]*np.sin(t3_free)
    r2 = wx**2 + wy**2
    cos_t2 = (r2 - L[0]**2 - L[1]**2) / (2*L[0]*L[1])
    if abs(cos_t2) > 1:
        return None
    t2 = np.arccos(np.clip(cos_t2, -1, 1))
    alpha = np.arctan2(wy, wx)
    beta  = np.arctan2(L[1]*np.sin(t2), L[0]+L[1]*np.cos(t2))
    t1 = alpha - beta
    t3_joint = t3_free - t1 - t2
    return t1, t2, t3_joint
 
# --- Drawing ---
COLORS = ['#1E88E5', '#43A047', '#E53935']   # link 1, 2, 3
 
def draw_arm(ax, joints, style='-', alpha=1.0, lw=3.5):
    for i in range(3):
        ax.plot([joints[i][0], joints[i+1][0]],
                [joints[i][1], joints[i+1][1]],
                color=COLORS[i], linestyle=style, linewidth=lw, alpha=alpha,
                solid_capstyle='round')
    for j in joints:
        ax.plot(*j, 'o', color='#222222', markersize=7, alpha=alpha, zorder=5)
 
def style_ax(ax, title):
    R = sum(L) + 0.4
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle=':')
    ax.set_xlabel('x (m)', fontsize=9); ax.set_ylabel('y (m)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(labelsize=8)
    ax.add_patch(plt.Circle((0,0), sum(L), fill=False,
                             linestyle='--', color='#AAAAAA', linewidth=1.2, zorder=1))
 
def mark_target(ax, x, y):
    ax.plot(x, y, '+', color='#FF4444', markersize=13, markeredgewidth=2.5, zorder=10)
 
# --- Main plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
 
# --- a) Unique solution: fully extended ---
ax = axes[0]
style_ax(ax, 'a) Unique solution\n(workspace boundary)')
 
t1_a = np.arctan2(0.6, 1.0)
t2_a, t3_a = 0.0, 0.0
joints_a = fk(t1_a, t2_a, t3_a)
draw_arm(ax, joints_a)
mark_target(ax, *joints_a[-1])
 
ax.legend(handles=[
    Line2D([0],[0], color='#AAAAAA', lw=1.2, ls='--', label='Workspace boundary'),
    Line2D([0],[0], marker='+', color='#FF4444', lw=0, markersize=10,
           markeredgewidth=2, label='End-effector / target'),
], fontsize=8, loc='lower left', framealpha=0.85)
 
# --- b) Two solutions: elbow up / elbow down ---
ax = axes[1]
style_ax(ax, 'b) Two solutions\n(elbow-up / elbow-down)')

x_b, y_b, phi_b = 1.2, 0.8, 0.5
t1u, t2u, t3u = ik_full(x_b, y_b, phi_b, elbow_up=True)
t1d, t2d, t3d = ik_full(x_b, y_b, phi_b, elbow_up=False)

joints_u = fk(t1u, t2u, t3u)
joints_d = fk(t1d, t2d, t3d)

wx_b = x_b - L[2]*np.cos(phi_b)
wy_b = y_b - L[2]*np.sin(phi_b)

# Assign solid to whichever is geometrically elbow-up
def side_of_line(ex, ey, lx, ly):
    return lx * ey - ly * ex

if joints_u[2][1] < joints_d[2][1]:
    arm_up, arm_down = joints_d, joints_u
else:
    arm_up, arm_down = joints_u, joints_d

draw_arm(ax, arm_up,   style='-',  alpha=1.0)
draw_arm(ax, arm_down, style='--', alpha=0.65)
mark_target(ax, x_b, y_b)

ax.plot(wx_b, wy_b, 'D', color='#FF9900', markersize=7, zorder=8)

ax.legend(handles=[
    Line2D([0],[0], color='#1E88E5', lw=2.5, ls='-',  label='Elbow-down'),
    Line2D([0],[0], color='#1E88E5', lw=2.5, ls='--', label='Elbow-up'),
    Line2D([0],[0], marker='D', color='#FF9900', lw=0, markersize=7, label='Wrist P₃ (fixed)'),
    Line2D([0],[0], marker='+', color='#FF4444', lw=0, markersize=10,
           markeredgewidth=2, label='End-effector / target'),
    Line2D([0],[0], color='#AAAAAA', lw=1.2, ls='--', label='Workspace boundary'),
], fontsize=8, loc='lower left', framealpha=0.85)

# --- c) Infinite solutions: position only, phi free ---
ax = axes[2]
style_ax(ax, 'c) Infinite solutions\n(position only, φ free)')
 
x_c, y_c = 0.9, 0.6
for t3_free, sty, tr in zip([-1.0, -0.2, 0.4, 1.8], ['-', ':', ':', ':'], [1.0, 0.75, 0.5, 0.35]):
    sol = ik_pos_only(x_c, y_c, t3_free)
    if sol is not None:
        draw_arm(ax, fk(*sol), style=sty, alpha=tr)
 
mark_target(ax, x_c, y_c)
 
ax.legend(handles=[
    Line2D([0],[0], color='#1E88E5', lw=2.5, ls='-',  label='Config 1'),
    Line2D([0],[0], color='#1E88E5', lw=2.5, ls=':',  label='Other Configs (infinitely more)'),
    Line2D([0],[0], marker='+', color='#FF4444', lw=0, markersize=10,
           markeredgewidth=2, label='End-effector / target'),
    Line2D([0],[0], color='#AAAAAA', lw=1.2, ls='--', label='Workspace boundary'),
], fontsize=8, loc='lower left', framealpha=0.85)
 
plt.tight_layout()
plt.show()