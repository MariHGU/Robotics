import numpy as np

def dubins_path(q0, q1, rmin, step_size=0.1):
    """
    Pure-numpy Dubins shortest path between two poses.
    q0, q1: (x, y, theta)
    Returns list of [x, y] waypoints, or [] if no valid path.
    """
    x0, y0, t0 = q0
    x1, y1, t1 = q1

    dx, dy = x1 - x0, y1 - y0
    d = np.hypot(dx, dy) / rmin          # normalised distance
    theta = np.arctan2(dy, dx)
    alpha = _wrap(t0 - theta)
    beta  = _wrap(t1 - theta)

    best_len = np.inf
    best_segs = None
    best_type = None

    for path_type, fn in [("LSL", _LSL), ("RSR", _RSR),
                           ("LSR", _LSR), ("RSL", _RSL),
                           ("RLR", _RLR), ("LRL", _LRL)]:
        segs = fn(alpha, beta, d)
        if segs is None:
            continue
        total = sum(abs(s) for s in segs)
        if total < best_len:
            best_len = total
            best_segs = segs
            best_type = path_type

    if best_segs is None:
        return []

    return _sample_path(x0, y0, t0, best_segs, best_type, rmin, step_size)


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

# ── Dubins word solvers ──────────────────────────────────────────────────────

def _LSL(a, b, d):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    tmp = 2 + d*d - 2*np.cos(a-b) + 2*d*(sa - sb)
    if tmp < 0: return None
    p = np.sqrt(tmp)
    theta = np.arctan2(cb - ca, d + sa - sb)
    t = _wrap(-a + theta)
    q = _wrap(b - theta)
    return t, p, q

def _RSR(a, b, d):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    tmp = 2 + d*d - 2*np.cos(a-b) + 2*d*(sb - sa)
    if tmp < 0: return None
    p = np.sqrt(tmp)
    theta = np.arctan2(ca - cb, d - sa + sb)
    t = _wrap(a - theta)
    q = _wrap(-b + theta)
    return t, p, q

def _LSR(a, b, d):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    tmp = -2 + d*d + 2*np.cos(a-b) + 2*d*(sa + sb)
    if tmp < 0: return None
    p = np.sqrt(tmp)
    theta = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2, p)
    t = _wrap(-a + theta)
    q = _wrap(-_wrap(b) + theta)
    return t, p, q

def _RSL(a, b, d):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    tmp = -2 + d*d + 2*np.cos(a-b) - 2*d*(sa + sb)
    if tmp < 0: return None
    p = np.sqrt(tmp)
    theta = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2, p)
    t = _wrap(a - theta)
    q = _wrap(b - theta)
    return t, p, q

def _RLR(a, b, d):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    tmp = (6 - d*d + 2*np.cos(a-b) + 2*d*(sa - sb)) / 8
    if abs(tmp) > 1: return None
    p = _wrap(np.arccos(np.clip(tmp, -1, 1)))
    theta = np.arctan2(ca - cb, d - sa + sb)
    t = _wrap(a - theta + p/2)
    q = _wrap(a - b - t + p)
    return t, -p, q        # R = positive turn, so negate middle

def _LRL(a, b, d):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    tmp = (6 - d*d + 2*np.cos(a-b) + 2*d*(sb - sa)) / 8
    if abs(tmp) > 1: return None
    p = _wrap(np.arccos(np.clip(tmp, -1, 1)))
    theta = np.arctan2(ca - cb, d + sa - sb)
    t = _wrap(-a + theta + p/2)
    q = _wrap(b - a + t - p)
    return -t, p, -q

# ── Path sampling ────────────────────────────────────────────────────────────

def _sample_path(x, y, theta, segs, path_type, rmin, step_size):
    points = []
    arc_len = [abs(s) * rmin for s in segs]   # actual lengths

    for seg_idx, (seg, ptype) in enumerate(zip(segs, path_type)):
        length = arc_len[seg_idx]
        n_steps = max(2, int(length / step_size))
        for k in range(n_steps):
            points.append(np.array([x, y]))
            ds = (length / n_steps)
            if ptype == 'S':
                x += ds * np.cos(theta)
                y += ds * np.sin(theta)
            elif ptype == 'L':                 # left turn
                dtheta = ds / rmin
                cx = x - rmin * np.sin(theta)
                cy = y + rmin * np.cos(theta)
                theta += dtheta
                x = cx + rmin * np.sin(theta)
                y = cy - rmin * np.cos(theta)
            elif ptype == 'R':                 # right turn
                dtheta = ds / rmin
                cx = x + rmin * np.sin(theta)
                cy = y - rmin * np.cos(theta)
                theta -= dtheta
                x = cx - rmin * np.sin(theta)
                y = cy + rmin * np.cos(theta)

    points.append(np.array([x, y]))
    return points