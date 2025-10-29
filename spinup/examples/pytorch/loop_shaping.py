import numpy as np

###############################################################################
# 1) Discrete-time first-order joint dynamics and simplified plant construction
###############################################################################

def joint_first_order_gz(alpha, dt):
    """
    Discrete-time first-order joint velocity TF under ZOH:
        g_j(z) = (1 - a_j) / (z - a_j),  a_j = exp(-dt/alpha_j)
    Returns:
        a: (6,) pole locations
        b0: (6,) numerator constants
    """
    alpha = np.asarray(alpha, dtype=float)
    a = np.exp(-dt / alpha)
    b0 = 1.0 - a
    return a, b0

def eval_gj_on_unit_circle(a, b0, z):
    """
    Evaluate diagonal joint TFs g_j(z) on the unit circle z = e^{j w dt}.
    Inputs:
        a, b0: (6,)
        z: complex scalar or array on the unit circle
    Returns:
        g_diag: (6, len(z)) complex array if z is array, else (6,) complex
    """
    # g_j(z) = b0_j / (z - a_j)
    return b0[:, None] / (z[None, :] - a[:, None]) if np.ndim(z) else b0 / (z - a)

def eval_Gz(z, dt):
    """
    Discrete integrator: G(z) = dt / (1 - z^{-1})
    """
    return dt / (1.0 - 1.0 / z)

def build_P_of_omega(J0, alpha, dt, wgrid):
    """
    Frequency response of P(z) = G(z) * J0 * Gq(z) on a frequency grid.
    Inputs:
        J0: (3,6)
        alpha: (6,)
        dt: float
        wgrid: (Nw,) array of frequencies in rad/s
    Returns:
        Pjw: list length Nw of complex matrices (3x6), P(e^{j w dt})
    """
    J0 = np.asarray(J0, dtype=float)
    a, b0 = joint_first_order_gz(alpha, dt)
    z = np.exp(1j * wgrid * dt)  # unit circle points
    Gz = eval_Gz(z, dt)          # (Nw,)
    gj = eval_gj_on_unit_circle(a, b0, z)  # (6,Nw)

    Pjw = []
    for k in range(len(wgrid)):
        Gq_k = np.diag(gj[:, k])             # (6,6)
        Pk = Gz[k] * (J0 @ Gq_k)             # (3,6)
        Pjw.append(Pk)
    return Pjw

###############################################################################
# 2) Per-axis SISO loop approximation and PI tuning by classical loop shaping
###############################################################################

def effective_axis_miso_row(J0, axis_index):
    """
    Row mapping from joint velocities to task velocity along axis i:
        r_i^T = e_i^T * J0
    Returns:
        r: (6,) row vector
    """
    e = np.zeros((3,))
    e[axis_index] = 1.0
    r = e @ J0  # (6,)
    return r

def effective_axis_open_loop(Pjw_k, Jhat_dag, axis_index, Kp, Ki, dt, w):
    """
    Correct scalar loop along axis i:
      L_i(z) = e_i^T P(z) Jhat_dag e_i * (Kp + Ki * G(z))
    """
    e_i = np.zeros((3,))
    e_i[axis_index] = 1.0
    z = np.exp(1j * w * dt)
    Cz = Kp + Ki * eval_Gz(z, dt)                       # controller (scalar)
    Mii = e_i @ (Pjw_k @ Jhat_dag) @ e_i                # scalar effective plant
    return Mii * Cz


def find_crossover_and_pm(Lw, phase_unwrap=True):
    """
    Given frequency response array L(e^{jw dt}) on wgrid, find crossover and phase margin.
    Crossover at |L| = 1. Phase margin = 180 + phase(L at wc) [deg].
    Returns:
        wc (rad/s), pm_deg (deg). If no crossing, returns (None, None).
    """
    mag = np.abs(Lw)
    ph = np.angle(Lw)
    if phase_unwrap:
        ph = np.unwrap(ph)

    # Find indices where magnitude crosses 1
    idx = np.where((mag[:-1] < 1.0) & (mag[1:] >= 1.0) | (mag[:-1] > 1.0) & (mag[1:] <= 1.0))[0]
    if len(idx) == 0:
        return None, None

    # Linear interpolation for crossing between idx[0] and idx[0]+1
    i = idx[0]
    m0, m1 = mag[i], mag[i+1]
    if m1 == m0:
        t = 0.0
    else:
        t = (1.0 - m0) / (m1 - m0)  # fraction in [0,1]
    return t, ph, i  # return details to compute wc, pm outside

def tune_axis(J0, Jhat_dag, alpha, dt, wgrid, wc_target=None, pm_target_deg=60.0,
              kp_range=(1e-4, 5.0), ki_range=(1e-4, 50.0),
              axis_index=0, coarse_points=25):
    """
    Tune kP,kI for a single axis via grid + refinement to meet wc_target and pm_target.
    Returns:
        kP, kI, achieved_wc, achieved_pm_deg
    """
    # Build P(jw) on grid:
    Pjw = build_P_of_omega(J0, alpha, dt, wgrid)

    # Default target crossover: conservative fraction of min(1/alpha)
    if wc_target is None:
        wc_target = 1.0 / (4* np.max(np.asarray(alpha)))

    # Precompute scalar kinematic gains for the axis:
    e_i = np.zeros((3,))
    e_i[axis_index] = 1.0
    M = J0 @ Jhat_dag
    g_axis = float(e_i @ M @ e_i)

    # Helper: evaluate objective for a candidate (kP,kI)
    def eval_candidate(kP, kI):
        Lw = np.zeros_like(wgrid, dtype=complex)
        for k, w in enumerate(wgrid):
            Lw[k] = effective_axis_open_loop(Pjw[k], Jhat_dag, axis_index, kP, kI, dt, w)
        # Find crossover and PM
        mag = np.abs(Lw)
        ph = np.unwrap(np.angle(Lw))
        crossings = np.where((mag[:-1] - 1.0) * (mag[1:] - 1.0) <= 0)[0]

        if len(crossings) == 0:
            # No crossing: penalize
            return 1e6, None, None

        i = crossings[0]
        # Linear interpolate magnitude and phase at crossing
        m0, m1 = mag[i], mag[i+1]
        t = 0.0 if m1 == m0 else (1.0 - m0) / (m1 - m0)
        w_c = (1 - t) * wgrid[i] + t * wgrid[i+1]
        ph_c = (1 - t) * ph[i] + t * ph[i+1]
        pm = 180.0 + (ph_c * 180.0 / np.pi)  # deg

        # Objective: match wc, penalize PM shortfall
        obj = (np.log10((w_c + 1e-9) / wc_target)) ** 2
        if pm < pm_target_deg:
            obj += 10.0 * (pm_target_deg - pm) ** 2  # heavy penalty
        return obj, w_c, pm

    # Coarse grid search
    kP_vals = np.geomspace(kp_range[0], kp_range[1], num=coarse_points)
    kI_vals = np.geomspace(ki_range[0], ki_range[1], num=coarse_points)
    best = (np.inf, None, None, None, None)  # obj, kP, kI, wc, pm
    for kP in kP_vals:
        for kI in kI_vals:
            obj, wc, pm = eval_candidate(kP, kI)
            if obj < best[0]:
                best = (obj, kP, kI, wc, pm)

    # Simple local refinement around best on a small log grid
    kP0, kI0 = best[1], best[2]
    for scaleP in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for scaleI in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            kP = np.clip(kP0 * scaleP, kp_range[0], kp_range[1])
            kI = np.clip(kI0 * scaleI, ki_range[0], ki_range[1])
            obj, wc, pm = eval_candidate(kP, kI)
            if obj < best[0]:
                best = (obj, kP, kI, wc, pm)

    _, kP_opt, kI_opt, wc_opt, pm_opt = best
    return kP_opt, kI_opt, wc_opt, pm_opt

def tune_PI_gains(J0, Jhat_dag, alpha, dt,
                  wc_target=None, pm_target_deg=60.0,
                  kp_range=(1e-4, 5.0), ki_range=(1e-4, 50.0),
                  wmin=1e-1, wmax=None, n_w=600):
    """
    Top-level tuner: returns diagonal Kp, Ki for the 3 task axes.
    """
    alpha = np.asarray(alpha, dtype=float)
    if wmax is None:
        # cap well below the tightest joint bandwidth
        wmax = 0.8 * np.min(1.0 / alpha)

    wgrid = np.linspace(wmin, wmax, n_w)

    Kp = np.zeros((3,))
    Ki = np.zeros((3,))
    wc = np.zeros((3,))
    pm = np.zeros((3,))

    for i in range(3):
        kP_i, kI_i, wc_i, pm_i = tune_axis(
            J0, Jhat_dag, alpha, dt, wgrid,
            wc_target=wc_target, pm_target_deg=pm_target_deg,
            kp_range=kp_range, ki_range=ki_range,
            axis_index=i, coarse_points=25
        )
        Kp[i], Ki[i], wc[i], pm[i] = kP_i, kI_i, wc_i, pm_i

    Kp_mat = np.diag(Kp)
    Ki_mat = np.diag(Ki)
    return Kp_mat, Ki_mat, wc, pm

def estimate_kp_from_wc(J0, Jhat_dag, alpha, dt, wc):
    # build P at single frequency wc
    Pjw = build_P_of_omega(J0, alpha, dt, np.array([wc]))[0]
    kp_est = np.zeros(3)
    for i in range(3):
        e_i = np.zeros((3,)); e_i[i] = 1.0
        Mii = e_i @ (Pjw @ Jhat_dag) @ e_i
        kp_est[i] = 1.0 / (np.abs(Mii) + 1e-12)  # ignore Ki at crossover
    return kp_est
###############################################################################
# Example usage (fill with your data)
###############################################################################
if __name__ == "__main__":
    dt = 0.1
    J0 = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/J_k0.npy")
    J0hat_dag = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/pihatJ_k0.npy")
    J0_dag = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/piJ_k0.npy")
    alpha=np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/alpha_dt0004.npy")

    pm_target_deg = 60.0
    wc_target = 1/(4*np.mean(alpha))

    print("wc_target=",wc_target)
    wgrid = np.linspace(0.1, 0.6/(np.max(alpha)), 400)
    Pjw = build_P_of_omega(J0, alpha, dt, wgrid)  # list of 3x6 matrices over w

    # --- TUNE GAINS ---
    Kp, Ki, wc, pm = tune_PI_gains(
        J0, J0hat_dag, alpha, dt,
        wc_target=wc_target,
        pm_target_deg=pm_target_deg,
        kp_range=(0.1, 10.0),
        ki_range=(0.1, 10.0),
        wmin=0.1,
        wmax=0.6 * 1/(np.max(alpha)),
        n_w=800
    )

    print("Kp =\n", Kp)
    print("Ki =\n", Ki)
    print("Per-axis achieved crossover (rad/s):", wc)
    print("Per-axis achieved phase margin (deg):", pm)
    print("")

