# import numpy as np
#
# ###############################################################################
# # 1) Discrete-time first-order joint dynamics and simplified plant construction
# ###############################################################################
#
# def joint_first_order_gz(alpha, dt):
#     """
#     Discrete-time first-order joint velocity TF under ZOH:
#         g_j(z) = (1 - a_j) / (z - a_j),  a_j = exp(-dt/alpha_j)
#     Returns:
#         a: (6,) pole locations
#         b0: (6,) numerator constants
#     """
#     alpha = np.asarray(alpha, dtype=float)
#     a = np.exp(-dt / alpha)
#     b0 = 1.0 - a
#     return a, b0
#
# def eval_gj_on_unit_circle(a, b0, z):
#     """
#     Evaluate diagonal joint TFs g_j(z) on the unit circle z = e^{j w dt}.
#     Inputs:
#         a, b0: (6,)
#         z: complex scalar or array on the unit circle
#     Returns:
#         g_diag: (6, len(z)) complex array if z is array, else (6,) complex
#     """
#     # g_j(z) = b0_j / (z - a_j)
#     return b0[:, None] / (z[None, :] - a[:, None]) if np.ndim(z) else b0 / (z - a)
#
# def eval_Gz(z, dt):
#     """
#     Discrete integrator: G(z) = dt / (1 - z^{-1})
#     """
#     return dt / (1.0 - 1.0 / z)
#
# def build_P_of_omega(J0, alpha, dt, wgrid):
#     """
#     Frequency response of P(z) = G(z) * J0 * Gq(z) on a frequency grid.
#     Inputs:
#         J0: (3,6)
#         alpha: (6,)
#         dt: float
#         wgrid: (Nw,) array of frequencies in rad/s
#     Returns:
#         Pjw: list length Nw of complex matrices (3x6), P(e^{j w dt})
#     """
#     J0 = np.asarray(J0, dtype=float)
#     a, b0 = joint_first_order_gz(alpha, dt)
#     z = np.exp(1j * wgrid * dt)  # unit circle points
#     Gz = eval_Gz(z, dt)          # (Nw,)
#     gj = eval_gj_on_unit_circle(a, b0, z)  # (6,Nw)
#
#     Pjw = []
#     for k in range(len(wgrid)):
#         Gq_k = np.diag(gj[:, k])             # (6,6)
#         Pk = Gz[k] * (J0 @ Gq_k)             # (3,6)
#         Pjw.append(Pk)
#     return Pjw
#
# ###############################################################################
# # 2) Per-axis SISO loop approximation and PI tuning by classical loop shaping
# ###############################################################################
#
# def effective_axis_miso_row(J0, axis_index):
#     """
#     Row mapping from joint velocities to task velocity along axis i:
#         r_i^T = e_i^T * J0
#     Returns:
#         r: (6,) row vector
#     """
#     e = np.zeros((3,))
#     e[axis_index] = 1.0
#     r = e @ J0  # (6,)
#     return r
#
# def effective_axis_open_loop(Pjw_k, Jhat_dag, axis_index, Kp, Ki, dt, w):
#     """
#     Correct scalar loop along axis i:
#       L_i(z) = e_i^T P(z) Jhat_dag e_i * (Kp + Ki * G(z))
#     """
#     e_i = np.zeros((3,))
#     e_i[axis_index] = 1.0
#     z = np.exp(1j * w * dt)
#     Cz = Kp + Ki * eval_Gz(z, dt)                       # controller (scalar)
#     Mii = e_i @ (Pjw_k @ Jhat_dag) @ e_i                # scalar effective plant
#     return Mii * Cz
#
#
# def find_crossover_and_pm(Lw, phase_unwrap=True):
#     """
#     Given frequency response array L(e^{jw dt}) on wgrid, find crossover and phase margin.
#     Crossover at |L| = 1. Phase margin = 180 + phase(L at wc) [deg].
#     Returns:
#         wc (rad/s), pm_deg (deg). If no crossing, returns (None, None).
#     """
#     mag = np.abs(Lw)
#     ph = np.angle(Lw)
#     if phase_unwrap:
#         ph = np.unwrap(ph)
#
#     # Find indices where magnitude crosses 1
#     idx = np.where((mag[:-1] < 1.0) & (mag[1:] >= 1.0) | (mag[:-1] > 1.0) & (mag[1:] <= 1.0))[0]
#     if len(idx) == 0:
#         return None, None
#
#     # Linear interpolation for crossing between idx[0] and idx[0]+1
#     i = idx[0]
#     m0, m1 = mag[i], mag[i+1]
#     if m1 == m0:
#         t = 0.0
#     else:
#         t = (1.0 - m0) / (m1 - m0)  # fraction in [0,1]
#     return t, ph, i  # return details to compute wc, pm outside
#
# def tune_axis(J0, Jhat_dag, alpha, dt, wgrid, wc_target=None, pm_target_deg=60.0,
#               kp_range=(1e-4, 5.0), ki_range=(1e-4, 50.0),
#               axis_index=0, coarse_points=25):
#     """
#     Tune kP,kI for a single axis via grid + refinement to meet wc_target and pm_target.
#     Returns:
#         kP, kI, achieved_wc, achieved_pm_deg
#     """
#     # Build P(jw) on grid:
#     Pjw = build_P_of_omega(J0, alpha, dt, wgrid)
#
#     # Default target crossover: conservative fraction of min(1/alpha)
#     if wc_target is None:
#         wc_target = 1.0 / (4* np.max(np.asarray(alpha)))
#
#     # Precompute scalar kinematic gains for the axis:
#     e_i = np.zeros((3,))
#     e_i[axis_index] = 1.0
#     M = J0 @ Jhat_dag
#     g_axis = float(e_i @ M @ e_i)
#
#     # Helper: evaluate objective for a candidate (kP,kI)
#     def eval_candidate(kP, kI):
#         Lw = np.zeros_like(wgrid, dtype=complex)
#         for k, w in enumerate(wgrid):
#             Lw[k] = effective_axis_open_loop(Pjw[k], Jhat_dag, axis_index, kP, kI, dt, w)
#         # Find crossover and PM
#         mag = np.abs(Lw)
#         ph = np.unwrap(np.angle(Lw))
#         crossings = np.where((mag[:-1] - 1.0) * (mag[1:] - 1.0) <= 0)[0]
#
#         if len(crossings) == 0:
#             # No crossing: penalize
#             return 1e6, None, None
#
#         i = crossings[0]
#         # Linear interpolate magnitude and phase at crossing
#         m0, m1 = mag[i], mag[i+1]
#         t = 0.0 if m1 == m0 else (1.0 - m0) / (m1 - m0)
#         w_c = (1 - t) * wgrid[i] + t * wgrid[i+1]
#         ph_c = (1 - t) * ph[i] + t * ph[i+1]
#         pm = 180.0 + (ph_c * 180.0 / np.pi)  # deg
#
#         # Objective: match wc, penalize PM shortfall
#         obj = (np.log10((w_c + 1e-9) / wc_target)) ** 2
#         if pm < pm_target_deg:
#             obj += 10.0 * (pm_target_deg - pm) ** 2  # heavy penalty
#         return obj, w_c, pm
#
#     # Coarse grid search
#     kP_vals = np.geomspace(kp_range[0], kp_range[1], num=coarse_points)
#     kI_vals = np.geomspace(ki_range[0], ki_range[1], num=coarse_points)
#     best = (np.inf, None, None, None, None)  # obj, kP, kI, wc, pm
#     for kP in kP_vals:
#         for kI in kI_vals:
#             obj, wc, pm = eval_candidate(kP, kI)
#             if obj < best[0]:
#                 best = (obj, kP, kI, wc, pm)
#
#     # Simple local refinement around best on a small log grid
#     kP0, kI0 = best[1], best[2]
#     for scaleP in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
#         for scaleI in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
#             kP = np.clip(kP0 * scaleP, kp_range[0], kp_range[1])
#             kI = np.clip(kI0 * scaleI, ki_range[0], ki_range[1])
#             obj, wc, pm = eval_candidate(kP, kI)
#             if obj < best[0]:
#                 best = (obj, kP, kI, wc, pm)
#
#     _, kP_opt, kI_opt, wc_opt, pm_opt = best
#     return kP_opt, kI_opt, wc_opt, pm_opt
#
# def tune_PI_gains(J0, Jhat_dag, alpha, dt,
#                   wc_target=None, pm_target_deg=60.0,
#                   kp_range=(1e-4, 5.0), ki_range=(1e-4, 50.0),
#                   wmin=1e-1, wmax=None, n_w=600):
#     """
#     Top-level tuner: returns diagonal Kp, Ki for the 3 task axes.
#     """
#     alpha = np.asarray(alpha, dtype=float)
#     if wmax is None:
#         # cap well below the tightest joint bandwidth
#         wmax = 0.8 * np.min(1.0 / alpha)
#
#     wgrid = np.linspace(wmin, wmax, n_w)
#
#     Kp = np.zeros((3,))
#     Ki = np.zeros((3,))
#     wc = np.zeros((3,))
#     pm = np.zeros((3,))
#
#     for i in range(3):
#         kP_i, kI_i, wc_i, pm_i = tune_axis(
#             J0, Jhat_dag, alpha, dt, wgrid,
#             wc_target=wc_target, pm_target_deg=pm_target_deg,
#             kp_range=kp_range, ki_range=ki_range,
#             axis_index=i, coarse_points=25
#         )
#         Kp[i], Ki[i], wc[i], pm[i] = kP_i, kI_i, wc_i, pm_i
#
#     Kp_mat = np.diag(Kp)
#     Ki_mat = np.diag(Ki)
#     return Kp_mat, Ki_mat, wc, pm
#
# def estimate_kp_from_wc(J0, Jhat_dag, alpha, dt, wc):
#     # build P at single frequency wc
#     Pjw = build_P_of_omega(J0, alpha, dt, np.array([wc]))[0]
#     kp_est = np.zeros(3)
#     for i in range(3):
#         e_i = np.zeros((3,)); e_i[i] = 1.0
#         Mii = e_i @ (Pjw @ Jhat_dag) @ e_i
#         kp_est[i] = 1.0 / (np.abs(Mii) + 1e-12)  # ignore Ki at crossover
#     return kp_est
# ###############################################################################
# # Example usage (fill with your data)
# ###############################################################################
# if __name__ == "__main__":
#     dt = 0.1
#     J0 = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/J_k0.npy")
#     J0hat_dag = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/pihatJ_k0.npy")
#     J0_dag = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/piJ_k0.npy")
#     alpha=np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/alpha_dt0004.npy")
#
#     pm_target_deg = 60.0
#     wc_target = 1/(4*np.mean(alpha))
#
#     print("wc_target=",wc_target)
#     wgrid = np.linspace(0.1, 0.6/(np.max(alpha)), 400)
#     Pjw = build_P_of_omega(J0, alpha, dt, wgrid)  # list of 3x6 matrices over w
#
#     # --- TUNE GAINS ---
#     Kp, Ki, wc, pm = tune_PI_gains(
#         J0, J0hat_dag, alpha, dt,
#         wc_target=wc_target,
#         pm_target_deg=pm_target_deg,
#         kp_range=(0.1, 10.0),
#         ki_range=(0.1, 10.0),
#         wmin=0.1,
#         wmax=0.6 * 1/(np.max(alpha)),
#         n_w=800
#     )
#
#     print("Kp =\n", Kp)
#     print("Ki =\n", Ki)
#     print("Per-axis achieved crossover (rad/s):", wc)
#     print("Per-axis achieved phase margin (deg):", pm)
#     print("")
#
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Matplotlib / LaTeX
# ============================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 14,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# ============================================================
# USER SETTINGS
# ============================================================

dt = 0.1
fs = 1.0 / dt

# Sliding-window parameters (rectangular window, no taper)
T_win = 1.0        # [s]
overlap = 0.9
nperseg = max(4, int(round(T_win / dt)))   # e.g. 10 samples
noverlap = int(round(overlap * nperseg))   # e.g. 9 samples
hop = max(1, nperseg - noverlap)

# Zero-padding for dense frequency sampling (df = fs / nfft)
nfft = 1000

# Frequency band for gamma cost [Hz]
fmin_hz = 0.0
fmax_hz = 5.0

# Switch: include or exclude omega = 0 (DC) in band for gamma cost & plots
exclude_dc = True   # set False if you want to include DC

# Frequency range for plotting (in rad/s, log scale)
omega_min_plot = 0.628    # [rad/s]
omega_max_plot = 31.4     # [rad/s]

# k-range (sample index) for plotting on x-axis
k_min = 10
k_max = 109

# Colorbar ranges
# Relative error gain spectrograms (dimensionless)
vmin_rel = 0.0
vmax_rel = 0.5   # adjust based on your data

# Gamma spectrogram (dimensionless)
vmin_gamma = 0.0
vmax_gamma = 1.0  # adjust if needed

# Colormaps
cmap_rel = "viridis"
cmap_gamma = "magma"

# Base directory with data files
base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

# PI gain search range (SCALAR gains, same on all 3 axes)
Kp_min, Kp_max, n_Kp = 10.0, 10.0, 1
Ki_min, Ki_max, n_Ki = 5.0, 5.0, 1

# Metric for robust LB optimization on sensitivity:
# "max" -> sup over freq
# "mean" -> average over freq
lb_metric = "max"

# Small epsilon to avoid division by zero when normalizing by ||P^*||
eps_ref = 1e-12

# ============================================================
# Simplified plant model helpers
# ============================================================

def joint_first_order_gz(alpha, dt):
    """
    Discrete-time first-order joint velocity poles and numerators under ZOH:
        a_j = exp(-dt/alpha_j),  b0_j = 1 - a_j
    """
    alpha = np.asarray(alpha, dtype=float)
    a = np.exp(-dt / alpha)
    b0 = 1.0 - a
    return a, b0

def eval_gj_on_unit_circle(a, b0, z):
    """
    Evaluate diagonal joint TFs g_j(z) on the unit circle z = e^{j w dt}.
    g_j(z) = b0_j / (z - a_j)
    Inputs:
        a, b0: (6,)
        z: complex array on the unit circle, shape (Nw,)
    Returns:
        gj: (6, Nw) complex array, gj[j, k] = g_j(z_k)
    """
    a = np.asarray(a).reshape(-1)
    b0 = np.asarray(b0).reshape(-1)
    z = np.asarray(z)
    return b0[:, None] / (z[None, :] - a[:, None])

def eval_Gz(z, dt):
    """
    Discrete integrator: G(z) = dt / (1 - z^{-1})
    """
    return dt / (1.0 - 1.0 / z)

def build_P_of_omega(J0, alpha, dt, wgrid):
    """
    Frequency response of simplified plant:
        P(z) = G(z) * J0 * Gq(z),
    where Gq(z) = diag(g_1(z), ..., g_6(z)).
    Inputs:
        J0    : (3,6), Jacobian at nominal posture
        alpha : (6,), joint time constants
        dt    : float
        wgrid : (Nw,) array of frequencies in rad/s
    Returns:
        Pjw: list length Nw of complex matrices (3x6), P(e^{j w dt})
    """
    J0 = np.asarray(J0, dtype=float)
    a, b0 = joint_first_order_gz(alpha, dt)
    z = np.exp(1j * wgrid * dt)          # unit circle points
    Gz_vec = eval_Gz(z, dt)              # (Nw,)
    gj = eval_gj_on_unit_circle(a, b0, z)  # (6,Nw)

    Pjw = []
    for k in range(len(wgrid)):
        Gq_k = np.diag(gj[:, k])         # (6,6)
        Pk = Gz_vec[k] * (J0 @ Gq_k)     # (3,6)
        Pjw.append(Pk)
    return Pjw

# ============================================================
# Controller + norms helpers
# ============================================================

def Gz_from_omega(omega, dt):
    """
    G(z) = dt / (1 - z^{-1}) evaluated on the unit circle z = e^{j omega dt}.
    """
    z = np.exp(1j * omega * dt)
    return eval_Gz(z, dt)

def make_C_PI_diag(Kp_vec, Ki_vec, Gz):
    """
    Build diagonal PI controller C(omega) in task space:

        C(omega) = diag_i( Kp_i + Ki_i * Gz(omega) ).

    Kp_vec, Ki_vec: shape (3,)
    Gz           : shape (F,), integrator in z-domain
    Returns      : C[frequency, 3, 3]
    """
    Kp_vec = np.asarray(Kp_vec, float).reshape(3)
    Ki_vec = np.asarray(Ki_vec, float).reshape(3)
    F = len(Gz)
    C_PI = np.zeros((F, 3, 3), dtype=complex)
    for i in range(3):
        C_PI[:, i, i] = Kp_vec[i] + Ki_vec[i] * Gz
    return C_PI

def sigma_min_2norm(M):
    return np.linalg.svd(M, compute_uv=False)[-1]

def sigma_max_2norm(M):
    return np.linalg.svd(M, compute_uv=False)[0]

def band_mask_freq_rad(omega, omega_min, omega_max, exclude_dc=True):
    """
    Build a boolean mask over angular frequencies omega [rad/s].
    """
    m = (omega >= omega_min) & (omega <= omega_max)
    if exclude_dc:
        m &= ~np.isclose(omega, 0.0)
    return m

# ============================================================
# Sliding-window DFT helpers (rectangular windows, no taper)
# ============================================================

def sliding_dft_vecnorm(
    x_mat,              # shape (T, D)
    dt, nperseg, noverlap, nfft
):
    """
    Sliding-window DFT (rectangular) of a multi-dimensional signal x_mat (T,D).

    For each window, compute rFFT along time (axis=0), then 2-norm across D.
    DFT is normalized by nperseg, so the unit matches x_mat.
    Returns:
        f_hz: frequency grid (Hz), shape (F,)
        t_end: window end times (s), shape (W,)
        mag: array of shape (F, W) with ||X_k(omega)||_2
    """
    x_mat = np.asarray(x_mat, float)
    T, D = x_mat.shape
    hop = max(1, nperseg - noverlap)
    ends = np.arange(nperseg - 1, T, hop)   # window end indices
    t_end = ends * dt

    f_hz = np.fft.rfftfreq(nfft, d=dt)
    F = len(f_hz)

    mag = np.zeros((F, len(ends)), dtype=float)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg + 1
        seg = x_mat[kstart:kend+1, :]    # (nperseg, D)
        Xf = np.fft.rfft(seg, n=nfft, axis=0) / nperseg  # normalize
        mag[:, wi] = np.linalg.norm(Xf, axis=1)  # 2-norm across D

    return f_hz, t_end, mag

def sliding_dft_scalar(
    x, dt, nperseg, noverlap, nfft
):
    """
    Sliding-window DFT (rectangular) of a scalar signal x (T,).

    DFT is normalized by nperseg, so the unit matches x.
    Returns:
        f_hz: frequency grid (Hz), shape (F,)
        t_end: window end times (s), shape (W,)
        mag: array of shape (F, W) with |X_k(omega)|
    """
    x = np.asarray(x, float).reshape(-1)
    T = x.shape[0]
    hop = max(1, nperseg - noverlap)
    ends = np.arange(nperseg - 1, T, hop)   # window end indices
    t_end = ends * dt

    f_hz = np.fft.rfftfreq(nfft, d=dt)
    F = len(f_hz)

    mag = np.zeros((F, len(ends)), dtype=float)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg + 1
        seg = x[kstart:kend+1]                     # (nperseg,)
        Xf = np.fft.rfft(seg, n=nfft) / nperseg    # normalize
        mag[:, wi] = np.abs(Xf)

    return f_hz, t_end, mag

# ============================================================
# Robust lower bound gamma(omega) using simplified plant
# ============================================================

def compute_gamma_simple_and_cost(
    dt, J0, J0_dag, J0hat_dag, alpha,
    Kp_vec, Ki_vec,
    nfft, fmin_hz, fmax_hz, exclude_dc=True,
    lb_metric="max"
):
    """
    Compute robust lower-bound factor gamma(omega) for simplified plant:

        P(omega) = G(omega) J0 Gq(omega),
        L0(omega) = P(omega) J0_dag C(omega),
        Delta_a = J0hat_dag - J0_dag,
        Delta_L(omega) = P(omega) Delta_a C(omega),
        S0(omega) = (I + L0(omega))^{-1},

        gamma(omega) = sigma_min(S0(omega))
                       / (1 + || Delta_L(omega) S0(omega) ||_2 ).

    Kp_vec, Ki_vec: diag entries in task space (3,).
    Returns:
        f_hz: frequency grid (Hz), shape (F,)
        gamma_omega: gamma(omega) over that grid, shape (F,)
        gamma_cost: scalar cost over [fmin_hz, fmax_hz].
    """
    I3 = np.eye(3)

    # Frequency grid
    f_hz = np.fft.rfftfreq(nfft, d=dt)
    omega = 2 * np.pi * f_hz
    F = len(f_hz)

    # Simplified plant frequency response
    P_list = build_P_of_omega(J0, alpha, dt, omega)  # list of 3x6

    # Integrator and controller
    Gz_vec = Gz_from_omega(omega, dt)           # (F,)
    C_all = make_C_PI_diag(Kp_vec, Ki_vec, Gz_vec)  # (F,3,3)

    Delta_a = J0hat_dag - J0_dag               # (6,3)

    gamma_omega = np.zeros(F, dtype=float)

    for kf in range(F):
        P = P_list[kf]                         # (3,6)
        Ck = C_all[kf]                         # (3,3)

        # Nominal loop L0 = P J0_dag C
        L0 = P @ J0_dag @ Ck                   # (3,3)
        A0 = I3 + L0
        try:
            S0 = np.linalg.inv(A0)
        except np.linalg.LinAlgError:
            S0 = np.linalg.pinv(A0)

        # Perturbation Delta_L = P Delta_a C
        Delta_L = P @ Delta_a @ Ck             # (3,3)

        smin_S0 = sigma_min_2norm(S0)
        DeltaS_norm = sigma_max_2norm(Delta_L @ S0)
        denom = 1.0 + DeltaS_norm
        if denom == 0.0:
            gamma_omega[kf] = 0.0
        else:
            gamma_omega[kf] = smin_S0 / denom

    # Cost over frequency band
    mask_band = (f_hz >= fmin_hz) & (f_hz <= fmax_hz)
    if exclude_dc:
        mask_band &= ~np.isclose(f_hz, 0.0)

    band_gamma = gamma_omega[mask_band]
    if lb_metric == "max":
        gamma_cost = np.max(band_gamma)
    else:
        gamma_cost = np.mean(band_gamma)

    return f_hz, gamma_omega, gamma_cost

# ============================================================
# Load data
# ============================================================

# Time-series signals
pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))    # [m], shape (T,3)
e_pi_mm    = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
e_sac_mm   = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()

assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1

e_pi_m  = 1e-3 * e_pi_mm
e_sac_m = 1e-3 * e_sac_mm

# Simplified plant parameters at nominal posture
J0        = np.load(os.path.join(base_dir, "J_k0.npy"))        # (3,6)
J0hat_dag = np.load(os.path.join(base_dir, "pihatJ_k0.npy"))   # (6,3)
J0_dag    = np.load(os.path.join(base_dir, "piJ_k0.npy"))      # (6,3)
alpha     = np.load(os.path.join(base_dir, "alpha_dt0004.npy"))  # (6,)

# Truncate time signals to same length for consistency
T_common = min(
    pstar_seq.shape[0],
    e_pi_m.shape[0],
    e_sac_m.shape[0],
)
pstar_seq = pstar_seq[:T_common]
e_pi_m    = e_pi_m[:T_common]
e_sac_m   = e_sac_m[:T_common]

# ============================================================
# Grid search for PI gains based on simplified plant gamma
# ============================================================

Kp_vals = np.linspace(Kp_min, Kp_max, n_Kp)
Ki_vals = np.linspace(Ki_min, Ki_max, n_Ki)

best_cost = np.inf
best_Kp_vec = None
best_Ki_vec = None
best_f_hz = None
best_gamma_omega = None

print("Searching over PI gains for robust lower bound (gamma) with simplified plant...")
for kp in Kp_vals:
    for ki in Ki_vals:
        Kp_vec = np.array([kp, kp, kp])
        Ki_vec = np.array([ki, ki, ki])

        f_hz_tmp, gamma_omega_tmp, gamma_cost_tmp = compute_gamma_simple_and_cost(
            dt, J0, J0_dag, J0hat_dag, alpha,
            Kp_vec, Ki_vec,
            nfft, fmin_hz, fmax_hz, exclude_dc=exclude_dc,
            lb_metric=lb_metric
        )

        if gamma_cost_tmp < best_cost:
            best_cost = gamma_cost_tmp
            best_Kp_vec = Kp_vec.copy()
            best_Ki_vec = Ki_vec.copy()
            best_f_hz = f_hz_tmp
            best_gamma_omega = gamma_omega_tmp

print("===================================================")
print("Optimal PI gains for PI-class lower bound (simplified plant, within search grid):")
print(f"Kp_opt = {best_Kp_vec}")
print(f"Ki_opt = {best_Ki_vec}")
print(f"LB cost ({lb_metric}) = {best_cost:.4g}")
print("===================================================\n")

# ============================================================
# Sliding-window DFTs (rectangular, no tapering)
# ============================================================

# 1) Reference trajectory -> vector-norm DFT magnitude [m]
f_ref, t_end_ref, P_ref_mag = sliding_dft_vecnorm(
    pstar_seq, dt, nperseg, noverlap, nfft
)

# 2) Loop-shaping PI error -> scalar DFT magnitude [m]
f_pi, t_end_pi, A_pi = sliding_dft_scalar(
    e_pi_m, dt, nperseg, noverlap, nfft
)

# 3) Hybrid SAC error -> scalar DFT magnitude [m]
f_sac, t_end_sac, A_sac = sliding_dft_scalar(
    e_sac_m, dt, nperseg, noverlap, nfft
)

# Consistency checks on frequency grids and window end times
if not (np.allclose(f_ref, best_f_hz) and np.allclose(f_pi, best_f_hz) and np.allclose(f_sac, best_f_hz)):
    raise ValueError("Frequency grids differ; ensure consistent dt, nperseg, nfft.")

if not (np.allclose(t_end_ref, t_end_pi) and np.allclose(t_end_ref, t_end_sac)):
    raise ValueError("Window end times differ; ensure consistent nperseg/noverlap and signal lengths.")

f_hz = best_f_hz
t_end = t_end_ref
omega = 2 * np.pi * f_hz
k_end = (t_end / dt).astype(int)

# ============================================================
# Relative error spectrograms: error / ||P^*||_2
# ============================================================

P_ref_safe = np.maximum(P_ref_mag, eps_ref)

G_pi  = A_pi  / P_ref_safe   # dimensionless
G_sac = A_sac / P_ref_safe   # dimensionless

# Gamma is time-invariant for simplified plant -> tile across windows
W = P_ref_mag.shape[1]
gamma_local = np.tile(best_gamma_omega[:, None], (1, W))  # shape (F,W)

# ============================================================
# Frequency & time masks for plotting
# ============================================================
mask_freq = band_mask_freq_rad(
    omega, omega_min_plot, omega_max_plot, exclude_dc=exclude_dc
)
mask_time = (k_end >= k_min) & (k_end <= k_max)

omega_plot = omega[mask_freq]
k_plot = k_end[mask_time]

gamma_plot = gamma_local[mask_freq, :][:, mask_time]
G_pi_plot  = G_pi[mask_freq, :][:, mask_time]
G_sac_plot = G_sac[mask_freq, :][:, mask_time]

# ============================================================
# Figure 1: Spectrogram of gamma(k, omega) (dimensionless)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im1 = ax1.pcolormesh(
    k_plot,
    omega_plot,
    gamma_plot,
    shading="gouraud",
    cmap=cmap_gamma,
    vmin=vmin_gamma, vmax=vmax_gamma
)
ax1.set_title(r"Robust Lower-Bound Factor $\gamma(k,\omega)$ (simplified plant)")
ax1.set_xlabel(r"Sample index $k$")
ax1.set_ylabel(r"$\omega$ [rad/s]")
ax1.set_xlim((k_min, k_max))
ax1.set_yscale("log")
ax1.set_ylim((omega_min_plot, omega_max_plot))
cbar1 = fig1.colorbar(im1, ax=ax1)
cbar1.set_label(r"$\gamma(k,\omega)$")

# ============================================================
# Figure 2: Spectrogram of relative PI error gain
# ============================================================
fig2, ax2 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im2 = ax2.pcolormesh(
    k_plot,
    omega_plot,
    G_pi_plot,
    shading="gouraud",
    cmap=cmap_rel,
    vmin=vmin_rel, vmax=vmax_rel
)
ax2.set_title(r"Loop-Shaping PI Relative Error Gain")
ax2.set_xlabel(r"Sample index $k$")
ax2.set_ylabel(r"$\omega$ [rad/s]")
ax2.set_xlim((k_min, k_max))
ax2.set_yscale("log")
ax2.set_ylim((omega_min_plot, omega_max_plot))
cbar2 = fig2.colorbar(im2, ax=ax2)
cbar2.set_label(r"$\|E_{\mathrm{PI}}(k,\omega)\| / \|\tilde{\mathbf P}^*_k(\omega)\|_2$")

# ============================================================
# Figure 3: Spectrogram of relative hybrid error gain
# ============================================================
fig3, ax3 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im3 = ax3.pcolormesh(
    k_plot,
    omega_plot,
    G_sac_plot,
    shading="gouraud",
    cmap=cmap_rel,
    vmin=vmin_rel, vmax=vmax_rel
)
ax3.set_title(r"Hybrid SAC--PI Relative Error Gain")
ax3.set_xlabel(r"Sample index $k$")
ax3.set_ylabel(r"$\omega$ [rad/s]")
ax3.set_xlim((k_min, k_max))
ax3.set_yscale("log")
ax3.set_ylim((omega_min_plot, omega_max_plot))
cbar3 = fig3.colorbar(im3, ax=ax3)
cbar3.set_label(r"$\|E_{\mathrm{hyb}}(k,\omega)\| / \|\tilde{\mathbf P}^*_k(\omega)\|_2$")

plt.show()

print("Done.")
