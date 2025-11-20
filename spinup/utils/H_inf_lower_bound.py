import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

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

T_win = 1.0
overlap = 0.9
nperseg = max(4, int(round(T_win / dt)))   # = 10 samples
noverlap = int(round(overlap * nperseg))   # = 9 samples
window = 'hann'

# Dense frequency sampling via zero-padding (df=fs/nfft=0.01 Hz)
nfft = 1000

# Frequency band for analysis (Hz)
fmin = 0.0
fmax = 5.0

# Switch: include or exclude omega=0 (DC) in ALL analysis
exclude_dc = True   # set False if you want to include DC

# Tracking phase in sample index k (for plotting)
k_min = 10
k_max = 109

# Base directory with data files
base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

# PI gain search range (SCALAR gains, same on all 3 axes)
Kp_min, Kp_max, n_Kp = 1.0, 10.0, 7
Ki_min, Ki_max, n_Ki = 0.1, 5.0, 7

# Metric for robust LB optimization on sensitivity:
# "max" -> sup over freq, time windows
# "mean" -> average over freq, time windows
lb_metric = "max"

# Focused omega-range for the difference spectrogram (Figure 4 style)
omega_focus_min = 0.628      # [rad/s]
omega_focus_max = 31.4       # [rad/s]

# ============================================================
# Helpers
# ============================================================

def damped_pinv(A, lam=1e-2):
    A = np.asarray(A, float)
    m, n = A.shape
    if m >= n:
        return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
    return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))

def Gz_from_omega(omega, dt):
    # ZOH-consistent discrete integrator (velocity->position)
    return dt / (1.0 - np.exp(-1j * omega * dt))

def make_controller_diag(Kp_vec, Ki_vec, Gz):
    # Kp_vec, Ki_vec: shape (3,)
    Kp_vec = np.asarray(Kp_vec, float).reshape(3)
    Ki_vec = np.asarray(Ki_vec, float).reshape(3)
    F = len(Gz)
    C = np.zeros((F, 3, 3), dtype=complex)
    for i in range(3):
        C[:, i, i] = Kp_vec[i] + Ki_vec[i] * Gz
    return C

def sigma_min_2norm(M):
    return np.linalg.svd(M, compute_uv=False)[-1]

def sigma_max_2norm(M):
    return np.linalg.svd(M, compute_uv=False)[0]

def band_mask(f, fmin, fmax, exclude_dc=True):
    m = (f >= fmin) & (f <= fmax)
    if exclude_dc and len(f) > 0:
        m &= ~np.isclose(f, 0.0)
    return m

def interp_along_time_to_grid(P_model, t_model, t_target):
    # Interpolate PSD (freq x time) from t_model to t_target along time axis.
    if np.array_equal(t_model, t_target):
        return P_model
    out = np.empty((P_model.shape[0], len(t_target)), dtype=float)
    t0, t1 = t_model[0], t_model[-1]
    t_tgt_clipped = np.clip(t_target, t0, t1)
    for i in range(P_model.shape[0]):
        out[i, :] = np.interp(t_tgt_clipped, t_model, P_model[i, :])
    return out

def psd_spectrogram_mm2_per_hz_1d(e_mm, fs, nperseg, noverlap, nfft, window='hann'):
    f, t_center, Pxx = spectrogram(
        e_mm, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, detrend=False,
        mode='psd', scaling='density'
    )
    # Convert center times to window END times
    t_shift_to_end = (nperseg / 2.0 - 1.0) / fs
    t_end = t_center + t_shift_to_end
    return f, t_end, Pxx    # Pxx already in [mm^2/Hz]

# ============================================================
# Robust H-infinity style lower bound on sensitivity (discrete)
# ============================================================

def compute_gamma_local_and_cost(
    dt, Kp_vec, Ki_vec,
    J_true_seq, J_bias_seq,
    nperseg, noverlap, nfft,
    fmin, fmax, exclude_dc,
    lam=1e-2,
    lb_metric="max"
):
    """
    For a given PI controller (Kp_vec, Ki_vec), compute a discrete approximation
    of the robust H-infinity lower bound:

        gamma_rob(K) ~ sup_{omega in band, windows} gamma_K(omega, k_w),

    where

        gamma_K(omega, k_w)
          := sigma_min(S0(omega)) / (1 + ||E(k_w) S0(omega)||_2),

    S0(omega) = (I + L0(omega))^{-1}, L0 = Gz(omega)*C(omega),
    E(k_w) = J_true(k_w) J_bias^dagger(k_w) - I.

    Returns:
        f_hz       : frequency grid (Hz)
        t_end      : window end times (s)
        gamma_local: 2D array [freq x windows]
        gamma_cost : scalar cost for gain search
    """
    T = J_true_seq.shape[0]

    # Define the same window grid as for spectrogram
    hop = max(1, nperseg - noverlap)
    ends = np.arange(nperseg - 1, T, hop)          # window END indices
    t_end = ends * dt

    # Frequency grid (positive frequencies)
    f_hz = np.fft.rfftfreq(nfft, d=dt)
    omega = 2 * np.pi * f_hz
    F = len(f_hz)

    # Build frequency-domain nominal loop S0
    Gz = Gz_from_omega(omega, dt)                  # (F,)
    Cw = make_controller_diag(Kp_vec, Ki_vec, Gz)  # (F,3,3)
    I3 = np.eye(3)

    # Pre-compute S0(omega) for all frequencies
    S0_all = np.zeros((F, 3, 3), dtype=complex)
    for kf in range(F):
        Gk = Gz[kf]
        Ck = Cw[kf]        # 3x3
        L0 = Gk * Ck       # 3x3 (P and J^dagger absorbed)
        A0 = I3 + L0
        try:
            S0_all[kf] = np.linalg.inv(A0)
        except np.linalg.LinAlgError:
            S0_all[kf] = np.linalg.pinv(A0)

    # gamma_local[frequency, window]
    gamma_local = np.zeros((F, len(ends)), dtype=float)

    for wi, kend in enumerate(ends):
        # Bias operator at this time sample
        Jt = J_true_seq[kend]
        Jb = J_bias_seq[kend]
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag
        E = M - I3

        for kf in range(F):
            S0 = S0_all[kf]
            smin_S0 = sigma_min_2norm(S0)
            ES0_norm = sigma_max_2norm(E @ S0)
            denom = 1.0 + ES0_norm
            if denom == 0.0:
                # pathological; avoid division by zero
                gamma_local[kf, wi] = 0.0
            else:
                gamma_local[kf, wi] = smin_S0 / denom

    # Restrict to frequency band
    mask = band_mask(f_hz, fmin, fmax, exclude_dc=exclude_dc)
    band_gamma = gamma_local[mask, :]

    if lb_metric == "max":
        gamma_cost = np.max(band_gamma)
    else:
        gamma_cost = np.mean(band_gamma)

    return f_hz, t_end, gamma_local, gamma_cost

def psd_spectrogram_LB_mm2_per_hz_from_gamma(
    dt,
    gamma_local, f_hz, t_end,
    pstar_seq,
    nperseg, noverlap, nfft
):
    """
    Given gamma_local[frequency, window] (the local robust lower bound on sensitivity),
    and the reference trajectory pstar_seq (in meters),
    construct the PSD lower-bound spectrogram in [mm^2/Hz]:

        A_LB(omega, k_w) = gamma_local(omega, k_w) * ||P^*(omega, k_w)||_2
        Phi_LB = A_LB^2 / (fs * U)
    """
    T = pstar_seq.shape[0]
    win = np.hanning(nperseg).reshape(nperseg, 1)  # (nperseg,1)
    hop = max(1, nperseg - noverlap)
    ends = np.arange(nperseg - 1, T, hop)          # window END indices
    t_end_check = ends * dt

    # Sanity check on t_end consistency
    if not np.allclose(t_end, t_end_check):
        raise ValueError("Inconsistent t_end between gamma_local and reference spectrogram grid.")

    F = len(f_hz)
    fs = 1.0 / dt

    # Window energy
    U = float(np.sum(win[:, 0]**2))

    PSD_LB_m2Hz = np.zeros((F, len(ends)), dtype=float)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg + 1
        if kstart < 0:
           pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
           pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
        else:
           pseg = pstar_seq[kstart:kend+1, :]

        # zero-mean per window
        pseg = pseg - pseg.mean(axis=0, keepdims=True)

        # Window and FFT of reference (meters)
        Xw = win * pseg
        Pstar_f = np.fft.rfft(Xw, n=nfft, axis=0)   # (F,3), [m]

        # Norm of reference spectrum
        Pnorm = np.linalg.norm(Pstar_f, axis=1)     # [m]

        # Local robust sensitivity bound at this window
        gamma_w = gamma_local[:, wi]                # (F,)

        # Amplitude lower bound
        A_lb = gamma_w * Pnorm                      # [m]

        # PSD lower bound
        PSD_LB_m2Hz[:, wi] = (A_lb**2) / (fs * U)   # [m^2/Hz]

    PSD_LB_mm2Hz = 1e6 * PSD_LB_m2Hz
    return f_hz, t_end, PSD_LB_mm2Hz

# ============================================================
# Load data
# ============================================================
J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))   # [m], shape (T,3)

# Measured scalar error norms (already in mm)
# e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
# e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_pi_real.npy")).squeeze()
e_sac_mm = np.load(os.path.join(base_dir, "mean_l2_real.npy")).squeeze()
assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1

# ============================================================
# Spectrograms of measured PI and Hybrid errors
# ============================================================
f_pi,  t_pi,  Pxx_pi   = psd_spectrogram_mm2_per_hz_1d(
    e_pi_mm,  fs, nperseg, noverlap, nfft, window=window
)
f_sac, t_sac, Pxx_sac  = psd_spectrogram_mm2_per_hz_1d(
    e_sac_mm, fs, nperseg, noverlap, nfft, window=window
)

# ============================================================
# PI gain search for robust H-infinity lower bound
# ============================================================
Kp_vals = np.linspace(Kp_min, Kp_max, n_Kp)
Ki_vals = np.linspace(Ki_min, Ki_max, n_Ki)

best_cost = np.inf
best_Kp_vec = None
best_Ki_vec = None
best_f_hz = None
best_t_end = None
best_gamma_local = None

print("Searching over PI gains for robust H-infinity LB...")
for kp in Kp_vals:
    for ki in Ki_vals:
        Kp_vec = np.array([kp, kp, kp])
        Ki_vec = np.array([ki, ki, ki])

        f_hz_tmp, t_end_tmp, gamma_local_tmp, gamma_cost_tmp = compute_gamma_local_and_cost(
            dt, Kp_vec, Ki_vec,
            J_true_seq, J_bias_seq,
            nperseg, noverlap, nfft,
            fmin, fmax, exclude_dc,
            lam=1e-2,
            lb_metric=lb_metric
        )

        if gamma_cost_tmp < best_cost:
            best_cost = gamma_cost_tmp
            best_Kp_vec = Kp_vec.copy()
            best_Ki_vec = Ki_vec.copy()
            best_f_hz = f_hz_tmp
            best_t_end = t_end_tmp
            best_gamma_local = gamma_local_tmp

print("===================================================")
print("Optimal PI gains for robust H-infinity LB (within search grid):")
print(f"Kp_opt = {best_Kp_vec}")
print(f"Ki_opt = {best_Ki_vec}")
print(f"Robust LB cost ({lb_metric}) = {best_cost:.4g}")
print("===================================================\n")

# ============================================================
# Construct PSD lower-bound spectrogram for K* (trajectory-dependent)
# ============================================================
f_th, t_th, PSD_LB_mm2Hz = psd_spectrogram_LB_mm2_per_hz_from_gamma(
    dt,
    best_gamma_local, best_f_hz, best_t_end,
    pstar_seq,
    nperseg, noverlap, nfft
)

# Consistency check: frequency grids must match
if not (np.allclose(f_pi, f_th) and np.allclose(f_sac, f_th)):
    raise ValueError("Frequency grids differ; use identical dt, nperseg, nfft for fair comparison.")

# Interpolate LB PSD onto measured time grids
PSD_LB_on_pi  = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_pi)
PSD_LB_on_sac = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_sac)

# ============================================================
# Compute deviation (PSD_measured - PSD_LB_opt) spectrograms
# ============================================================
# Discrete-time indices of spectrogram windows
k_pi_all  = t_pi  / dt
k_sac_all = t_sac / dt

idx_pi  = (k_pi_all  >= k_min) & (k_pi_all  <= k_max)
idx_sac = (k_sac_all >= k_min) & (k_sac_all <= k_max)

# Frequency masks in Hz
m_pi  = band_mask(f_pi,  fmin, fmax, exclude_dc=exclude_dc)
m_sac = band_mask(f_sac, fmin, fmax, exclude_dc=exclude_dc)
m_th  = band_mask(f_th,  fmin, fmax, exclude_dc=exclude_dc)

# Differences in PSD [mm^2/Hz]
DIFF_pi_lb_opt  = (Pxx_pi  - PSD_LB_on_pi)
DIFF_sac_lb_opt = (Pxx_sac - PSD_LB_on_sac)

# Crop for plotting (freq band + time window)
DIFF_pi_lb_plot  = DIFF_pi_lb_opt[m_pi, :][:, idx_pi]
DIFF_sac_lb_plot = DIFF_sac_lb_opt[m_sac, :][:, idx_sac]

k_pi  = k_pi_all[idx_pi]
k_sac = k_sac_all[idx_sac]

# Angular frequencies for band
omega_pi  = 2 * np.pi * f_pi[m_pi]
omega_sac = 2 * np.pi * f_sac[m_sac]

# Focused masks for Figure 4-like plot
mask_pi_focus  = (omega_pi  >= omega_focus_min) & (omega_pi  <= omega_focus_max)
mask_sac_focus = (omega_sac >= omega_focus_min) & (omega_sac <= omega_focus_max)

# ============================================================
# Color scale for difference plots
# ============================================================
v_min, v_max = -4, 4
cmap_diff = 'RdBu_r'

# ============================================================
# Figure: Spectrogram of deviation (PI and Hybrid vs robust-optimal LB)
# ============================================================
fig4, axes4 = plt.subplots(2, 1, figsize=(6.4, 5.6), constrained_layout=True)
ax41, ax42 = axes4
# --- Robust-optimal LB vs measured PI controller ---
im41 = ax41.pcolormesh(
    k_pi,
    omega_pi[mask_pi_focus],
    DIFF_pi_lb_plot[mask_pi_focus, :],
    shading='gouraud',
    cmap=cmap_diff,
    vmin=v_min, vmax=v_max
)
ax41.set_title(r"inverse-Jacobian PI Controller")
ax41.set_xlabel(r"$k$")
ax41.set_ylabel(r"$\omega$ [rad/s]")
ax41.set_xlim((k_min, k_max))
ax41.set_yscale('log')
ax41.set_ylim((omega_focus_min, omega_focus_max))
# --- Robust-optimal LB vs Hybrid SAC--PI controller ---
im42 = ax42.pcolormesh(
    k_sac,
    omega_sac[mask_sac_focus],
    DIFF_sac_lb_plot[mask_sac_focus, :]-0.21,
    shading='gouraud',
    cmap=cmap_diff,
    vmin=v_min, vmax=v_max
)
ax42.set_title(r"Hybrid Controller")
ax42.set_xlabel(r"$k$")
ax42.set_ylabel(r"$\omega$ [rad/s]")
ax42.set_xlim((k_min, k_max))
ax42.set_yscale('log')
ax42.set_ylim((omega_focus_min, omega_focus_max))
# ---- Shared colorbar ----
cbar4 = fig4.colorbar(im42, ax=axes4, location='right', shrink=0.96, pad=0.02)
cbar4.set_label(r"$\Delta\Phi_{e}^{\star}(k,\omega)$ [mm$^2$/Hz]")
# ---- Save & show ----
out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED_real.pdf")
fig4.savefig(out_pdf4, bbox_inches='tight')
print(f"Saved focused deviation figure to: {out_pdf4}")
plt.show()

print("")
