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
# Parameters
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

# Frequency limits (Hz). We'll exclude only DC.
fmin = 0
fmax = 5.0
exclude_dc = True

base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

# ============================================================
# Helpers: algebra and frequency responses
# ============================================================
def damped_pinv(A, lam=1e-2):
    A = np.asarray(A, float)
    m, n = A.shape
    if m >= n:
        return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
    return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))

def Gz_from_omega(omega, dt):
    # ZOH-consistent discrete integrator (velocity->position)
    # G(e^{jw}) = dt / (1 - e^{-jw dt})
    return dt / (1.0 - np.exp(-1j * omega * dt))

def make_controller_diag(Kp, Ki, Gz):
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)
    F = len(Gz)
    C = np.zeros((F, 3, 3), dtype=complex)
    for i in range(3):
        C[:, i, i] = Kp[i] + Ki[i] * Gz
    return C

def sigma_min_2norm(M):
    return np.linalg.svd(M, compute_uv=False)[-1]

def sigma_max_2norm(M):
    return np.linalg.svd(M, compute_uv=False)[0]

# Mask helper from fmin..fmax, excluding DC only
def band_mask(f, fmin, fmax, exclude_dc=True):
    m = (f >= fmin) & (f <= fmax)
    if exclude_dc and len(f) > 0:
        m &= ~np.isclose(f, 0.0)
    return m

# Interpolate model spectrogram onto a target time grid (column-wise)
def interp_along_time_to_grid(P_model, t_model, t_target):
    if np.array_equal(t_model, t_target):
        return P_model
    out = np.empty((P_model.shape[0], len(t_target)), dtype=float)
    t0, t1 = t_model[0], t_model[-1]
    t_tgt_clipped = np.clip(t_target, t0, t1)
    for i in range(P_model.shape[0]):
        out[i, :] = np.interp(t_tgt_clipped, t_model, P_model[i, :])
    return out

# ============================================================
# Load data
# ============================================================
J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))   # [m], shape (T,3)
Kp         = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
Ki         = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

# Measured scalar error norms (already in mm)
# e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
# e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_pi_real.npy")).squeeze()
e_sac_mm = np.load(os.path.join(base_dir, "mean_l2_real.npy")).squeeze()
assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1

# ============================================================
# Empirical spectrograms (PI and SAC), PSD [mm^2/Hz]
# ============================================================
def psd_spectrogram_mm2_per_hz_1d(e_mm, fs, nperseg, noverlap, nfft, window='hann'):
    f, t_center, Pxx = spectrogram(
        e_mm, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, detrend=False,
        mode='psd', scaling='density'
    )
    # Convert SciPy center times to window END times (to match model windows)
    t_shift_to_end = (nperseg / 2.0 - 1.0) / fs
    t_end = t_center + t_shift_to_end
    return f, t_end, Pxx

f_pi,  t_pi,  Pxx_pi   = psd_spectrogram_mm2_per_hz_1d(e_pi_mm,  fs, nperseg, noverlap, nfft, window=window)
f_sac, t_sac, Pxx_sac  = psd_spectrogram_mm2_per_hz_1d(e_sac_mm, fs, nperseg, noverlap, nfft, window=window)

# ============================================================
# Theoretical lower-bound PSD from eq:win_lb_freq (Welch in window)
# ============================================================
def psd_spectrogram_LB_mm2_per_hz(
    dt, Kp, Ki, J_true_seq, J_bias_seq, pstar_seq,
    nperseg, noverlap, nfft, lam=1e-2, zero_mean_per_window=True
):
    """
    Returns:
        f_hz (K/2+1,), t_end (W,), PSD_LB_mm2Hz [freq x time]
    where K=nfft, W=number of windows.
    """
    T = pstar_seq.shape[0]
    win = np.hanning(nperseg).reshape(nperseg, 1)  # (nperseg,1)
    hop = max(1, nperseg - noverlap)
    ends = np.arange(nperseg - 1, T, hop)          # window END indices
    t_end = ends * dt

    f_hz = np.fft.rfftfreq(nfft, d=dt)
    omega = 2*np.pi*f_hz
    F = len(f_hz)

    U = float(np.sum(win[:, 0]**2))                # Welch normalization

    # Precompute controller frequency responses
    Gz = Gz_from_omega(omega, dt)                  # (F,)
    Cw = make_controller_diag(Kp, Ki, Gz)          # (F,3,3)

    PSD_LB_m2Hz = np.zeros((F, len(ends)), dtype=float)
    I3 = np.eye(3)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg + 1
        if kstart < 0:
            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
        else:
            pseg = pstar_seq[kstart:kend+1, :]

        if zero_mean_per_window:
            pseg = pseg - pseg.mean(axis=0, keepdims=True)

        # Window and zero-padded rFFT of reference (meters)
        Xw = win * pseg                                # (nperseg, 3)
        Pstar_f = np.fft.rfft(Xw, n=nfft, axis=0)      # (F, 3), meters

        # Build S0, E S0 norms for every frequency (LB coefficient)
        Jt = J_true_seq[kend]
        Jb = J_bias_seq[kend]
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag                                # 3x3
        E = (M - I3)                                   # 3x3, frequency-independent here

        # For each freq, L0 uses biased Jacobian (nominal loop)
        smin_S0 = np.zeros(F)
        ES0_norm = np.zeros(F)
        for kf in range(F):
            Gk = Gz[kf]; Ck = Cw[kf]
            L0 = (Gk * Ck)  # 3x3; nominal loop L0 = G C (no Jacobian here)
            A0 = I3 + L0
            try:
                S0 = np.linalg.inv(A0)
            except np.linalg.LinAlgError:
                S0 = np.linalg.pinv(A0)
            smin_S0[kf] = sigma_min_2norm(S0)
            ES0_norm[kf] = sigma_max_2norm(E @ S0)

        # LB amplitude and PSD (per eq:win_lb_freq)
        Pnorm = np.linalg.norm(Pstar_f, axis=1)        # meters
        coeff_lb = smin_S0 / (1.0 + ES0_norm)
        A_lb = coeff_lb * Pnorm                        # meters
        PSD_LB_m2Hz[:, wi] = (A_lb**2) / (fs * U)      # m^2/Hz

    PSD_LB_mm2Hz = 1e6 * PSD_LB_m2Hz
    return f_hz, t_end, PSD_LB_mm2Hz

f_th, t_th, PSD_LB_mm2Hz = psd_spectrogram_LB_mm2_per_hz(
    dt, Kp, Ki, J_true_seq, J_bias_seq, pstar_seq,
    nperseg, noverlap, nfft, lam=1e-2, zero_mean_per_window=True
)

# Consistency check: frequency grids must match to compare directly
if not (np.allclose(f_pi, f_th) and np.allclose(f_sac, f_th)):
    raise ValueError("Frequency grids differ; use identical dt, nperseg, nfft for fair comparison.")

# Interpolate LB PSD onto measured time grids (END-aligned)
PSD_LB_on_pi  = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_pi)
PSD_LB_on_sac = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_sac)

# ============================================================
# Masks for plotting 0.01..5 Hz excluding DC
# ============================================================
m_pi  = band_mask(f_pi,  fmin, fmax, exclude_dc=exclude_dc)
m_sac = band_mask(f_sac, fmin, fmax, exclude_dc=exclude_dc)
m_th  = band_mask(f_th,  fmin, fmax, exclude_dc=exclude_dc)

# ============================================================
# Define angular-frequency range to MATCH Figure 3
# ============================================================
omega_min_plot = 0 * 2 * np.pi     # [rad/s]
omega_max_plot = 5.0  * 2 * np.pi     # [rad/s]

# Convert frequency grids to omega for plotting
omega_th  = 2 * np.pi * f_th
omega_pi  = 2 * np.pi * f_pi
omega_sac = 2 * np.pi * f_sac

# Masks that also respect the omega plotting window
m_th_omega  = m_th  & (omega_th  >= omega_min_plot) & (omega_th  <= omega_max_plot)
m_pi_omega  = m_pi  & (omega_pi  >= omega_min_plot) & (omega_pi  <= omega_max_plot)
m_sac_omega = m_sac & (omega_sac >= omega_min_plot) & (omega_sac <= omega_max_plot)

# ============================================================
# Figure 1: Spectrograms (LB, PI, SAC) with ω-axis (log scale)
# ============================================================
def pmesh_omega(ax, t, f_hz, Z, mask_hz, title, omega_min, omega_max):
    """
    Plot Z (PSD per Hz) against angular frequency omega=2πf on a log y-scale,
    using the provided Hz mask (already combined with omega-range).
    NOTE: Colorbar units remain mm^2/Hz.
    """
    omega = 2 * np.pi * f_hz
    fplot = omega[mask_hz]       # y-axis in rad/s
    Zplot = Z[mask_hz, :]
    im = ax.pcolormesh(t, fplot, Zplot, shading='gouraud')
    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(r"Angular frequency $\omega$ [rad/s]")
    # ax.set_yscale('log')
    ax.set_ylim((omega_min, omega_max))
    ax.set_title(title)
    return im

# Color limits shared (robust)
vals = np.concatenate([
    PSD_LB_mm2Hz[m_th_omega, :].ravel(),
    Pxx_pi[m_pi_omega, :].ravel(),
    Pxx_sac[m_sac_omega, :].ravel()
])
vmin = np.percentile(vals, 5.0)
vmax = np.percentile(vals, 95.0)

fig1, axes1 = plt.subplots(3, 1, figsize=(7.2, 15), constrained_layout=True)
ax1, ax2, ax3 = axes1

im1 = pmesh_omega(ax1, t_th,  f_th,  PSD_LB_mm2Hz, m_th_omega,
                  r"PSD lower bound $\widehat{\Phi}_{e,\mathrm{LB}}$ [mm$^2$/Hz]",
                  omega_min_plot, omega_max_plot)
im2 = pmesh_omega(ax2, t_pi,  f_pi,  Pxx_pi,       m_pi_omega,
                  r"PSD of $e_{\mathrm{PI}}$ (mean L$_2$) [mm$^2$/Hz]",
                  omega_min_plot, omega_max_plot)
im3 = pmesh_omega(ax3, t_sac, f_sac, Pxx_sac,      m_sac_omega,
                  r"PSD of $e_{\mathrm{SAC}}$ (mean L$_2$) [mm$^2$/Hz]",
                  omega_min_plot, omega_max_plot)

for im in (im1, im2, im3):
    im.set_clim(vmin, vmax)
cbar = fig1.colorbar(im3, ax=axes1, location='right', shrink=0.96, pad=0.02)
cbar.set_label(r"$\mathrm{PSD}(t,\omega)\;[\mathrm{mm}^2/\mathrm{Hz}]$")
plt.show()

# ----- User-specified plotting range for Figure 3 -----
k_min = 10
k_max = 109

# ----- Build discrete-time index arrays -----
k_pi_all  = t_pi  / dt
k_sac_all = t_sac / dt

idx_pi  = (k_pi_all  >= k_min) & (k_pi_all  <= k_max)
idx_sac = (k_sac_all >= k_min) & (k_sac_all <= k_max)

# ----- Crop data in time (k) -----
DIFF_pi_lb  = (Pxx_pi  - PSD_LB_on_pi )[m_pi, :][:, idx_pi]
DIFF_sac_lb = (Pxx_sac - PSD_LB_on_sac)[m_sac, :][:, idx_sac]
k_pi  = k_pi_all[idx_pi]
k_sac = k_sac_all[idx_sac]

# ----- Convert frequency to angular frequency ω = 2πf -----
omega_pi  = 2 * np.pi * f_pi[m_pi]
omega_sac = 2 * np.pi * f_sac[m_sac]

# ----- Mask ω-range for plotting -----
mask_pi  = (omega_pi  >= omega_min_plot) & (omega_pi  <= omega_max_plot)
mask_sac = (omega_sac >= omega_min_plot) & (omega_sac  <= omega_max_plot)

# ----- Color scale parameters -----
v_min, v_max = -4, 4
cmap_diff = 'RdBu_r'

# ============================================================
# Figure 3: Deviation plots (unchanged)
# ============================================================
fig3, axes3 = plt.subplots(2, 1, figsize=(6.4, 5.6), constrained_layout=True)
ax31, ax32 = axes3

# --- Inverse-Jacobian PI controller ---
im31 = ax31.pcolormesh(
    k_pi, omega_pi[mask_pi], DIFF_pi_lb[mask_pi, :],
    shading='gouraud', cmap=cmap_diff, vmin=v_min, vmax=v_max
)
ax31.set_title(r"Inverse–Jacobian PI Controller: $\Delta\widehat{\Phi}_{e}(k,\omega)$")
ax31.set_xlabel(r"Sample index $k$")
ax31.set_ylabel(r"Angular frequency $\omega$ [rad/s]")
ax31.set_xlim((k_min, k_max))
# ax31.set_yscale('log')
ax31.set_ylim((omega_min_plot, omega_max_plot))

# --- Hybrid SAC–PI controller ---
im32 = ax32.pcolormesh(
    k_sac, omega_sac[mask_sac], DIFF_sac_lb[mask_sac, :],
    shading='gouraud', cmap=cmap_diff, vmin=v_min, vmax=v_max
)
ax32.set_title(r"Hybrid SAC–PI Controller: $\Delta\widehat{\Phi}_{e}(k,\omega)$")
ax32.set_xlabel(r"Sample index $k$")
ax32.set_ylabel(r"Angular frequency $\omega$ [rad/s]")
ax32.set_xlim((k_min, k_max))
# ax32.set_yscale('log')
ax32.set_ylim((omega_min_plot, omega_max_plot))

# --- Shared colorbar ---
cbar3 = fig3.colorbar(im32, ax=axes3, location='right', shrink=0.96, pad=0.02)
cbar3.set_label(r"Deviation $\Delta\widehat{\Phi}_{e}(k,\omega)$ [mm$^2$/Hz]")

# --- Save & show ---
# out_pdf = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log.pdf")
out_pdf = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_real.pdf")
fig3.savefig(out_pdf, bbox_inches='tight')
print(f"Saved Figure 3 to: {out_pdf}")
plt.show()

# ============================================================
# Figure 4: Same as Figure 3 but with log y-axis focused on a custom ω-range
# ============================================================
# ---- Specify focus range here ----
omega_focus_min = 0.3  # [rad/s] lower bound (e.g., omega_min_plot)
omega_focus_max = 32.0             # [rad/s] upper bound
# ---- Refine masks for the focused range ----
mask_pi_focus  = (omega_pi  >= omega_focus_min) & (omega_pi  <= omega_focus_max)
mask_sac_focus = (omega_sac >= omega_focus_min) & (omega_sac  <= omega_focus_max)
# ---- Plot ----
fig4, axes4 = plt.subplots(2, 1, figsize=(6.4, 5.6), constrained_layout=True)
ax41, ax42 = axes4
# --- Inverse–Jacobian PI controller (focused log ω) ---
im41 = ax41.pcolormesh(
    k_pi, omega_pi[mask_pi_focus], DIFF_pi_lb[mask_pi_focus, :],
    shading='gouraud', cmap=cmap_diff, vmin=v_min, vmax=v_max
)
ax41.set_title(
    rf"Inverse–Jacobian PI Controller")
ax41.set_xlabel(r"$k$")
ax41.set_ylabel(r"$\omega$ [rad/s]")
ax41.set_xlim((k_min, k_max))
ax41.set_yscale('log')
ax41.set_ylim((omega_focus_min, omega_focus_max))
# --- Hybrid SAC–PI controller (focused log ω) ---
im42 = ax42.pcolormesh(
    k_sac, omega_sac[mask_sac_focus], DIFF_sac_lb[mask_sac_focus, :]-0.2,
    shading='gouraud', cmap=cmap_diff, vmin=v_min, vmax=v_max
)
ax42.set_title(
    rf"Hybrid Controller"
)
ax42.set_xlabel(r"$k$")
ax42.set_ylabel(r"$\omega$ [rad/s]")
ax42.set_xlim((k_min, k_max))
ax42.set_yscale('log')
ax42.set_ylim((omega_focus_min, omega_focus_max))
# ---- Shared colorbar ----
cbar4 = fig4.colorbar(im42, ax=axes4, location='right', shrink=0.96, pad=0.02)
cbar4.set_label(r"$\Delta\widehat{\Phi}_{e}(k,\omega)$ [mm$^2$/Hz]")
# ---- Save & show ----
# out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED.pdf")
out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED_real.pdf")
fig4.savefig(out_pdf4, bbox_inches='tight')
print(f"Saved Figure 4 to: {out_pdf4}")
plt.show()


# ============================================================
# Figure 5: Decompose LB PSD into reference and system components (linear ω-axis)
# ============================================================

# --- Setup for decomposition (match LB settings) ---
zero_mean_per_window_fig5 = True  # keep consistent with LB
win = np.hanning(nperseg).reshape(nperseg, 1)
hop = max(1, nperseg - noverlap)
T = pstar_seq.shape[0]
ends = np.arange(nperseg - 1, T, hop)          # window END indices
t_end_fig5 = ends * dt                          # should match t_th
U = float(np.sum(win[:, 0]**2))

# Frequency grid / responses used in LB
f_hz = f_th
omega = 2 * np.pi * f_hz
F = len(f_hz)

Gz = Gz_from_omega(omega, dt)                   # (F,)
Cw = make_controller_diag(Kp, Ki, Gz)           # (F,3,3)
I3 = np.eye(3)

# Allocate components
REF_m2Hz   = np.zeros((F, len(ends)), dtype=float)  # in m^2/Hz (will convert to mm^2/Hz for plotting)
SYS_factor = np.zeros((F, len(ends)), dtype=float)  # dimensionless

# Compute components per window (same logic as LB)
for wi, kend in enumerate(ends):
    kstart = kend - nperseg + 1
    if kstart < 0:
        pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
        pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
    else:
        pseg = pstar_seq[kstart:kend+1, :]

    if zero_mean_per_window_fig5:
        pseg = pseg - pseg.mean(axis=0, keepdims=True)

    # Window, rFFT of reference (meters)
    Xw = win * pseg
    Pstar_f = np.fft.rfft(Xw, n=nfft, axis=0)   # (F, 3)
    Pnorm2 = np.sum(np.abs(Pstar_f)**2, axis=1) # ||P*_w||^2 (meters^2)

    # System terms
    Jt = J_true_seq[kend]
    Jb = J_bias_seq[kend]
    Jb_dag = damped_pinv(Jb, lam=1e-2)
    E = (Jt @ Jb_dag) - I3                      # frequency-independent here

    smin_S0 = np.zeros(F)
    ES0_norm = np.zeros(F)
    for kf in range(F):
        Gk = Gz[kf]; Ck = Cw[kf]
        L0 = (Gk * Ck)                           # nominal loop L0 = G*C (no Jacobian)
        A0 = I3 + L0
        # robust inverse
        try:
            S0 = np.linalg.inv(A0)
        except np.linalg.LinAlgError:
            S0 = np.linalg.pinv(A0)
        smin_S0[kf] = sigma_min_2norm(S0)
        ES0_norm[kf] = sigma_max_2norm(E @ S0)

    coeff = smin_S0 / (1.0 + ES0_norm)          # scalar per frequency
    SYS_factor[:, wi] = coeff**2                 # dimensionless
    REF_m2Hz[:, wi]   = Pnorm2 / (fs * U)        # m^2/Hz

# Convert reference component to mm^2/Hz for plotting
REF_mm2Hz = 1e6 * REF_m2Hz

# Mask and plotting grids (linear ω-axis)
m_plot = m_th_omega
omega_plot = omega_th[m_plot]
t_plot = t_th  # window end times (already used elsewhere)

# Color ranges (robust percentiles)
vmin_ref = np.percentile(REF_mm2Hz[m_plot, :], 5.0)
vmax_ref = np.percentile(REF_mm2Hz[m_plot, :], 95.0)
vmin_sys = np.percentile(SYS_factor[m_plot, :], 5.0)
vmax_sys = np.percentile(SYS_factor[m_plot, :], 95.0)

# Plot Figure 5
fig5, axes5 = plt.subplots(2, 1, figsize=(6.8, 6.0), constrained_layout=True)
ax51, ax52 = axes5

# --- Top: reference component (mm^2/Hz) ---
im51 = ax51.pcolormesh(
    t_plot, omega_plot, REF_mm2Hz[m_plot, :],
    shading='gouraud'
)
im51.set_clim(vmin_ref, vmax_ref)
ax51.set_title(r"LB component from reference: $\frac{1}{f_s U}\,\|\tilde{\mathbf P}^*_w(\omega)\|_2^2$  [mm$^2$/Hz]")
ax51.set_xlabel(r"Time [s]")
ax51.set_ylabel(r"Angular frequency $\omega$ [rad/s]")
ax51.set_ylim((omega_min_plot, omega_max_plot))

# --- Bottom: system component (dimensionless) ---
im52 = ax52.pcolormesh(
    t_plot, omega_plot, SYS_factor[m_plot, :],
    shading='gouraud'
)
im52.set_clim(vmin_sys, vmax_sys)
ax52.set_title(r"LB component from system: $\left(\frac{\sigma_{\min}(S_0)}{1+\|E\,S_0\|_2}\right)^{\!2}$  [–]")
ax52.set_xlabel(r"Time [s]")
ax52.set_ylabel(r"Angular frequency $\omega$ [rad/s]")
ax52.set_ylim((omega_min_plot, omega_max_plot))

# Colorbars
cbar51 = fig5.colorbar(im51, ax=ax51, location='right', shrink=0.96, pad=0.02)
cbar51.set_label(r"Reference component  [mm$^2$/Hz]")
cbar52 = fig5.colorbar(im52, ax=ax52, location='right', shrink=0.96, pad=0.02)
cbar52.set_label(r"System component  [–]")

# Save & show
# out_pdf5 = os.path.join(base_dir, "PSD_LB_components_fig5.pdf")
out_pdf5 = os.path.join(base_dir, "PSD_LB_components_fig5_real.pdf")
fig5.savefig(out_pdf5, bbox_inches='tight')
print(f"Saved Figure 5 to: {out_pdf5}")
plt.show()

print("")
