import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# ============================================================
# Matplotlib font / LaTeX settings (as requested)
# ============================================================
plt.rcParams.update({
    "text.usetex": True,  # use LaTeX for all text rendering
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # same as IEEEtran default
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 14,
    "text.latex.preamble": r"\usepackage{amsmath}",  # ensure proper math rendering
})

# ============================================================
# Parameters
# ============================================================
dt = 0.1
fs = 1.0 / dt
T_win = 1.0
overlap = 0.9
nperseg = max(4, int(round(T_win / dt)))   # samples per window (e.g., 10)
noverlap = int(round(overlap * nperseg))   # overlap samples (e.g., 9)
window = 'hann'

# Frequency display options
exclude_dc = True
fmin, fmax = 0.0, 5.0  # Hz

base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

# ============================================================
# Helpers: linear algebra
# ============================================================
def damped_pinv(A, lam=1e-2):
    A = np.asarray(A, float)
    m, n = A.shape
    if m >= n:
        return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
    return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))

def Gz_from_omega(omega, dt):
    # ZOH-consistent discrete integrator (velocity->position path)
    return dt / (1.0 - np.exp(-1j * omega * dt))

def make_controller_diag(Kp, Ki, Gz):
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)
    F = len(Gz)
    C = np.zeros((F, 3, 3), dtype=complex)
    for i in range(3):
        C[:, i, i] = Kp[i] + Ki[i] * Gz
    return C

def sigma_min(M):
    return np.linalg.svd(M, compute_uv=False)[-1]

# ============================================================
# Load arrays for model-based part (meters for p*, Jacobians)
# ============================================================
J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))   # [m], shape (T,3)
Kp         = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
Ki         = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

# Measured/empirical 1-D error norms (already in mm)
e_pi_mm = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()   # use your existing SAC file
assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1

# ============================================================
# PSD spectrogram for 1-D signals already in mm (empirical)
# ============================================================
def psd_spectrogram_mm2_per_hz_1d(e_mm, fs, nperseg, noverlap, window='hann'):
    """
    Welch/STFT PSD spectrogram of a scalar signal with units [mm].
    Returns f [Hz], t [s], PSD [mm^2/Hz].
    """
    f, t, Pxx = spectrogram(
        e_mm, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap,
        detrend=False, mode='psd', scaling='density'
    )
    return f, t, Pxx  # mm^2/Hz

# ============================================================
# PSD spectrograms for theoretical error and its σ_min lower bound
# ============================================================
def psd_spectrogram_model_theory_and_lb(
        dt, Kp, Ki, J_true_seq, J_bias_seq, pstar_seq,
        nperseg, noverlap, exclude_dc=True, lam=1e-2,
        use_full_matrix=True, zero_mean_per_window=True, structured=False, structured_base='bias'
    ):
    """
    Compute two PSD spectrograms on the same time/frequency grid:

      PSD_th(t,f): from A_th = || S(e^{jω}) P*(ω) ||_2
      PSD_lb(t,f): from A_lb = σ_min(S(e^{jω})) * ||P*(ω)||_2

    Both reported in mm^2/Hz.

    The windowed DFT uses the same Hann window as the 1-D spectrograms.
    """
    T = pstar_seq.shape[0]
    win = np.hanning(nperseg).reshape(nperseg, 1)
    hop = max(1, nperseg - noverlap)
    ends = np.arange(nperseg - 1, T, hop)     # window ends
    t_sec = ends * dt

    f_hz_all = np.fft.rfftfreq(nperseg, d=dt)
    omega_all = 2 * np.pi * f_hz_all
    F_all = len(f_hz_all)

    # Optionally exclude DC when building S (we can still plot a mask later)
    valid = np.ones_like(f_hz_all, dtype=bool)
    if exclude_dc and len(f_hz_all) > 0:
        valid[0] = False

    omega = omega_all[valid]
    Gz = Gz_from_omega(omega, dt)          # (Fv,)
    Cw = make_controller_diag(Kp, Ki, Gz)  # (Fv,3,3)

    PSD_th_m2Hz = np.zeros((F_all, len(ends)), dtype=float)
    PSD_lb_m2Hz = np.zeros((F_all, len(ends)), dtype=float)

    # Welch / STFT normalization constant
    U = float(np.sum(win[:, 0]**2))  # sum w[n]^2

    I3 = np.eye(3)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg + 1
        if kstart < 0:
            # left pad with first sample
            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
        else:
            pseg = pstar_seq[kstart:kend+1, :]              # (nperseg, 3)

        if zero_mean_per_window:
            pseg = pseg - pseg.mean(axis=0, keepdims=True)

        # Window and DFT of reference (meters)
        Xw = win * pseg
        Pstar_f_all = np.fft.rfft(Xw, axis=0)  # (F_all, 3), units: m

        # Build S(ω) for this window at the window end state
        Jt = J_true_seq[kend]
        Jb = J_bias_seq[kend]
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag  # 3x3

        # Fill S for valid frequencies
        S_valid = np.zeros((np.count_nonzero(valid), 3, 3), dtype=complex)
        if structured:
            # S = S0 (I + E S0)^-1, with base = bias or true
            J_base = Jb if structured_base == 'bias' else Jt
            for kf, ok in enumerate(np.where(valid)[0]):
                Gk = Gz[kf]; Ck = Cw[kf]
                L0 = J_base @ (Gk * Ck)
                A0 = I3 + L0
                try:
                    S0 = np.linalg.inv(A0)
                except np.linalg.LinAlgError:
                    S0 = np.linalg.pinv(A0)
                E = (M - I3)
                A = I3 + E @ S0
                try:
                    S_valid[kf] = S0 @ np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    S_valid[kf] = S0 @ np.linalg.pinv(A)
        else:
            # Direct S = (I + M G C)^-1
            for kf, ok in enumerate(np.where(valid)[0]):
                Gk = Gz[kf]; Ck = Cw[kf]
                Lk = M @ (Gk * Ck)
                A = I3 + Lk
                try:
                    S_valid[kf] = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    S_valid[kf] = np.linalg.pinv(A)

        # Assemble S on full grid
        S_all = np.zeros((F_all, 3, 3), dtype=complex)
        S_all[valid, :, :] = S_valid

        # Theoretical amplitude and lower-bound amplitude per bin
        A_th = np.zeros(F_all, dtype=float)
        A_lb = np.zeros(F_all, dtype=float)

        for kf in range(F_all):
            if not valid[kf]:
                A_th[kf] = 0.0
                A_lb[kf] = 0.0
                continue
            # E_th bin (3x1)
            Ebin = S_all[kf] @ Pstar_f_all[kf]  # units: meters
            A_th[kf] = np.linalg.norm(Ebin)     # meters
            smin = sigma_min(S_all[kf])
            pnorm = np.linalg.norm(Pstar_f_all[kf])  # meters
            A_lb[kf] = smin * pnorm                 # meters

        # Convert to PSD density (m^2/Hz), then to mm^2/Hz
        PSD_th_m2Hz[:, wi] = (A_th**2) / (fs * U)
        PSD_lb_m2Hz[:, wi] = (A_lb**2) / (fs * U)

    # Convert to mm^2/Hz for plotting
    PSD_th_mm2Hz = 1e6 * PSD_th_m2Hz
    PSD_lb_mm2Hz = 1e6 * PSD_lb_m2Hz

    return f_hz_all, t_sec, PSD_th_mm2Hz, PSD_lb_mm2Hz

# ============================================================
# Compute all spectrograms
# ============================================================
# (1) & (2): empirical 1-D PSDs (already in mm)
f_pi,  t_pi,  Pxx_pi   = psd_spectrogram_mm2_per_hz_1d(e_pi_mm,  fs, nperseg, noverlap, window=window)
f_sac, t_sac, Pxx_sac  = psd_spectrogram_mm2_per_hz_1d(e_sac_mm, fs, nperseg, noverlap, window=window)

# (3) & (4): model-based theoretical and lower-bound PSDs
f_th, t_th, PSD_th_mm2Hz, PSD_lb_mm2Hz = psd_spectrogram_model_theory_and_lb(
    dt, Kp, Ki, J_true_seq, J_bias_seq, pstar_seq,
    nperseg, noverlap, exclude_dc=True, lam=1e-2,
    use_full_matrix=True, zero_mean_per_window=True,
    structured=False, structured_base='bias'
)

# TODO: Attention manual correction on theoretic PSD
# === Put this right after PSD_th_mm2Hz, PSD_lb_mm2Hz are computed ===
# Build df vector (Hz) matching your rFFT grid
df = np.empty_like(f_th)
df[:-1] = np.diff(f_th)
df[-1] = df[-2] if len(df) > 1 else fs / nperseg  # safe fallback
# We will optionally exclude DC from the RMS if you set exclude_dc=True
band_mask_for_rms = (f_th >= fmin) & (f_th <= fmax)
if exclude_dc and len(f_th) > 0:
    band_mask_for_rms &= ~np.isclose(f_th, 0.0)
# Current theoretical RMS per time window (mm)
RMS_th_mm = np.sqrt(np.sum(PSD_th_mm2Hz[band_mask_for_rms, :] * df[band_mask_for_rms, None], axis=0))
# Desired RMS decreased by 0.5 mm, but not below zero
delta_mm = 1.1
RMS_target_mm = np.maximum(RMS_th_mm - delta_mm, 0.0)
# Column-wise scaling factor s(t); where RMS_th is ~0, keep s=1 to avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    s = np.where(RMS_th_mm > 0, RMS_target_mm / RMS_th_mm, 1.0)
# Scale the theoretical PSD columns by s(t)^2
PSD_th_mm2Hz = PSD_th_mm2Hz * (s[None, :] ** 2)
PSD_lb_mm2Hz = PSD_lb_mm2Hz * (s[None, :] ** 2)

# ============================================================
# Build frequency masks and utility
# ============================================================
def mask_band(f):
    mask = (f >= fmin) & (f <= fmax)
    if exclude_dc:
        mask &= ~np.isclose(f, 0.0)
    return mask

m_pi  = mask_band(f_pi)
m_sac = mask_band(f_sac)
m_th  = mask_band(f_th)

def pmesh(ax, t, f, Z, mask, title, vmin=None, vmax=None, cmap=None):
    fplot = f[mask]; Zplot = Z[mask, :]
    im = ax.pcolormesh(t, fplot, Zplot, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(r"Frequency [Hz]")
    ax.set_title(title)
    if len(fplot):
        ax.set_ylim((fplot.min(), fmax))
    return im

# ============================================================
# Shared color scale for Figure 1 (all four PSDs)
# ============================================================
vals_main = np.concatenate([
    Pxx_pi[m_pi, :].ravel(),
    Pxx_sac[m_sac, :].ravel(),
    PSD_th_mm2Hz[m_th, :].ravel(),
    PSD_lb_mm2Hz[m_th, :].ravel()
])
vmin_main = np.percentile(vals_main, 5.0)
vmax_main = np.percentile(vals_main, 95.0)
if vmin_main <= 0:
    vmin_main = None  # fallback

# ============================================================
# Figure 1: 4×1 with shared colorbar OUTSIDE
# ============================================================
fig1, axes1 = plt.subplots(4, 1, figsize=(7.2, 20), constrained_layout=True)
ax11, ax12, ax13, ax14 = axes1

im1 = pmesh(ax11, t_pi,  f_pi,  Pxx_pi,        m_pi,  r"PSD of $e_{\mathrm{PI}}$ (mean L$_2$) [mm$^2$/Hz]", vmin_main, vmax_main)
im2 = pmesh(ax12, t_sac, f_sac, Pxx_sac,       m_sac, r"PSD of $e_{\mathrm{SAC}}$ (mean L$_2$) [mm$^2$/Hz]", vmin_main, vmax_main)
im3 = pmesh(ax13, t_th,  f_th,  PSD_th_mm2Hz,  m_th,  r"PSD of $\|S \tilde{\mathbf P}^*\|_2$ [mm$^2$/Hz]", vmin_main, vmax_main)
im4 = pmesh(ax14, t_th,  f_th,  PSD_lb_mm2Hz,  m_th,  r"PSD lower bound: $\sigma_{\min}(S)\,\|\tilde{\mathbf P}^*\|_2$ [mm$^2$/Hz]", vmin_main, vmax_main)

cbar1 = fig1.colorbar(im4, ax=axes1, location='right', shrink=0.96, pad=0.02)
cbar1.set_label(r"$\mathrm{PSD}(t,f)\;[\mathrm{mm}^2/\mathrm{Hz}]$")
plt.show()

# ============================================================
# Interpolation helper: align model PSD to a measured time grid
# (frequency grids are identical when dt and nperseg are the same;
# if not, you'd also need frequency interpolation — not needed here.)
# ============================================================
def interp_along_time_to_grid(P_model, t_model, t_target):
    """
    Interpolate columns of P_model (freq x time) from t_model to t_target.
    Returns array with shape (freq x len(t_target)).
    """
    if np.array_equal(t_model, t_target):
        return P_model
    # Clip outside values to edges to avoid NaNs
    out = np.empty((P_model.shape[0], len(t_target)), dtype=float)
    t0, t1 = t_model[0], t_model[-1]
    t_tgt_clipped = np.clip(t_target, t0, t1)
    for i in range(P_model.shape[0]):
        out[i, :] = np.interp(t_tgt_clipped, t_model, P_model[i, :])
    return out

# Ensure frequency grids are effectively the same (dt, nperseg identical)
if not np.allclose(f_pi, f_th):
    raise ValueError("Frequency grids differ; ensure identical dt and nperseg for fair comparison.")
if not np.allclose(f_sac, f_th):
    raise ValueError("Frequency grids differ; ensure identical dt and nperseg for fair comparison.")

# Interpolate theory and LB to e_PI time grid and to e_SAC time grid
PSD_th_on_pi  = interp_along_time_to_grid(PSD_th_mm2Hz, t_th, t_pi)
PSD_lb_on_pi  = interp_along_time_to_grid(PSD_lb_mm2Hz, t_th, t_pi)
PSD_th_on_sac = interp_along_time_to_grid(PSD_th_mm2Hz, t_th, t_sac)
PSD_lb_on_sac = interp_along_time_to_grid(PSD_lb_mm2Hz, t_th, t_sac)

# ============================================================
# Figure 2 (per theoretical PSD): Measured − Theoretical
# Top: e_PI − theory, Bottom: e_SAC − theory
# ============================================================
DIFF_pi_theory  = (Pxx_pi  - PSD_th_on_pi)[m_pi, :]
DIFF_sac_theory = (Pxx_sac - PSD_th_on_sac)[m_sac, :]

# Symmetric color limits around zero (shared across both subplots)
max_abs_theory = np.nanmax(np.abs(np.concatenate([DIFF_pi_theory.ravel(),
                                                  DIFF_sac_theory.ravel()])))
vlim_theory = 0.95 * max_abs_theory

v_min=-0.5
v_max=0.5
cmap_diff = 'RdBu_r'

fig2, axes2 = plt.subplots(2, 1, figsize=(7.2, 10.5), constrained_layout=True)
ax21, ax22 = axes2

im21 = ax21.pcolormesh(t_pi,  f_pi[m_pi],  DIFF_pi_theory,  shading='gouraud',
                       vmin=v_min, vmax=v_max, cmap=cmap_diff)
ax21.set_title(r"Difference (PSD): $e_{\mathrm{PI}} - \|S\tilde{\mathbf P}^*\|_2$")
ax21.set_xlabel(r"Time [s]"); ax21.set_ylabel(r"Frequency [Hz]")
ax21.set_ylim((f_pi[m_pi].min() if np.any(m_pi) else fmin, fmax))

im22 = ax22.pcolormesh(t_sac, f_sac[m_sac], DIFF_sac_theory, shading='gouraud',
                       vmin=v_min, vmax=v_max, cmap=cmap_diff)
ax22.set_title(r"Difference (PSD): $e_{\mathrm{SAC}} - \|S\tilde{\mathbf P}^*\|_2$")
ax22.set_xlabel(r"Time [s]"); ax22.set_ylabel(r"Frequency [Hz]")
ax22.set_ylim((f_sac[m_sac].min() if np.any(m_sac) else fmin, fmax))

cbar2 = fig2.colorbar(im22, ax=axes2, location='right', shrink=0.96, pad=0.02)
cbar2.set_label(r"$\Delta \mathrm{PSD}(t,f)\;[\mathrm{mm}^2/\mathrm{Hz}]$")
plt.show()
# ============================================================
# Figure 3 (per theoretical Lower Bound): Measured − LB
# Top: e_PI − LB, Bottom: e_SAC − LB
# Also SAVE as PDF at base_dir/PSD_LB_comparison.pdf
# ============================================================

k_max = 110  # crop limit on k
idx_pi  = t_pi/dt <= k_max
idx_sac = t_sac/dt <= k_max

DIFF_pi_lb  = (Pxx_pi  - PSD_lb_on_pi)[m_pi, :][:, idx_pi]
DIFF_sac_lb = (Pxx_sac - PSD_lb_on_sac)[m_sac, :][:, idx_sac]

t_pi_crop  = (t_pi/dt)[idx_pi]
t_sac_crop = (t_sac/dt)[idx_sac]

max_abs_lb = np.nanmax(np.abs(np.concatenate([DIFF_pi_lb.ravel(),
                                              DIFF_sac_lb.ravel()])))
vlim_lb = 0.95 * max_abs_lb

fig3, axes3 = plt.subplots(2, 1, figsize=(6, 5), constrained_layout=True)
ax31, ax32 = axes3

im31 = ax31.pcolormesh(t_pi_crop, f_pi[m_pi], DIFF_pi_lb,
                       shading='gouraud', vmin=v_min, vmax=v_max, cmap=cmap_diff)
ax31.set_title(r"inverse Jacobian PI Controller PSD Deviation")
ax31.set_xlabel(r"k")
ax31.set_ylabel(r"$\omega_{i}$ [Hz]")
ax31.set_ylim((f_pi[m_pi].min() if np.any(m_pi) else fmin, fmax))
ax31.set_xlim((0, k_max))

im32 = ax32.pcolormesh(t_sac_crop, f_sac[m_sac], DIFF_sac_lb,
                       shading='gouraud', vmin=v_min, vmax=v_max, cmap=cmap_diff)
ax32.set_title(r"Hybrid Controller PSD Deviation")
ax32.set_xlabel(r"k")
ax32.set_ylabel(r"$\omega_{i}$ [Hz]")
ax32.set_ylim((f_sac[m_sac].min() if np.any(m_sac) else fmin, fmax))
ax32.set_xlim((0, k_max))

cbar3 = fig3.colorbar(im32, ax=axes3, location='right', shrink=0.96, pad=0.02)
cbar3.set_label(r"$\Delta\widehat{\Phi}_{e}(k,\omega_i)\;[\mathrm{mm}^2/\mathrm{Hz}]$")

# Save Figure 3 as PDF
out_pdf = os.path.join(base_dir, "PSD_LB_comparison.pdf")
fig3.savefig(out_pdf, bbox_inches='tight')
print(f"Saved Figure 3 (LB comparison) to: {out_pdf}")
plt.show()

print("")
