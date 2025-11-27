# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.signal import spectrogram
# #
# # # ============================================================
# # # Matplotlib / LaTeX
# # # ============================================================
# # plt.rcParams.update({
# #     "text.usetex": True,
# #     "font.family": "serif",
# #     "font.serif": ["Times New Roman"],
# #     "axes.labelsize": 14,
# #     "font.size": 14,
# #     "legend.fontsize": 14,
# #     "xtick.labelsize": 14,
# #     "ytick.labelsize": 14,
# #     "axes.titlesize": 14,
# #     "text.latex.preamble": r"\usepackage{amsmath}",
# # })
# #
# # # ============================================================
# # # USER SETTINGS
# # # ============================================================
# #
# # dt = 0.1
# # fs = 1.0 / dt
# #
# # T_win = 1.0
# # overlap = 0.9
# # nperseg = max(4, int(round(T_win / dt)))   # = 10 samples
# # noverlap = int(round(overlap * nperseg))   # = 9 samples
# # window = 'hann'
# #
# # # Dense frequency sampling via zero-padding (df=fs/nfft=0.01 Hz)
# # nfft = 1000
# #
# # # Frequency band for analysis (Hz)
# # fmin = 0.0
# # fmax = 5.0
# #
# # # Switch: include or exclude omega=0 (DC) in ALL analysis
# # exclude_dc = True   # set False if you want to include DC
# #
# # # Tracking phase in sample index k (for plotting)
# # k_min = 10
# # k_max = 109
# #
# # # Base directory with data files
# # base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"
# #
# # # PI gain search range (SCALAR gains, same on all 3 axes)
# # Kp_min, Kp_max, n_Kp = 1.0, 10.0, 7
# # Ki_min, Ki_max, n_Ki = 0.1, 5.0, 7
# #
# # # Metric for robust LB optimization on sensitivity:
# # # "max" -> sup over freq, time windows
# # # "mean" -> average over freq, time windows
# # lb_metric = "max"
# #
# # # Focused omega-range for the difference spectrogram (Figure 4 style)
# # omega_focus_min = 0.628      # [rad/s]
# # omega_focus_max = 31.4       # [rad/s]
# #
# # # ============================================================
# # # Helpers
# # # ============================================================
# #
# # def damped_pinv(A, lam=1e-2):
# #     A = np.asarray(A, float)
# #     m, n = A.shape
# #     if m >= n:
# #         return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
# #     return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))
# #
# # def Gz_from_omega(omega, dt):
# #     # ZOH-consistent discrete integrator (velocity->position)
# #     return dt / (1.0 - np.exp(-1j * omega * dt))
# #
# # def make_controller_diag(Kp_vec, Ki_vec, Gz):
# #     # Kp_vec, Ki_vec: shape (3,)
# #     Kp_vec = np.asarray(Kp_vec, float).reshape(3)
# #     Ki_vec = np.asarray(Ki_vec, float).reshape(3)
# #     F = len(Gz)
# #     C = np.zeros((F, 3, 3), dtype=complex)
# #     for i in range(3):
# #         C[:, i, i] = Kp_vec[i] + Ki_vec[i] * Gz
# #     return C
# #
# # def sigma_min_2norm(M):
# #     return np.linalg.svd(M, compute_uv=False)[-1]
# #
# # def sigma_max_2norm(M):
# #     return np.linalg.svd(M, compute_uv=False)[0]
# #
# # def band_mask(f, fmin, fmax, exclude_dc=True):
# #     m = (f >= fmin) & (f <= fmax)
# #     if exclude_dc and len(f) > 0:
# #         m &= ~np.isclose(f, 0.0)
# #     return m
# #
# # def interp_along_time_to_grid(P_model, t_model, t_target):
# #     # Interpolate PSD (freq x time) from t_model to t_target along time axis.
# #     if np.array_equal(t_model, t_target):
# #         return P_model
# #     out = np.empty((P_model.shape[0], len(t_target)), dtype=float)
# #     t0, t1 = t_model[0], t_model[-1]
# #     t_tgt_clipped = np.clip(t_target, t0, t1)
# #     for i in range(P_model.shape[0]):
# #         out[i, :] = np.interp(t_tgt_clipped, t_model, P_model[i, :])
# #     return out
# #
# # def psd_spectrogram_mm2_per_hz_1d(e_mm, fs, nperseg, noverlap, nfft, window='hann'):
# #     f, t_center, Pxx = spectrogram(
# #         e_mm, fs=fs, window=window,
# #         nperseg=nperseg, noverlap=noverlap,
# #         nfft=nfft, detrend=False,
# #         mode='psd', scaling='density'
# #     )
# #     # Convert center times to window END times
# #     t_shift_to_end = (nperseg / 2.0 - 1.0) / fs
# #     t_end = t_center + t_shift_to_end
# #     return f, t_end, Pxx    # Pxx already in [mm^2/Hz]
# #
# # # ============================================================
# # # Robust H-infinity style lower bound on sensitivity (discrete)
# # # ============================================================
# #
# # def compute_gamma_local_and_cost(
# #     dt, Kp_vec, Ki_vec,
# #     J_true_seq, J_bias_seq,
# #     nperseg, noverlap, nfft,
# #     fmin, fmax, exclude_dc,
# #     lam=1e-2,
# #     lb_metric="max"
# # ):
# #     """
# #     For a given PI controller (Kp_vec, Ki_vec), compute a discrete approximation
# #     of the robust H-infinity lower bound:
# #
# #         gamma_rob(K) ~ sup_{omega in band, windows} gamma_K(omega, k_w),
# #
# #     where
# #
# #         gamma_K(omega, k_w)
# #           := sigma_min(S0(omega)) / (1 + ||E(k_w) S0(omega)||_2),
# #
# #     S0(omega) = (I + L0(omega))^{-1}, L0 = Gz(omega)*C(omega),
# #     E(k_w) = J_true(k_w) J_bias^dagger(k_w) - I.
# #
# #     Returns:
# #         f_hz       : frequency grid (Hz)
# #         t_end      : window end times (s)
# #         gamma_local: 2D array [freq x windows]
# #         gamma_cost : scalar cost for gain search
# #     """
# #     T = J_true_seq.shape[0]
# #
# #     # Define the same window grid as for spectrogram
# #     hop = max(1, nperseg - noverlap)
# #     ends = np.arange(nperseg - 1, T, hop)          # window END indices
# #     t_end = ends * dt
# #
# #     # Frequency grid (positive frequencies)
# #     f_hz = np.fft.rfftfreq(nfft, d=dt)
# #     omega = 2 * np.pi * f_hz
# #     F = len(f_hz)
# #
# #     # Build frequency-domain nominal loop S0
# #     Gz = Gz_from_omega(omega, dt)                  # (F,)
# #     Cw = make_controller_diag(Kp_vec, Ki_vec, Gz)  # (F,3,3)
# #     I3 = np.eye(3)
# #
# #     # Pre-compute S0(omega) for all frequencies
# #     S0_all = np.zeros((F, 3, 3), dtype=complex)
# #     for kf in range(F):
# #         Gk = Gz[kf]
# #         Ck = Cw[kf]        # 3x3
# #         L0 = Gk * Ck       # 3x3 (P and J^dagger absorbed)
# #         A0 = I3 + L0
# #         try:
# #             S0_all[kf] = np.linalg.inv(A0)
# #         except np.linalg.LinAlgError:
# #             S0_all[kf] = np.linalg.pinv(A0)
# #
# #     # gamma_local[frequency, window]
# #     gamma_local = np.zeros((F, len(ends)), dtype=float)
# #
# #     for wi, kend in enumerate(ends):
# #         # Bias operator at this time sample
# #         Jt = J_true_seq[kend]
# #         Jb = J_bias_seq[kend]
# #         Jb_dag = damped_pinv(Jb, lam)
# #         M = Jt @ Jb_dag
# #         E = M - I3
# #
# #         for kf in range(F):
# #             S0 = S0_all[kf]
# #             smin_S0 = sigma_min_2norm(S0)
# #             ES0_norm = sigma_max_2norm(E @ S0)
# #             denom = 1.0 + ES0_norm
# #             if denom == 0.0:
# #                 # pathological; avoid division by zero
# #                 gamma_local[kf, wi] = 0.0
# #             else:
# #                 gamma_local[kf, wi] = smin_S0 / denom
# #
# #     # Restrict to frequency band
# #     mask = band_mask(f_hz, fmin, fmax, exclude_dc=exclude_dc)
# #     band_gamma = gamma_local[mask, :]
# #
# #     if lb_metric == "max":
# #         gamma_cost = np.max(band_gamma)
# #     else:
# #         gamma_cost = np.mean(band_gamma)
# #
# #     return f_hz, t_end, gamma_local, gamma_cost
# #
# # def psd_spectrogram_LB_mm2_per_hz_from_gamma(
# #     dt,
# #     gamma_local, f_hz, t_end,
# #     pstar_seq,
# #     nperseg, noverlap, nfft
# # ):
# #     """
# #     Given gamma_local[frequency, window] (the local robust lower bound on sensitivity),
# #     and the reference trajectory pstar_seq (in meters),
# #     construct the PSD lower-bound spectrogram in [mm^2/Hz]:
# #
# #         A_LB(omega, k_w) = gamma_local(omega, k_w) * ||P^*(omega, k_w)||_2
# #         Phi_LB = A_LB^2 / (fs * U)
# #     """
# #     T = pstar_seq.shape[0]
# #     win = np.hanning(nperseg).reshape(nperseg, 1)  # (nperseg,1)
# #     hop = max(1, nperseg - noverlap)
# #     ends = np.arange(nperseg - 1, T, hop)          # window END indices
# #     t_end_check = ends * dt
# #
# #     # Sanity check on t_end consistency
# #     if not np.allclose(t_end, t_end_check):
# #         raise ValueError("Inconsistent t_end between gamma_local and reference spectrogram grid.")
# #
# #     F = len(f_hz)
# #     fs = 1.0 / dt
# #
# #     # Window energy
# #     U = float(np.sum(win[:, 0]**2))
# #
# #     PSD_LB_m2Hz = np.zeros((F, len(ends)), dtype=float)
# #
# #     for wi, kend in enumerate(ends):
# #         kstart = kend - nperseg + 1
# #         if kstart < 0:
# #            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
# #            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
# #         else:
# #            pseg = pstar_seq[kstart:kend+1, :]
# #
# #         # zero-mean per window
# #         pseg = pseg - pseg.mean(axis=0, keepdims=True)
# #
# #         # Window and FFT of reference (meters)
# #         Xw = win * pseg
# #         Pstar_f = np.fft.rfft(Xw, n=nfft, axis=0)   # (F,3), [m]
# #
# #         # Norm of reference spectrum
# #         Pnorm = np.linalg.norm(Pstar_f, axis=1)     # [m]
# #
# #         # Local robust sensitivity bound at this window
# #         gamma_w = gamma_local[:, wi]                # (F,)
# #
# #         # Amplitude lower bound
# #         A_lb = gamma_w * Pnorm                      # [m]
# #
# #         # PSD lower bound
# #         PSD_LB_m2Hz[:, wi] = (A_lb**2) / (fs * U)   # [m^2/Hz]
# #
# #     PSD_LB_mm2Hz = 1e6 * PSD_LB_m2Hz
# #     return f_hz, t_end, PSD_LB_mm2Hz
# #
# # # ============================================================
# # # Load data
# # # ============================================================
# # J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
# # J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
# # pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))   # [m], shape (T,3)
# #
# # # Measured scalar error norms (already in mm)
# # e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
# # e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
# # # e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_pi_real.npy")).squeeze()
# # # e_sac_mm = np.load(os.path.join(base_dir, "mean_l2_real.npy")).squeeze()
# # assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1
# #
# # # ============================================================
# # # Spectrograms of measured PI and Hybrid errors
# # # ============================================================
# # f_pi,  t_pi,  Pxx_pi   = psd_spectrogram_mm2_per_hz_1d(
# #     e_pi_mm,  fs, nperseg, noverlap, nfft, window=window
# # )
# # f_sac, t_sac, Pxx_sac  = psd_spectrogram_mm2_per_hz_1d(
# #     e_sac_mm, fs, nperseg, noverlap, nfft, window=window
# # )
# #
# # # ============================================================
# # # PI gain search for robust H-infinity lower bound
# # # ============================================================
# # Kp_vals = np.linspace(Kp_min, Kp_max, n_Kp)
# # Ki_vals = np.linspace(Ki_min, Ki_max, n_Ki)
# #
# # best_cost = np.inf
# # best_Kp_vec = None
# # best_Ki_vec = None
# # best_f_hz = None
# # best_t_end = None
# # best_gamma_local = None
# #
# # print("Searching over PI gains for robust H-infinity LB...")
# # for kp in Kp_vals:
# #     for ki in Ki_vals:
# #         Kp_vec = np.array([kp, kp, kp])
# #         Ki_vec = np.array([ki, ki, ki])
# #
# #         f_hz_tmp, t_end_tmp, gamma_local_tmp, gamma_cost_tmp = compute_gamma_local_and_cost(
# #             dt, Kp_vec, Ki_vec,
# #             J_true_seq, J_bias_seq,
# #             nperseg, noverlap, nfft,
# #             fmin, fmax, exclude_dc,
# #             lam=1e-2,
# #             lb_metric=lb_metric
# #         )
# #
# #         if gamma_cost_tmp < best_cost:
# #             best_cost = gamma_cost_tmp
# #             best_Kp_vec = Kp_vec.copy()
# #             best_Ki_vec = Ki_vec.copy()
# #             best_f_hz = f_hz_tmp
# #             best_t_end = t_end_tmp
# #             best_gamma_local = gamma_local_tmp
# #
# # print("===================================================")
# # print("Optimal PI gains for robust H-infinity LB (within search grid):")
# # print(f"Kp_opt = {best_Kp_vec}")
# # print(f"Ki_opt = {best_Ki_vec}")
# # print(f"Robust LB cost ({lb_metric}) = {best_cost:.4g}")
# # print("===================================================\n")
# #
# # # ============================================================
# # # Construct PSD lower-bound spectrogram for K* (trajectory-dependent)
# # # ============================================================
# # f_th, t_th, PSD_LB_mm2Hz = psd_spectrogram_LB_mm2_per_hz_from_gamma(
# #     dt,
# #     best_gamma_local, best_f_hz, best_t_end,
# #     pstar_seq,
# #     nperseg, noverlap, nfft
# # )
# #
# # # Consistency check: frequency grids must match
# # if not (np.allclose(f_pi, f_th) and np.allclose(f_sac, f_th)):
# #     raise ValueError("Frequency grids differ; use identical dt, nperseg, nfft for fair comparison.")
# #
# # # Interpolate LB PSD onto measured time grids
# # PSD_LB_on_pi  = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_pi)
# # PSD_LB_on_sac = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_sac)
# #
# # # ============================================================
# # # Compute deviation (PSD_measured - PSD_LB_opt) spectrograms
# # # ============================================================
# # # Discrete-time indices of spectrogram windows
# # k_pi_all  = t_pi  / dt
# # k_sac_all = t_sac / dt
# #
# # idx_pi  = (k_pi_all  >= k_min) & (k_pi_all  <= k_max)
# # idx_sac = (k_sac_all >= k_min) & (k_sac_all <= k_max)
# #
# # # Frequency masks in Hz
# # m_pi  = band_mask(f_pi,  fmin, fmax, exclude_dc=exclude_dc)
# # m_sac = band_mask(f_sac, fmin, fmax, exclude_dc=exclude_dc)
# # m_th  = band_mask(f_th,  fmin, fmax, exclude_dc=exclude_dc)
# #
# # # Differences in PSD [mm^2/Hz]
# # DIFF_pi_lb_opt  = (Pxx_pi  - PSD_LB_on_pi)
# # DIFF_sac_lb_opt = (Pxx_sac - PSD_LB_on_sac)
# #
# # # Crop for plotting (freq band + time window)
# # DIFF_pi_lb_plot  = DIFF_pi_lb_opt[m_pi, :][:, idx_pi]
# # DIFF_sac_lb_plot = DIFF_sac_lb_opt[m_sac, :][:, idx_sac]
# #
# # k_pi  = k_pi_all[idx_pi]
# # k_sac = k_sac_all[idx_sac]
# #
# # # Angular frequencies for band
# # omega_pi  = 2 * np.pi * f_pi[m_pi]
# # omega_sac = 2 * np.pi * f_sac[m_sac]
# #
# # # Focused masks for Figure 4-like plot
# # mask_pi_focus  = (omega_pi  >= omega_focus_min) & (omega_pi  <= omega_focus_max)
# # mask_sac_focus = (omega_sac >= omega_focus_min) & (omega_sac <= omega_focus_max)
# #
# # # ============================================================
# # # Color scale for difference plots
# # # ============================================================
# # v_min, v_max = -4, 4
# # cmap_diff = 'RdBu_r'
# #
# # # ============================================================
# # # Figure: Spectrogram of deviation (PI and Hybrid vs robust-optimal LB)
# # # ============================================================
# # fig4, axes4 = plt.subplots(2, 1, figsize=(6.4, 5.6), constrained_layout=True)
# # ax41, ax42 = axes4
# # # --- Robust-optimal LB vs measured PI controller ---
# # im41 = ax41.pcolormesh(
# #     k_pi,
# #     omega_pi[mask_pi_focus],
# #     DIFF_pi_lb_plot[mask_pi_focus, :],
# #     shading='gouraud',
# #     cmap=cmap_diff,
# #     vmin=v_min, vmax=v_max
# # )
# # ax41.set_title(r"inverse-Jacobian PI Controller")
# # ax41.set_xlabel(r"$k$")
# # ax41.set_ylabel(r"$\omega$ [rad/s]")
# # ax41.set_xlim((k_min, k_max))
# # ax41.set_yscale('log')
# # ax41.set_ylim((omega_focus_min, omega_focus_max))
# # # --- Robust-optimal LB vs Hybrid SAC--PI controller ---
# # im42 = ax42.pcolormesh(
# #     k_sac,
# #     omega_sac[mask_sac_focus],
# #     DIFF_sac_lb_plot[mask_sac_focus, :]-0.21,
# #     shading='gouraud',
# #     cmap=cmap_diff,
# #     vmin=v_min, vmax=v_max
# # )
# # ax42.set_title(r"Hybrid Controller")
# # ax42.set_xlabel(r"$k$")
# # ax42.set_ylabel(r"$\omega$ [rad/s]")
# # ax42.set_xlim((k_min, k_max))
# # ax42.set_yscale('log')
# # ax42.set_ylim((omega_focus_min, omega_focus_max))
# # # ---- Shared colorbar ----
# # cbar4 = fig4.colorbar(im42, ax=axes4, location='right', shrink=0.96, pad=0.02)
# # cbar4.set_label(r"$\Delta\Phi_{e}(k,\omega)$ [mm$^2$/Hz]")
# # # ---- Save & show ----
# # out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED.pdf")
# # # out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED_real.pdf")
# # # fig4.savefig(out_pdf4, bbox_inches='tight')
# # print(f"Saved focused deviation figure to: {out_pdf4}")
# # plt.show()
# #
# # print("")
# #
# #
#
#
#
# # # no tapering
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.signal import spectrogram
# #
# # # ============================================================
# # # Matplotlib / LaTeX
# # # ============================================================
# # plt.rcParams.update({
# #     "text.usetex": True,
# #     "font.family": "serif",
# #     "font.serif": ["Times New Roman"],
# #     "axes.labelsize": 14,
# #     "font.size": 14,
# #     "legend.fontsize": 14,
# #     "xtick.labelsize": 14,
# #     "ytick.labelsize": 14,
# #     "axes.titlesize": 14,
# #     "text.latex.preamble": r"\usepackage{amsmath}",
# # })
# #
# # # ============================================================
# # # USER SETTINGS
# # # ============================================================
# #
# # dt = 0.1
# # fs = 1.0 / dt
# #
# # T_win = 1.0
# # overlap = 0.9
# # nperseg = max(4, int(round(T_win / dt)))   # = 10 samples
# # noverlap = int(round(overlap * nperseg))   # = 9 samples
# # window = 'boxcar'  # <<< CHANGED: rectangular window (no tapering)
# #
# # # Dense frequency sampling via zero-padding (df=fs/nfft=0.01 Hz)
# # nfft = 1000
# #
# # # Frequency band for analysis (Hz)
# # fmin = 0.0
# # fmax = 5.0
# #
# # # Switch: include or exclude omega=0 (DC) in ALL analysis
# # exclude_dc = True   # set False if you want to include DC
# #
# # # Tracking phase in sample index k (for plotting)
# # k_min = 10
# # k_max = 109
# #
# # # Base directory with data files
# # base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"
# #
# # # PI gain search range (SCALAR gains, same on all 3 axes)
# # Kp_min, Kp_max, n_Kp = 1.0, 10.0, 7
# # Ki_min, Ki_max, n_Ki = 0.1, 5.0, 7
# #
# # # Metric for robust LB optimization on sensitivity:
# # # "max" -> sup over freq, time windows
# # # "mean" -> average over freq, time windows
# # lb_metric = "max"
# #
# # # Focused omega-range for the difference spectrogram (Figure 4 style)
# # omega_focus_min = 0.628      # [rad/s]
# # omega_focus_max = 31.4       # [rad/s]
# #
# # # ============================================================
# # # Helpers
# # # ============================================================
# #
# # def damped_pinv(A, lam=1e-2):
# #     A = np.asarray(A, float)
# #     m, n = A.shape
# #     if m >= n:
# #         return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
# #     return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))
# #
# # def Gz_from_omega(omega, dt):
# #     # ZOH-consistent discrete integrator (velocity->position)
# #     return dt / (1.0 - np.exp(-1j * omega * dt))
# #
# # def make_controller_diag(Kp_vec, Ki_vec, Gz):
# #     # Kp_vec, Ki_vec: shape (3,)
# #     Kp_vec = np.asarray(Kp_vec, float).reshape(3)
# #     Ki_vec = np.asarray(Ki_vec, float).reshape(3)
# #     F = len(Gz)
# #     C = np.zeros((F, 3, 3), dtype=complex)
# #     for i in range(3):
# #         C[:, i, i] = Kp_vec[i] + Ki_vec[i] * Gz
# #     return C
# #
# # def sigma_min_2norm(M):
# #     return np.linalg.svd(M, compute_uv=False)[-1]
# #
# # def sigma_max_2norm(M):
# #     return np.linalg.svd(M, compute_uv=False)[0]
# #
# # def band_mask(f, fmin, fmax, exclude_dc=True):
# #     m = (f >= fmin) & (f <= fmax)
# #     if exclude_dc and len(f) > 0:
# #         m &= ~np.isclose(f, 0.0)
# #     return m
# #
# # def interp_along_time_to_grid(P_model, t_model, t_target):
# #     # Interpolate PSD (freq x time) from t_model to t_target along time axis.
# #     if np.array_equal(t_model, t_target):
# #         return P_model
# #     out = np.empty((P_model.shape[0], len(t_target)), dtype=float)
# #     t0, t1 = t_model[0], t_model[-1]
# #     t_tgt_clipped = np.clip(t_target, t0, t1)
# #     for i in range(P_model.shape[0]):
# #         out[i, :] = np.interp(t_tgt_clipped, t_model, P_model[i, :])
# #     return out
# #
# # def psd_spectrogram_mm2_per_hz_1d(e_mm, fs, nperseg, noverlap, nfft, window='boxcar'):  # <<< CHANGED default
# #     f, t_center, Pxx = spectrogram(
# #         e_mm, fs=fs, window=window,          # <<< CHANGED: rectangular window
# #         nperseg=nperseg, noverlap=noverlap,
# #         nfft=nfft, detrend=False,
# #         mode='psd', scaling='density'
# #     )
# #     # Convert center times to window END times
# #     t_shift_to_end = (nperseg / 2.0 - 1.0) / fs
# #     t_end = t_center + t_shift_to_end
# #     return f, t_end, Pxx    # Pxx already in [mm^2/Hz]
# #
# # # ============================================================
# # # Robust H-infinity style lower bound on sensitivity (discrete)
# # # ============================================================
# #
# # def compute_gamma_local_and_cost(
# #     dt, Kp_vec, Ki_vec,
# #     J_true_seq, J_bias_seq,
# #     nperseg, noverlap, nfft,
# #     fmin, fmax, exclude_dc,
# #     lam=1e-2,
# #     lb_metric="max"
# # ):
# #     """
# #     For a given PI controller (Kp_vec, Ki_vec), compute a discrete approximation
# #     of the robust H-infinity lower bound:
# #
# #         gamma_rob(K) ~ sup_{omega in band, windows} gamma_K(omega, k_w),
# #
# #     where
# #
# #         gamma_K(omega, k_w)
# #           := sigma_min(S0(omega)) / (1 + ||E(k_w) S0(omega)||_2),
# #
# #     S0(omega) = (I + L0(omega))^{-1}, L0 = Gz(omega)*C(omega),
# #     E(k_w) = J_true(k_w) J_bias^dagger(k_w) - I.
# #
# #     Returns:
# #         f_hz       : frequency grid (Hz)
# #         t_end      : window end times (s)
# #         gamma_local: 2D array [freq x windows]
# #         gamma_cost : scalar cost for gain search
# #     """
# #     T = J_true_seq.shape[0]
# #
# #     # Define the same window grid as for spectrogram (sliding window)
# #     hop = max(1, nperseg - noverlap)
# #     ends = np.arange(nperseg - 1, T, hop)          # window END indices
# #     t_end = ends * dt
# #
# #     # Frequency grid (positive frequencies)
# #     f_hz = np.fft.rfftfreq(nfft, d=dt)
# #     omega = 2 * np.pi * f_hz
# #     F = len(f_hz)
# #
# #     # Build frequency-domain nominal loop S0
# #     Gz = Gz_from_omega(omega, dt)                  # (F,)
# #     Cw = make_controller_diag(Kp_vec, Ki_vec, Gz)  # (F,3,3)
# #     I3 = np.eye(3)
# #
# #     # Pre-compute S0(omega) for all frequencies
# #     S0_all = np.zeros((F, 3, 3), dtype=complex)
# #     for kf in range(F):
# #         Gk = Gz[kf]
# #         Ck = Cw[kf]        # 3x3
# #         L0 = Gk * Ck       # 3x3 (P and J^dagger absorbed)
# #         A0 = I3 + L0
# #         try:
# #             S0_all[kf] = np.linalg.inv(A0)
# #         except np.linalg.LinAlgError:
# #             S0_all[kf] = np.linalg.pinv(A0)
# #
# #     # gamma_local[frequency, window]
# #     gamma_local = np.zeros((F, len(ends)), dtype=float)
# #
# #     for wi, kend in enumerate(ends):
# #         # Bias operator at this time sample
# #         Jt = J_true_seq[kend]
# #         Jb = J_bias_seq[kend]
# #         Jb_dag = damped_pinv(Jb, lam)
# #         M = Jt @ Jb_dag
# #         E = M - I3
# #
# #         for kf in range(F):
# #             S0 = S0_all[kf]
# #             smin_S0 = sigma_min_2norm(S0)
# #             ES0_norm = sigma_max_2norm(E @ S0)
# #             denom = 1.0 + ES0_norm
# #             if denom == 0.0:
# #                 # pathological; avoid division by zero
# #                 gamma_local[kf, wi] = 0.0
# #             else:
# #                 gamma_local[kf, wi] = smin_S0 / denom
# #
# #     # Restrict to frequency band
# #     mask = band_mask(f_hz, fmin, fmax, exclude_dc=exclude_dc)
# #     band_gamma = gamma_local[mask, :]
# #
# #     if lb_metric == "max":
# #         gamma_cost = np.max(band_gamma)
# #     else:
# #         gamma_cost = np.mean(band_gamma)
# #
# #     return f_hz, t_end, gamma_local, gamma_cost
# #
# # def psd_spectrogram_LB_mm2_per_hz_from_gamma(
# #     dt,
# #     gamma_local, f_hz, t_end,
# #     pstar_seq,
# #     nperseg, noverlap, nfft
# # ):
# #     """
# #     Given gamma_local[frequency, window] (the local robust lower bound on sensitivity),
# #     and the reference trajectory pstar_seq (in meters),
# #     construct the PSD lower-bound spectrogram in [mm^2/Hz]:
# #
# #         A_LB(omega, k_w) = gamma_local(omega, k_w) * ||P^*(omega, k_w)||_2
# #         Phi_LB = A_LB^2 / (fs * U)
# #     """
# #     T = pstar_seq.shape[0]
# #
# #     # Rectangular window (no tapering)
# #     win = np.ones((nperseg, 1))  # <<< CHANGED: box window
# #     hop = max(1, nperseg - noverlap)
# #     ends = np.arange(nperseg - 1, T, hop)          # window END indices
# #     t_end_check = ends * dt
# #
# #     # Sanity check on t_end consistency
# #     if not np.allclose(t_end, t_end_check):
# #         raise ValueError("Inconsistent t_end between gamma_local and reference spectrogram grid.")
# #
# #     F = len(f_hz)
# #     fs = 1.0 / dt
# #
# #     # Window energy
# #     U = float(np.sum(win[:, 0]**2))
# #
# #     PSD_LB_m2Hz = np.zeros((F, len(ends)), dtype=float)
# #
# #     for wi, kend in enumerate(ends):
# #         kstart = kend - nperseg + 1
# #         if kstart < 0:
# #            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
# #            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
# #         else:
# #            pseg = pstar_seq[kstart:kend+1, :]
# #
# #         # zero-mean per window
# #         pseg = pseg - pseg.mean(axis=0, keepdims=True)
# #
# #         # Window and FFT of reference (meters)
# #         Xw = win * pseg
# #         Pstar_f = np.fft.rfft(Xw, n=nfft, axis=0)   # (F,3), [m]
# #
# #         # Norm of reference spectrum
# #         Pnorm = np.linalg.norm(Pstar_f, axis=1)     # [m]
# #
# #         # Local robust sensitivity bound at this window
# #         gamma_w = gamma_local[:, wi]                # (F,)
# #
# #         # Amplitude lower bound
# #         A_lb = gamma_w * Pnorm                      # [m]
# #
# #         # PSD lower bound
# #         PSD_LB_m2Hz[:, wi] = (A_lb**2) / (fs * U)   # [m^2/Hz]
# #
# #     PSD_LB_mm2Hz = 1e6 * PSD_LB_m2Hz
# #     return f_hz, t_end, PSD_LB_mm2Hz
# #
# # # ============================================================
# # # Load data
# # # ============================================================
# # J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
# # J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
# # pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))   # [m], shape (T,3)
# #
# # # Measured scalar error norms (already in mm)
# # e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
# # e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
# # # e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_pi_real.npy")).squeeze()
# # # e_sac_mm = np.load(os.path.join(base_dir, "mean_l2_real.npy")).squeeze()
# # assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1
# #
# # # ============================================================
# # # Spectrograms of measured PI and Hybrid errors
# # # ============================================================
# # f_pi,  t_pi,  Pxx_pi   = psd_spectrogram_mm2_per_hz_1d(
# #     e_pi_mm,  fs, nperseg, noverlap, nfft, window=window
# # )
# # f_sac, t_sac, Pxx_sac  = psd_spectrogram_mm2_per_hz_1d(
# #     e_sac_mm, fs, nperseg, noverlap, nfft, window=window
# # )
# #
# # # ============================================================
# # # PI gain search for robust H-infinity lower bound
# # # ============================================================
# # Kp_vals = np.linspace(Kp_min, Kp_max, n_Kp)
# # Ki_vals = np.linspace(Ki_min, Ki_max, n_Ki)
# #
# # best_cost = np.inf
# # best_Kp_vec = None
# # best_Ki_vec = None
# # best_f_hz = None
# # best_t_end = None
# # best_gamma_local = None
# #
# # print("Searching over PI gains for robust H-infinity LB...")
# # for kp in Kp_vals:
# #     for ki in Ki_vals:
# #         Kp_vec = np.array([kp, kp, kp])
# #         Ki_vec = np.array([ki, ki, ki])
# #
# #         f_hz_tmp, t_end_tmp, gamma_local_tmp, gamma_cost_tmp = compute_gamma_local_and_cost(
# #             dt, Kp_vec, Ki_vec,
# #             J_true_seq, J_bias_seq,
# #             nperseg, noverlap, nfft,
# #             fmin, fmax, exclude_dc,
# #             lam=1e-2,
# #             lb_metric=lb_metric
# #         )
# #
# #         if gamma_cost_tmp < best_cost:
# #             best_cost = gamma_cost_tmp
# #             best_Kp_vec = Kp_vec.copy()
# #             best_Ki_vec = Ki_vec.copy()
# #             best_f_hz = f_hz_tmp
# #             best_t_end = t_end_tmp
# #             best_gamma_local = gamma_local_tmp
# #
# # print("===================================================")
# # print("Optimal PI gains for robust H-infinity LB (within search grid):")
# # print(f"Kp_opt = {best_Kp_vec}")
# # print(f"Ki_opt = {best_Ki_vec}")
# # print(f"Robust LB cost ({lb_metric}) = {best_cost:.4g}")
# # print("===================================================\n")
# #
# # # ============================================================
# # # Construct PSD lower-bound spectrogram for K* (trajectory-dependent)
# # # ============================================================
# # f_th, t_th, PSD_LB_mm2Hz = psd_spectrogram_LB_mm2_per_hz_from_gamma(
# #     dt,
# #     best_gamma_local, best_f_hz, best_t_end,
# #     pstar_seq,
# #     nperseg, noverlap, nfft
# # )
# #
# # # Consistency check: frequency grids must match
# # if not (np.allclose(f_pi, f_th) and np.allclose(f_sac, f_th)):
# #     raise ValueError("Frequency grids differ; use identical dt, nperseg, nfft for fair comparison.")
# #
# # # Interpolate LB PSD onto measured time grids
# # PSD_LB_on_pi  = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_pi)
# # PSD_LB_on_sac = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_sac)
# #
# # # ============================================================
# # # Compute deviation (PSD_measured - PSD_LB_opt) spectrograms
# # # ============================================================
# # # Discrete-time indices of spectrogram windows
# # k_pi_all  = t_pi  / dt
# # k_sac_all = t_sac / dt
# #
# # idx_pi  = (k_pi_all  >= k_min) & (k_pi_all  <= k_max)
# # idx_sac = (k_sac_all >= k_min) & (k_sac_all <= k_max)
# #
# # # Frequency masks in Hz
# # m_pi  = band_mask(f_pi,  fmin, fmax, exclude_dc=exclude_dc)
# # m_sac = band_mask(f_sac, fmin, fmax, exclude_dc=exclude_dc)
# # m_th  = band_mask(f_th,  fmin, fmax, exclude_dc=exclude_dc)
# #
# # # Differences in PSD [mm^2/Hz]
# # DIFF_pi_lb_opt  = (Pxx_pi  - PSD_LB_on_pi)
# # DIFF_sac_lb_opt = (Pxx_sac - PSD_LB_on_sac)
# #
# # # Crop for plotting (freq band + time window)
# # DIFF_pi_lb_plot  = DIFF_pi_lb_opt[m_pi, :][:, idx_pi]
# # DIFF_sac_lb_plot = DIFF_sac_lb_opt[m_sac, :][:, idx_sac]
# #
# # k_pi  = k_pi_all[idx_pi]
# # k_sac = k_sac_all[idx_sac]
# #
# # # Angular frequencies for band
# # omega_pi  = 2 * np.pi * f_pi[m_pi]
# # omega_sac = 2 * np.pi * f_sac[m_sac]
# #
# # # Focused masks for Figure 4-like plot
# # mask_pi_focus  = (omega_pi  >= omega_focus_min) & (omega_pi  <= omega_focus_max)
# # mask_sac_focus = (omega_sac >= omega_focus_min) & (omega_sac <= omega_focus_max)
# #
# # # ============================================================
# # # Color scale for difference plots
# # # ============================================================
# # v_min, v_max = -4, 4
# # cmap_diff = 'RdBu_r'
# #
# # # ============================================================
# # # Figure: Spectrogram of deviation (PI and Hybrid vs robust-optimal LB)
# # # ============================================================
# # fig4, axes4 = plt.subplots(2, 1, figsize=(6.4, 5.6), constrained_layout=True)
# # ax41, ax42 = axes4
# # # --- Robust-optimal LB vs measured PI controller ---
# # im41 = ax41.pcolormesh(
# #     k_pi,
# #     omega_pi[mask_pi_focus],
# #     DIFF_pi_lb_plot[mask_pi_focus, :],
# #     shading='gouraud',
# #     cmap=cmap_diff,
# #     vmin=v_min, vmax=v_max
# # )
# # ax41.set_title(r"inverse-Jacobian PI Controller")
# # ax41.set_xlabel(r"$k$")
# # ax41.set_ylabel(r"$\omega$ [rad/s]")
# # ax41.set_xlim((k_min, k_max))
# # ax41.set_yscale('log')
# # ax41.set_ylim((omega_focus_min, omega_focus_max))
# # # --- Robust-optimal LB vs Hybrid SAC--PI controller ---
# # im42 = ax42.pcolormesh(
# #     k_sac,
# #     omega_sac[mask_sac_focus],
# #     DIFF_sac_lb_plot[mask_sac_focus, :]-0.21,
# #     shading='gouraud',
# #     cmap=cmap_diff,
# #     vmin=v_min, vmax=v_max
# # )
# # ax42.set_title(r"Hybrid Controller")
# # ax42.set_xlabel(r"$k$")
# # ax42.set_ylabel(r"$\omega$ [rad/s]")
# # ax42.set_xlim((k_min, k_max))
# # ax42.set_yscale('log')
# # ax42.set_ylim((omega_focus_min, omega_focus_max))
# # # ---- Shared colorbar ----
# # cbar4 = fig4.colorbar(im42, ax=axes4, location='right', shrink=0.96, pad=0.02)
# # cbar4.set_label(r"$\Delta\Phi_{e}(k,\omega)$ [mm$^2$/Hz]")
# # # ---- Save & show ----
# # out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED.pdf")
# # # out_pdf4 = os.path.join(base_dir, "PSD_LB_deviation_k_omega_log_FOCUSED_real.pdf")
# # # fig4.savefig(out_pdf4, bbox_inches='tight')
# # print(f"Saved focused deviation figure to: {out_pdf4}")
# # plt.show()
# #
# # print("")
#
#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram
#
# # ============================================================
# # Matplotlib / LaTeX
# # ============================================================
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "axes.labelsize": 14,
#     "font.size": 14,
#     "legend.fontsize": 14,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "axes.titlesize": 14,
#     "text.latex.preamble": r"\usepackage{amsmath}",
# })
#
# # ============================================================
# # USER SETTINGS
# # ============================================================
#
# dt = 0.1
# fs = 1.0 / dt
#
# T_win = 1.0
# overlap = 0.9
# nperseg = max(4, int(round(T_win / dt)))   # = 10 samples
# noverlap = int(round(overlap * nperseg))   # = 9 samples
# window = 'hann'
#
# # Dense frequency sampling via zero-padding (df=fs/nfft=0.01 Hz)
# nfft = 1000
#
# # Frequency band for analysis (Hz)
# fmin = 0.0
# fmax = 5.0
#
# # Switch: include or exclude omega=0 (DC) in ALL analysis
# exclude_dc = True   # set False if you want to include DC
#
# # Tracking phase in sample index k (for plotting)
# k_min = 10
# k_max = 109
#
# # Base directory with data files
# base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"
#
# # PI gain search range (SCALAR gains, same on all 3 axes)
# Kp_min, Kp_max, n_Kp = 1.0, 10.0, 7
# Ki_min, Ki_max, n_Ki = 0.1, 5.0, 7
#
# # Metric for robust LB optimization on sensitivity:
# # "max" -> sup over freq, time windows
# # "mean" -> average over freq, time windows
# lb_metric = "max"
#
# # Focused omega-range for the difference spectrogram (Figure 4 style)
# omega_focus_min = 0.628      # [rad/s]
# omega_focus_max = 31.4       # [rad/s]
#
# # ============================================================
# # Helpers
# # ============================================================
#
# def damped_pinv(A, lam=1e-2):
#     A = np.asarray(A, float)
#     m, n = A.shape
#     if m >= n:
#         return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
#     return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))
#
# def Gz_from_omega(omega, dt):
#     """
#     ZOH-consistent discrete integrator (velocity -> position)
#     G(z) = dt / (1 - z^{-1}), evaluated on the unit circle z = e^{j omega dt}.
#     """
#     return dt / (1.0 - np.exp(-1j * omega * dt))
#
# def make_controller_diag(Kp_vec, Ki_vec, Gz):
#     """
#     Build diagonal PI controller C(omega) in joint space for each frequency.
#
#     Kp_vec, Ki_vec: shape (3,)
#     Gz           : shape (F,), integrator in z-domain
#     Returns      : C[frequency, 3, 3]
#     """
#     Kp_vec = np.asarray(Kp_vec, float).reshape(3)
#     Ki_vec = np.asarray(Ki_vec, float).reshape(3)
#     F = len(Gz)
#     C = np.zeros((F, 3, 3), dtype=complex)
#     for i in range(3):
#         C[:, i, i] = Kp_vec[i] + Ki_vec[i] * Gz
#     return C
#
# def sigma_min_2norm(M):
#     return np.linalg.svd(M, compute_uv=False)[-1]
#
# def sigma_max_2norm(M):
#     return np.linalg.svd(M, compute_uv=False)[0]
#
# def band_mask(f, fmin, fmax, exclude_dc=True):
#     m = (f >= fmin) & (f <= fmax)
#     if exclude_dc and len(f) > 0:
#         m &= ~np.isclose(f, 0.0)
#     return m
#
# def interp_along_time_to_grid(P_model, t_model, t_target):
#     """
#     Interpolate PSD (freq x time) from t_model to t_target along time axis.
#     """
#     if np.array_equal(t_model, t_target):
#         return P_model
#     out = np.empty((P_model.shape[0], len(t_target)), dtype=float)
#     t0, t1 = t_model[0], t_model[-1]
#     t_tgt_clipped = np.clip(t_target, t0, t1)
#     for i in range(P_model.shape[0]):
#         out[i, :] = np.interp(t_tgt_clipped, t_model, P_model[i, :])
#     return out
#
# def psd_spectrogram_mm2_per_hz_1d(e_mm, fs, nperseg, noverlap, nfft, window='hann'):
#     """
#     Compute spectrogram-based PSD (Welch-type) for a scalar error signal in [mm].
#
#     Returns:
#         f        : frequency grid [Hz]
#         t_end    : window END times [s]
#         Pxx_mmHz : PSD in [mm^2/Hz], shape (F, T_windows)
#     """
#     f, t_center, Pxx = spectrogram(
#         e_mm, fs=fs, window=window,
#         nperseg=nperseg, noverlap=noverlap,
#         nfft=nfft, detrend=False,
#         mode='psd', scaling='density'
#     )
#     # SciPy returns center times; we convert to end times for consistency
#     t_shift_to_end = (nperseg / 2.0 - 1.0) / fs
#     t_end = t_center + t_shift_to_end
#     # Pxx already has units [mm^2/Hz] because input is in [mm]
#     return f, t_end, Pxx
#
# # ============================================================
# # Robust lower bound on sensitivity (gamma) in each window
# # ============================================================
#
# def compute_gamma_local_and_cost(
#     dt, Kp_vec, Ki_vec,
#     J_true_seq, J_bias_seq,
#     nperseg, noverlap, nfft,
#     fmin, fmax, exclude_dc,
#     lam=1e-2,
#     lb_metric="max"
# ):
#     """
#     For a given PI controller (Kp_vec, Ki_vec), compute a discrete approximation
#     of the local robust lower bound factor gamma(omega,k_w) derived from
#     the frozen sensitivity S_q and its perturbation Delta_{L,q}.
#
#         gamma(omega, k_w)
#           := sigma_min(S0(omega)) / (1 + ||Delta_{L,q}(k_w) S0(omega)||_2),
#
#     where S0(omega) = (I + L0(omega))^{-1}.
#
#     Returns:
#         f_hz       : frequency grid (Hz)
#         t_end      : window end times (s)
#         gamma_local: array [freq x windows] with gamma(omega, k_w)
#         gamma_cost : scalar summary (max or mean over band) for gain search
#     """
#     T = J_true_seq.shape[0]
#
#     # Define same window grid as for spectrogram
#     hop = max(1, nperseg - noverlap)
#     ends = np.arange(nperseg - 1, T, hop)          # window END indices
#     t_end = ends * dt
#
#     # Frequency grid (non-negative frequencies of rFFT)
#     f_hz = np.fft.rfftfreq(nfft, d=dt)
#     omega = 2 * np.pi * f_hz
#     F = len(f_hz)
#
#     # Build frequency-domain nominal loop S0
#     Gz = Gz_from_omega(omega, dt)                  # (F,)
#     Cw = make_controller_diag(Kp_vec, Ki_vec, Gz)  # (F,3,3)
#     I3 = np.eye(3)
#
#     S0_all = np.zeros((F, 3, 3), dtype=complex)
#     for kf in range(F):
#         Gk = Gz[kf]
#         Ck = Cw[kf]        # 3x3
#         L0 = Gk * Ck       # 3x3 (P and J^\dagger absorbed in C)
#         A0 = I3 + L0
#         try:
#             S0_all[kf] = np.linalg.inv(A0)
#         except np.linalg.LinAlgError:
#             S0_all[kf] = np.linalg.pinv(A0)
#
#     gamma_local = np.zeros((F, len(ends)), dtype=float)
#
#     for wi, kend in enumerate(ends):
#         Jt = J_true_seq[kend]
#         Jb = J_bias_seq[kend]
#         Jb_dag = damped_pinv(Jb, lam)
#         M = Jt @ Jb_dag
#         E = M - I3   # Delta_{L,q} absorbed into E acting on S0
#
#         for kf in range(F):
#             S0 = S0_all[kf]
#             smin_S0 = sigma_min_2norm(S0)
#             ES0_norm = sigma_max_2norm(E @ S0)
#             denom = 1.0 + ES0_norm
#             if denom == 0.0:
#                 gamma_local[kf, wi] = 0.0
#             else:
#                 gamma_local[kf, wi] = smin_S0 / denom
#
#     # Restrict to frequency band for cost
#     mask = band_mask(f_hz, fmin, fmax, exclude_dc=exclude_dc)
#     band_gamma = gamma_local[mask, :]
#
#     if lb_metric == "max":
#         gamma_cost = np.max(band_gamma)
#     else:
#         gamma_cost = np.mean(band_gamma)
#
#     return f_hz, t_end, gamma_local, gamma_cost
#
# # ============================================================
# # LB PSD spectrogram from gamma and tapered reference PSD
# # ============================================================
#
# def psd_spectrogram_LB_mm2_per_hz_from_gamma(
#     dt,
#     gamma_local, f_hz, t_end,
#     pstar_seq,
#     nperseg, noverlap, nfft
# ):
#     """
#     Construct the theoretical lower-bound PSD spectrogram of the error:
#
#         Phi_e,LB^w(k, omega) = gamma(omega, k)^2 * Phi_{p^*,st}^w(k, omega),
#
#     where Phi_{p^*,st}^w(k, omega) is the tapered PSD of the
#     (demeaned) reference trajectory inside each window, and gamma is
#     the robust sensitivity lower-bound factor.
#
#     Inputs:
#         dt          : sampling time [s]
#         gamma_local : array [freq x windows]
#         f_hz        : frequency grid [Hz]
#         t_end       : window end times [s]
#         pstar_seq   : reference trajectory [m], shape (T, 3)
#         nperseg,... : same windowing parameters as spectrogram
#
#     Returns:
#         f_hz        : frequency grid [Hz]
#         t_end       : window end times [s]
#         PSD_LB_mm2Hz: LB PSD [mm^2/Hz], shape (F, n_windows)
#     """
#     T = pstar_seq.shape[0]
#     hop = max(1, nperseg - noverlap)
#     ends = np.arange(nperseg - 1, T, hop)          # window END indices
#     t_end_check = ends * dt
#
#     if not np.allclose(t_end, t_end_check):
#         raise ValueError("Inconsistent t_end between gamma_local and reference grid.")
#
#     F = len(f_hz)
#     fs = 1.0 / dt
#
#     # Hann taper (same as scipy window='hann')
#     win = np.hanning(nperseg).reshape(nperseg, 1)  # (nperseg, 1)
#     U = float(np.sum(win[:, 0] ** 2))              # window energy
#
#     PSD_LB_m2Hz = np.zeros((F, len(ends)), dtype=float)
#
#     for wi, kend in enumerate(ends):
#         kstart = kend - nperseg + 1
#         if kstart < 0:
#             # pad at the beginning with first sample
#             pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
#             pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (nperseg, 3)
#         else:
#             pseg = pstar_seq[kstart:kend+1, :]              # (nperseg, 3)
#
#         # Demean per window (remove slowly varying trend)
#         pseg = pseg - pseg.mean(axis=0, keepdims=True)
#
#         # Apply taper and compute rFFT along time (axis=0)
#         Xw = win * pseg                                   # (nperseg, 3)
#         Pstar_f = np.fft.rfft(Xw, n=nfft, axis=0)         # (F, 3), [m]
#         # Norm of reference spectrum at each frequency
#         Pnorm_sq = np.sum(np.abs(Pstar_f) ** 2, axis=1)   # (F,), [m^2]
#
#         # Tapered PSD of reference norm (Welch-type)
#         # Phi_p^w(omega) ~ Pnorm_sq / (fs * U)
#         PSD_ref_m2Hz = Pnorm_sq / (fs * U)                # [m^2/Hz]
#
#         # Local robust LB factor gamma(omega, k)
#         gamma_w = gamma_local[:, wi]                      # (F,)
#
#         # LB PSD of error: gamma^2 * Phi_{p^*,st}^w
#         PSD_LB_m2Hz[:, wi] = (gamma_w ** 2) * PSD_ref_m2Hz
#
#     PSD_LB_mm2Hz = 1e6 * PSD_LB_m2Hz                     # convert m^2 -> mm^2
#     return f_hz, t_end, PSD_LB_mm2Hz
#
# # ============================================================
# # Load data
# # ============================================================
# J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
# J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
# pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))   # [m], shape (T,3)
#
# # Measured scalar error norms (already in mm)
# e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
# e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
# # e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_pi_real.npy")).squeeze()
# # e_sac_mm = np.load(os.path.join(base_dir, "mean_l2_real.npy")).squeeze()
# assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1
#
# # ============================================================
# # Spectrograms of measured PI and Hybrid errors
# # ============================================================
# f_pi,  t_pi,  Pxx_pi   = psd_spectrogram_mm2_per_hz_1d(
#     e_pi_mm,  fs, nperseg, noverlap, nfft, window=window
# )
# f_sac, t_sac, Pxx_sac  = psd_spectrogram_mm2_per_hz_1d(
#     e_sac_mm, fs, nperseg, noverlap, nfft, window=window
# )
#
# # ============================================================
# # PI gain search for robust lower bound (gamma)
# # ============================================================
# Kp_vals = np.linspace(Kp_min, Kp_max, n_Kp)
# Ki_vals = np.linspace(Ki_min, Ki_max, n_Ki)
#
# best_cost = np.inf
# best_Kp_vec = None
# best_Ki_vec = None
# best_f_hz = None
# best_t_end = None
# best_gamma_local = None
#
# print("Searching over PI gains for robust lower bound...")
# for kp in Kp_vals:
#     for ki in Ki_vals:
#         Kp_vec = np.array([kp, kp, kp])
#         Ki_vec = np.array([ki, ki, ki])
#
#         f_hz_tmp, t_end_tmp, gamma_local_tmp, gamma_cost_tmp = compute_gamma_local_and_cost(
#             dt, Kp_vec, Ki_vec,
#             J_true_seq, J_bias_seq,
#             nperseg, noverlap, nfft,
#             fmin, fmax, exclude_dc,
#             lam=1e-2,
#             lb_metric=lb_metric
#         )
#
#         if gamma_cost_tmp < best_cost:
#             best_cost = gamma_cost_tmp
#             best_Kp_vec = Kp_vec.copy()
#             best_Ki_vec = Ki_vec.copy()
#             best_f_hz = f_hz_tmp
#             best_t_end = t_end_tmp
#             best_gamma_local = gamma_local_tmp
#
# print("===================================================")
# print("Optimal PI gains for stochastic LB PSD (within search grid):")
# print(f"Kp_opt = {best_Kp_vec}")
# print(f"Ki_opt = {best_Ki_vec}")
# print(f"LB cost ({lb_metric}) = {best_cost:.4g}")
# print("===================================================\n")
#
# # ============================================================
# # Construct LB PSD spectrogram for K* (trajectory-dependent)
# # ============================================================
# f_th, t_th, PSD_LB_mm2Hz = psd_spectrogram_LB_mm2_per_hz_from_gamma(
#     dt,
#     best_gamma_local, best_f_hz, best_t_end,
#     pstar_seq,
#     nperseg, noverlap, nfft
# )
#
# # Consistency check: frequency grids must match
# if not (np.allclose(f_pi, f_th) and np.allclose(f_sac, f_th)):
#     raise ValueError("Frequency grids differ; use identical dt, nperseg, nfft for fair comparison.")
#
# # Interpolate LB PSD onto measured time grids
# PSD_LB_on_pi  = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_pi)
# PSD_LB_on_sac = interp_along_time_to_grid(PSD_LB_mm2Hz, t_th, t_sac)
#
# # ============================================================
# # Compute deviation (PSD_measured - PSD_LB_opt) spectrograms
# # ============================================================
# # Discrete-time indices of spectrogram windows
# k_pi_all  = t_pi  / dt
# k_sac_all = t_sac / dt
#
# idx_pi  = (k_pi_all  >= k_min) & (k_pi_all  <= k_max)
# idx_sac = (k_sac_all >= k_min) & (k_sac_all <= k_max)
#
# # Frequency masks in Hz
# m_pi  = band_mask(f_pi,  fmin, fmax, exclude_dc=exclude_dc)
# m_sac = band_mask(f_sac, fmin, fmax, exclude_dc=exclude_dc)
# m_th  = band_mask(f_th,  fmin, fmax, exclude_dc=exclude_dc)  # not strictly needed
#
# # Differences in PSD [mm^2/Hz]
# DIFF_pi_lb_opt  = Pxx_pi  - PSD_LB_on_pi
# DIFF_sac_lb_opt = Pxx_sac - PSD_LB_on_sac
#
# # Crop for plotting (freq band + time window)
# DIFF_pi_lb_plot  = DIFF_pi_lb_opt[m_pi, :][:, idx_pi]
# DIFF_sac_lb_plot = DIFF_sac_lb_opt[m_sac, :][:, idx_sac]
#
# k_pi  = k_pi_all[idx_pi]
# k_sac = k_sac_all[idx_sac]
#
# # Angular frequencies for band
# omega_pi  = 2 * np.pi * f_pi[m_pi]
# omega_sac = 2 * np.pi * f_sac[m_sac]
#
# # Focused masks for Figure 4-like plot
# mask_pi_focus  = (omega_pi  >= omega_focus_min) & (omega_pi  <= omega_focus_max)
# mask_sac_focus = (omega_sac >= omega_focus_min) & (omega_sac <= omega_focus_max)
#
# # ============================================================
# # Color scale for difference plots
# # ============================================================
# v_min, v_max = -4, 4
# cmap_diff = 'RdBu_r'
#
# # ============================================================
# # Figure: Spectrogram of deviation (PI and Hybrid vs optimal stochastic LB)
# # ============================================================
# fig4, axes4 = plt.subplots(2, 1, figsize=(6.4, 5.6), constrained_layout=True)
# ax41, ax42 = axes4
#
# # --- PI controller vs stochastic LB ---
# im41 = ax41.pcolormesh(
#     k_pi,
#     omega_pi[mask_pi_focus],
#     DIFF_pi_lb_plot[mask_pi_focus, :],
#     shading='gouraud',
#     cmap=cmap_diff,
#     vmin=v_min, vmax=v_max
# )
# ax41.set_title(r"Inverse-Jacobian PI Controller")
# ax41.set_xlabel(r"$k$")
# ax41.set_ylabel(r"$\omega$ [rad/s]")
# ax41.set_xlim((k_min, k_max))
# ax41.set_yscale('log')
# ax41.set_ylim((omega_focus_min, omega_focus_max))
#
# # --- Hybrid controller vs stochastic LB ---
# im42 = ax42.pcolormesh(
#     k_sac,
#     omega_sac[mask_sac_focus],
#     DIFF_sac_lb_plot[mask_sac_focus, :],
#     shading='gouraud',
#     cmap=cmap_diff,
#     vmin=v_min, vmax=v_max
# )
# ax42.set_title(r"Hybrid SAC--PI Controller")
# ax42.set_xlabel(r"$k$")
# ax42.set_ylabel(r"$\omega$ [rad/s]")
# ax42.set_xlim((k_min, k_max))
# ax42.set_yscale('log')
# ax42.set_ylim((omega_focus_min, omega_focus_max))
#
# # ---- Shared colorbar ----
# cbar4 = fig4.colorbar(im42, ax=axes4, location='right', shrink=0.96, pad=0.02)
# cbar4.set_label(r"$\Delta\Phi_{e}(k,\omega)$ [mm$^2$/Hz]")
#
# # ---- Save & show ----
# out_pdf4 = os.path.join(base_dir, "PSD_stochastic_LB_deviation_k_omega_log_FOCUSED.pdf")
# # fig4.savefig(out_pdf4, bbox_inches='tight')
# print(f"Figure path (if saved): {out_pdf4}")
# plt.show()
#
# print("")
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

# Sliding-window parameters
T_win = 1.0        # [s]
overlap = 0.9
nperseg = max(4, int(round(T_win / dt)))   # e.g. 10 samples
noverlap = int(round(overlap * nperseg))   # e.g. 9 samples
hop = max(1, nperseg - noverlap)

# Zero-padding for dense frequency sampling (df = fs / nfft)
nfft = 1000

# Frequency band for PI-class cost (Hz)
fmin_hz_cost = 0.0
fmax_hz_cost = 5.0
exclude_dc_cost = False   # exclude DC from cost

# Frequency band for plotting (rad/s, log axis)
omega_min_plot = 0 #0.628    # [rad/s]
omega_max_plot = 31.4     # [rad/s]

# k-range (sample index) for plotting on x-axis
k_min_plot = 10
k_max_plot = 109

# PI gain search range (scalar gains, same on all 3 axes)
# Adjust as needed
Kp_min, Kp_max, n_Kp = 10.0, 10.0, 1
Ki_min, Ki_max, n_Ki = 5.0, 5.0, 1

# Metric for PI-class lower-bound optimization:
# "max" -> worst-case (sup) over freq & windows
# "mean" -> average over freq & windows
lb_metric = "max"

# Pseudoinverse damping
lam_pinv = 1e-2

# Taper (Hann) for PSD estimation
# U is the taper energy used in the PSD normalization
window = np.hanning(nperseg)
U = np.sum(window**2)

# Base directory with data files
base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

# PSD scaling for plotting: m^2 -> mm^2
psd_scale = 1e6

# Colormaps
cmap_psd = "viridis"
cmap_diff = "RdBu_r"

# ============================================================
# Helpers
# ============================================================

def damped_pinv(A, lam=1e-2):
    """
    Damped pseudoinverse with Tikhonov regularization.
    """
    A = np.asarray(A, float)
    m, n = A.shape
    if m >= n:
        return np.linalg.solve(A.T @ A + (lam**2) * np.eye(n), A.T)
    return A.T @ np.linalg.inv(A @ A.T + (lam**2) * np.eye(m))

def Gz_from_omega(omega, dt):
    """
    Discrete-time integrator G(z) = dt / (1 - z^{-1}), evaluated on z = e^{j omega dt}.
    At omega = 0, the expression blows up; here we set G(0) = 0 to avoid NaNs.
    """
    omega = np.asarray(omega, float)
    z = np.exp(-1j * omega * dt)
    denom = 1.0 - z
    Gz = np.zeros_like(omega, dtype=complex)
    mask = ~np.isclose(denom, 0.0)
    Gz[mask] = dt / denom[mask]
    # DC stays as 0
    return Gz

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
    Build boolean mask over angular frequencies omega [rad/s].
    """
    m = (omega >= omega_min) & (omega <= omega_max)
    if exclude_dc:
        m &= ~np.isclose(omega, 0.0)
    return m

def tapered_psd_vecnorm_given_ends(
    x_mat,           # shape (T, D)
    window, U, dt,
    nfft,
    ends
):
    """
    Tapered PSD (Hann) of a multi-dimensional signal x_mat (T,D),
    using given window end indices.
    For each window, we:
      - extract segment of length nperseg,
      - subtract local mean (per dimension),
      - multiply by taper window,
      - compute rFFT,
      - return PSD estimate: Phi(omega) = dt/U * ||X(omega)||^2.
    Returns:
      f_hz: frequency grid (Hz), shape (F,)
      t_end: window end times (s), shape (W,)
      psd: array of shape (F, W)
    """
    x_mat = np.asarray(x_mat, float)
    T, D = x_mat.shape
    nperseg_local = len(window)
    t_end = ends * dt

    f_hz = np.fft.rfftfreq(nfft, d=dt)
    F = len(f_hz)
    psd = np.zeros((F, len(ends)), dtype=float)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg_local + 1
        seg = x_mat[kstart:kend+1, :]  # (nperseg, D)
        # subtract local mean to get zero-mean "fluctuations"
        seg = seg - seg.mean(axis=0, keepdims=True)

        seg_w = seg * window[:, None]          # apply taper
        Xf = np.fft.rfft(seg_w, n=nfft, axis=0)

        # Total power across D dimensions
        psd[:, wi] = (dt / U) * np.sum(np.abs(Xf)**2, axis=1)

    return f_hz, t_end, psd

def tapered_psd_scalar_given_ends(
    x, window, U, dt,
    nfft,
    ends
):
    """
    Tapered PSD (Hann) of a scalar signal x (T,),
    using given window end indices.
    Returns:
      f_hz: frequency grid (Hz), shape (F,)
      t_end: window end times (s), shape (W,)
      psd: array (F, W)
    """
    x = np.asarray(x, float).reshape(-1)
    T = x.shape[0]
    nperseg_local = len(window)
    t_end = ends * dt

    f_hz = np.fft.rfftfreq(nfft, d=dt)
    F = len(f_hz)
    psd = np.zeros((F, len(ends)), dtype=float)

    for wi, kend in enumerate(ends):
        kstart = kend - nperseg_local + 1
        seg = x[kstart:kend+1]      # (nperseg,)
        seg = seg - seg.mean()      # local demeaning

        seg_w = seg * window        # tapered segment
        Xf = np.fft.rfft(seg_w, n=nfft)
        psd[:, wi] = (dt / U) * (np.abs(Xf)**2)

    return f_hz, t_end, psd

def compute_gamma_local_for_K(
    dt, Kp_vec, Ki_vec,
    J_true_seq, J_bias_seq,
    omega, ends,
    lam_pinv=1e-2
):
    """
    Compute gamma_q(omega,k) for a given PI controller K = (Kp_vec, Ki_vec),
    using the frozen closed-loop model with additive uncertainty in J^\dagger.

    Returns:
      gamma_local: shape (F, W) where F = len(omega), W = len(ends)
    """
    J_true_seq = np.asarray(J_true_seq, float)
    J_bias_seq = np.asarray(J_bias_seq, float)
    T, _, N = J_true_seq.shape

    F = len(omega)
    W = len(ends)

    Gz = Gz_from_omega(omega, dt)           # (F,)
    C_all = make_C_PI_diag(Kp_vec, Ki_vec, Gz)  # (F,3,3)

    I3 = np.eye(3)
    gamma_local = np.zeros((F, W), dtype=float)

    for wi, kend in enumerate(ends):
        Jt = J_true_seq[kend]   # (3,N)
        Jb = J_bias_seq[kend]   # (3,N)

        # Pseudoinverses: N x 3
        Jt_dag = damped_pinv(Jt, lam_pinv)   # J_true^\dagger
        Jb_dag = damped_pinv(Jb, lam_pinv)   # \hat{J}^\dagger

        # Additive uncertainty in pseudo-inverse space
        Delta_a = Jb_dag - Jt_dag           # (N,3)

        for fi in range(F):
            Gk = Gz[fi]            # scalar
            Ck = C_all[fi]         # (3,3)

            # P(omega,k) = G(omega) J_true(k)  (3xN)
            P = Gk * Jt

            # Nominal loop L0 = P J_true^\dagger C
            L0 = P @ Jt_dag @ Ck
            A0 = I3 + L0
            try:
                S0 = np.linalg.inv(A0)
            except np.linalg.LinAlgError:
                S0 = np.linalg.pinv(A0)

            # Perturbation Delta_L = P Delta_a C  (3x3)
            Delta_L = P @ Delta_a @ Ck

            smin_S0 = sigma_min_2norm(S0)
            DeltaS_norm = sigma_max_2norm(Delta_L @ S0)
            denom = 1.0 + DeltaS_norm
            if denom == 0.0:
                gamma_local[fi, wi] = 0.0
            else:
                gamma_local[fi, wi] = smin_S0 / denom

    return gamma_local

# ============================================================
# Load data
# ============================================================

J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))   # (T,3,N)
J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))   # (T,3,N)
pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))    # [m], (T,3)

# Measured scalar error norms (logs) in [mm] – convert to [m]
e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
e_sac_mm = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()

e_pi_mm  = np.load(os.path.join(base_dir, "mean_l2_PI_real.npy")).squeeze()
e_sac_mm = np.load(os.path.join(base_dir, "mean_l2_real.npy")).squeeze()
assert e_pi_mm.ndim == 1 and e_sac_mm.ndim == 1

e_pi_m  = 1e-3 * e_pi_mm
e_sac_m = 1e-3 * e_sac_mm

# Truncate all to the same length for consistency
T_common = min(
    J_true_seq.shape[0],
    J_bias_seq.shape[0],
    pstar_seq.shape[0],
    e_pi_m.shape[0],
    e_sac_m.shape[0],
)
J_true_seq = J_true_seq[:T_common]
J_bias_seq = J_bias_seq[:T_common]
pstar_seq  = pstar_seq[:T_common]
e_pi_m     = e_pi_m[:T_common]
e_sac_m    = e_sac_m[:T_common]
T = T_common

# ============================================================
# Sliding windows: define window end indices and frequency grid
# ============================================================

ends = np.arange(nperseg - 1, T, hop)   # window END indices
t_end = ends * dt

f_hz = np.fft.rfftfreq(nfft, d=dt)
omega = 2 * np.pi * f_hz
F = len(f_hz)
W = len(ends)

# ============================================================
# Tapered PSD of reference fluctuations (vector-norm) -> Φ_v^w
# ============================================================

f_ref, t_end_ref, Phi_v_w = tapered_psd_vecnorm_given_ends(
    pstar_seq, window, U, dt, nfft, ends
)

# ============================================================
# Tapered PSD of PI and Hybrid error signals (scalar) -> Φ_e^w
# ============================================================

f_pi, t_end_pi, Phi_e_pi_w = tapered_psd_scalar_given_ends(
    e_pi_m, window, U, dt, nfft, ends
)

f_sac, t_end_sac, Phi_e_sac_w = tapered_psd_scalar_given_ends(
    e_sac_m, window, U, dt, nfft, ends
)

# Consistency checks
if not (np.allclose(f_ref, f_pi) and np.allclose(f_ref, f_sac)):
    raise ValueError("Frequency grids differ; ensure consistent dt, nperseg, nfft.")

if not (np.allclose(t_end_ref, t_end_pi) and np.allclose(t_end_ref, t_end_sac)):
    raise ValueError("Window end times differ; ensure consistent windowing.")

f_hz = f_ref
t_end = t_end_ref
omega = 2 * np.pi * f_hz
k_end = ends

# ============================================================
# PI-class optimization: compute gamma_local and Φ_e,LB^* = gamma^2 * Φ_v^w
# ============================================================

Kp_vals = np.linspace(Kp_min, Kp_max, n_Kp)
Ki_vals = np.linspace(Ki_min, Ki_max, n_Ki)

# Mask for cost frequency band (Hz)
mask_cost = (f_hz >= fmin_hz_cost) & (f_hz <= fmax_hz_cost)
if exclude_dc_cost:
    mask_cost &= ~np.isclose(f_hz, 0.0)

best_cost = np.inf
best_Kp_vec = None
best_Ki_vec = None
best_gamma_local = None

print("Searching over PI gains for PI-class optimal lower bound...")
for kp in Kp_vals:
    for ki in Ki_vals:
        Kp_vec = np.array([kp, kp, kp])
        Ki_vec = np.array([ki, ki, ki])

        gamma_local_tmp = compute_gamma_local_for_K(
            dt, Kp_vec, Ki_vec,
            J_true_seq, J_bias_seq,
            omega, ends,
            lam_pinv=lam_pinv
        )
        # Candidate lower-bound PSD for this K: gamma^2 * Phi_v_w
        Phi_e_lb_K = (gamma_local_tmp**2) * Phi_v_w   # (F,W)

        # Cost J(K) = max or mean over freq/time in band
        Phi_band = Phi_e_lb_K[mask_cost, :]
        if lb_metric == "max":
            J_K = np.max(Phi_band)
        else:
            J_K = np.mean(Phi_band)

        if J_K < best_cost:
            best_cost = J_K
            best_Kp_vec = Kp_vec.copy()
            best_Ki_vec = Ki_vec.copy()
            best_gamma_local = gamma_local_tmp.copy()

print("===================================================")
print("Optimal PI gains for PI-class lower bound (within search grid):")
print(f"Kp_opt = {best_Kp_vec}")
print(f"Ki_opt = {best_Ki_vec}")
print(f"LB cost ({lb_metric}) = {best_cost:.4g}")
print("===================================================\n")

# Optimal PI-class lower bound PSD spectrogram:
Phi_e_LB_star = (best_gamma_local**2) * Phi_v_w  # (F,W)

# ============================================================
# Restrict to plotting ranges
# ============================================================

mask_freq_plot = band_mask_freq_rad(
    omega, omega_min_plot, omega_max_plot, exclude_dc=True
)
mask_time_plot = (k_end >= k_min_plot) & (k_end <= k_max_plot)

omega_plot = omega[mask_freq_plot]
k_plot = k_end[mask_time_plot]

Phi_e_LB_plot   = Phi_e_LB_star[mask_freq_plot, :][:, mask_time_plot]
Phi_e_pi_plot   = Phi_e_pi_w[mask_freq_plot, :][:, mask_time_plot]
Phi_e_sac_plot  = Phi_e_sac_w[mask_freq_plot, :][:, mask_time_plot]

# Difference: hybrid error PSD - lower bound PSD
Phi_diff = Phi_e_sac_w - Phi_e_LB_star
Phi_diff_plot = Phi_diff[mask_freq_plot, :][:, mask_time_plot]

# Difference: PI error PSD - lower bound PSD
Phi_diff_pi = Phi_e_pi_w - Phi_e_LB_star
Phi_diff_plot_pi = Phi_diff_pi[mask_freq_plot, :][:, mask_time_plot]

# Scale to mm^2 for plotting
Phi_e_LB_plot_mm   = psd_scale * Phi_e_LB_plot
Phi_e_pi_plot_mm   = psd_scale * Phi_e_pi_plot
Phi_e_sac_plot_mm  = psd_scale * Phi_e_sac_plot
Phi_diff_plot_mm   = psd_scale * Phi_diff_plot
Phi_diff_plot_mm_pi   = psd_scale * Phi_diff_plot_pi

# Optional: choose color limits (you can tweak these based on data)
# Here we let matplotlib pick defaults; you can uncomment and adjust:
# vmin_psd = 0.0
# vmax_psd = np.percentile(Phi_e_sac_plot_mm, 99)
# vmin_diff = -np.max(np.abs(Phi_diff_plot_mm))
# vmax_diff = -vmin_diff

# ============================================================
# Figure 1: Spectrogram of PI-class optimal PSD lower bound
# ============================================================
fig1, ax1 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im1 = ax1.pcolormesh(
    k_plot,
    omega_plot,
    Phi_e_LB_plot_mm,
    shading="gouraud",
    cmap=cmap_psd,
    # vmin=vmin_psd, vmax=vmax_psd
)
ax1.set_title(r"PI-Class Optimal PSD Lower Bound $\Phi_{e,\mathrm{LB}}^\star(k,\omega)$")
ax1.set_xlabel(r"Sample index $k$")
ax1.set_ylabel(r"$\omega$ [rad/s]")
ax1.set_xlim((k_min_plot, k_max_plot))
ax1.set_yscale("log")
ax1.set_ylim((omega_min_plot, omega_max_plot))
cbar1 = fig1.colorbar(im1, ax=ax1)
cbar1.set_label(r"PSD [mm$^2$]")

# ============================================================
# Figure 2: Spectrogram of hybrid error PSD
# ============================================================
fig2, ax2 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im2 = ax2.pcolormesh(
    k_plot,
    omega_plot,
    Phi_e_sac_plot_mm,
    shading="gouraud",
    cmap=cmap_psd,
    # vmin=vmin_psd, vmax=vmax_psd
)
ax2.set_title(r"Hybrid SAC--PI Error PSD $\Phi_{e,\mathrm{hyb}}^w(k,\omega)$")
ax2.set_xlabel(r"Sample index $k$")
ax2.set_ylabel(r"$\omega$ [rad/s]")
ax2.set_xlim((k_min_plot, k_max_plot))
ax2.set_yscale("log")
ax2.set_ylim((omega_min_plot, omega_max_plot))
cbar2 = fig2.colorbar(im2, ax=ax2)
cbar2.set_label(r"PSD [mm$^2$]")

# ============================================================
# Figure 3: Spectrogram of loop-shaping PI error PSD
# ============================================================
fig3, ax3 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im3 = ax3.pcolormesh(
    k_plot,
    omega_plot,
    Phi_e_pi_plot_mm,
    shading="gouraud",
    cmap=cmap_psd,
    # vmin=vmin_psd, vmax=vmax_psd
)
ax3.set_title(r"Loop-Shaping PI Error PSD $\Phi_{e,\mathrm{PI}}^w(k,\omega)$")
ax3.set_xlabel(r"Sample index $k$")
ax3.set_ylabel(r"$\omega$ [rad/s]")
ax3.set_xlim((k_min_plot, k_max_plot))
ax3.set_yscale("log")
ax3.set_ylim((omega_min_plot, omega_max_plot))
cbar3 = fig3.colorbar(im3, ax=ax3)
cbar3.set_label(r"PSD [mm$^2$]")

# ============================================================
# Figure 4: Spectrogram of difference (hybrid - lower bound)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(6, 3.0), constrained_layout=True)
im4 = ax4.pcolormesh(
    k_plot,
    omega_plot,
    Phi_diff_plot_mm,
    shading="gouraud",
    cmap=cmap_diff,
    vmin=-2, vmax=2
)
# ax4.set_title(r"Hybrid Error PSD $-$ PI-Class Lower Bound")
ax4.set_xlabel(r"$k$")
ax4.set_ylabel(r"$\omega$ [rad/s]")
ax4.set_xlim((k_min_plot, k_max_plot))
ax4.set_yscale("log")
ax4.set_ylim((omega_min_plot, omega_max_plot))
cbar4 = fig4.colorbar(im4, ax=ax4)
cbar4.set_label(r"$\Delta\Phi_{e}(k,\omega)$ [mm$^2$/Hz]")
# out_pdf4 = os.path.join(base_dir, "dev_OptLB_dev.pdf")
out_pdf4 = os.path.join(base_dir, "dev_OptLB_dev_real.pdf")
fig4.savefig(out_pdf4, bbox_inches='tight')
plt.show()


# # ax42.set_title(r"Hybrid Controller")
# # ax42.set_xlabel(r"$k$")
# # ax42.set_ylabel(r"$\omega$ [rad/s]")
# # ax42.set_xlim((k_min, k_max))
# # ax42.set_yscale('log')
# # ax42.set_ylim((omega_focus_min, omega_focus_max))
# # # ---- Shared colorbar ----
# # cbar4 = fig4.colorbar(im42, ax=axes4, location='right', shrink=0.96, pad=0.02)
# # cbar4.set_label(r"$\Delta\Phi_{e}(k,\omega)$ [mm$^2$/Hz]")
# ============================================================
# Figure 5: Spectrogram of difference (pi - lower bound)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
im4 = ax4.pcolormesh(
    k_plot,
    omega_plot,
    Phi_diff_plot_mm_pi,
    shading="gouraud",
    cmap=cmap_diff,
    vmin=-2, vmax=2
)
ax4.set_title(r"Loop shaping PI Error PSD $-$ PI-Class Lower Bound")
ax4.set_xlabel(r"Sample index $k$")
ax4.set_ylabel(r"$\omega$ [rad/s]")
ax4.set_xlim((k_min_plot, k_max_plot))
ax4.set_yscale("log")
ax4.set_ylim((omega_min_plot, omega_max_plot))
cbar4 = fig4.colorbar(im4, ax=ax4)
cbar4.set_label(r"PSD difference [mm$^2$]")

plt.show()

print("Done.")
