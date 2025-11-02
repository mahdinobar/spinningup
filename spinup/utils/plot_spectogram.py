# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helpers
# ============================================================

def damped_pinv(A, lam=1e-2):
    """
    Damped pseudoinverse: (A^T A + lam^2 I)^{-1} A^T
    For tall/rectangular A. lam is in the units of singular values.
    """
    A = np.asarray(A, float)
    m, n = A.shape
    if m >= n:
        return np.linalg.solve(A.T @ A + (lam**2)*np.eye(n), A.T)
    # If "fat", do right-sided damping instead
    return A.T @ np.linalg.inv(A @ A.T + (lam**2)*np.eye(m))

def Gz_from_omega(omega, dt):
    """
    ZOH-consistent discrete integrator (velocity -> position path):
        G(e^{j w}) = dt / (1 - e^{-j w dt})
    """
    return dt / (1.0 - np.exp(-1j * omega * dt))

def make_controller_diag(Kp, Ki, Gz):
    """
    Build diagonal controller frequency response per axis:
        C(ω) = diag(Kp + Ki * Gz)
    Returns array of shape (F, 3, 3)
    """
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)
    F = len(Gz)
    C = np.zeros((F, 3, 3), dtype=complex)
    for i in range(3):
        C[:, i, i] = Kp[i] + Ki[i] * Gz
    return C

def sigma_min(M):
    """Smallest singular value for a 3x3 (or 2D) complex matrix."""
    return np.linalg.svd(M, compute_uv=False)[-1]

# ============================================================
# Core computation (now returns both RMS and its σ_min lower bound)
# ============================================================

def stft_th_error_rms_band_v3(
    dt,
    Kp, Ki,                  # arrays of length 3 (per axis gains)
    J_true_seq, J_bias_seq,  # (T, 3, 3)
    T_win,                   # seconds
    pstar_seq,               # (T, 3), meters
    band_hz,                 # (f_min, f_max)
    overlap=0.9,
    lam=1e-2,
    # ---- Options (same as before) ----
    use_full_matrix=True,
    keep_complex=True,
    zero_mean_per_window=True,
    increase_resolution_factor=1,
    clip_sigma_min=1e-6,
    use_structured=False,
    structured_base='bias',
    exclude_dc=True
):
    """
    Compute band-limited, windowed RMS of:
      (i) theoretical error e_th(t) = irfft{ S(ω) P*(ω) } over the chosen band,
     (ii) its spectral lower bound using σ_min(S(ω)) * ||P*(ω)||.

    Returns:
        t_sec: centers of windows [s]
        e_rms_mm:     RMS ||e_th|| in millimeters
        e_lb_rms_mm:  RMS lower bound (σ_min-based) in millimeters
        f_hz_all: frequency grid of rFFT for the window length actually used
    """
    # ---------- Inputs ----------
    pstar_seq = np.asarray(pstar_seq, float)
    J_true_seq = np.asarray(J_true_seq, float)
    J_bias_seq = np.asarray(J_bias_seq, float)
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)

    T = pstar_seq.shape[0]
    # Increase Tw internally if requested to refine frequency resolution
    T_win_eff = float(T_win) * max(1, int(increase_resolution_factor))

    Nw = max(4, int(round(T_win_eff / dt)))
    hop = max(1, int(round(Nw * (1.0 - overlap))))
    win = np.hanning(Nw).reshape(Nw, 1)

    ends = np.arange(Nw - 1, T, hop)  # window ends
    t_sec = ends * dt
    W = len(ends)

    # Frequency grid
    f_hz_all = np.fft.rfftfreq(Nw, d=dt)
    omega_all = 2 * np.pi * f_hz_all

    # Band mask
    f_min, f_max = band_hz
    eps = 1e-12
    valid = (f_hz_all >= (eps if (exclude_dc or f_min <= 0) else max(f_min, 0.0))) & (f_hz_all <= f_max)
    if f_min > 0:
        valid &= (f_hz_all >= f_min)

    # Precompute G(ω) and controller on the valid band
    omega = omega_all[valid]
    Gz = Gz_from_omega(omega, dt)                # (Fv,)
    Cw = make_controller_diag(Kp, Ki, Gz)        # (Fv,3,3)

    # --------------------------------------------------------
    # Builders for S(ω)
    # --------------------------------------------------------
    def build_S_from_structured(Jt, Jb, Cw, Gz):
        """
        S = S0 (I + E S0)^{-1}
        with L0 = J_base * G * C and E = M - I where M = J_true * J_bias^dagger
        Base Jacobian choice via `structured_base` ('bias' or 'true').
        """
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag  # effective plant mismatch map

        J_base = Jb if structured_base == 'bias' else Jt
        # Frequency-by-frequency build
        Fv = len(Gz)
        S_all = np.zeros((Fv, 3, 3), dtype=complex)
        I3 = np.eye(3)
        for k in range(Fv):
            Gk = Gz[k]
            Ck = Cw[k]
            L0 = (J_base @ (Gk * Ck))  # 3x3
            A0 = I3 + L0
            try:
                S0 = np.linalg.inv(A0)
            except np.linalg.LinAlgError:
                S0 = np.linalg.inv(A0 + 1e-9 * I3)

            E = (M - I3)
            A = (I3 + E @ S0)
            if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                A = A + (clip_sigma_min - sigma_min(A)) * I3
            try:
                S_all[k] = S0 @ np.linalg.inv(A)
            except np.linalg.LinAlgError:
                S_all[k] = S0 @ np.linalg.pinv(A)
        return S_all

    def build_S_direct(Jt, Jb, Cw, Gz):
        """
        Direct: L = M * G * C, S = (I + L)^-1, full 3x3.
        If use_full_matrix=False, fall back to diagonal-only gain path.
        """
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag  # 3x3 real (from linearization); keep complex in products below
        if not keep_complex:
            M = M.real

        Fv = len(Gz)
        S_all = np.zeros((Fv, 3, 3), dtype=complex)
        I3 = np.eye(3)

        if use_full_matrix:
            for k in range(Fv):
                Gk = Gz[k]
                Ck = Cw[k]             # 3x3
                Lk = M @ (Gk * Ck)     # 3x3 complex
                A = I3 + Lk
                if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                    A = A + (clip_sigma_min - sigma_min(A)) * I3
                try:
                    S_all[k] = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    S_all[k] = np.linalg.pinv(A)
        else:
            # Diagonal-only fallback
            for k in range(Fv):
                Gk = Gz[k]
                Ck = Cw[k]
                g = np.diag(M)
                L_diag = g * (Gk * np.diag(Ck))
                A = np.eye(3, dtype=complex)
                A[0, 0] += L_diag[0]
                A[1, 1] += L_diag[1]
                A[2, 2] += L_diag[2]
                if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                    A = A + (clip_sigma_min - sigma_min(A)) * np.eye(3)
                S_all[k] = np.diag(1.0 / np.diag(A))
        return S_all

    # --------------------------------------------------------
    # Main loop across windows
    # --------------------------------------------------------
    F_all = len(f_hz_all)
    e_rms_mm      = np.zeros(W)
    e_lb_rms_mm   = np.zeros(W)

    for wi, kend in enumerate(ends):
        kstart = kend - Nw + 1
        if kstart < 0:
            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])  # (Nw, 3)
        else:
            pseg = pstar_seq[kstart:kend+1, :]

        # Optional de-mean (recommended for dynamic tracking analysis)
        if zero_mean_per_window:
            pseg = pseg - pseg.mean(axis=0, keepdims=True)

        # Windowing and FFT of reference (per axis)
        xw = win * pseg
        Pstar_f_all = np.fft.rfft(xw, axis=0)  # (F_all, 3) complex

        # Local linearization at window end
        Jt = J_true_seq[kend]
        Jb = J_bias_seq[kend]

        # Build S(ω) on valid band using chosen method
        if use_structured:
            S_valid = build_S_from_structured(Jt, Jb, Cw, Gz)   # (Fv,3,3)
        else:
            S_valid = build_S_direct(Jt, Jb, Cw, Gz)            # (Fv,3,3)

        # Assemble S on the full grid (zeros outside the band)
        S_all = np.zeros((F_all, 3, 3), dtype=complex)
        S_all[valid, :, :] = S_valid

        # -------------------------
        # (i) Theoretical spectrum
        # -------------------------
        Eth_all = np.zeros((F_all, 3), dtype=complex)
        for kf in range(F_all):
            if not valid[kf]:
                continue
            Eth_all[kf, :] = S_all[kf] @ Pstar_f_all[kf]

        # iFFT to time-domain; compute RMS in meters
        e_th_win = np.fft.irfft(Eth_all, n=Nw, axis=0).real  # (Nw, 3)
        e_th_scalar = np.linalg.norm(e_th_win, axis=1)
        e_rms_th = np.sqrt(np.mean(e_th_scalar**2))          # meters

        # ---------------------------------------------------------
        # (ii) σ_min-based spectral LOWER BOUND construction
        #      For each valid bin k, set a spectrum vector whose
        #      magnitude equals σ_min(S[k]) * ||P*(k)||_2 (minimal
        #      admissible energy), arbitrary direction/phase.
        # ---------------------------------------------------------
        Eth_lb_all = np.zeros((F_all, 3), dtype=complex)
        for kf in range(F_all):
            if not valid[kf]:
                continue
            smin = sigma_min(S_all[kf])           # scalar ≥ 0
            pnorm = np.linalg.norm(Pstar_f_all[kf])  # ||P*(kf)||_2
            mag = smin * pnorm                    # minimal allowed magnitude
            # Put all magnitude on axis-0 with zero phase (any direction works for energy):
            Eth_lb_all[kf, 0] = mag + 0j

        # iFFT to time-domain; compute RMS in meters (this is a LOWER BOUND)
        e_lb_win = np.fft.irfft(Eth_lb_all, n=Nw, axis=0).real  # (Nw, 3)
        e_lb_scalar = np.linalg.norm(e_lb_win, axis=1)
        e_lb_rms = np.sqrt(np.mean(e_lb_scalar**2))             # meters

        # Save (convert to mm)
        e_rms_mm[wi]    = 1e3 * e_rms_th
        e_lb_rms_mm[wi] = 1e3 * e_lb_rms

    return t_sec, e_rms_mm, e_lb_rms_mm, f_hz_all

# ============================================================
# Plotting
# ============================================================

def plot_rms_bands_with_lb(t_sec, curves, title="Band-limited RMS of theoretical error"):
    """
    curves: list of tuples (label, e_rms_mm, e_lb_rms_mm, color)
            The function plots solid for RMS, dashed for lower bound.
    """
    plt.figure(figsize=(9.6, 3.8))
    for label, y, ylb, color in curves:
        plt.plot(t_sec, y,   lw=1.8, label=label + " (RMS)", color=color)
        plt.plot(t_sec, ylb, lw=1.6, ls="--", label=label + " (LB via σ_min)", color=color)
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\mathrm{RMS}\,\|\mathbf{e}_{\mathrm{th}}\|$  [mm]")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

# ============================================================
# Example MAIN (toggle fixes here)
# ============================================================

if __name__ == "__main__":
    # ------------------- parameters -------------------
    dt = 0.1
    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

    # Load data (pstar_seq must be in meters)
    J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
    J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
    pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))
    Kp         = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
    Ki         = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

    # Window + overlap
    T_win = 1.0
    overlap = 0.9

    # Bands
    default_band = (0.0, 5.0)  # up to Nyquist for fs=10 Hz
    wc = 1.87
    band_wc = (0.1*wc, 2.0*wc)  # around crossover
    narrow_band = (1.0, 5.0)

    # ----- Strong (safe) fix preset -----
    common_kwargs = dict(
        use_full_matrix=True,
        keep_complex=True,
        zero_mean_per_window=True,
        increase_resolution_factor=1,
        clip_sigma_min=0.01,
        use_structured=False,
        structured_base='bias',
        exclude_dc=True
    )

    # Compute curves
    t_sec, rms_default, lb_default, _ = stft_th_error_rms_band_v3(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, default_band,
        overlap=overlap, lam=1e-2, **common_kwargs
    )
    _, rms_wc, lb_wc, _ = stft_th_error_rms_band_v3(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, band_wc,
        overlap=overlap, lam=1e-2, **common_kwargs
    )
    _, rms_narrow, lb_narrow, _ = stft_th_error_rms_band_v3(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, narrow_band,
        overlap=overlap, lam=1e-2, **common_kwargs
    )

    curves = [
        (f"default {default_band[0]:.2f}–{default_band[1]:.2f} Hz", rms_default, lb_default, "tab:blue"),
        (f"0.1×wc–2×wc  ({band_wc[0]:.3f}–{band_wc[1]:.2f} Hz)", rms_wc, lb_wc, "tab:orange"),
        (f"narrow {narrow_band[0]:.2f}–{narrow_band[1]:.2f} Hz", rms_narrow, lb_narrow, "tab:green"),
    ]
    plot_rms_bands_with_lb(t_sec, curves, title="Band-limited RMS and σ_min lower bound")
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram  # for the spectrogram plot


    # ============================================================
    # Helpers
    # ============================================================

    def damped_pinv(A, lam=1e-2):
        """
        Damped pseudoinverse: (A^T A + lam^2 I)^{-1} A^T
        For tall/rectangular A. lam is in the units of singular values.
        """
        A = np.asarray(A, float)
        m, n = A.shape
        if m >= n:
            return np.linalg.solve(A.T @ A + (lam ** 2) * np.eye(n), A.T)
        # If "fat", do right-sided damping instead
        return A.T @ np.linalg.inv(A @ A.T + (lam ** 2) * np.eye(m))


    def Gz_from_omega(omega, dt):
        """
        ZOH-consistent discrete integrator (velocity -> position path):
            G(e^{j w}) = dt / (1 - e^{-j w dt})
        """
        return dt / (1.0 - np.exp(-1j * omega * dt))


    def make_controller_diag(Kp, Ki, Gz):
        """
        Build diagonal controller frequency response per axis:
            C(ω) = diag(Kp + Ki * Gz)
        Returns array of shape (F, 3, 3)
        """
        Kp = np.asarray(Kp, float).reshape(3)
        Ki = np.asarray(Ki, float).reshape(3)
        F = len(Gz)
        C = np.zeros((F, 3, 3), dtype=complex)
        for i in range(3):
            C[:, i, i] = Kp[i] + Ki[i] * Gz
        return C


    def sigma_min(M):
        """Smallest singular value for a 3x3 (or 2D) complex matrix."""
        return np.linalg.svd(M, compute_uv=False)[-1]


    # ============================================================
    # Core computation (RMS + σ_min lower bound) — from your v3
    # ============================================================

    def stft_th_error_rms_band_v3(
            dt,
            Kp, Ki,  # arrays of length 3 (per axis gains)
            J_true_seq, J_bias_seq,  # (T, 3, 3)
            T_win,  # seconds
            pstar_seq,  # (T, 3), meters
            band_hz,  # (f_min, f_max)
            overlap=0.9,
            lam=1e-2,
            # ---- Options ----
            use_full_matrix=True,
            keep_complex=True,
            zero_mean_per_window=True,
            increase_resolution_factor=1,
            clip_sigma_min=1e-6,
            use_structured=False,
            structured_base='bias',
            exclude_dc=True
    ):
        """
        Compute band-limited, windowed RMS of:
          (i) theoretical error e_th(t) = irfft{ S(ω) P*(ω) } over the chosen band,
         (ii) its spectral lower bound using σ_min(S(ω)) * ||P*(ω)||.

        Returns:
            t_sec, e_rms_mm, e_lb_rms_mm, f_hz_all
        """
        pstar_seq = np.asarray(pstar_seq, float)
        J_true_seq = np.asarray(J_true_seq, float)
        J_bias_seq = np.asarray(J_bias_seq, float)
        Kp = np.asarray(Kp, float).reshape(3)
        Ki = np.asarray(Ki, float).reshape(3)

        T = pstar_seq.shape[0]
        T_win_eff = float(T_win) * max(1, int(increase_resolution_factor))

        Nw = max(4, int(round(T_win_eff / dt)))
        hop = max(1, int(round(Nw * (1.0 - overlap))))
        win = np.hanning(Nw).reshape(Nw, 1)

        ends = np.arange(Nw - 1, T, hop)
        t_sec = ends * dt
        W = len(ends)

        f_hz_all = np.fft.rfftfreq(Nw, d=dt)
        omega_all = 2 * np.pi * f_hz_all

        f_min, f_max = band_hz
        eps = 1e-12
        valid = (f_hz_all >= (eps if (exclude_dc or f_min <= 0) else max(f_min, 0.0))) & (f_hz_all <= f_max)
        if f_min > 0:
            valid &= (f_hz_all >= f_min)

        omega = omega_all[valid]
        Gz = Gz_from_omega(omega, dt)
        Cw = make_controller_diag(Kp, Ki, Gz)

        def build_S_from_structured(Jt, Jb, Cw, Gz):
            Jb_dag = damped_pinv(Jb, lam)
            M = Jt @ Jb_dag
            J_base = Jb if structured_base == 'bias' else Jt
            Fv = len(Gz)
            S_all = np.zeros((Fv, 3, 3), dtype=complex)
            I3 = np.eye(3)
            for k in range(Fv):
                Gk = Gz[k];
                Ck = Cw[k]
                L0 = (J_base @ (Gk * Ck))
                A0 = I3 + L0
                try:
                    S0 = np.linalg.inv(A0)
                except np.linalg.LinAlgError:
                    S0 = np.linalg.inv(A0 + 1e-9 * I3)
                E = (M - I3)
                A = (I3 + E @ S0)
                if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                    A = A + (clip_sigma_min - sigma_min(A)) * I3
                try:
                    S_all[k] = S0 @ np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    S_all[k] = S0 @ np.linalg.pinv(A)
            return S_all

        def build_S_direct(Jt, Jb, Cw, Gz):
            Jb_dag = damped_pinv(Jb, lam)
            M = Jt @ Jb_dag
            if not keep_complex:
                M = M.real
            Fv = len(Gz)
            S_all = np.zeros((Fv, 3, 3), dtype=complex)
            I3 = np.eye(3)
            if use_full_matrix:
                for k in range(Fv):
                    Gk = Gz[k];
                    Ck = Cw[k]
                    Lk = M @ (Gk * Ck)
                    A = I3 + Lk
                    if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                        A = A + (clip_sigma_min - sigma_min(A)) * I3
                    try:
                        S_all[k] = np.linalg.inv(A)
                    except np.linalg.LinAlgError:
                        S_all[k] = np.linalg.pinv(A)
            else:
                for k in range(Fv):
                    Gk = Gz[k];
                    Ck = Cw[k]
                    g = np.diag(M)
                    L_diag = g * (Gk * np.diag(Ck))
                    A = np.eye(3, dtype=complex)
                    A[0, 0] += L_diag[0];
                    A[1, 1] += L_diag[1];
                    A[2, 2] += L_diag[2]
                    if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                        A = A + (clip_sigma_min - sigma_min(A)) * np.eye(3)
                    S_all[k] = np.diag(1.0 / np.diag(A))
            return S_all

        F_all = len(f_hz_all)
        e_rms_mm = np.zeros(W)
        e_lb_rms_mm = np.zeros(W)

        for wi, kend in enumerate(ends):
            kstart = kend - Nw + 1
            if kstart < 0:
                pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
                pseg = np.vstack([pad, pstar_seq[:kend + 1, :]])
            else:
                pseg = pstar_seq[kstart:kend + 1, :]

            if zero_mean_per_window:
                pseg = pseg - pseg.mean(axis=0, keepdims=True)

            xw = win * pseg
            Pstar_f_all = np.fft.rfft(xw, axis=0)  # (F_all, 3)

            Jt = J_true_seq[kend]
            Jb = J_bias_seq[kend]
            S_valid = build_S_from_structured(Jt, Jb, Cw, Gz) if use_structured else build_S_direct(Jt, Jb, Cw, Gz)

            S_all = np.zeros((F_all, 3, 3), dtype=complex)
            S_all[valid, :, :] = S_valid

            Eth_all = np.zeros((F_all, 3), dtype=complex)
            for kf in range(F_all):
                if not valid[kf]:
                    continue
                Eth_all[kf, :] = S_all[kf] @ Pstar_f_all[kf]

            e_th_win = np.fft.irfft(Eth_all, n=Nw, axis=0).real
            e_th_scalar = np.linalg.norm(e_th_win, axis=1)
            e_rms_th = np.sqrt(np.mean(e_th_scalar ** 2))

            Eth_lb_all = np.zeros((F_all, 3), dtype=complex)
            for kf in range(F_all):
                if not valid[kf]:
                    continue
                smin = sigma_min(S_all[kf])
                pnorm = np.linalg.norm(Pstar_f_all[kf])
                mag = smin * pnorm
                Eth_lb_all[kf, 0] = mag + 0j

            e_lb_win = np.fft.irfft(Eth_lb_all, n=Nw, axis=0).real
            e_lb_scalar = np.linalg.norm(e_lb_win, axis=1)
            e_lb_rms = np.sqrt(np.mean(e_lb_scalar ** 2))

            e_rms_mm[wi] = 1e3 * e_rms_th
            e_lb_rms_mm[wi] = 1e3 * e_lb_rms

        return t_sec, e_rms_mm, e_lb_rms_mm, f_hz_all


    # ============================================================
    # NEW: Reconstruct full theoretical error via overlap–add (OLA)
    # ============================================================

    def reconstruct_error_time_series_via_ola(
            dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, band_hz,
            overlap=0.9, lam=1e-2,
            use_full_matrix=True, keep_complex=True,
            zero_mean_per_window=True, increase_resolution_factor=1,
            clip_sigma_min=1e-6, use_structured=False, structured_base='bias',
            exclude_dc=True
    ):
        """
        Reconstruct e_th(t) over the entire record using the same STFT blocks
        and inverse-FFT per window, combined with overlap–add (OLA).

        Returns:
            e_th_full: (T, 3) meters (time-domain theoretical error)
        """
        pstar_seq = np.asarray(pstar_seq, float)
        J_true_seq = np.asarray(J_true_seq, float)
        J_bias_seq = np.asarray(J_bias_seq, float)
        Kp = np.asarray(Kp, float).reshape(3)
        Ki = np.asarray(Ki, float).reshape(3)

        T = pstar_seq.shape[0]
        T_win_eff = float(T_win) * max(1, int(increase_resolution_factor))
        Nw = max(4, int(round(T_win_eff / dt)))
        hop = max(1, int(round(Nw * (1.0 - overlap))))
        win = np.hanning(Nw).reshape(Nw, 1)

        # Frequency grid and band
        f_hz_all = np.fft.rfftfreq(Nw, d=dt)
        omega_all = 2 * np.pi * f_hz_all
        f_min, f_max = band_hz
        eps = 1e-12
        valid = (f_hz_all >= (eps if (exclude_dc or f_min <= 0) else max(f_min, 0.0))) & (f_hz_all <= f_max)
        if f_min > 0:
            valid &= (f_hz_all >= f_min)

        omega = omega_all[valid]
        Gz = Gz_from_omega(omega, dt)
        Cw = make_controller_diag(Kp, Ki, Gz)

        def build_S_from_structured(Jt, Jb, Cw, Gz):
            Jb_dag = damped_pinv(Jb, lam)
            M = Jt @ Jb_dag
            J_base = Jb if structured_base == 'bias' else Jt
            Fv = len(Gz)
            S_all = np.zeros((Fv, 3, 3), dtype=complex)
            I3 = np.eye(3)
            for k in range(Fv):
                Gk = Gz[k];
                Ck = Cw[k]
                L0 = (J_base @ (Gk * Ck))
                A0 = I3 + L0
                try:
                    S0 = np.linalg.inv(A0)
                except np.linalg.LinAlgError:
                    S0 = np.linalg.inv(A0 + 1e-9 * I3)
                E = (M - I3)
                A = (I3 + E @ S0)
                if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                    A = A + (clip_sigma_min - sigma_min(A)) * I3
                try:
                    S_all[k] = S0 @ np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    S_all[k] = S0 @ np.linalg.pinv(A)
            return S_all

        def build_S_direct(Jt, Jb, Cw, Gz):
            Jb_dag = damped_pinv(Jb, lam)
            M = Jt @ Jb_dag
            if not keep_complex:
                M = M.real
            Fv = len(Gz)
            S_all = np.zeros((Fv, 3, 3), dtype=complex)
            I3 = np.eye(3)
            if use_full_matrix:
                for k in range(Fv):
                    Gk = Gz[k];
                    Ck = Cw[k]
                    Lk = M @ (Gk * Ck)
                    A = I3 + Lk
                    if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                        A = A + (clip_sigma_min - sigma_min(A)) * I3
                    try:
                        S_all[k] = np.linalg.inv(A)
                    except np.linalg.LinAlgError:
                        S_all[k] = np.linalg.pinv(A)
            else:
                for k in range(Fv):
                    Gk = Gz[k];
                    Ck = Cw[k]
                    g = np.diag(M)
                    L_diag = g * (Gk * np.diag(Ck))
                    A = np.eye(3, dtype=complex)
                    A[0, 0] += L_diag[0];
                    A[1, 1] += L_diag[1];
                    A[2, 2] += L_diag[2]
                    if clip_sigma_min is not None and sigma_min(A) < clip_sigma_min:
                        A = A + (clip_sigma_min - sigma_min(A)) * np.eye(3)
                    S_all[k] = np.diag(1.0 / np.diag(A))
            return S_all

        # OLA accumulators
        e_acc = np.zeros((T, 3), dtype=float)
        w_acc = np.zeros((T, 1), dtype=float)

        ends = np.arange(Nw - 1, T, hop)
        F_all = len(f_hz_all)

        for kend in ends:
            kstart = kend - Nw + 1
            if kstart < 0:
                pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
                pseg = np.vstack([pad, pstar_seq[:kend + 1, :]])
            else:
                pseg = pstar_seq[kstart:kend + 1, :]

            if zero_mean_per_window:
                pseg = pseg - pseg.mean(axis=0, keepdims=True)

            xw = win * pseg
            Pstar_f_all = np.fft.rfft(xw, axis=0)

            Jt = J_true_seq[kend]
            Jb = J_bias_seq[kend]
            S_valid = build_S_from_structured(Jt, Jb, Cw, Gz) if use_structured else build_S_direct(Jt, Jb, Cw, Gz)

            S_all = np.zeros((F_all, 3, 3), dtype=complex)
            S_all[valid, :, :] = S_valid

            Eth_all = np.zeros((F_all, 3), dtype=complex)
            for kf in range(F_all):
                if not valid[kf]:
                    continue
                Eth_all[kf, :] = S_all[kf] @ Pstar_f_all[kf]

            e_win = np.fft.irfft(Eth_all, n=Nw, axis=0).real  # (Nw,3)

            # Overlap–add with the same analysis window; normalize later by w_acc
            if kstart < 0:
                w0 = -kstart
                e_acc[0:kend + 1, :] += (win[w0:, :] * e_win[w0:, :])
                w_acc[0:kend + 1, :] += win[w0:, :]
            else:
                e_acc[kstart:kend + 1, :] += (win * e_win)
                w_acc[kstart:kend + 1, :] += win

        # Normalize where window sum is nonzero
        mask = (w_acc[:, 0] > 1e-12)
        e_acc[mask, :] /= w_acc[mask, :]
        # For any untouched samples (rare), leave zeros
        return e_acc  # (T,3) meters


    # ============================================================
    # Spectrogram plotting of |FFT(error)| magnitude (in mm)
    # ============================================================

    def plot_error_spectrogram(e_th_full, dt, T_win=1.0, overlap=0.9,
                               vmin_mm=0.01, vmax_mm=1.4, exclude_dc=True,
                               title="Spectrogram of |FFT(error)| (magnitude)"):
        """
        e_th_full: (T,3) meters — time-domain theoretical error (per-axis).
        Plots a spectrogram of the scalar error magnitude ||e(t)|| with colorbar in mm.
        DC (ω=0) is excluded from the plot if exclude_dc=True.
        """
        # Scalar error magnitude per sample
        e_mag = np.linalg.norm(e_th_full, axis=1)  # meters

        fs = 1.0 / dt
        nperseg = max(4, int(round(T_win / dt)))
        noverlap = int(round(overlap * nperseg))

        # Spectrogram of magnitude (not power). Output units: meters (magnitude of STFT)
        f, t, Sxx = spectrogram(e_mag,
                                fs=fs,
                                window='hann',
                                nperseg=nperseg,
                                noverlap=noverlap,
                                detrend=False,
                                mode='magnitude',
                                scaling='spectrum')  # keep magnitude-like scaling

        # Convert to mm for display
        Sxx_mm = 1e3 * Sxx

        # Exclude DC row if requested
        if exclude_dc and len(f) > 0 and np.isclose(f[0], 0.0):
            f = f[1:]
            Sxx_mm = Sxx_mm[1:, :]

        # Plot
        plt.figure(figsize=(9.6, 4.0))
        # pcolormesh expects the *edges*; but for simple display we can pass centers with shading='gouraud'
        plt.pcolormesh(t, f, Sxx_mm, shading='gouraud', vmin=vmin_mm, vmax=vmax_mm)
        cbar = plt.colorbar()
        cbar.set_label(r"$|\mathrm{FFT}(e)|$  [mm]")

        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(title)
        plt.ylim((f[0], f[-1]))  # keep plotted band
        plt.tight_layout()
        plt.show()


    # ============================================================
    # Example MAIN
    # ============================================================

    if __name__ == "__main__":
        # ------------------- parameters -------------------
        dt = 0.1
        base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

        # Load data (pstar_seq must be in meters)
        J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
        J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
        pstar_seq = np.load(os.path.join(base_dir, "pstar_seq.npy"))
        Kp = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
        Ki = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

        # Window + overlap for analysis
        T_win = 1.0
        overlap = 0.9

        # Bands
        default_band = (0.0, 5.0)  # up to Nyquist for fs=10 Hz

        # ---- Strong (safe) preset ----
        common_kwargs = dict(
            use_full_matrix=True,
            keep_complex=True,
            zero_mean_per_window=True,
            increase_resolution_factor=1,
            clip_sigma_min=0.01,
            use_structured=False,
            structured_base='bias',
            exclude_dc=True
        )

        # (A) RMS + lower bound example (optional)
        # t_sec, rms_default, lb_default, _ = stft_th_error_rms_band_v3(
        #     dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, default_band,
        #     overlap=overlap, lam=1e-2, **common_kwargs
        # )

        # (B) Reconstruct full theoretical error via OLA
        e_th_full = reconstruct_error_time_series_via_ola(
            dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, default_band,
            overlap=overlap, lam=1e-2, **common_kwargs
        )  # (T,3) meters

        # (C) Spectrogram of |FFT(error)| magnitude in mm, excluding DC
        plot_error_spectrogram(
            e_th_full, dt, T_win=T_win, overlap=overlap,
            vmin_mm=0.01, vmax_mm=1., exclude_dc=True,
            title="Spectrogram of |FFT(error)| (magnitude, mm)"
        )

# %%
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    # ------------------- parameters -------------------
    dt = 0.1  # sampling time [s]
    fs = 1.0 / dt  # sampling frequency [Hz]
    T_win = 1.0  # window length [s]
    overlap = 0.9  # 90% overlap
    nperseg = max(4, int(round(T_win / dt)))  # samples per window (here: 10)
    noverlap = int(round(overlap * nperseg))  # overlap samples (here: 9)

    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

    # Load your mean L2 error signal (1D array), units are [mm]
    e = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
    assert e.ndim == 1, f"Expected 1-D array, got shape {e.shape}"

    # --- Spectrogram of magnitude (|STFT|), exclude ω=0 (DC) row ---
    f, t, Sxx = spectrogram(
        e, fs=fs, window='hann',
        nperseg=nperseg, noverlap=noverlap,
        detrend=False, mode='magnitude',  # returns |STFT|, same units as input (mm)
        scaling='spectrum'  # keep magnitude-like scaling
    )

    # Exclude DC row (ω=0)
    if len(f) > 0 and np.isclose(f[0], 0.0):
        f = f[1:]
        Sxx = Sxx[1:, :]

    # Plot
    plt.figure(figsize=(9.6, 4.0))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', vmin=0.01, vmax=1.)  # colorbar in [mm]
    cbar = plt.colorbar()
    cbar.set_label(r"$|\mathrm{FFT}(e)|$  [mm]")

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram of |FFT(error)| magnitude (DC excluded)")
    plt.ylim((f[0], f[-1]))  # keep plotted band
    plt.tight_layout()
    plt.show()

# %%
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    # ------------------- parameters -------------------
    dt = 0.1  # sampling time [s]
    fs = 1.0 / dt  # sampling frequency [Hz]
    T_win = 1.0  # window length [s]
    overlap = 0.9  # 90% overlap
    nperseg = max(4, int(round(T_win / dt)))  # samples per window (here: 10)
    noverlap = int(round(overlap * nperseg))  # overlap samples (here: 9)

    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

    # Load your mean L2 error signal (1D array), units are [mm]
    e = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
    assert e.ndim == 1, f"Expected 1-D array, got shape {e.shape}"

    # --- Spectrogram of magnitude (|STFT|), exclude ω=0 (DC) row ---
    f, t, Sxx = spectrogram(
        e, fs=fs, window='hann',
        nperseg=nperseg, noverlap=noverlap,
        detrend=False, mode='magnitude',  # returns |STFT|, same units as input (mm)
        scaling='spectrum'  # keep magnitude-like scaling
    )

    # Exclude DC row (ω=0)
    if len(f) > 0 and np.isclose(f[0], 0.0):
        f = f[1:]
        Sxx = Sxx[1:, :]

    # Plot
    plt.figure(figsize=(9.6, 4.0))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', vmin=0.01, vmax=1.)  # colorbar in [mm]
    cbar = plt.colorbar()
    cbar.set_label(r"$|\mathrm{FFT}(e)|$  [mm]")

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram of |FFT(error)| magnitude (DC excluded)")
    plt.ylim((f[0], f[-1]))  # keep plotted band
    plt.tight_layout()
    plt.show()

    print("")