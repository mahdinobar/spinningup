# ##############################################################################################


import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# Helper functions
# ============================================================

def damped_pinv(J, lam=1e-2):
    """3x6 -> 6x3 Tikhonov-damped pseudoinverse (slightly larger lam for conditioning)."""
    J = np.asarray(J, float)
    m, n = J.shape
    if m <= n:
        JJt = J @ J.T
        return J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(m), np.eye(m))
    else:
        JtJ = J.T @ J
        return np.linalg.solve(JtJ + (lam**2)*np.eye(n), J.T)

def Gz_from_omega(omega, dt):
    """Discrete integrator G(z) = dt / (1 - z^{-1}) on unit circle z=e^{jωdt}."""
    z_inv = np.exp(-1j * omega * dt)
    return dt / (1.0 - z_inv)

def _centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        step = 1.0
        return np.array([centers[0] - step/2, centers[0] + step/2])
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0]  = centers[0]  - (edges[1]  - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges

# ============================================================
# Core computation with BAND LIMITING
# ============================================================

def stft_th_error_rms_band(
    dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, band_hz,
    overlap=0.9, lam=1e-2
):
    """
    Compute time-domain RMS of theoretical error with frequency band limiting.

    band_hz: tuple (f_min, f_max) in Hz. Bins outside band are zeroed before irfft.
    Returns:
        t_sec: centers of windows [s]
        e_rms_mm: band-limited RMS ||e_th|| in millimeters
    """
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)
    J_true_seq = np.asarray(J_true_seq, float)
    J_bias_seq = np.asarray(J_bias_seq, float)
    pstar_seq = np.asarray(pstar_seq, float)  # meters

    T = pstar_seq.shape[0]
    Nw = max(4, int(round(T_win / dt)))
    hop = max(1, int(round(Nw * (1.0 - overlap))))
    win = np.hanning(Nw).reshape(Nw, 1)

    ends = np.arange(Nw - 1, T, hop)
    t_sec = ends * dt
    W = len(ends)

    # Frequency grid
    f_hz_all = np.fft.rfftfreq(Nw, d=dt)
    omega_all = 2 * np.pi * f_hz_all

    # Build band mask (and also remove DC robustly)
    f_min, f_max = band_hz
    eps = 1e-12
    valid = (f_hz_all >= max(f_min, eps)) & (f_hz_all <= f_max)

    # Precompute frequency response terms on the valid band
    omega = omega_all[valid]
    Gz = Gz_from_omega(omega, dt)              # shape (Fv,)
    # To form C(ω) per axis: Kp + Ki * Gz
    Cw = np.stack([Kp[i] + Ki[i] * Gz for i in range(3)], axis=1)  # (Fv, 3)
    Gw = Gz[:, None]                                               # (Fv, 1)

    e_rms_th = np.zeros(W)

    for wi, kend in enumerate(ends):
        kstart = kend - Nw + 1
        if kstart < 0:
            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])
        else:
            pseg = pstar_seq[kstart:kend+1, :]

        # Window and FFT of reference (per axis)
        xw = win * pseg
        Pstar_f_all = np.fft.rfft(xw, axis=0)  # (F_all, 3)

        # Local linearization at window end
        Jt = J_true_seq[kend]
        Jb = J_bias_seq[kend]
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag
        g_axis = np.diag(M).real                # (3,)
        g = g_axis[None, :]                     # (1,3)

        # Loop and sensitivity on VALID band
        Lw = (g * Gw) * Cw                      # (Fv,3)
        den = 1.0 + Lw
        tiny = 1e-8
        den = np.where(np.abs(den) < tiny, tiny * den/np.maximum(np.abs(den), tiny), den)
        Sw = 1.0 / den                          # (Fv,3)

        # Assemble S(ω) on full grid, but keep only valid band nonzero
        F_all = len(f_hz_all)
        Sw_all = np.zeros((F_all, 3), dtype=complex)
        Sw_all[valid, :] = Sw

        # Theoretical error spectrum, then band-limit (bins outside band are already zero)
        Eth_all = Sw_all * Pstar_f_all

        # Invert to time domain
        e_th_win = np.fft.irfft(Eth_all, n=Nw, axis=0).real  # (Nw, 3) meters
        e_th_scalar = np.linalg.norm(e_th_win, axis=1)       # ||e|| per sample
        e_rms_th[wi] = np.sqrt(np.mean(e_th_scalar**2))      # meters

    # Convert once to mm for plotting
    e_rms_mm = 1e3 * e_rms_th
    return t_sec, e_rms_mm, f_hz_all

# ============================================================
# Plotting
# ============================================================

def plot_rms_bands(t_sec, curves, title="Band-limited RMS of theoretical error"):
    """
    curves: list of tuples (label, e_rms_mm, color)
    """
    plt.figure(figsize=(9, 3.4))
    for label, y, color in curves:
        plt.plot(t_sec, y, lw=1.6, label=label, color=color)
    plt.xlabel("Time (s)")
    plt.ylabel(r"RMS $\|\mathbf{e}_{\mathrm{th}}\|$  [mm]")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Example MAIN
# ============================================================

if __name__ == "__main__":
    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"
    dt = 0.1
    T_win = 1.0
    overlap = 0.9

    # Load your data (meters for pstar_seq)
    J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
    J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
    pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))
    Kp         = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
    Ki         = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

    # --- Suggested bands ---
    # Reasonable default (tune to your setup):
    # - Exclude near-DC where the integrator dominates & the bound is trivial.
    # - Keep within your sensing/actuation bandwidth (e.g., camera 10 Hz -> Nyquist ~5 Hz).
    default_band = (0, 5)   # Hz: start above ~0.2 Hz, stop below Nyquist margin

    # Band around wc:
    wc = 1.87  # Hz (given)
    band_wc = (0.1 * wc, 2.0 * wc)  # [0.187, 3.74] Hz

    # Optionally a narrower “tracking” band (if your conveyor speed is slow):
    # narrow_band = (0.3, 2.0)
    narrow_band = (4, 5.0)

    # Compute RMS curves
    t_sec, rms_default, _ = stft_th_error_rms_band(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, default_band, overlap=overlap
    )
    _, rms_wc, _ = stft_th_error_rms_band(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, band_wc, overlap=overlap
    )
    _, rms_narrow, _ = stft_th_error_rms_band(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, narrow_band, overlap=overlap
    )

    # Plot comparison
    curves = [
        (f"default {default_band[0]:.2f}–{default_band[1]:.2f} Hz", rms_default, "tab:blue"),
        (f"0.1×wc–2×wc  ({band_wc[0]:.3f}–{band_wc[1]:.2f} Hz)", rms_wc, "tab:orange"),
        (f"narrow {narrow_band[0]:.2f}–{narrow_band[1]:.2f} Hz", rms_narrow, "tab:green"),
    ]
    plot_rms_bands(t_sec, curves, title="Band-limited RMS (converted to mm)")







# ##############################################################################################
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ============================================================
# Helper functions
# ============================================================

def damped_pinv(J, lam=1e-3):
    """3x6 -> 6x3 Tikhonov-damped pseudoinverse."""
    J = np.asarray(J, float)
    m, n = J.shape
    if m <= n:
        JJt = J @ J.T
        return J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(m), np.eye(m))
    else:
        JtJ = J.T @ J
        return np.linalg.solve(JtJ + (lam**2)*np.eye(n), J.T)

def Gz_from_omega(omega, dt):
    """Discrete integrator G(z) = dt / (1 - z^{-1}) on unit circle z=e^{jωdt}."""
    z_inv = np.exp(-1j * omega * dt)
    return dt / (1.0 - z_inv)

def _centers_to_edges(centers):
    """Convert center coordinates to edges for pcolormesh."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        step = 1.0
        return np.array([centers[0] - step/2, centers[0] + step/2])
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0]  = centers[0]  - (edges[1]  - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges

# ============================================================
# Core computation
# ============================================================

def stft_theoretical_error_spectrogram_and_rms(
    dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, overlap=0.9, lam=1e-3
):
    """
    Build spectrogram of ||E_th(ω; k)||_2 and compute RMS of time-domain theoretical error.
    DC term is set to zero (S(0)=0).
    """
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)
    J_true_seq = np.asarray(J_true_seq, float)
    J_bias_seq = np.asarray(J_bias_seq, float)
    pstar_seq = np.asarray(pstar_seq, float)

    T = pstar_seq.shape[0]
    Nw = max(4, int(round(T_win / dt)))
    hop = max(1, int(round(Nw * (1.0 - overlap))))
    win = np.hanning(Nw).reshape(Nw, 1)

    ends = np.arange(Nw - 1, T, hop)
    t_sec = ends * dt
    W = len(ends)

    # Frequency grid (RFFT)
    f_hz_all = np.fft.rfftfreq(Nw, d=dt)
    omega_all = 2 * np.pi * f_hz_all
    valid = f_hz_all > 0.0  # drop DC
    f_hz = f_hz_all[valid]
    omega = omega_all[valid]
    F_all = len(f_hz_all)
    F_bins = len(f_hz)

    Gz = Gz_from_omega(omega, dt)

    Fmag = np.zeros((F_bins, W))
    e_rms_th = np.zeros(W)

    for wi, kend in enumerate(ends):
        kstart = kend - Nw + 1
        if kstart < 0:
            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
            pseg = np.vstack([pad, pstar_seq[:kend+1, :]])
        else:
            pseg = pstar_seq[kstart:kend+1, :]
        xw = win * pseg
        Pstar_f_all = np.fft.rfft(xw, axis=0)

        Jt = J_true_seq[kend]
        Jb = J_bias_seq[kend]
        Jb_dag = damped_pinv(Jb, lam)
        M = Jt @ Jb_dag
        g_axis = np.diag(M).real

        Cw = np.stack([Kp[i] + Ki[i] * Gz for i in range(3)], axis=1)
        Gw = Gz[:, None]
        g = g_axis[None, :]
        Lw = (g * Gw) * Cw
        Sw = 1.0 / (1.0 + Lw)

        Sw_all = np.zeros((F_all, 3), dtype=complex)
        Sw_all[valid, :] = Sw

        Eth_all = Sw_all * Pstar_f_all
        e_th_win = np.fft.irfft(Eth_all, n=Nw, axis=0).real
        e_th_scalar = np.linalg.norm(e_th_win, axis=1)
        e_rms_th[wi] = np.sqrt(np.mean(e_th_scalar**2))

        Eth_noDC = Eth_all[valid, :]
        Fmag[:, wi] = np.linalg.norm(Eth_noDC, axis=1)

    return Fmag, f_hz, t_sec, e_rms_th

# ============================================================
# Plotting
# ============================================================

def plot_spectrogram(Fmag, f_hz, t_sec, *, vmin=0.01, vmax=1.2, log=False, title_suffix=""):
    f_edges = _centers_to_edges(f_hz)
    t_edges = _centers_to_edges(t_sec)
    Z = np.maximum(Fmag, vmin if log else 0.0)

    plt.figure(figsize=(9, 4.5))
    if log:
        pcm = plt.pcolormesh(t_edges, f_edges, Z, shading="auto",
                             cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        pcm = plt.pcolormesh(t_edges, f_edges, Z, shading="auto",
                             cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label(r"$\|E_{\mathrm{th}}(f;t)\|_2$" +
                   (" (log scale)" if log else " (linear scale)"))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Theoretical error spectrogram {title_suffix}")
    plt.tight_layout()
    plt.show()

def plot_rms(t_sec, e_rms_th):
    plt.figure(figsize=(9, 3))
    plt.plot(t_sec, e_rms_th, lw=1.8, color="tab:blue")
    plt.xlabel("Time (s)")
    plt.ylabel(r"RMS $\|\mathbf{e}_{\mathrm{th}}\|$")
    plt.title("Time-domain RMS of theoretical error")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"
    dt = 0.1
    T_win = 1.0
    overlap = 0.9

    # Load
    J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
    J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
    pstar_seq  = np.load(os.path.join(base_dir, "pstar_seq.npy"))
    Kp         = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
    Ki         = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

    Fmag, f_hz, t_sec, e_rms_th = stft_theoretical_error_spectrogram_and_rms(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, overlap=overlap
    )

    # Spectrograms
    plot_spectrogram(Fmag, f_hz, t_sec, vmin=0.01, vmax=1.2, log=False,
                     title_suffix="(linear, 0.01–1.2)")
    plot_spectrogram(Fmag, f_hz, t_sec, vmin=0.01, vmax=1.2, log=True,
                     title_suffix="(log, 0.01–1.2)")

    # Separate RMS plot
    plot_rms(t_sec, e_rms_th)







##############################################################################################
    def _centers_to_edges(c):
        """Convert center coordinates to edges for pcolormesh."""
        c = np.asarray(c, float)
        if c.size == 1:
            step = 1.0
            return np.array([c[0] - step / 2, c[0] + step / 2])
        e = np.empty(c.size + 1, float)
        e[1:-1] = 0.5 * (c[:-1] + c[1:])
        e[0] = c[0] - (e[1] - c[0])
        e[-1] = c[-1] + (c[-1] - e[-2])
        return e


    def windowed_rms(x, nperseg, noverlap, fs):
        """Compute sliding-window RMS using a simple loop (compatible with all NumPy versions)."""
        step = nperseg - noverlap
        if step <= 0:
            raise ValueError("noverlap must be < nperseg")
        idx_starts = np.arange(0, len(x) - nperseg + 1, step, dtype=int)
        rms = np.empty(len(idx_starts))
        for i, start in enumerate(idx_starts):
            seg = x[start:start + nperseg]
            rms[i] = np.sqrt(np.mean(seg ** 2))
        t_centers = (idx_starts + 0.5 * nperseg) / fs
        return t_centers, rms


    # ------------------- parameters -------------------
    dt = 0.1
    fs = 1.0 / dt  # sampling frequency [Hz]
    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

    # Load your mean L2 error signal (1D array)
    e = np.load(os.path.join(base_dir, "mean_l2_PI.npy")).squeeze()
    assert e.ndim == 1, f"Expected 1-D array, got shape {e.shape}"

    # Spectrogram parameters
    T_win = 1.0  # seconds
    overlap = 0.90
    nperseg = max(4, int(round(T_win / dt)))  # samples per window
    noverlap = int(round(overlap * nperseg))  # overlap samples

    # Compute spectrogram (magnitude)
    f, t, Sxx = spectrogram(
        e, fs=fs, window='hann', nperseg=nperseg,
        noverlap=noverlap, detrend=False, mode='magnitude'
    )

    # Exclude DC (ω = 0)
    mask = f > 0.0
    f = f[mask]
    Sxx = Sxx[mask, :]

    # Build edges for pcolormesh
    f_edges = _centers_to_edges(f)
    t_edges = _centers_to_edges(t)

    # Compute time-domain RMS aligned with spectrogram
    t_rms, rms = windowed_rms(e, nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Fixed color limits
    cmin, cmax = 0.01, 1.20
    S_linear = np.clip(Sxx, cmin, cmax)
    S_log = np.clip(Sxx, cmin, cmax)

    # -------------------------------------------------------------------------
    # (1) Linear scale spectrogram
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    pcm = plt.pcolormesh(
        t_edges, f_edges, S_linear, shading='auto', cmap='viridis',
        vmin=cmin, vmax=cmax
    )
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label(r'$|X(f,t)|$ (linear scale)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram of Tracking Error (Linear Scale, DC removed)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # (2) Logarithmic scale spectrogram
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    pcm = plt.pcolormesh(
        t_edges, f_edges, S_log, shading='auto', cmap='viridis',
        norm=LogNorm(vmin=cmin, vmax=cmax)
    )
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label(r'$|X(f,t)|$ (logarithmic scale)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram of Tracking Error (Logarithmic Scale, DC removed)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # (3) RMS over time (separate plot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 3))
    plt.plot(t_rms, rms, linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Error (mm)")
    plt.title("Sliding-Window RMS of Tracking Error - iJPI - biased kinematics")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



    # ------------------- parameters -------------------
    dt = 0.1
    fs = 1.0 / dt  # sampling frequency [Hz]
    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"

    # Load your mean L2 error signal (1D array)
    e = np.load(os.path.join(base_dir, "mean_l2.npy")).squeeze()
    assert e.ndim == 1, f"Expected 1-D array, got shape {e.shape}"

    # Spectrogram parameters
    T_win = 1.0  # seconds
    overlap = 0.90
    nperseg = max(4, int(round(T_win / dt)))  # samples per window
    noverlap = int(round(overlap * nperseg))  # overlap samples

    # Compute spectrogram (magnitude)
    f, t, Sxx = spectrogram(
        e, fs=fs, window='hann', nperseg=nperseg,
        noverlap=noverlap, detrend=False, mode='magnitude'
    )

    # Exclude DC (ω = 0)
    mask = f > 0.0
    f = f[mask]
    Sxx = Sxx[mask, :]

    # Build edges for pcolormesh
    f_edges = _centers_to_edges(f)
    t_edges = _centers_to_edges(t)

    # Compute time-domain RMS aligned with spectrogram
    t_rms, rms = windowed_rms(e, nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Fixed color limits
    cmin, cmax = 0.01, 1.20
    S_linear = np.clip(Sxx, cmin, cmax)
    S_log = np.clip(Sxx, cmin, cmax)

    # -------------------------------------------------------------------------
    # (1) Linear scale spectrogram
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    pcm = plt.pcolormesh(
        t_edges, f_edges, S_linear, shading='auto', cmap='viridis',
        vmin=cmin, vmax=cmax
    )
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label(r'$|X(f,t)|$ (linear scale)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram of Tracking Error (Linear Scale, DC removed)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # (2) Logarithmic scale spectrogram
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    pcm = plt.pcolormesh(
        t_edges, f_edges, S_log, shading='auto', cmap='viridis',
        norm=LogNorm(vmin=cmin, vmax=cmax)
    )
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label(r'$|X(f,t)|$ (logarithmic scale)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram of Tracking Error (Logarithmic Scale, DC removed)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # (3) RMS over time (separate plot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 3))
    plt.plot(t_rms, rms, linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Error (mm)")
    plt.title("Sliding-Window RMS of Tracking Error - SAC+iJPI - biased kinematics")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



















import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ------------------------- helpers -------------------------

def damped_pinv(J, lam=1e-3):
    """3x6 -> 6x3 Tikhonov-damped pseudoinverse."""
    J = np.asarray(J, float)
    m, n = J.shape
    if m <= n:
        JJt = J @ J.T
        return J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(m), np.eye(m))
    else:
        JtJ = J.T @ J
        return np.linalg.solve(JtJ + (lam ** 2) * np.eye(n), J.T)


def Gz_from_omega(omega, dt):
    """Discrete integrator G(z) = dt / (1 - z^{-1}) on unit circle z=e^{jωdt}."""
    z_inv = np.exp(-1j * omega * dt)
    return dt / (1.0 - z_inv)


def _centers_to_edges(centers):
    """Convert center coordinates to edges for pcolormesh."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        step = 1.0
        return np.array([centers[0] - step / 2, centers[0] + step / 2])
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


# --------------------- core computation ---------------------

def lower_bound_spectrogram_and_rms_mm(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq,
        overlap=0.9, lam=1e-3
):
    """
    Compute:
      - LB_mag_mm(f, k) = sigma_min(S(f;k)) * ||P*(f;k)||_2  (in mm)
      - RMS lower bound per window from LB_mag via Parseval (one-sided RFFT)
    Returns:
      LB_mag_mm: (F_bins_noDC, W)
      f_hz:      (F_bins_noDC,)
      t_sec:     (W,)
      rms_lb_mm: (W,)
    """
    # ---- inputs to arrays ----
    Kp = np.asarray(Kp, float).reshape(3)
    Ki = np.asarray(Ki, float).reshape(3)
    J_true_seq = np.asarray(J_true_seq, float)
    J_bias_seq = np.asarray(J_bias_seq, float)
    pstar_seq = np.asarray(pstar_seq, float)  # (T,3) in meters

    T = pstar_seq.shape[0]
    Nw = max(4, int(round(T_win / dt)))
    hop = max(1, int(round(Nw * (1.0 - overlap))))
    win = np.hanning(Nw).reshape(Nw, 1)

    ends = np.arange(Nw - 1, T, hop)
    t_sec = ends * dt
    W = len(ends)

    # RFFT frequencies and dropping DC
    f_hz_all = np.fft.rfftfreq(Nw, d=dt)  # 0..Nyq [Hz]
    omega_all = 2 * np.pi * f_hz_all
    valid = f_hz_all > 0.0
    f_hz = f_hz_all[valid]
    omega = omega_all[valid]
    F_all = len(f_hz_all)
    F_bins = len(f_hz)

    # Precompute G(z) for non-DC bins
    Gz = Gz_from_omega(omega, dt)  # (F_bins,)

    # Outputs
    LB_mag_mm = np.zeros((F_bins, W))  # lower-bound spectrogram in mm
    rms_lb_mm = np.zeros(W)  # RMS lower bound in mm

    # One-sided RFFT weights for Parseval (matching np.fft.rfft)
    # Indices: 0..M, we drop 0 (DC). For even Nw, Nyquist=M has weight 1, interior weight 2.
    M = Nw // 2
    if Nw % 2 == 0:
        # even length: bins 0..M exist
        weights_valid = np.ones(F_bins)
        if F_bins >= 2:
            weights_valid[:-1] *= 2.0  # bins 1..M-1 doubled
        # last one (Nyquist) stays 1.0
    else:
        # odd length: bins 0..M, all valid bins (1..M) doubled
        weights_valid = np.full(F_bins, 2.0)

    for wi, kend in enumerate(ends):
        # windowed segment of reference (meters)
        kstart = kend - Nw + 1
        if kstart < 0:
            pad = np.repeat(pstar_seq[0:1, :], -kstart, axis=0)
            pseg = np.vstack([pad, pstar_seq[:kend + 1, :]])
        else:
            pseg = pstar_seq[kstart:kend + 1, :]  # (Nw,3)
        xw = win * pseg

        # RFFT per axis (meters)
        Pstar_f_all = np.fft.rfft(xw, axis=0)  # (F_all,3)
        Pstar_f = Pstar_f_all[valid, :]  # (F_bins,3)
        Pnorm_m = np.linalg.norm(Pstar_f, axis=1)  # (F_bins,) meters

        # Freeze Jacobians at window end
        Jt = J_true_seq[kend]  # (3,6)
        Jb = J_bias_seq[kend]  # (3,6)
        Jb_dag = damped_pinv(Jb, lam)  # (6,3)
        Mmat = Jt @ Jb_dag  # (3,3), dimensionless
        g_axis = np.diag(Mmat).real  # (3,), dimensionless

        # Diagonal per-axis sensitivity S_i(ω;k) = 1 / (1 + g_i * G * (Kp_i + Ki_i*G))
        Cw = np.stack([Kp[i] + Ki[i] * Gz for i in range(3)], axis=1)  # (F_bins,3) [1/s]
        Gw = Gz[:, None]  # (F_bins,1) [s]
        g = g_axis[None, :]  # (1,3)
        Lw = (g * Gw) * Cw  # (F_bins,3), dimensionless
        Sw = 1.0 / (1.0 + Lw)  # (F_bins,3), dimensionless

        # σ_min(S) for diagonal S is min over axes of |S_i|
        sigma_min_S = np.min(np.abs(Sw), axis=1)  # (F_bins,)

        # Lower-bound magnitude per bin (meters)
        LB_mag_m = sigma_min_S * Pnorm_m  # (F_bins,)

        # Store spectrogram in mm
        LB_mag_mm[:, wi] = 1000.0 * LB_mag_m

        # ---- RMS lower bound via Parseval (one-sided RFFT, NumPy scaling) ----
        # RMS_lb^2 = (1/Nw^2) * sum(weights * LB_mag_m^2)
        rms_lb_m = np.sqrt((weights_valid @ (LB_mag_m ** 2)) / (Nw ** 2))
        rms_lb_mm[wi] = 1000.0 * rms_lb_m

    return LB_mag_mm, f_hz, t_sec, rms_lb_mm


# --------------------------- plotting ---------------------------

def plot_lb_spectrogram(LB_mag_mm, f_hz, t_sec, *, vmin=0.01, vmax=1.2, log=False, title_suffix=""):
    f_edges = _centers_to_edges(f_hz)
    t_edges = _centers_to_edges(t_sec)
    Z = np.maximum(LB_mag_mm, vmin if log else 0.0)

    plt.figure(figsize=(9, 4.6))
    if log:
        pcm = plt.pcolormesh(t_edges, f_edges, Z, shading="auto",
                             cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        pcm = plt.pcolormesh(t_edges, f_edges, Z, shading="auto",
                             cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label(r"$\sigma_{\min}(S)\,\|\tilde\mathbf{P}^*\|_2$  (mm)" +
                   (" — log scale" if log else " — linear scale"))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Lower-bound spectrogram {title_suffix}")
    plt.tight_layout()
    plt.show()


def plot_lb_rms(t_sec, rms_lb_mm):
    plt.figure(figsize=(9, 3.2))
    plt.plot(t_sec, rms_lb_mm, lw=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS lower bound (mm)")
    plt.title("Time-domain RMS lower bound from per-frequency bound")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


# ------------------------------ main ------------------------------

if __name__ == "__main__":
    # ---- paths & params ----
    base_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds"
    dt = 0.1
    T_win = 1.0
    # If you move by 1 sample with a 1 s window, overlap ≈ 0.9
    overlap = 0.9

    # ---- load data (meters) ----
    J_true_seq = np.load(os.path.join(base_dir, "J_true_seq.npy"))
    J_bias_seq = np.load(os.path.join(base_dir, "J_bias_seq.npy"))
    pstar_seq = np.load(os.path.join(base_dir, "pstar_seq.npy"))
    Kp = np.diag(np.load(os.path.join(base_dir, "Kp.npy")))
    Ki = np.diag(np.load(os.path.join(base_dir, "Ki.npy")))

    # ---- compute lower-bound spectrogram & RMS (mm) ----
    LB_mag_mm, f_hz, t_sec, rms_lb_mm = lower_bound_spectrogram_and_rms_mm(
        dt, Kp, Ki, J_true_seq, J_bias_seq, T_win, pstar_seq, overlap=overlap
    )

    # ---- plots ----
    # linear scale, fixed 0.01–1.2 mm
    plot_lb_spectrogram(LB_mag_mm, f_hz, t_sec, vmin=0.01, vmax=1.2, log=False,
                        title_suffix="(linear, 0.01–1.2 mm)")
    # log scale, same limits
    plot_lb_spectrogram(LB_mag_mm, f_hz, t_sec, vmin=0.01, vmax=1.2, log=True,
                        title_suffix="(log, 0.01–1.2 mm)")

    # separate RMS (mm)
    plot_lb_rms(t_sec, rms_lb_mm)

print("")
