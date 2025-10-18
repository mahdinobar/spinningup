# # ---------- identify_alphas.py ----------
# import numpy as np
# from typing import Tuple, Dict
#
# def identify_alphas(u_log: np.ndarray, dq_log: np.ndarray, dt: float,
#                     clip_a: Tuple[float, float] = (1e-6, 0.999999),
#                     check_kappa: bool = True) -> Tuple[np.ndarray, Dict]:
#     """
#     Identify first-order time constants alpha_j (kappa=1 assumed) per joint
#     using a discrete ARX(1,1) model:
#         dq[k] ≈ a * dq[k-1] + b * u[k-1]
#       -> a = exp(-dt/alpha)  =>  alpha = -dt / ln(a)
#       -> (optional) kappa_hat = b / (1 - a)  (should be ~1 if kappa=1)
#
#     Inputs
#     ------
#     u_log : (N, 6) ndarray
#         Commanded joint velocities (rad/s), columns correspond to joints 1..6.
#     dq_log : (N, 6) ndarray
#         Measured joint velocities (rad/s), same sampling and alignment as u_log.
#     dt : float
#         Sampling time in seconds (constant).
#     clip_a : (float, float)
#         Safety bounds to clip the estimated 'a' before taking the log.
#     check_kappa : bool
#         If True, also estimate kappa_hat and report deviations from 1.
#
#     Returns
#     -------
#     alpha : (6,) ndarray
#         Identified time constants (s) for joints 1..6.
#     info : dict
#         Contains per-joint dicts with keys: 'a', 'b', 'kappa_hat', 'R2'.
#     """
#     u = np.asarray(u_log, float)
#     y = np.asarray(dq_log, float)
#     assert u.shape == y.shape and u.ndim == 2 and u.shape[1] == 6, \
#         "u_log and dq_log must be (N,6) arrays with the same shape."
#     N = u.shape[0]
#     if N < 50:
#         raise ValueError("Not enough samples; provide at least ~50-100 for a stable estimate.")
#
#     alpha = np.zeros(6, dtype=float)
#     info = {"per_joint": []}
#
#     for j in range(6):
#         # Build regression for joint j
#         X = np.column_stack([y[:-1, j], u[:-1, j]])  # [dq_{k-1}, u_{k-1}]
#         Y = y[1:, j]                                 # dq_k
#         theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
#         a_hat, b_hat = map(float, theta)
#
#         # Clip 'a' and compute alpha
#         a_hat = float(np.clip(a_hat, clip_a[0], clip_a[1]))
#         if 1.0 - a_hat < 1e-12:
#             # series approx ln(a) ~ -(1-a) for a -> 1-
#             alpha_j = dt / (1.0 - a_hat + 1e-12)
#         else:
#             alpha_j = -dt / np.log(a_hat)
#
#         # Optional diagnostic: DC gain estimate kappa_hat = b / (1 - a)
#         kappa_hat = b_hat / (1.0 - a_hat + 1e-12) if check_kappa else np.nan
#
#         # Fit quality (coefficient of determination)
#         Y_pred = X @ theta
#         ss_res = float(np.sum((Y - Y_pred)**2))
#         ss_tot = float(np.sum((Y - np.mean(Y))**2) + 1e-12)
#         R2 = 1.0 - ss_res / ss_tot
#
#         alpha[j] = alpha_j
#         info["per_joint"].append({
#             "a": a_hat, "b": b_hat, "kappa_hat": kappa_hat, "R2": R2
#         })
#
#     if check_kappa:
#         for j, d in enumerate(info["per_joint"], 1):
#             if not (0.8 <= d["kappa_hat"] <= 1.2):
#                 print(f"[warn] joint {j}: kappa_hat={d['kappa_hat']:.2f} deviates from 1. "
#                       f"(Your model assumes kappa=1.)")
#
#     return alpha, info
#
#
# if __name__ == "__main__":
#     # Example (replace with your real logs):
#     N = 600
#     u_log=np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/u_log_cor.npy")
#     dq_log=np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/dq_log_cor.npy")
#
#     # u_log = np.zeros((N, 6))
#     # dq_log = np.zeros((N, 6))
#
#     # dt = 0.004
#     dt = 0.004
#     alpha, info = identify_alphas(u_log, dq_log, dt)
#     print("alpha (s):", alpha)
#     # alpha[3] =alpha[3]/2
#     print("R2 per joint:", [round(d["R2"], 3) for d in info["per_joint"]])
#     np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/alpha_dt0004.npy",alpha)
#     # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/alpha_dt01.npy",alpha)
#     print("")
#
#
#


# ---------- identify_alphas.py (update) ----------
import numpy as np
from typing import Tuple, Dict, Optional

def _safe_alpha_from_a(a: float, dt: float) -> float:
    a = float(a)
    if 1.0 - a < 1e-12:
        return dt / (1.0 - a + 1e-12)
    return -dt / np.log(a)

def _split_train_valid(M: int, valid_frac: float = 0.25):
    m_valid = max(20, int(M * valid_frac))
    m_train = M - m_valid
    return m_train, m_valid

def _xcorr_delay_hint(u: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    """
    Rough delay hint: argmax cross-correlation of u -> y over lags [0..max_lag].
    Returns a non-negative delay (u lags y by 'd' samples in ARX form u[k-1-d]).
    """
    u = u - u.mean()
    y = y - y.mean()
    best_d, best_c = 0, -np.inf
    for d in range(max_lag + 1):
        # correlate u[k-1-d] with y[k]
        up = u[:len(u)-1-d]
        yp = y[1+d:len(y)]
        if len(up) < 8:
            continue
        c = float(np.dot(up, yp)) / (np.linalg.norm(up) * np.linalg.norm(yp) + 1e-12)
        if c > best_c:
            best_c, best_d = c, d
    return best_d

def identify_alphas(u_log: np.ndarray,
                    dq_log: np.ndarray,
                    dt: float,
                    clip_a: Tuple[float, float] = (1e-6, 0.9990),
                    max_delay: int = 8,
                    include_intercept: bool = True,
                    downsample_every: Optional[int] = None,
                    enforce_kappa1: bool = True,
                    alpha_penalty: float = 1e-3,
                    max_alpha: Optional[float] = 0.6,     # seconds; set None to disable
                    delay_hint_from_xcorr: bool = True,
                    delay_window: int = 2                 # search hint±window
                    ) -> Tuple[np.ndarray, Dict]:
    """
    Identify first-order time constants with delay search, DC-gain option, alpha cap, and x-corr delay hint.
    """
    u = np.asarray(u_log, float)
    y = np.asarray(dq_log, float)
    assert u.shape == y.shape and u.ndim == 2 and u.shape[1] == 6, \
        "u_log and dq_log must be (N,6) arrays with the same shape."

    if downsample_every is not None and downsample_every > 1:
        u = u[::downsample_every, :]
        y = y[::downsample_every, :]

    N = u.shape[0]
    if N < 80:
        raise ValueError("Not enough samples; provide at least ~80 for a robust fit.")

    # Convert max_alpha -> max allowed 'a'
    a_upper_cap = clip_a[1]
    if max_alpha is not None and max_alpha > 0:
        a_upper_cap = min(a_upper_cap, float(np.exp(-dt / max_alpha)))  # tighter than clip_a[1]

    alpha = np.zeros(6, dtype=float)
    info = {"per_joint": []}

    for j in range(6):
        # Delay search range
        if delay_hint_from_xcorr:
            hint = _xcorr_delay_hint(u[:, j], y[:, j], max_delay)
            d_start = max(0, hint - delay_window)
            d_end   = min(max_delay, hint + delay_window)
            delays = list(range(d_start, d_end + 1))
        else:
            delays = list(range(0, max_delay + 1))

        best = None
        candidates = []

        for d in delays:
            y_prev = y[d:N-1, j]      # dq_{k-1}
            u_prev = u[:N-1-d, j]     # u_{k-1-d}
            Y     = y[d+1:N, j]       # dq_k
            M = len(Y)
            if M < 40:
                continue

            m_train, m_valid = _split_train_valid(M)
            if m_train < 30:
                continue

            yp_tr, up_tr, Y_tr = y_prev[:m_train], u_prev[:m_train], Y[:m_train]
            yp_v,  up_v,  Y_v  = y_prev[m_train:], u_prev[m_train:], Y[m_train:]

            if enforce_kappa1:
                # (Y - u_prev) = a * (y_prev - u_prev) + c
                X_tr = (yp_tr - up_tr).reshape(-1, 1)
                X_v  = (yp_v  - up_v ).reshape(-1, 1)
                if include_intercept:
                    X_tr = np.column_stack([X_tr, np.ones(len(X_tr))])
                    X_v  = np.column_stack([X_v,  np.ones(len(X_v))])
                Y_tr_p = (Y_tr - up_tr)
                Y_v_p  = (Y_v  - up_v)

                theta, *_ = np.linalg.lstsq(X_tr, Y_tr_p, rcond=None)
                a_hat = float(theta[0])
                c_hat = float(theta[1]) if include_intercept else 0.0

                # Enforce stability and alpha cap: clip a ∈ [clip_a[0], a_upper_cap]
                a_hat = float(np.clip(a_hat, clip_a[0], a_upper_cap))
                b_hat = (1.0 - a_hat)
                kappa_hat = 1.0

                Yhat_tr = a_hat * yp_tr + b_hat * up_tr + c_hat
                Yhat_v  = a_hat * yp_v  + b_hat * up_v  + c_hat
            else:
                # Free b
                X_tr = np.column_stack([yp_tr, up_tr])
                X_v  = np.column_stack([yp_v,  up_v ])
                if include_intercept:
                    X_tr = np.column_stack([X_tr, np.ones(len(X_tr))])
                    X_v  = np.column_stack([X_v,  np.ones(len(X_v))])
                theta, *_ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
                a_hat = float(np.clip(theta[0], clip_a[0], a_upper_cap))
                b_hat = float(theta[1])
                c_hat = float(theta[2]) if include_intercept else 0.0
                kappa_hat = float(b_hat / (1.0 - a_hat + 1e-12))

                Yhat_tr = X_tr @ theta
                Yhat_v  = X_v  @ theta

            # Metrics
            ss_res_v = float(np.sum((Y_v - Yhat_v) ** 2))
            ss_tot_v = float(np.sum((Y_v - np.mean(Y_v)) ** 2) + 1e-12)
            R2_v = 1.0 - ss_res_v / ss_tot_v
            mse_v = ss_res_v / max(1, len(Y_v))

            alpha_j = _safe_alpha_from_a(a_hat, dt)
            score = mse_v + alpha_penalty * max(0.0, alpha_j)

            cand = {
                "delay": d, "a": a_hat, "b": b_hat, "c": c_hat,
                "alpha": alpha_j, "kappa_hat": kappa_hat,
                "R2_valid": R2_v, "mse_valid": mse_v,
            }
            candidates.append(cand)

            if (best is None) or (score < best["score"] - 1e-12) or \
               (abs(score - best["score"]) <= 1e-12 and R2_v > best["R2_valid"]):
                best = dict(cand)
                best["score"] = score

        if best is None:
            raise RuntimeError(f"Joint {j+1}: insufficient data for delay search.")
        alpha[j] = best["alpha"]

        # Notes if we hit the alpha cap
        a_cap_hit = (max_alpha is not None) and (best["a"] >= a_upper_cap - 1e-12)
        if a_cap_hit:
            print(f"[note] joint {j+1}: a clipped to {best['a']:.5f} "
                  f"(alpha≈{best['alpha']:.3f}s, cap={max_alpha}s).")

        best["candidates"] = candidates
        best["enforce_kappa1"] = enforce_kappa1
        best["a_upper_cap"] = a_upper_cap
        info["per_joint"].append(best)

    return alpha, info

if __name__ == "__main__":
    u_log = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/u_log_cor.npy")
    dq_log = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/dq_log_cor.npy")
    dt = 0.004

    alpha, info = identify_alphas(
        u_log, dq_log, dt,
        max_delay=5,
        include_intercept=True,
        enforce_kappa1=False,
        downsample_every=2,          # helps when oversampled vs. bandwidth
        max_alpha=0.1,               # cap alpha at 0.6 s (~1.7 Hz bandwidth)
        delay_hint_from_xcorr=True,
        delay_window=2,
        alpha_penalty=1e-3
    )

    print("alpha (s):", np.array2string(alpha, precision=8))
    print("delay per joint:", [d["delay"] for d in info["per_joint"]])
    print("R2_valid per joint:", [round(d["R2_valid"], 3) for d in info["per_joint"]])
    print("kappa_hat per joint:", [round(d["kappa_hat"], 3) for d in info["per_joint"]])
    np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/alpha_dt0004.npy",alpha)