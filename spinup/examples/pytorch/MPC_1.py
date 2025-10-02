# mpc_tracking.py
import numpy as np
import cvxpy as cp

# ---------- build linear model from Option-A plant ----------
def build_linear_model(J0: np.ndarray, alpha: np.ndarray, dt: float):
    """
    Returns A(9x9), B(9x6), C(3x9) for:
        dq[k+1] = A_q dq[k] + B_q u[k]
        p[k+1]  = p[k] + dt * J0 * dq[k+1]
    with A = [[A_q, 0],
              [dt J0 A_q, I3]],
         B = [[B_q],
              [dt J0 B_q]],
         C = [0_{3x6}  I3].
    """
    J0 = np.asarray(J0, float)
    alpha = np.asarray(alpha, float).reshape(6,)
    assert J0.shape == (3, 6)
    assert np.all(alpha > 0) and dt > 0

    a = np.exp(-dt / alpha)           # (6,)
    b0 = 1.0 - a                      # (6,)
    A_q = np.diag(a)                  # (6,6)
    B_q = np.diag(b0)                 # (6,6)

    A = np.zeros((9, 9), float)
    B = np.zeros((9, 6), float)
    C = np.zeros((3, 9), float)

    # dq block
    A[:6, :6] = A_q
    B[:6, :6] = B_q

    # p block (uses dq[k+1])
    A[6:, 6:] = np.eye(3)
    A[6:, :6] = dt * (J0 @ A_q)
    B[6:, :6] = dt * (J0 @ B_q)

    # Output y = p
    C[:, 6:] = np.eye(3)
    return A, B, C

# ---------- one MPC step (QP) ----------
def mpc_step(A, B, C, x0, p_ref_seq, u_prev,
             u_min, u_max, dt,
             Qp, Qf, R, Qdq=None, S=None,
             du_max=None, N=30, solver=cp.OSQP):
    """
    Solve the QP and return u0 (6,), predicted trajectories (optional).
    Shapes:
      x0: (9,), p_ref_seq: (N+1, 3), u_prev: (6,), u_min/u_max: (6,)
      Qp,Qf: (3,3), R: (6,6), Qdq,S: (6,6) or None
    """
    # Variables
    x = cp.Variable((9, N+1))
    u = cp.Variable((6, N))

    # Parameters from inputs
    x0 = np.asarray(x0, float).reshape(9,)
    p_ref_seq = np.asarray(p_ref_seq, float).reshape(N+1, 3)
    u_prev = np.asarray(u_prev, float).reshape(6,)
    u_min = np.asarray(u_min, float).reshape(6,)
    u_max = np.asarray(u_max, float).reshape(6,)

    # Weights (as matrices)
    Qp = np.asarray(Qp, float)
    Qf = np.asarray(Qf, float)
    R  = np.asarray(R,  float)
    if Qdq is None:
        Qdq = np.zeros((6,6))
    else:
        Qdq = np.asarray(Qdq, float)
    if S is None:
        S = np.zeros((6,6))
    else:
        S = np.asarray(S, float)

    cost = 0
    cons = []

    # Initial condition
    cons += [x[:, 0] == x0]

    # Build stage costs and dynamics
    for k in range(N):
        p_k = C @ x[:, k]                 # (3,)
        dq_k = x[:6, k]                   # (6,)

        # Tracking and regularization
        e_k = p_k - p_ref_seq[k]          # (3,)
        du_k = u[:, k] - (u_prev if k == 0 else u[:, k-1])

        cost += cp.quad_form(e_k, Qp) \
                + cp.quad_form(dq_k, Qdq) \
                + cp.quad_form(u[:, k], R) \
                + cp.quad_form(du_k, S)

        # Dynamics
        cons += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

        # Input box constraints
        cons += [u[:, k] <= u_max, u[:, k] >= u_min]

        # Optional rate limits
        if du_max is not None:
            cons += [cp.abs(du_k) <= du_max]

    # Terminal cost
    p_N = C @ x[:, N]
    e_N = p_N - p_ref_seq[N]
    cost += cp.quad_form(e_N, Qf)

    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=solver, warm_start=True, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MPC infeasible or solver failed: {prob.status}")

    u0 = u[:, 0].value
    return u0, {"x": x.value, "u": u.value, "status": prob.status}

# ---------- tiny closed-loop demo (replace with your plant/measurements) ----------
def simulate_one_step(x, u, A, B):
    """ State update x_{k+1} = A x_k + B u_k """
    return A @ x + B @ u

if __name__ == "__main__":
    # --- USER INPUTS ---
    dt = 0.004
    # Provide your real J0 and alpha:
    J0 = np.load("J0.npy")           # shape (3,6)
    alpha = np.load("alpha.npy")     # shape (6,)

    # Velocity limits:
    u_max = np.array([2.1750,2.1750,2.1750,2.1750,2.6100,2.6100])
    u_min = -u_max

    # Build model
    A, B, C = build_linear_model(J0, alpha, dt)

    # Horizon & weights
    N = 30
    Qp = np.diag([1.0, 1.0, 1.0])     # position tracking
    Qf = 5.0 * Qp                     # terminal
    Qdq = 0.01 * np.eye(6)            # joint velocity soft penalty
    R  = 1e-3 * np.eye(6)             # input effort
    S  = 1e-2 * np.eye(6)             # input rate smoothing
    du_max = 0.5 * np.ones(6)         # (rad/s) per step, optional

    # Initial state and reference preview
    x0 = np.zeros(9)                  # [dq0(6); p0(3)]
    p_ref_seq = np.zeros((N+1, 3))
    # e.g., track a small step on x-axis after a few samples
    p_ref_seq[:, 0] = 0.1             # 10 cm

    u_prev = np.zeros(6)

    # Solve one MPC step
    u0, info = mpc_step(A, B, C, x0, p_ref_seq, u_prev,
                        u_min, u_max, dt,
                        Qp, Qf, R, Qdq=Qdq, S=S, du_max=du_max, N=N)

    print("u0 (rad/s):", np.round(u0, 4))

    # Example: apply to the model (in real use, send u0 to robot and measure x1)
    x1 = simulate_one_step(x0, u0, A, B)
    print("x1[:6] dq (rad/s):", np.round(x1[:6], 4))
    print("x1[6:]  p  (m):   ", np.round(x1[6:], 4))
