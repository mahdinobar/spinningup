import pybullet as p
import pybullet_data
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

# === Sim and Robot Setup ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf")
robot = p.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf", useFixedBase=True)

controlled_joints = [0, 1, 2, 3, 4, 5]  # Use 6 DOF
nq = len(controlled_joints)
dt = 0.1
horizon = 10
lambda_reg = 1e-2

# === Target Trajectory ===
def target_position(t):
    return np.array([0.5 + 0.05 * t, 0.3 + 0.05 * t, 0.5])

# === Forward Kinematics Wrapper ===
def compute_ee_position(q):
    for i, joint in enumerate(controlled_joints):
        p.resetJointState(robot, joint, q[i])
    state = p.getLinkState(robot, 11)  # Link 11 is the hand
    return np.array(state[0])

# === MPC Solver ===
def mpc_controller(q0, t0):
    opti = ca.Opti()
    Q = opti.variable(nq, horizon + 1)
    dQ = opti.variable(nq, horizon)
    opti.subject_to(Q[:, 0] == q0)

    cost = 0
    for k in range(horizon):
        qk = Q[:, k]
        dqk = dQ[:, k]
        qk_next = Q[:, k + 1]
        qk_np = qk.full().flatten() if hasattr(qk, 'full') else np.array([float(qk[i]) for i in range(nq)])
        ee_pos = compute_ee_position(qk_np)
        target = target_position(t0 + k * dt)
        cost += ca.sumsqr(ee_pos[:2] - target[:2]) + lambda_reg * ca.sumsqr(dqk)
        opti.subject_to(Q[:, k + 1] == qk + dqk * dt)
        opti.subject_to(opti.bounded(-1.0, dqk, 1.0))  # velocity limits

    opti.minimize(cost)
    opti.solver("ipopt", {"print_level": 0, "sb": "yes"})
    try:
        sol = opti.solve()
        return sol.value(dQ[:, 0])
    except:
        print("MPC failed at time", t0)
        return np.zeros(nq)

# === Logging ===
log_data = {
    "time": [],
    "q": [],
    "ee_pos": [],
    "target_pos": []
}

# === Main Simulation Loop ===
q_real = np.zeros(nq)
for step in range(100):
    t_sim = step * dt
    dq_ref = mpc_controller(q_real, t_sim)
    q_real += dq_ref * dt

    # Apply joint positions to sim
    for i, joint in enumerate(controlled_joints):
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL,
                                targetPosition=q_real[i], force=87.0)

    # Logging
    ee_pos = compute_ee_position(q_real)
    target_pos = target_position(t_sim)
    log_data["time"].append(t_sim)
    log_data["q"].append(q_real.copy())
    log_data["ee_pos"].append(ee_pos.copy())
    log_data["target_pos"].append(target_pos.copy())

    time.sleep(dt)

# === Plotting ===
log_data["q"] = np.array(log_data["q"])
log_data["ee_pos"] = np.array(log_data["ee_pos"])
log_data["target_pos"] = np.array(log_data["target_pos"])

# XY Tracking
plt.figure(figsize=(8, 6))
plt.plot(log_data["target_pos"][:, 0], log_data["target_pos"][:, 1], 'r--', label="Target XY")
plt.plot(log_data["ee_pos"][:, 0], log_data["ee_pos"][:, 1], 'b-', label="EE XY")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("End Effector vs Target Trajectory (XY)")
plt.legend()
plt.grid(True)

# Joint Trajectories
plt.figure(figsize=(10, 6))
for j in range(nq):
    plt.plot(log_data["time"], log_data["q"][:, j], label=f"Joint {j}")
plt.xlabel("Time [s]")
plt.ylabel("Joint Angle [rad]")
plt.title("Joint Position Trajectories")
plt.legend()
plt.grid(True)

plt.show()
