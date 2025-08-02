import pybullet as p
import pybullet_data
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

# === Sim and Robot Setup ===
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
    useFixedBase=True
)

controlled_joints = [0, 1, 2, 3, 4, 5, 6]
nq = len(controlled_joints)
ee_link_index = 9  # end effector link ID in PyBullet
dt = 0.1
horizon = 10
lambda_reg = 1e-2
p.setTimeStep(timeStep=dt)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


# === Target Trajectory ===
def target_position(t):
    return np.array([0.5 + 0.05 * t, 0.3 + 0.05 * t, 0.5])

# === Forward Kinematics from PyBullet ===
def pb_forward_kinematics(q):
    for i, joint in enumerate(controlled_joints):
        p.resetJointState(robot, joint, q[i])
    state = p.getLinkState(robot, ee_link_index, computeForwardKinematics=True)
    return np.array(state[0])  # XYZ position

# === CasADi FK Wrapper Using PyBullet ===
def casadi_fk_pybullet():
    q_sym = ca.MX.sym('q', nq)
    fk_vals = []
    # Evaluate FK for each CasADi variable set via a CasADi "callback"
    def fk_callback(q_np):
        return np.array(pb_forward_kinematics(q_np)).reshape((3,))
    fk_func = ca.external('fk_func', fk_callback)
    # We wrap it in a CasADi function
    fk_casadi = ca.Function('fk', [q_sym], [fk_func(q_sym)])
    return fk_casadi

# fk_casadi = casadi_fk_pybullet()

def dh_transform(a, alpha, d, theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)*ca.cos(alpha),  ca.sin(theta)*ca.sin(alpha), a*ca.cos(theta)),
        ca.horzcat(ca.sin(theta),  ca.cos(theta)*ca.cos(alpha), -ca.cos(theta)*ca.sin(alpha), a*ca.sin(theta)),
        ca.horzcat(0,              ca.sin(alpha),                ca.cos(alpha),               d),
        ca.horzcat(0,              0,                            0,                           1)
    )
def casadi_fk_3d():
    q = ca.MX.sym('q', nq)

    # DH parameters from Panda URDF
    dh_params = [
        (0,        ca.pi/2, 0.333, q[0]),
        (0,       -ca.pi/2, 0.0,   q[1]),
        (0,        ca.pi/2, 0.316, q[2]),
        (0.0825,   ca.pi/2, 0.0,   q[3]),
        (-0.0825, -ca.pi/2, 0.384, q[4]),
        (0,        ca.pi/2, 0.0,   q[5]),
        (0.088,    ca.pi/2, 0.0,   q[6]),
        (0,        0.0,     0.107, 0)  # flange
    ]

    T = ca.MX.eye(4)
    for a, alpha, d, theta in dh_params:
        T = ca.mtimes(T, dh_transform(a, alpha, d, theta))

    ee_pos = T[:3, 3]
    return ca.Function('fk_3d', [q], [ee_pos])

fk_casadi = casadi_fk_3d()

# === MPC Solver in full 3D ===
def mpc_controller(q0, t0):
    opti = ca.Opti()
    Q = opti.variable(nq, horizon + 1)
    dQ = opti.variable(nq, horizon)
    opti.subject_to(Q[:, 0] == q0)

    cost = 0
    for k in range(horizon):
        qk = Q[:, k]
        dqk = dQ[:, k]
        target = target_position(t0 + k * dt)
        # ee_pos = fk_casadi(qk)  # Now uses PyBullet FK model
        # cost += ca.sumsqr(ee_pos - target) + lambda_reg * ca.sumsqr(dqk)
        ee_pos = fk_casadi(qk)
        cost += ca.sumsqr(ee_pos - target) + lambda_reg * ca.sumsqr(dqk)
        opti.subject_to(Q[:, k + 1] == qk + dqk * dt)
        opti.subject_to(opti.bounded(-1.0, dqk, 1.0))

    opti.minimize(cost)
    opti.solver("ipopt")

    try:
        sol = opti.solve()
        return sol.value(dQ[:, 0])
    except RuntimeError as e:
        print(f"MPC failed at time {t0} due to error: {e}")
        return np.zeros(nq)

# === Logging ===
log_data = {"time": [], "q": [], "ee_pos": [], "target_pos": []}

# === Main Simulation Loop ===
q_real = np.array([-0.20, 0.40, -0.20, -2.05, -0.12, 2.41, 0.0])
for step in range(100):
    t_sim = step * dt
    dq_ref = mpc_controller(q_real, t_sim)
    q_real += dq_ref * dt

    # Apply joint velocities
    p.setJointMotorControlArray(
        robot,
        controlled_joints,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=list(dq_ref),
        velocityGains=[1]*nq,
        forces=[87, 87, 87, 87, 12, 12, 12]
    )
    p.stepSimulation()

    # Logging
    ee_pos = pb_forward_kinematics(q_real)
    target_pos = target_position(t_sim)
    log_data["time"].append(t_sim)
    log_data["q"].append(q_real.copy())
    log_data["ee_pos"].append(ee_pos.copy())
    log_data["target_pos"].append(target_pos.copy())

    time.sleep(dt)

# === Convert to arrays ===
log_data["q"] = np.array(log_data["q"])
log_data["ee_pos"] = np.array(log_data["ee_pos"])
log_data["target_pos"] = np.array(log_data["target_pos"])

# === Absolute tracking errors ===
errors = np.abs(log_data["ee_pos"] - log_data["target_pos"])

# === Plot tracking errors ===
plt.figure(figsize=(10, 6))
labels = ["X error", "Y error", "Z error"]
for i in range(3):
    plt.plot(log_data["time"], errors[:, i] * 1000, label=f"{labels[i]} [mm]")
plt.xlabel("Time [s]")
plt.ylabel("Absolute Error [mm]")
plt.title("End-Effector Absolute Tracking Errors (MPC with PyBullet FK)")
plt.legend()
plt.grid(True)

# === XY Tracking plot ===
plt.figure(figsize=(8, 6))
plt.plot(log_data["target_pos"][:, 0], log_data["target_pos"][:, 1], 'r--', label="Target XY")
plt.plot(log_data["ee_pos"][:, 0], log_data["ee_pos"][:, 1], 'b-', label="EE XY")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("End Effector vs Target Trajectory (XY)")
plt.legend()
plt.grid(True)

plt.show()

# import pybullet as p
# import pybullet_data
# import numpy as np
# import casadi as ca
# import matplotlib.pyplot as plt
# import time
#
# # === Sim and Robot Setup ===
# p.connect(p.DIRECT)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# plane = p.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf")
# robot = p.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf", useFixedBase=True)
#
# controlled_joints = [0, 1, 2, 3, 4, 5, 6]  # Use 6 DOF
# nq = len(controlled_joints)
# dt = 0.1
# horizon = 10
# lambda_reg = 1e-2
#
# # === Target Trajectory ===
# def target_position(t):
#     return np.array([0.5 + 0.05 * t, 0.3 + 0.05 * t, 0.5])
#
# # === Forward Kinematics Wrapper ===
# def compute_ee_position(q):
#     # for i, joint in enumerate(controlled_joints):
#     #     p.resetJointState(robot, joint, q[i])
#     state = p.getLinkState(robot, 9, computeForwardKinematics=True,
#                                                 computeLinkVelocity=True)
#     return np.array(state[0])
#
# def dh_transform(a, alpha, d, theta):
#     return ca.vertcat(
#         ca.horzcat(ca.cos(theta), -ca.sin(theta)*ca.cos(alpha),  ca.sin(theta)*ca.sin(alpha), a*ca.cos(theta)),
#         ca.horzcat(ca.sin(theta),  ca.cos(theta)*ca.cos(alpha), -ca.cos(theta)*ca.sin(alpha), a*ca.sin(theta)),
#         ca.horzcat(0,              ca.sin(alpha),                ca.cos(alpha),               d),
#         ca.horzcat(0,              0,                            0,                           1)
#     )
#
# # === CasADi Forward Kinematics in 3D ===
# def casadi_fk_3d():
#     q = ca.MX.sym('q', 7)
#
#     # DH parameters for Panda
#     dh_params = [
#         (0,        ca.pi/2, 0.333, q[0]),
#         (0,       -ca.pi/2, 0.0,   q[1]),
#         (0,        ca.pi/2, 0.316, q[2]),
#         (0.0825,   ca.pi/2, 0.0,   q[3]),
#         (-0.0825, -ca.pi/2, 0.384, q[4]),
#         (0,        ca.pi/2, 0.0,   q[5]),
#         (0.088,    ca.pi/2, 0.0,   q[6]),
#         (0,        0.0,     0.107, 0)  # flange
#     ]
#
#     T = ca.MX.eye(4)
#     for a, alpha, d, theta in dh_params:
#         T = ca.mtimes(T, dh_transform(a, alpha, d, theta))
#
#     ee_pos = T[:3, 3]
#     return ca.Function('fk_3d', [q], [ee_pos])
#
#
# # === MPC Solver in full 3D ===
# def mpc_controller(q0, t0):
#     opti = ca.Opti()
#     Q = opti.variable(nq, horizon + 1)
#     dQ = opti.variable(nq, horizon)
#     opti.subject_to(Q[:, 0] == q0)
#
#     cost = 0
#     for k in range(horizon):
#         qk = Q[:, k]
#         dqk = dQ[:, k]
#         target = target_position(t0 + k * dt)
#         ee_pos = fk_3d(qk)
#         cost += ca.sumsqr(ee_pos - target) + lambda_reg * ca.sumsqr(dqk)
#         opti.subject_to(Q[:, k + 1] == qk + dqk * dt)
#         opti.subject_to(opti.bounded(-1.0, dqk, 1.0))  # vel limits
#
#     opti.minimize(cost)
#     opti.solver("ipopt")
#
#     try:
#         sol = opti.solve()
#         return sol.value(dQ[:, 0])
#     except RuntimeError as e:
#         print(f"MPC failed at time {t0} due to error: {e}")
#         return np.zeros(nq)
#
# # === Logging ===
# log_data = {
#     "time": [],
#     "q": [],
#     "ee_pos": [],
#     "target_pos": []
# }
#
# joint_lower_limits = np.array(
#     [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])  # example panda limits
# joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# # === Main Simulation Loop ===
# # q_real = np.zeros(nq)
# q_real = np.array([-0.20040717459772067, 0.4015263130901932, -0.20536087173827033, -2.0534489870435384, -0.12694184090841293, 2.413510859720994,0])
# fk_3d = casadi_fk_3d()
# for step in range(100):
#     t_sim = step * dt
#     dq_ref = mpc_controller(q_real, t_sim)
#     q_real += dq_ref * dt
#
#     p.setJointMotorControlArray(
#         robot,
#         [0, 1, 2, 3, 4, 5, 6],
#         controlMode=p.VELOCITY_CONTROL,
#         targetVelocities=list(dq_ref),
#         velocityGains=[1, 1, 2, 1, 1, 1, 1],
#         forces=[87, 87, 87, 87, 12, 12, 12])
#     p.stepSimulation()
#
#     # Logging
#     ee_pos = compute_ee_position(q_real)
#     target_pos = target_position(t_sim)
#     log_data["time"].append(t_sim)
#     log_data["q"].append(q_real.copy())
#     log_data["ee_pos"].append(ee_pos.copy())
#     log_data["target_pos"].append(target_pos.copy())
#
#     time.sleep(dt)
#
# # === Plotting ===
# log_data["q"] = np.array(log_data["q"])
# log_data["ee_pos"] = np.array(log_data["ee_pos"])
# log_data["target_pos"] = np.array(log_data["target_pos"])
#
# # XY Tracking
# plt.figure(figsize=(8, 6))
# plt.plot(log_data["target_pos"][:, 0], log_data["target_pos"][:, 1], 'r--', label="Target XY")
# plt.plot(log_data["ee_pos"][:, 0], log_data["ee_pos"][:, 1], 'b-', label="EE XY")
# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.title("End Effector vs Target Trajectory (XY)")
# plt.legend()
# plt.grid(True)
# # Joint Trajectories
# plt.figure(figsize=(10, 6))
# for j in range(nq):
#     plt.plot(log_data["time"], log_data["q"][:, j], label=f"Joint {j}")
# plt.xlabel("Time [s]")
# plt.ylabel("Joint Angle [rad]")
# plt.title("Joint Position Trajectories")
# plt.legend()
# plt.grid(True)
#
# # Absolute tracking errors
# errors = np.abs(log_data["ee_pos"] - log_data["target_pos"])
# plt.figure(figsize=(10, 6))
# labels = ["X error", "Y error", "Z error"]
# for i in range(3):
#     plt.plot(log_data["time"], errors[:, i] * 1000, label=f"{labels[i]} [mm]")
# plt.xlabel("Time [s]")
# plt.ylabel("Absolute Error [mm]")
# plt.title("End-Effector Absolute Tracking Errors")
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
#
#
# plt.show()
#
# print("")
