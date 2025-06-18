import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import pinv, svd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import svd, pinv, norm
import pybullet_data

# Start PyBullet in DIRECT mode (no GUI) or GUI if you want visualization
p.connect(p.DIRECT)  # or p.GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
# Disable verbose logging:
p.setPhysicsEngineParameter(enableConeFriction=0)  # optional

def load_jacobian(robot_id, q_t, dq_t):

    # # default timestep is 1/240 second (search fixedTimeStep)
    # p.setTimeStep(timeStep=dt_pb_sim, physicsClientId=physics_client)
    # # Set gravity
    p.setGravity(0, 0, -9.81)
    # Load URDFs
    # Load robot, target object and plane urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # # Let the simulation run a bit to initialize
    # for _ in range(10):
    #     p.stepSimulation()

    # # command joint speeds (only 6 joints)
    # p.setJointMotorControlArray(
    #     robot_id,
    #     [0, 1, 2, 3, 4, 5],
    #     controlMode=p.VELOCITY_CONTROL,
    #     targetVelocities=list(dqc_t),
    #     velocityGains=[1, 1, 2, 1, 1, 1],
    #     forces=[87, 87, 87, 87, 12, 12]
    # )
    # # TODO pay attention to number of repetition (e.g., use 24 for period 24*1/240*1000=100 [ms])
    # for _ in range(24):
    #     # default timestep is 1/240 second
    #     p.stepSimulation()

    # Define joint indices for movable joints
    # num_joints = p.getNumJoints(robot_id)
    # joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    #
    # # Set joint angles (you can use your actual desired joint configuration)
    # # joint_positions = [0.0 for _ in joint_indices]  # e.g. home pose
    # for i, j in enumerate(joint_indices):
    #     p.resetJointState(robot_id, j, q[i])
    #
    # # Get the end-effector link index (change to your actual EE link name if needed)
    # ee_link_index = joint_indices[-1]  # assuming last joint is the end-effector
    #
    # # Get current joint states
    # joint_states = p.getJointStates(robot_id, joint_indices)
    # joint_angles = [s[0] for s in joint_states]

    # # Get link state
    # link_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    # link_pos = link_state[0]

    for i in range(12):
        if i < 6:
            p.resetJointState(robot_id, i, q[i])
        else:
            p.resetJointState(robot_id, i, 0)

    # Compute Jacobian
    # J_lin, J_ang = p.calculateJacobian(
    #     bodyUniqueId=robot_id,
    #     linkIndex=ee_link_index,
    #     localPosition=[0.0, 0.0, 0.0],  # point on the link (usually origin of link frame)
    #     objPositions=joint_angles,
    #     objVelocities=zero_vec,
    #     objAccelerations=zero_vec
    # )
    # print("q=",q)
    # print("dq=",dq)
    LinkState = p.getLinkState(robot_id, 9, computeForwardKinematics=True, computeLinkVelocity=True)
    [J_lin, J_ang] = p.calculateJacobian(robot_id,
                                         10,
                                         list(LinkState[2]),
                                         list(q),
                                         list(dq),
                                         list(np.zeros(9)))

    # Convert to numpy arrays
    J_linear = np.array(J_lin)
    J_angular = np.array(J_ang)

    # Stack to get 6xN Jacobian (linear + angular)
    J_geo = np.vstack((J_linear, J_angular))

    # print("Geometric Jacobian (6xN):")
    # print(J_geo)
    return J_geo

urdf_path_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf"
urdf_path_biased_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_1.urdf"
robot_id_true = p.loadURDF(urdf_path_, useFixedBase=True)
robot_id_biased = p.loadURDF(urdf_path_biased_, useFixedBase=True)
q_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/q.npy")
dq_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/dq.npy")
rd_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/rd.npy")
r_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/r.npy")
drd_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/drd.npy")


# Assume the following arrays are loaded (length T = 136)
# Each element has the shape: J_true[k] and J_tilde[k] -> (3,6), u_d[k] -> (3,)
# You need to define or load these variables:
# J_true_list, J_tilde_list, u_d_list
# For demonstration: Create dummy data
T = 136
np.random.seed(0)
# J_true_list = [np.random.randn(3,6) for _ in range(T)]
# J_tilde_list = [J + 0.05*np.random.randn(3,6) for J in J_true_list]  # biased
# u_d_list = [np.random.randn(3) for _ in range(T)]
# Gain matrix
K_p = 0.1 * np.eye(3)
# Store outputs
error_bounds = []
e_r_ss_xyz = []
for k in range(T):
    # J_true = J_true_list[k]
    # J_tilde = J_tilde_list[k]
    q = np.hstack((q_[k, :], np.zeros(3)))
    dq = np.hstack((dq_[k, :], np.zeros(3)))
    J_true = load_jacobian(robot_id_true, q, dq)
    J = J_true[:3, :6]
    J_biased = load_jacobian(robot_id_biased, q, dq)
    J_tilde = J_biased[:3, :6]

    u_d = drd_[k,:]

    # Compute pseudoinverse
    J_tilde_pinv = np.linalg.pinv(J_tilde)

    # Operator P = J_true @ J_tilde_pinv
    P = J @ J_tilde_pinv

    # Identity
    I = np.eye(3)

    # Compute numerator: || (I - P) @ u_d ||
    num = np.linalg.norm((I - P) @ u_d)

    # Compute denominator: lambda_min of symmetric part of P @ K_p
    PKp_sym = (P @ K_p + (P @ K_p).T) / 2  # ensure symmetry
    eigvals = np.linalg.eigvalsh(PKp_sym)  # use eigvalsh for symmetric matrices
    valid_eigs = [eig for eig in eigvals if eig > 1e-6]
    lambda_min = np.min(valid_eigs) if valid_eigs else 1e-6

    # Error bound
    bound = num / lambda_min
    error_bounds.append(bound)

    # Steady-state position error direction (not the true value, just visualized)
    e_r_ss = np.linalg.pinv(P @ K_p) @ (I - P) @ u_d
    e_r_ss_xyz.append(e_r_ss)

e_r_ss_xyz = np.array(e_r_ss_xyz)  # shape (T, 3)
error_bounds = np.array(error_bounds)
# Plot total error bound
plt.figure(figsize=(8, 4))
plt.rcParams.update({
    'font.size': 14,  # overall font size
    'axes.labelsize': 16,  # x and y axis labels
    'xtick.labelsize': 12,  # x-axis tick labels
    'ytick.labelsize': 12,  # y-axis tick labels
    'legend.fontsize': 12,  # legend text
    'font.family': 'Serif'
})
plt.plot(error_bounds*1000, "-ob", label=r"$\|\mathbf{e}_r^{ss}\|$ bound")
plt.xlabel("k")
plt.ylabel("Error Bound [mm]")
plt.title("Steady-State Position Error Bound Magnitude")
plt.grid(True)
plt.legend()
plt.savefig(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/e_r_ss_upper_bound.pdf", format="pdf",
    bbox_inches='tight')

# Subplots for XYZ components
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
# plt.rcParams['font.family'] = 'Serif'
plt.rcParams.update({
    'font.size': 14,  # overall font size
    'axes.labelsize': 16,  # x and y axis labels
    'xtick.labelsize': 12,  # x-axis tick labels
    'ytick.labelsize': 12,  # y-axis tick labels
    'legend.fontsize': 12,  # legend text
    'font.family': 'Serif'
})
labels = ['x', 'y', 'z']
for i in range(3):
    axs[i].plot(e_r_ss_xyz[:, i]*1000, "-ob", label=rf"$\mathbf{{e}}_r^{{ss}}[{labels[i]}]$")
    axs[i].set_ylabel(f"{labels[i]} [mm]")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("k")
axs[0].set_title("Steady-State Position Error Vector Components")
plt.tight_layout()
plt.savefig(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/e_r_ss.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Simulated dummy input data — replace with your real ones
T = 136
# np.random.seed(0)
# J_true_list = [np.random.randn(3,6) for _ in range(T)]
# J_tilde_list = [J + 0.05*np.random.randn(3,6) for J in J_true_list]
# u_d_list = [np.random.randn(3) for _ in range(T)]

# Simulation time step
delta_t = 100/1000 #[s]

# Initialize logs
e_v_list = []
e_v_norm_list = []
e_r_list = []
e_r_norm_list = []
e_r_bound_list = []

# Initialize cumulative position error
e_r = np.zeros(3)
e_r_bound = 0.0

for k in range(T):
    q = np.hstack((q_[k, :], np.zeros(3)))
    dq = np.hstack((dq_[k, :], np.zeros(3)))
    J_true = load_jacobian(robot_id_true, q, dq)
    J = J_true[:3, :6]
    J_biased = load_jacobian(robot_id_biased, q, dq)
    J_tilde = J_biased[:3, :6]

    u_d = drd_[k,:]

    # Pseudoinverse of biased Jacobian
    J_tilde_pinv = np.linalg.pinv(J_tilde)
    P = J @ J_tilde_pinv
    I = np.eye(3)

    # Velocity error: e_v(k) = (I - P) u_d(k)
    e_v = (I - P) @ u_d
    e_v_norm = np.linalg.norm(e_v)

    # Update cumulative position error and bound
    e_r += e_v * delta_t
    e_r_bound += e_v_norm * delta_t

    # Log
    e_v_list.append(e_v)
    e_v_norm_list.append(e_v_norm)
    e_r_list.append(e_r.copy())
    e_r_norm_list.append(np.linalg.norm(e_r))
    e_r_bound_list.append(e_r_bound)

# Convert to arrays
e_r_array = np.array(e_r_list)
e_r_norm_list=np.array(e_r_norm_list)
e_r_bound_list=np.array(e_r_bound_list)
# ------------- Plot 1: Error Norm and Bound -------------
plt.figure(figsize=(8, 4))
plt.plot(e_r_norm_list*1000, label=r"$\|\mathbf{e}_r(k)\|$ (actual)", color='blue')
plt.plot(e_r_bound_list*1000, label=r"Upper Bound $\sum \|\mathbf{e}_v(j)\| \Delta t$", color='red', linestyle='--')
plt.xlabel("Timestep $k$")
plt.ylabel("Position Error Norm [mm]")
plt.title("Cumulative End-Effector Position Error and Theoretical Bound")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/e_r_cumulative_upper_bound.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

# ------------- Plot 2: Position Error in XYZ -------------
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
labels = ['x', 'y', 'z']
for i in range(3):
    axs[i].plot(e_r_array[:, i]*1000, "-o", label=rf"$e_{{r,{labels[i]}}}(k)$", color=f"C{i}")
    axs[i].set_ylabel(f"{labels[i]} [mm]")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("k")
axs[0].set_title("Cumulative End-Effector Position Error Components")
plt.tight_layout()
plt.savefig(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/e_r_cumulative.pdf", format="pdf",
    bbox_inches='tight')
plt.show()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Dummy data initialization – Replace with your real data
T = 136
np.random.seed(42)
J_true_list = [np.random.randn(3,6) for _ in range(T)]
J_tilde_list = [J + 0.05*np.random.randn(3,6) for J in J_true_list]
q_list = [np.random.randn(6) for _ in range(T)]
u_d_list = [np.random.randn(3) for _ in range(T)]
r_list = [np.random.randn(3) for _ in range(T)]
r_d_list = [r + 0.1*np.random.randn(3) for r in r_list]

# Gains
K_p = 0.1 * np.eye(3)
K_i = 0.01 * np.eye(3)

# Initialize integral of error
int_err = np.zeros(3)

# Logging
e_v_norms = []
e_v_bounds = []
e_v_components = []

for k in range(T):
    # J = J_true_list[k]
    # J_tilde = J_tilde_list[k]
    # u_d = u_d_list[k]
    # r = r_list[k]
    # r_d = r_d_list[k]
    q = np.hstack((q_[k, :], np.zeros(3)))
    dq = np.hstack((dq_[k, :], np.zeros(3)))
    J_true = load_jacobian(robot_id_true, q, dq)
    J = J_true[:3, :6]
    J_biased = load_jacobian(robot_id_biased, q, dq)
    J_tilde = J_biased[:3, :6]

    u_d = drd_[k,:]
    r_d = rd_[k,:]
    r = r_[k,:]


    # Compute control vector u(k)
    delta_r = r_d - r
    int_err += delta_r  # integral update
    u = u_d + K_p @ delta_r + K_i @ int_err

    # Pseudoinverse of biased Jacobian
    J_tilde_pinv = np.linalg.pinv(J_tilde)

    # Mismatch operator
    P = J @ J_tilde_pinv
    I = np.eye(3)

    # Actual velocity error: e_v = (I - P) u
    e_v = (I - P) @ u
    e_v_components.append(e_v)
    e_v_norms.append(np.linalg.norm(e_v))

    # Worst-case upper bound: (1 - sigma_min(P)) * ||u||
    sigma_min = np.min(np.linalg.svd(P, compute_uv=False))
    e_v_bound = (1 - sigma_min) * np.linalg.norm(u)
    e_v_bounds.append(e_v_bound)

# Convert to arrays
e_v_components = np.array(e_v_components)
e_v_norms=np.array(e_v_norms)
e_v_bounds=np.array(e_v_bounds)
# ------------------ Plot 1: Norm and Bound ------------------
plt.figure(figsize=(8, 8))
plt.rcParams.update({
    'font.size': 14,  # overall font size
    'axes.labelsize': 16,  # x and y axis labels
    'xtick.labelsize': 12,  # x-axis tick labels
    'ytick.labelsize': 12,  # y-axis tick labels
    'legend.fontsize': 12,  # legend text
    'font.family': 'Serif'
})
plt.subplot(2, 1, 1)
plt.plot(e_v_norms*1000, "-o", label=r"$\|\mathbf{e}_v(k)\|$", color='b')
plt.ylabel("Velocity Error Norm [mm/s]")
plt.grid(True)
plt.legend()
plt.title("Task-Space Velocity Error and Theoretical Bound")
plt.subplot(2, 1, 2)
plt.plot(e_v_bounds*1000, "-o",label=r"$(1 - \sigma_\min) \|\mathbf{u}(k)\|$", color='r')
plt.xlabel("k")
plt.ylabel("Upper Bound [mm/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/e_v_bounds.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

# ------------------ Plot 2: e_v components ------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
plt.rcParams.update({
    'font.size': 14,  # overall font size
    'axes.labelsize': 16,  # x and y axis labels
    'xtick.labelsize': 12,  # x-axis tick labels
    'ytick.labelsize': 12,  # y-axis tick labels
    'legend.fontsize': 12,  # legend text
    'font.family': 'Serif'
})
labels = ['vx', 'vy', 'vz']
for i in range(3):
    axs[i].plot(e_v_components[:, i]*1000,"-o", label=rf"$\mathbf{{e}}_v[{labels[i]}]$", color='C'+str(i))
    axs[i].set_ylabel(f"{labels[i]} [mm/s]")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Timestep $k$")
axs[0].set_title("Task-Space Velocity Error Components")
plt.tight_layout()
plt.savefig(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/e_v.pdf", format="pdf",
    bbox_inches='tight')
plt.show()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k in range(0,136,10):
    q = np.hstack((q_[k, :], np.zeros(3)))
    dq = np.hstack((dq_[k, :], np.zeros(3)))

    J_true = load_jacobian(robot_id_true, q, dq)
    J = J_true[:3, :6]
    J_biased = load_jacobian(robot_id_biased, q, dq)
    J_tilde = J_biased[:3, :6]

    # -----------------------------
    # Step 1: Setup true Jacobian and biased model
    # -----------------------------
    # np.random.seed(42)
    # J = np.random.randn(3, 6)
    # J_tilde = J + 0.8 * np.random.randn(3, 6)
    P = J @ pinv(J_tilde)  # mismatch operator (3x3)
    # -----------------------------
    # Step 2: Generate unit vectors on the 3D sphere
    # -----------------------------
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    unit_vectors = np.vstack((x.ravel(), y.ravel(), z.ravel()))  # shape: (3, N)

    # -----------------------------
    # Step 3: Compute deformed directions and errors
    # -----------------------------
    deformed = P @ unit_vectors
    errors = unit_vectors - deformed
    error_norms = norm(errors, axis=0)
    error_field = error_norms.reshape(x.shape)
    x_def = deformed[0].reshape(x.shape)
    y_def = deformed[1].reshape(y.shape)
    z_def = deformed[2].reshape(z.shape)

    if k % 10 == 0:
    # if k==50:
        # -----------------------------
        # Step 4: Plot 1 - Deformation of Unit Sphere
        # -----------------------------
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_surface(x, y, z, color='gray', alpha=0.2, label='Ideal')
        ax1.plot_surface(x_def, y_def, z_def, color='red', alpha=0.6)
        ax1.set_title("Deformed Unit Sphere under P = J J̃⁺")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=30, azim=45)
        # -----------------------------
        # Step 5: Plot 2 - Error Magnitude on Unit Sphere
        # -----------------------------
        ax2 = fig.add_subplot(222, projection='3d')
        surf = ax2.plot_surface(x, y, z, facecolors=plt.cm.viridis(error_field / np.max(error_field)),
                                rstride=1, cstride=1, antialiased=False, alpha=0.9)
        m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        m.set_array(error_field)
        fig.colorbar(m, ax=ax2, shrink=0.5, label='Tracking Error Magnitude')
        ax2.set_title("Tracking Error Magnitude Across Directions")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.view_init(elev=30, azim=45)
        # -----------------------------
        # Step 6: Plot 3 - Principal Singular Directions
        # -----------------------------
        U, S, Vh = svd(P)  # P = U Σ V^T
        origin = np.zeros(3)
        colors = ['r', 'g', 'b']
        labels = ['1st', '2nd', '3rd']
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.plot_surface(x, y, z, color='gray', alpha=0.2)
        for i in range(3):
            v = Vh[i, :]  # right singular vector (input)
            Pv = P @ v  # transformed direction
            ax3.quiver(*origin, *v, color=colors[i], linewidth=2, label=f"v{i + 1} (input dir)")
            ax3.quiver(*origin, *Pv, color=colors[i], linestyle='dashed', linewidth=2, alpha=0.5,
                       label=f"P v{i + 1} (achieved)")
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])
        ax3.set_zlim([-1, 1])
        ax3.set_title("Principal Singular Directions and Distortion")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.legend()


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fig2 = plt.figure(figsize=(6, 12))
        # -----------------------------
        ax2 = fig2.add_subplot(3, 1, 1)
        circle = np.linspace(0, 2 * np.pi, 200)
        u_yz = np.vstack((np.zeros_like(circle), np.cos(circle), np.sin(circle))) / 100  # XZ plane
        deformed_yz = P @ u_yz
        errors_yz = norm(u_yz - deformed_yz, axis=0)
        ax2.plot(u_yz[1], u_yz[2], 'k--', label='Ideal (unit circle)')
        ax2.plot(deformed_yz[1], deformed_yz[2], 'r-', label='Deformed')
        sc = ax2.scatter(u_yz[1], u_yz[2], c=errors_yz, cmap='viridis', label='Error Magnitude')
        plt.colorbar(sc, ax=ax2, label='Tracking Error')
        ax2.set_aspect('equal')
        ax2.set_title("YZ Plane Deformation & Error (k={})".format(k))
        ax2.set_xlabel("VY [m/s]")
        ax2.set_ylabel("VZ [m/s]")
        ax2.grid(True)
        ax2.quiver(0, 0, Vh[0,1]/np.linalg.norm(Vh[0,:]) / 100, Vh[0,2]/np.linalg.norm(Vh[0,:]) / 100, color='r', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v1/|v1|,yz) (σ1={S[0]:.4f})')
        ax2.quiver(0, 0, Vh[1,1]/np.linalg.norm(Vh[1,:]) / 100, Vh[1,2]/np.linalg.norm(Vh[1,:]) / 100, color='g', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v2/|v2|,yz) (σ2={S[1]:.4f})')
        ax2.quiver(0, 0, Vh[2,1]/np.linalg.norm(Vh[2,:]) / 100, Vh[2,2]/np.linalg.norm(Vh[2,:]) / 100, color='b', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v3/|v3|,yz) (σ3={S[2]:.4f})')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        # -----------------------------
        ax3 = fig2.add_subplot(3, 1, 2)
        circle = np.linspace(0, 2 * np.pi, 200)
        u_xz = np.vstack((np.cos(circle), np.zeros_like(circle), np.sin(circle))) / 100  # XZ plane
        deformed_xz = P @ u_xz
        errors_xz = norm(u_xz - deformed_xz, axis=0)
        ax3.plot(u_xz[0], u_xz[2], 'k--', label='Ideal (unit circle)')
        ax3.plot(deformed_xz[0], deformed_xz[2], 'r-', label='Deformed')
        sc = ax3.scatter(u_xz[0], u_xz[2], c=errors_xz, cmap='viridis', label='Error Magnitude')
        plt.colorbar(sc, ax=ax3, label='Tracking Error')
        ax3.set_aspect('equal')
        ax3.set_title("XZ Plane Deformation & Error (k={})".format(k))
        ax3.set_xlabel("VX [m/s]")
        ax3.set_ylabel("VZ [m/s]")
        ax3.grid(True)
        # Project first two right singular vectors onto XY plane and normalize for plotting
        v1_xz = Vh[0, :3]
        v2_xz = Vh[2, :3]
        ax3.quiver(0, 0, Vh[0,0]/np.linalg.norm(Vh[0,:]) / 100, Vh[0,2]/np.linalg.norm(Vh[0,:]) / 100, color='r', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v1/|v1|,xz) (σ1={S[0]:.4f})')
        ax3.quiver(0, 0, Vh[1,0]/np.linalg.norm(Vh[1,:]) / 100, Vh[1,2]/np.linalg.norm(Vh[1,:]) / 100, color='g', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v2/|v2|,xz) (σ2={S[1]:.4f})')
        ax3.quiver(0, 0, Vh[2,0]/np.linalg.norm(Vh[2,:]) / 100, Vh[2,2]/np.linalg.norm(Vh[2,:]) / 100, color='b', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v3/|v3|,xz) (σ3={S[2]:.4f})')
        ax3.legend(loc='upper right')
        plt.tight_layout()
        # -----------------------------
        ax4 = fig2.add_subplot(3, 1, 3)
        circle = np.linspace(0, 2 * np.pi, 200)
        u_xy = np.vstack((np.cos(circle), np.sin(circle), np.zeros_like(circle))) / 100  # XY plane
        deformed_xy = P @ u_xy
        errors_xy = norm(u_xy - deformed_xy, axis=0)
        ax4.plot(u_xy[0], u_xy[1], 'k--', label='Ideal (unit circle)')
        ax4.plot(deformed_xy[0], deformed_xy[1], 'r-', label='Deformed')
        sc = ax4.scatter(u_xy[0], u_xy[1], c=errors_xy, cmap='viridis', label='Error Magnitude')
        plt.colorbar(sc, ax=ax4, label='Tracking Error')
        ax4.set_aspect('equal')
        ax4.set_title("XY Plane Deformation & Error (k={})".format(k))
        ax4.set_xlabel("VX [m/s]")
        ax4.set_ylabel("VY [m/s]")
        ax4.grid(True)
        ax4.quiver(0, 0, Vh[0,0]/np.linalg.norm(Vh[0,:]) / 100, Vh[0,1]/np.linalg.norm(Vh[0,:]) / 100, color='r', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v1/|v1|,xy) (σ1={S[0]:.4f})')
        ax4.quiver(0, 0, Vh[1,0]/np.linalg.norm(Vh[1,:]) / 100, Vh[1,1]/np.linalg.norm(Vh[1,:]) / 100, color='g', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v2/|v2|,xy) (σ2={S[1]:.4f})')
        ax4.quiver(0, 0, Vh[2,0]/np.linalg.norm(Vh[2,:]) / 100, Vh[2,1]/np.linalg.norm(Vh[2,:]) / 100, color='b', angles='xy', scale_units='xy', scale=1,
                   label=f'proj(v3/|v3|,xy) (σ3={S[2]:.4f})')
        ax4.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/model_errors_k_{}.pdf".format(
                str(k)), format="pdf",
            bbox_inches='tight')
        plt.show()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def tracking_ellipsoid(J_true, J_biased):
#     # Compute projection matrix
#     P = J_true @ np.linalg.pinv(J_biased)
#
#     # Sample directions
#     directions = np.eye(J_true.shape[0])
#     transformed = P @ directions
#
#     # Visualize
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     for i in range(transformed.shape[1]):
#         vec = transformed[:, i]
#         ax.quiver(0, 0, 0, *vec, color='b', label="Achievable" if i == 0 else "")
#
#     ax.set_title("Image space of J_true @ pseudo-inv(J_biased)")
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     plt.legend()
#     plt.show()
#
#     return P
#
#
# def tracking_error_bound(J_true, J_biased):
#     P = J_true @ np.linalg.pinv(J_biased)
#     max_error_gain = norm(np.eye(P.shape[0]) - P, 2)
#     print("Worst-case relative error gain:", max_error_gain)
#     return max_error_gain
#
#
# # Sample 3x6 Jacobians: J_true is correct; J_biased is from incorrect model
# np.random.seed(42)
# J_true = np.random.randn(3, 6)
# J_biased = J_true + 0.1 * np.random.randn(3, 6)
#
# P = tracking_ellipsoid(J_true, J_biased)
# max_error_gain = tracking_error_bound(J_true, J_biased)
# # Compute pseudoinverses
# J_biased_pinv = np.linalg.pinv(J_biased)
# J_true_pinv = np.linalg.pinv(J_true)
#
# # Operator mapping desired to achieved velocities
# P = J_true @ J_biased_pinv
#
# # Singular value decomposition of P
# U, S, Vt = np.linalg.svd(P)
#
# # Visualize ellipsoid: how P distorts unit circle in task space
# theta = np.linspace(0, 2 * np.pi, 100)
# circle = np.vstack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
#
# # Only project 2D for visualization (project to first two singular vectors)
# ellipsoid = P @ circle
#
# plt.figure(figsize=(8, 6))
# plt.plot(circle[0], circle[1], 'k--', label='Ideal Command Directions (unit circle)')
# plt.plot(ellipsoid[0], ellipsoid[1], 'r-', label='Distorted Directions by $P$')
# plt.quiver(np.zeros(3), np.zeros(3), U[:, 0], U[:, 1], angles='xy', scale_units='xy', scale=1, color='blue',
#            label='Principal Directions')
# plt.axis('equal')
# plt.grid(True)
# plt.title("Effect of Kinematic Mismatch on Achievable Task-space Velocities")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()
#
# # Print singular values
# print("Singular values of P (tracking quality per direction):", S)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Simulated Jacobians (3x6: from joint velocities to Cartesian velocities)
# np.random.seed(0)
# J_true = np.random.randn(3, 6)
# J_biased = J_true + 0.3 * np.random.randn(3, 6)  # introduce bias
#
# # Compute pseudo-inverse
# J_true_pinv = np.linalg.pinv(J_true)
# J_biased_pinv = np.linalg.pinv(J_biased)
#
# # Unit joint velocity sphere in R^6
# num_samples = 500
# U = np.random.randn(6, num_samples)
# U /= np.linalg.norm(U, axis=0)  # normalize to unit length
#
# # Project joint velocities to task-space velocities
# v_true = J_true @ U
# v_biased = J_biased @ U
#
# # Plot: 3x1 subplots for X, Y, Z directions
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#
# directions = ['x', 'y', 'z']
# for i in range(3):
#     axes[i].scatter(v_true[i, :], v_biased[i, :], alpha=0.6)
#     axes[i].set_xlabel(f'True $v_{directions[i]}$ [m/s]')
#     axes[i].set_ylabel(f'Biased $v_{directions[i]}$ [m/s]')
#     axes[i].set_title(f'{directions[i].upper()}-Axis Task-Space Velocity Comparison')
#     axes[i].grid(True)
#     axes[i].axis('equal')
#
# plt.tight_layout()
# plt.show()
#
# # 3D plot of task-space velocity ellipsoids
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(v_true[0, :], v_true[1, :], v_true[2, :], color='blue', alpha=0.4, label='True Jacobian')
# ax.scatter(v_biased[0, :], v_biased[1, :], v_biased[2, :], color='red', alpha=0.4, label='Biased Jacobian')
# ax.set_xlabel('vx [m/s]')
# ax.set_ylabel('vy [m/s]')
# ax.set_zlabel('vz [m/s]')
# ax.set_title('Task-Space Velocity Achievability (3D)')
# ax.legend()
# plt.tight_layout()
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Define example Jacobians (3x6)
# np.random.seed(0)
# J_true = np.random.randn(3, 6)
# J_biased = J_true + 0.2 * np.random.randn(3, 6)  # simulate biased kinematics
#
# # Compute pseudoinverses
# J_true_pinv = np.linalg.pinv(J_true)
# J_biased_pinv = np.linalg.pinv(J_biased)
#
# # Unit sphere of joint velocity commands
# num_samples = 100
# joint_vel_samples = np.random.randn(6, num_samples)
# joint_vel_samples /= np.linalg.norm(joint_vel_samples, axis=0)
#
# # Map joint velocities to task space
# v_true = J_true @ joint_vel_samples
# v_biased = J_biased @ joint_vel_samples
#
# # --- PLOT 2D COMPONENTS ---
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# labels = ['X', 'Y', 'Z']
# units = '[m/s]'
#
# for i in range(3):
#     axes[i].scatter(range(num_samples), v_true[i, :], label='True Jacobian', alpha=0.7)
#     axes[i].scatter(range(num_samples), v_biased[i, :], label='Biased Jacobian', alpha=0.7)
#     axes[i].set_title(f'{labels[i]}-Component of Task Space Velocity')
#     axes[i].set_xlabel('Sample Index')
#     axes[i].set_ylabel(f'Velocity {units}')
#     axes[i].legend()
#     axes[i].grid(True)
#
# plt.suptitle('Task-Space Velocity Components Achieved by True vs. Biased Jacobian')
# plt.tight_layout()
# plt.show()
#
# # --- PLOT 3D COMPARISON ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(v_true[0], v_true[1], v_true[2], label='True Jacobian', alpha=0.6)
# ax.scatter(v_biased[0], v_biased[1], v_biased[2], label='Biased Jacobian', alpha=0.6)
# ax.set_xlabel('X [m/s]')
# ax.set_ylabel('Y [m/s]')
# ax.set_zlabel('Z [m/s]')
# ax.set_title('3D Task-Space Velocities')
# ax.legend()
# plt.tight_layout()
# plt.show()
#
# # Start PyBullet in DIRECT mode (no GUI) or GUI if you want visualization
# p.connect(p.DIRECT)  # or p.GUI
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
#
#
# def load_jacobian(urdf_path):
#     robot_id = p.loadURDF(urdf_path, useFixedBase=True)
#
#     # Let the simulation run a bit to initialize
#     for _ in range(10):
#         p.stepSimulation()
#
#     # Define joint indices for movable joints
#     num_joints = p.getNumJoints(robot_id)
#     joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
#
#     # Set joint angles (you can use your actual desired joint configuration)
#     joint_positions = [0.0 for _ in joint_indices]  # e.g. home pose
#     for i, j in enumerate(joint_indices):
#         p.resetJointState(robot_id, j, joint_positions[i])
#
#     # Get the end-effector link index (change to your actual EE link name if needed)
#     ee_link_index = joint_indices[-1]  # assuming last joint is the end-effector
#
#     # Get current joint states
#     joint_states = p.getJointStates(robot_id, joint_indices)
#     joint_angles = [s[0] for s in joint_states]
#
#     # Get link state
#     link_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
#     link_pos = link_state[0]
#
#     # Compute Jacobian
#     zero_vec = [0.0] * len(joint_indices)
#     J_lin, J_ang = p.calculateJacobian(
#         bodyUniqueId=robot_id,
#         linkIndex=ee_link_index,
#         localPosition=[0.0, 0.0, 0.0],  # point on the link (usually origin of link frame)
#         objPositions=joint_angles,
#         objVelocities=zero_vec,
#         objAccelerations=zero_vec
#     )
#
#     # Convert to numpy arrays
#     J_linear = np.array(J_lin)
#     J_angular = np.array(J_ang)
#
#     # Stack to get 6xN Jacobian (linear + angular)
#     J_geo = np.vstack((J_linear, J_angular))
#
#     print("Geometric Jacobian (6xN):")
#     print(J_geo)
#     return J_geo
#
#
# def analyze_jacobian(J, label='J'):
#     # SVD
#     U, S, Vt = svd(J)
#     print(f"Singular values of {label}: {S}")
#
#     # Image space: U[:, i] scaled by S[i]
#     vectors = U @ np.diag(S)
#
#     return vectors, S, U
#
#
# # Example: define 3x6 Jacobians (e.g., for a 6-DOF robot arm in XYZ task space)
# # np.random.seed(42)
# # J_true = np.random.randn(3, 6)
# # J_biased = J_true + 0.3 * np.random.randn(3, 6)  # add bias
# urdf_path_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf"
# J_true = load_jacobian(urdf_path_)
# J_true = J_true[:3, :6]
# urdf_path_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_3.urdf"
# J_biased = load_jacobian(urdf_path_)
# J_biased = J_biased[:3, :6]
# # Analyze
# vecs_true, S_true, U_true = analyze_jacobian(J_true, 'J_true')
# vecs_biased, S_biased, U_biased = analyze_jacobian(J_biased, 'J_biased')
#
# # Plot task-space ellipsoids
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# # Plot true Jacobian image
# origin = np.zeros(2)
# colors = ['r', 'g', 'b']
# for i in range(2):  # XY only
#     ax.quiver(*origin, vecs_true[0, i], vecs_true[1, i], angles='xy', scale_units='xy', scale=1, color=colors[i],
#               label=f'True U{i + 1}')
# # Plot biased Jacobian image
# for i in range(2):
#     ax.quiver(*origin, vecs_biased[0, i], vecs_biased[1, i], angles='xy', scale_units='xy', scale=1, linestyle='dashed',
#               color=colors[i], alpha=0.6, label=f'Biased U{i + 1}')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_aspect('equal')
# ax.grid(True)
# ax.set_title("Image space directions (XY plane)")
# ax.legend()
# plt.show()
#
# # Plot task-space ellipsoids
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# # Plot true Jacobian image
# origin = np.zeros(2)
# colors = ['r', 'g', 'b']
# for i in range(2):  # XY only
#     ax.quiver(*origin, vecs_true[0, i], vecs_true[2, i], angles='xy', scale_units='xy', scale=1, color=colors[i],
#               label=f'True U{i + 1}')
# # Plot biased Jacobian image
# for i in range(2):
#     ax.quiver(*origin, vecs_biased[0, i], vecs_biased[2, i], angles='xy', scale_units='xy', scale=1, linestyle='dashed',
#               color=colors[i], alpha=0.6, label=f'Biased U{i + 1}')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_aspect('equal')
# ax.grid(True)
# ax.set_title("Image space directions (XZ plane)")
# ax.legend()
# plt.show()
#
# # Plot task-space ellipsoids
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# # Plot true Jacobian image
# origin = np.zeros(2)
# colors = ['r', 'g', 'b']
# for i in range(2):  # XY only
#     ax.quiver(*origin, vecs_true[1, i], vecs_true[2, i], angles='xy', scale_units='xy', scale=1, color=colors[i],
#               label=f'True U{i + 1}')
# # Plot biased Jacobian image
# for i in range(2):
#     ax.quiver(*origin, vecs_biased[1, i], vecs_biased[2, i], angles='xy', scale_units='xy', scale=1, linestyle='dashed',
#               color=colors[i], alpha=0.6, label=f'Biased U{i + 1}')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_aspect('equal')
# ax.grid(True)
# ax.set_title("Image space directions (YZ plane)")
# ax.legend()
# plt.show()
#
#
# def plot_jacobian_ellipsoid(J, ax, color='b', label=''):
#     # SVD of J
#     U, S, Vh = np.linalg.svd(J, full_matrices=False)
#
#     # Build ellipsoid axes
#     radii = S
#     directions = U  # columns are principal directions in task space
#
#     # Sample unit sphere
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 100)
#     x = np.outer(np.cos(u), np.sin(v))
#     y = np.outer(np.sin(u), np.sin(v))
#     z = np.outer(np.ones_like(u), np.cos(v))
#
#     # Transform sphere to ellipsoid using singular values and vectors
#     ellipsoid = np.array([x, y, z])  # shape: (3, N, N)
#     for i in range(3):
#         ellipsoid[i] *= radii[i]
#
#     # Rotate the ellipsoid with U (task-space directions)
#     ellipsoid_reshaped = ellipsoid.reshape(3, -1)  # (3, N*N)
#     ellipsoid_rotated = directions @ ellipsoid_reshaped  # (3, N*N)
#     x_e, y_e, z_e = ellipsoid_rotated.reshape(3, x.shape[0], x.shape[1])
#
#     ax.plot_surface(x_e, y_e, z_e, rstride=4, cstride=4, alpha=0.3, color=color)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title(f"Jacobian Image Ellipsoid {label}")
#
#
# # Example Jacobian (3x4)
# J_biased = np.array([
#     [0.5, 0.1, 0.0, 0.0],
#     [0.0, 0.2, 0.3, 0.0],
#     [0.0, 0.0, 0.1, 0.4]
# ])
#
# # Plotting
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# plot_jacobian_ellipsoid(J_biased, ax, color='r', label='Biased Jacobian')
# plt.show()
