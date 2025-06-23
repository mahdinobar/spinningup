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
p.setPhysicsEngineParameter(enableConeFriction=0)

def load_jacobian(robot_id, q_t, dq_t):
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for i in range(12):
        if i < 6:
            p.resetJointState(robot_id, i, q[i])
        else:
            p.resetJointState(robot_id, i, 0)
    LinkState = p.getLinkState(robot_id, 9, computeForwardKinematics=True, computeLinkVelocity=True)
    [J_lin, J_ang] = p.calculateJacobian(robot_id,
                                         10,
                                         list(LinkState[2]),
                                         list(q),
                                         list(dq),
                                         list(np.zeros(9)))
    J_linear = np.array(J_lin)
    J_angular = np.array(J_ang)
    J_geo = np.vstack((J_linear, J_angular))
    return J_geo

urdf_path_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf"
urdf_path_biased_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_3.urdf"
robot_id_true = p.loadURDF(urdf_path_, useFixedBase=True)
robot_id_biased = p.loadURDF(urdf_path_biased_, useFixedBase=True)
q_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_q.npy")
dq_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_dq.npy")
rd_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_rd.npy")
r_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_r.npy")
drd_ = np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_drd.npy")


T = 136
K_p = 1 * np.eye(3)
K_i = 0.1 * np.eye(3)
# Store outputs
error_bounds = []
e_r_ss_xyz = []
for k in range(T):
    q = np.hstack((q_[k, :], np.zeros(3)))
    dq = np.hstack((dq_[k, :], np.zeros(3)))
    J_true = load_jacobian(robot_id_true, q, dq)
    J = J_true[:3, :6]
    J_biased = load_jacobian(robot_id_biased, q, dq)
    J_tilde = J_biased[:3, :6]
    u_d = drd_[k,:]
    J_tilde_pinv = np.linalg.pinv(J_tilde)
    P = J @ J_tilde_pinv
    I = np.eye(3)
    num = np.linalg.norm((I - P) @ u_d)
    PKp_sym = (P @ K_p + (P @ K_p).T) / 2  # ensure symmetry
    eigvals = np.linalg.eigvalsh(PKp_sym)  # use eigvalsh for symmetric matrices
    valid_eigs = [eig for eig in eigvals if eig > 1e-6]
    lambda_min = np.min(valid_eigs) if valid_eigs else 1e-6
    bound = num / lambda_min
    error_bounds.append(bound)
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
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_r_ss_upper_bound.pdf", format="pdf",
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
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_r_ss.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta_t = 100/1000 #[s]
e_v_list = []
e_v_norm_list = []
e_r_list = []
e_r_norm_list = []
e_r_bound_list = []
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
    J_tilde_pinv = np.linalg.pinv(J_tilde)
    P = J @ J_tilde_pinv
    I = np.eye(3)
    e_v = (I - P) @ u_d
    e_v_norm = np.linalg.norm(e_v)
    e_r += e_v * delta_t
    e_r_bound += e_v_norm * delta_t
    e_v_list.append(e_v)
    e_v_norm_list.append(e_v_norm)
    e_r_list.append(e_r.copy())
    e_r_norm_list.append(np.linalg.norm(e_r))
    e_r_bound_list.append(e_r_bound)

e_r_array = np.array(e_r_list)
e_r_norm_list=np.array(e_r_norm_list)
e_r_bound_list=np.array(e_r_bound_list)
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
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_r_cumulative_upper_bound.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

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
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_r_cumulative.pdf", format="pdf",
    bbox_inches='tight')
plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int_err = np.zeros(3)
e_v_norms = []
e_v_bounds = []
e_v_components = []
for k in range(T):
    q = np.hstack((q_[k, :], np.zeros(3)))
    dq = np.hstack((dq_[k, :], np.zeros(3)))
    J_true = load_jacobian(robot_id_true, q, dq)
    J = J_true[:3, :6]
    J_biased = load_jacobian(robot_id_biased, q, dq)
    J_tilde = J_biased[:3, :6]
    u_d = drd_[k,:]
    r_d = rd_[k,:]
    r = r_[k,:]
    delta_r = r_d - r
    int_err += delta_r  # integral update
    u = u_d + K_p @ delta_r + K_i @ int_err
    J_tilde_pinv = np.linalg.pinv(J_tilde)
    P = J @ J_tilde_pinv
    I = np.eye(3)
    e_v = (I - P) @ u
    e_v_components.append(e_v)
    e_v_norms.append(np.linalg.norm(e_v))
    sigma_min = np.min(np.linalg.svd(P, compute_uv=False))
    e_v_bound = (1 - sigma_min) * np.linalg.norm(u)
    e_v_bounds.append(e_v_bound)

e_v_components = np.array(e_v_components)
e_v_norms=np.array(e_v_norms)
e_v_bounds=np.array(e_v_bounds)
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
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_bounds.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

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
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v.pdf", format="pdf",
    bbox_inches='tight')
plt.show()

np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_bounds.npy", e_v_bounds)
np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_norms.npy", e_v_norms)
np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_components.npy", e_v_components)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k in range(136):
    if k == 50:
        q = np.hstack((q_[k, :], np.zeros(3)))
        dq = np.hstack((dq_[k, :], np.zeros(3)))
        J_true = load_jacobian(robot_id_true, q, dq)
        J = J_true[:3, :6]
        J_biased = load_jacobian(robot_id_biased, q, dq)
        J_tilde = J_biased[:3, :6]
        P = J @ pinv(J_tilde)  # mismatch operator (3x3)
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        unit_vectors = np.vstack((x.ravel(), y.ravel(), z.ravel()))  # shape: (3, N)
        deformed = P @ unit_vectors
        errors = unit_vectors - deformed
        error_norms = norm(errors, axis=0)
        error_field = error_norms.reshape(x.shape)
        x_def = deformed[0].reshape(x.shape)
        y_def = deformed[1].reshape(y.shape)
        z_def = deformed[2].reshape(z.shape)
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_surface(x, y, z, color='gray', alpha=0.2, label='Ideal')
        ax1.plot_surface(x_def, y_def, z_def, color='red', alpha=0.6)
        ax1.set_title("Deformed Unit Sphere under P = J J̃⁺")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=30, azim=45)
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
        plt.savefig(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_3Derrors_k_{}.pdf".format(
                str(k)), format="pdf",
            bbox_inches='tight')
        plt.show()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fig2 = plt.figure(figsize=(8, 14))
        # -----------------------------
        ax2 = fig2.add_subplot(3, 1, 1)
        circle = np.linspace(0, 2 * np.pi, 200)
        u_yz = np.vstack((np.zeros_like(circle), np.cos(circle), np.sin(circle))) / 100  # XZ plane
        deformed_yz = P @ u_yz
        errors_yz = norm(u_yz - deformed_yz, axis=0) * 1000
        ax2.plot(u_yz[1], u_yz[2], 'k--', label='$\mathbf{u}$')
        ax2.plot(deformed_yz[1], deformed_yz[2], color='m', linestyle='-',
                 label='proj($\dot{\mathbf{p}}_{\t{achieved}}$,yz)')
        sc = ax2.scatter(u_yz[1], u_yz[2], c=errors_yz, cmap='viridis')
        plt.colorbar(sc, ax=ax2, label='Tracking Error [mm/s]')
        ax2.set_aspect('equal')
        # ax2.set_title("YZ Plane Deformation & Error")
        ax2.set_xlabel("$\mathbf{u}_y$ [m/s]")
        ax2.set_ylabel("$\mathbf{u}_z$ [m/s]")
        ax2.grid(True)
        ax2.quiver(0, 0, Vh[0, 1] / np.linalg.norm(Vh[0, :]) / 100, Vh[0, 2] / np.linalg.norm(Vh[0, :]) / 100,
                   color='r', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_1$,$yz$) ($\sigma_1$={S[0]:.3f})')
        ax2.quiver(0, 0, Vh[1, 1] / np.linalg.norm(Vh[1, :]) / 100, Vh[1, 2] / np.linalg.norm(Vh[1, :]) / 100,
                   color='g', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_2$,$yz$) ($\sigma_2$={S[1]:.3f})')
        ax2.quiver(0, 0, Vh[2, 1] / np.linalg.norm(Vh[2, :]) / 100, Vh[2, 2] / np.linalg.norm(Vh[2, :]) / 100,
                   color='b', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_3$,$yz$) ($\sigma_3$={S[2]:.3f})')
        ax2.legend(loc='lower right')
        plt.tight_layout()
        ax3 = fig2.add_subplot(3, 1, 2)
        circle = np.linspace(0, 2 * np.pi, 200)
        u_xz = np.vstack((np.cos(circle), np.zeros_like(circle), np.sin(circle))) / 100  # XZ plane
        deformed_xz = P @ u_xz
        errors_xz = norm(u_xz - deformed_xz, axis=0) * 1000
        ax3.plot(u_xz[0], u_xz[2], 'k--', label='$\mathbf{u}$')
        ax3.plot(deformed_xz[0], deformed_xz[2], color='m', linestyle='-',
                 label='proj($\dot{\mathbf{p}}_{\t{achieved}}$,xz)')
        sc = ax3.scatter(u_xz[0], u_xz[2], c=errors_xz, cmap='viridis')
        plt.colorbar(sc, ax=ax3, label='Tracking Error [mm/s]')
        ax3.set_aspect('equal')
        # ax3.set_title("XZ Plane Deformation & Error")
        ax3.set_xlabel("$\mathbf{u}_x$ [m/s]")
        ax3.set_ylabel("$\mathbf{u}_z$ [m/s]")
        ax3.grid(True)
        v1_xz = Vh[0, :3]
        v2_xz = Vh[2, :3]
        ax3.quiver(0, 0, Vh[0, 0] / np.linalg.norm(Vh[0, :]) / 100, Vh[0, 2] / np.linalg.norm(Vh[0, :]) / 100,
                   color='r', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_1$,$xz$) ($\sigma_1$={S[0]:.3f})')
        ax3.quiver(0, 0, Vh[1, 0] / np.linalg.norm(Vh[1, :]) /
                   100, Vh[1, 2] / np.linalg.norm(Vh[1, :]) / 100,
                   color='g', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_2$,$xz$) ($\sigma_2$={S[1]:.3f})')
        ax3.quiver(0, 0, Vh[2, 0] / np.linalg.norm(Vh[2, :]) / 100, Vh[2, 2] / np.linalg.norm(Vh[2, :]) / 100,
                   color='b', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_3$,$xz$) ($\sigma_3$={S[2]:.3f})')
        ax3.legend(loc='lower right')
        plt.tight_layout()
        ax4 = fig2.add_subplot(3, 1, 3)
        circle = np.linspace(0, 2 * np.pi, 200)
        u_xy = np.vstack((np.cos(circle), np.sin(circle), np.zeros_like(circle))) / 100  # XY plane
        deformed_xy = P @ u_xy
        errors_xy = norm(u_xy - deformed_xy, axis=0) * 1000
        ax4.plot(u_xy[0], u_xy[1], 'k--', label='$\mathbf{u}$')
        ax4.plot(deformed_xy[0], deformed_xy[1], color='m', linestyle='-',
                 label='proj($\dot{\mathbf{p}}_{\t{achieved}}$,xy)')
        sc = ax4.scatter(u_xy[0], u_xy[1], c=errors_xy, cmap='viridis')
        plt.colorbar(sc, ax=ax4, label='Tracking Error [mm/s]')
        ax4.set_aspect('equal')
        # ax4.set_title("XY Plane Deformation & Error")
        ax4.set_xlabel("$\mathbf{u}_x$ [m/s]")
        ax4.set_ylabel("$\mathbf{u}_y$ [m/s]")
        ax4.grid(True)
        ax4.quiver(0, 0, Vh[0, 0] / np.linalg.norm(Vh[0, :]) / 100, Vh[0, 1] / np.linalg.norm(Vh[0, :]) / 100,
                   color='r', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_1$,$xy$) ($\sigma_1$={S[0]:.3f})')
        ax4.quiver(0, 0, Vh[1, 0] / np.linalg.norm(Vh[1, :]) / 100, Vh[1, 1] / np.linalg.norm(Vh[1, :]) / 100,
                   color='g', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_2$,$xy$) ($\sigma_2$={S[1]:.3f})')
        ax4.quiver(0, 0, Vh[2, 0] / np.linalg.norm(Vh[2, :]) / 100, Vh[2, 1] / np.linalg.norm(Vh[2, :]) / 100,
                   color='b', angles='xy', scale_units='xy', scale=1,
                   label=f'proj($v_3$, $xy$) ($\sigma_3$ = {S[2]:.3f})')
        ax4.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_model_errors_k_{}.pdf".format(
                str(k)), format="pdf",
            bbox_inches='tight')
        plt.show()
print("")