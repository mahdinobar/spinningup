
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import pybullet_data
from gym import core, spaces
from gym.utils import seeding
import sys
import gpytorch
import joblib
import torch
import warnings
from typing import Optional, Tuple
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

sys.path.append('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch')
from myKalmanFilter import KalmanFilter

__copyright__ = "Copyright 2024, IfA https://control.ee.ethz.ch/"
__credits__ = ["Mahdi Nobar"]
__author__ = "Mahdi Nobar from ETH Zurich <mnobar@ethz.ch>"

# Connect to physics client
physics_client = pb.connect(pb.DIRECT)

# TDOO ATTENTION how you choose dt
dt = 100e-3
dt_startup = 1e-3
dt_pb_sim = 1 / 240

# renderer = pb.ER_TINY_RENDERER  # p.ER_BULLET_HARDWARE_OPENGL
# _width = 224
# _height = 224
# _cam_dist = 1.3
# _cam_yaw = 15
# _cam_pitch = -30
# _cam_roll = 0
# camera_target_pos = [0.2, 0, 0.]
# _screen_width = 3840  # 1920
# _screen_height = 2160  # 1080
# physics_client = pb.connect(pb.GUI,
#                             options='--mp4fps=10 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (
#                                 _screen_width, _screen_height))
# # pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4,
# #                      "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/simulation.mp4")
# # Initialise debug camera angle
# pb.resetDebugVisualizerCamera(
#     cameraDistance=1.2,
#     cameraYaw=5,
#     cameraPitch=-30,
#     cameraTargetPosition=camera_target_pos,
#     physicsClientId=physics_client)

# # default timestep is 1/240 second (search fixedTimeStep)
pb.setTimeStep(timeStep=dt_pb_sim, physicsClientId=physics_client)
# # Set gravity
pb.setGravity(0, 0, -9.81, physicsClientId=physics_client)
# Load URDFs
# Load robot, target object and plane urdf
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

# pb.setPhysicsEngineParameter(numSolverIterations=50)  # Increase for better accuracy
# /home/mahdi/ETHZ/codes/spinningup
arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
                  useFixedBase=True, physicsClientId=physics_client)

arm_auxiliary_mismatch = pb.loadURDF(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
    useFixedBase=True, physicsClientId=physics_client)

# # Create the second physics client
# client_auxilary = pb.connect(pb.DIRECT)  # Use p.GUI for visualization
# pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_auxilary)
# pb.loadURDF("plane.urdf", physicsClientId=client_auxilary)
# arm_auxilary = pb.loadURDF(
#     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
#     useFixedBase=True, physicsClientId=client_auxilary)
# # # default timestep is 1/240 second (search fixedTimeStep)
# pb.setTimeStep(timeStep=dt_pb_sim, physicsClientId=client_auxilary)
# # # Set gravity
# pb.setGravity(0, 0, -9.81, physicsClientId=client_auxilary)
import os

# arm_biased_kinematics = pb.loadURDF(
#     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_3.urdf",
#     useFixedBase=True, physicsClientId=physics_client)
arm_biased_kinematics = pb.loadURDF(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_3.urdf",
    useFixedBase=True, physicsClientId=physics_client)
# arm_biased_kinematics = pb.loadURDF(
#     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_biased_kinematics_3.urdf",
#     useFixedBase=True)

# import os
# import rospkg
# import subprocess
# rospack = rospkg.RosPack()
# xacro_filename = os.path.join("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep2/robots/panda/panda.urdf.xacro")
# urdf_filename = os.path.join("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep2/robots/panda/panda.urdf")
# urdf = open(urdf_filename, "w")
#
# # Recompile the URDF to make sure it's up to date
# subprocess.call(['rosrun', 'xacro', 'xacro.py', xacro_filename], stdout=urdf)
#
#
# arm2 = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep2/robots/panda/panda.urdf.xacro",
#                   useFixedBase=True)
target_object = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                            useFixedBase=True, physicsClientId=physics_client)
conveyor_object = pb.loadURDF(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
    useFixedBase=True, physicsClientId=physics_client)
plane = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                    useFixedBase=True, physicsClientId=physics_client)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x_shape, likelihood):
        dummy_train_x = torch.empty(train_x_shape)
        dummy_train_y = torch.empty(train_x_shape[0])

        super().__init__(dummy_train_x, dummy_train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FepEnv(core.Env):
    """
    Two-link planar arm with two revolut joints (based on simplified models at book "A Mathematical Introduction to
Robotic Manipulation" by Murry et al.
    """

    def __init__(self):
        seed = 1
        # self.n = 0
        # reset seed(here is where seed is reset to count 0)
        np.random.seed(seed)
        self.seed(seed=seed)
        # TODO: reward params
        self.lpx = 600
        self.lpy = 600
        self.lpz = 600
        self.lv = 10
        self.lddqc = 1
        self.reward_eta_p = 1
        self.reward_eta_v = 0
        self.reward_eta_ddqc = 0
        # TODO: User defined linear position gain
        self.K_p = 1
        self.K_i = 0.1
        self.K_d = 0
        self.korque_noise_max = 0.  # TODO
        self.viewer = None
        self.state = None
        self.state_buffer = None
        self.k = 0
        # Attention: this is the tf final of startup phase
        self.xd_init = 0.5345
        self.yd_init = -0.2455
        self.zd_init = 0.1392
        # TDOO ATTENTION how you choose MAX_TIMESTEPS
        self.MAX_TIMESTEPS = 136  # maximum timesteps per episode
        # TODO Attention: just the dimension of the observation space is enforced. The data here is not used. If you need to enforce them then modify the code.
        # Attention just 6 DOF is simulated (7th DOF is disabled)
        high_s = np.array([200, 200, 200,
                           1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                           2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100,
                           2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100,
                           0.435, 0.435, 0.435, 0.435, 0.522, 0.522])
        low_s = -high_s
        self.observation_space = spaces.Box(low=low_s, high=high_s, dtype=np.float32)
        # Attention just 6 DOF is simulated (7th DOF is disabled)
        # Attention: limits of SAC actions
        high_a = 0.2 * np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100,
                                 2.6100])  # TODO Attention: limits should be the same otherwise modify sac code
        low_a = -high_a
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)

        # for i in range(6):
        #     input_scaler = joblib.load('input_scaler{}.pkl'.format(str(joint_number)))
        #     target_scaler_q = joblib.load('target_scaler_q{}.pkl'.format(str(joint_number)))
        #     target_scaler_dq = joblib.load('target_scaler_dq{}.pkl'.format(str(joint_number)))
        #     # Re-instantiate model and likelihood for q
        #     likelihood_q = gpytorch.likelihoods.GaussianLikelihood()
        #     model_q = GPModel(train_x_shape=(1, input_dim), likelihood=likelihood_q)
        #     # Load q model
        #     checkpoint_q = torch.load('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/gp_model_q{}.pth'.format(str(joint_number)))
        #     model_q.load_state_dict(checkpoint_q['model_state_dict'])
        #     likelihood_q.load_state_dict(checkpoint_q['likelihood_state_dict'])
        #     input_scaler = checkpoint_q['input_scaler']
        #     target_scaler_q = checkpoint_q['target_scaler']
        #     self.model_q.eval()
        #     self.likelihood_q.eval()
        #     # Repeat for dq
        #     likelihood_dq = gpytorch.likelihoods.GaussianLikelihood()
        #     model_dq = GPModel(train_x_shape=(1, input_dim), likelihood=likelihood_dq)
        #     checkpoint_dq = torch.load('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/gp_model_dq{}.pth'.format(str(joint_number)))
        #     model_dq.load_state_dict(checkpoint_dq['model_state_dict'])
        #     likelihood_dq.load_state_dict(checkpoint_dq['likelihood_state_dict'])
        #     target_scaler_dq = checkpoint_dq['target_scaler']
        #     self.model_dq.eval()
        #     self.likelihood_dq.eval()

        # Initialize lists to hold models and scalers for all joints
        self.input_scalers = []
        self.target_scalers_q = []
        # self.target_scalers_dq = []
        self.models_q = []
        self.likelihoods_q = []
        # self.models_dq = []
        self.likelihoods_dq = []
        # GP_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/trainOnSAC1and2PI1and2and3_testOnSAC3_trackingPhaseOnly/"
        # GP_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/trainOnSAC_1_2_3_testOnSAC_5_trackingPhaseOnly/"
        # GP_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/trainOnSAC_1_2_3_testOnSAC_5_trackingPhaseOnly/"
        # GP_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/trainOnSAC_1_2_3_testOnSAC_5_trackingPhaseOnly/"
        GP_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/trainOnSAC_1_2_3_testOnSAC_5_trackingPhaseOnly/"
        GP_input_dim = 2
        for joint_number in range(6):
            # Load scalers
            input_scaler = joblib.load(GP_dir + f'input_scaler{joint_number}.pkl')
            target_scaler_q = joblib.load(GP_dir + f'target_scaler_q{joint_number}.pkl')
            # target_scaler_dq = joblib.load(GP_dir+f'target_scaler_dq{joint_number}.pkl')

            # Instantiate and load model for q
            likelihood_q = gpytorch.likelihoods.GaussianLikelihood()

            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    # self.covar_module = gpytorch.kernels.ScaleKernel(
                    #     gpytorch.kernels.RBFKernel()
                    # )
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel() +
                        gpytorch.kernels.MaternKernel(nu=2.5)
                    )

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            # model_q = GPModel(train_x_shape=(1, GP_input_dim), likelihood=likelihood_q)
            train_x_placeholder = torch.zeros((1, GP_input_dim))
            train_y_placeholder = torch.zeros((1,))
            model_q = ExactGPModel(train_x_placeholder, train_y_placeholder, likelihood_q)
            checkpoint_q = torch.load(
                GP_dir + f'gp_model_q{joint_number}.pth')
            model_q.load_state_dict(checkpoint_q['model_state_dict'])
            likelihood_q.load_state_dict(checkpoint_q['likelihood_state_dict'])
            input_scaler = checkpoint_q['input_scaler']  # overwrite with trained one
            target_scaler_q = checkpoint_q['target_scaler']

            model_q.eval()
            likelihood_q.eval()

            device = torch.device('cpu')
            model_q.to(device)
            likelihood_q.to(device)
            # # Instantiate and load model for dq
            # likelihood_dq = gpytorch.likelihoods.GaussianLikelihood()
            # model_dq = GPModel(train_x_shape=(1, GP_input_dim), likelihood=likelihood_dq)
            # checkpoint_dq = torch.load(
            #     GP_dir+f'gp_model_dq{joint_number}.pth')
            # model_dq.load_state_dict(checkpoint_dq['model_state_dict'])
            # likelihood_dq.load_state_dict(checkpoint_dq['likelihood_state_dict'])
            # target_scaler_dq = checkpoint_dq['target_scaler']
            # model_dq.eval()
            # likelihood_dq.eval()

            # X_test=torch.tensor([[0.1734,2.6176]])
            # with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
            #     pred_q = likelihood_q(model_q(X_test))
            #     mean_q = pred_q.mean.numpy()

            # Append to lists
            self.input_scalers.append(input_scaler)
            self.target_scalers_q.append(target_scaler_q)
            # self.target_scalers_dq.append(target_scaler_dq)
            self.models_q.append(model_q)
            self.likelihoods_q.append(likelihood_q)
            # self.models_dq.append(model_dq)
            # self.likelihoods_dq.append(likelihood_dq)

    def pseudoInverseMat(self, A, ld):
        # Input: Any m-by-n matrix, and a damping factor.
        # Output: An n-by-m pseudo-inverse of the input according to the Moore-Penrose formula
        # Get the number of rows (m) and columns (n) of A
        [m, n] = np.shape(A)
        # Compute the pseudo inverse for both left and right cases
        if (m > n):
            # Compute the left pseudoinverse.
            pinvA = np.linalg.lstsq((A.T * A + ld * ld * np.eye(n, n)), A.T)[0]
        elif (m <= n):
            # Compute the right pseudoinverse.
            pinvA = np.linalg.lstsq((np.matmul(A, A.T) + ld * ld * np.eye(m, m)).T, A, rcond=None)[0].T
        return pinvA

    def q_command(self, r_ee, v_ee, Jpinv, rd, vd, edt, deltaT):
        """
        PID Traj Tracking Feedback Controller
        Inputs:
            r_ee          : current end effector position
            rd       : desired end effector position
            vd       : desired end effector velocity
            Jpinv : current pseudo inverse jacobian matrix
        Output: joint-space velocity command of the robot.
        """
        edt_new = (rd - r_ee) * deltaT
        edt = edt + edt_new
        v_command = vd + self.K_p * (rd - r_ee) + self.K_i * edt + self.K_d * (vd - v_ee)
        dqc = np.dot(Jpinv, v_command)
        return dqc, edt

    def f_logistic(self, x, l):
        H = 2
        return H / (math.e ** (x * l) + math.e ** (-x * l))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _damped_pinv(self, J: np.ndarray, lam: float = 1e-3) -> np.ndarray:
        J = np.asarray(J);
        m, n = J.shape
        if m <= n:
            JJt = J @ J.T
            return J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(m, dtype=J.dtype), np.eye(m, dtype=J.dtype))
        else:
            JtJ = J.T @ J
            return np.linalg.solve(JtJ + (lam ** 2) * np.eye(n, dtype=J.dtype), J.T)

    def lower_bound_best_direction_no_injection(self,
            J_true_k: np.ndarray,  # (m,n)
            J_bias_k: np.ndarray,  # (m,n)
            e_k: np.ndarray,  # (m,)   current error
            Kp: np.ndarray,  # (m,m)
            Ki: np.ndarray,  # (m,m)  (you may pass zeros to ignore integral penalty)
            m_k: np.ndarray,  # (m,)   integral state
            dt: float,
            pinv_damping: float = 1e-3,
            include_integral_penalty: bool = True
    ):
        """
        Computes max over singular directions of
          LB_i = alpha_i*|s_i| - beta*||e_perp,i|| - (integral penalty if enabled),
        with the reference/mismatch injection term omitted.

        Returns:
          LB_best : float
          best    : dict (details for winning direction)
          per_dir : list of dicts with per-direction diagnostics (alpha, beta, s_i, e_perp_i, required_s_for_positive, etc.)
        """
        # Shapes
        e_k = np.asarray(e_k).reshape(-1)
        Kp = np.asarray(Kp);
        Ki = np.asarray(Ki)
        m_k = np.asarray(m_k).reshape(-1)
        m = e_k.size
        assert J_true_k.shape[0] == m and J_bias_k.shape[0] == m
        assert Kp.shape == (m, m) and Ki.shape == (m, m)

        # P = J_true * (J_bias)^†
        Jb_dag = self._damped_pinv(np.asarray(J_bias_k), lam=pinv_damping)
        P = np.asarray(J_true_k) @ Jb_dag

        # SVD
        U, S, Vt = np.linalg.svd(P, full_matrices=True)
        sigma_max = float(S[0])
        P_norm2 = sigma_max
        Kp_norm2 = float(np.linalg.norm(Kp, 2))
        Ki_norm2 = float(np.linalg.norm(Ki, 2))
        m_norm = float(np.linalg.norm(m_k, 2))

        e_norm = float(np.linalg.norm(e_k))
        beta = 1.0 + dt * P_norm2 * Kp_norm2
        int_penalty = dt * P_norm2 * Ki_norm2 * m_norm if include_integral_penalty else 0.0

        LB_best = -np.inf
        best = None
        per_dir = []

        for i in range(m):
            sigma_i = float(S[i])
            u_i = U[:, i];
            v_i = Vt.T[:, i]
            c_i = float(u_i @ v_i)
            kappa_i = float(v_i @ (Kp @ v_i))  # Rayleigh along v_i
            s_i = float(v_i @ e_k)
            eperp_i = float(np.sqrt(max(0.0, e_norm ** 2 - s_i ** 2)))
            alpha_i = abs(c_i - dt * sigma_i * kappa_i)

            LB_i = alpha_i * abs(s_i) - beta * eperp_i - int_penalty

            # compute the |s_i| needed to make LB_i > 0 (helpful diagnostic)
            denom = max(alpha_i, 1e-12)
            required_s = (beta * eperp_i + int_penalty) / denom

            entry = dict(
                i=i, sigma=sigma_i, c=c_i, kappa=kappa_i,
                s=s_i, eperp=eperp_i, alpha=alpha_i, beta=beta,
                LB=LB_i, required_abs_s_for_positive=required_s,
                P_norm2=P_norm2, Kp_norm2=Kp_norm2, Ki_norm2=Ki_norm2, m_norm=m_norm,
                LBmm=LB_i*1000, B1mm=alpha_i * abs(s_i)*1000, B2mm=beta * eperp_i*1000, B3mm=int_penalty*1000,
            )
            per_dir.append(entry)
            if LB_i > LB_best:
                LB_best = LB_i
                best = entry

        return float(LB_best), best, per_dir

    # =========================
    # =========================

    def damped_pinv(self,J: np.ndarray, lam: float = 1e-2) -> np.ndarray:
        """Tikhonov-damped pseudoinverse."""
        J = np.asarray(J);
        m, n = J.shape
        if m <= n:
            JJt = J @ J.T
            return J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(m), np.eye(m))
        else:
            JtJ = J.T @ J
            return np.linalg.solve(JtJ + (lam ** 2) * np.eye(n), J.T)

    def H_accumulator(self,z: complex) -> complex:
        """H(z) = 1 / (1 - z^{-1}) evaluated on the unit circle."""
        return 1.0 / (1.0 - 1.0 / z)

    def build_S0_ES0(self,omega: float, dt: float, Kp: np.ndarray, Ki: np.ndarray, Delta: np.ndarray):
        """
        For one frequency ω, build:
          S0 = (I + G*C)^(-1),
          ES0 = E*S0 with E = -G * Delta * C,
        where G = dt * H(z),  C = Kp + Ki * (dt * H(z)).
        """
        m = Kp.shape[0]
        z = np.exp(1j * omega * dt)
        H = self.H_accumulator(z)  # complex scalar
        Gs = dt * H  # scalar
        C = Kp + Ki * (dt * H)  # (m,m) complex
        I = np.eye(m, dtype=complex)
        L0 = Gs * C
        S0 = np.linalg.inv(I + L0)
        E = -(Gs) * (Delta @ C)
        ES0 = E @ S0
        return S0, ES0

    def make_omega_grid(self,dt: float, N: int = 2048, omega_min: float = 1e-6) -> np.ndarray:
        """Uniform grid in [omega_min, π/dt] (exclude DC)."""
        return np.linspace(max(omega_min, 1e-9), np.pi / dt, N)

    def detrend_window(self,r_win: np.ndarray, dt: float, mode: str = 'mean') -> np.ndarray:
        """Detrend window by removing mean or best affine fit (per channel)."""
        if mode is None:
            return r_win
        r = np.asarray(r_win, dtype=float).copy()
        if mode == 'mean':
            r -= np.mean(r, axis=0, keepdims=True)
        elif mode == 'linear':
            T, m = r.shape
            t = np.arange(T, dtype=float).reshape(-1, 1)
            X = np.hstack([np.ones((T, 1)), t])
            for j in range(m):
                theta, *_ = np.linalg.lstsq(X, r[:, j:j + 1], rcond=None)
                r[:, j] -= (X @ theta).ravel()
        else:
            raise ValueError("detrend mode must be None|'mean'|'linear'")
        return r

    def choose_signal_band_from_window(self,r_win, dt,
                                       energy_keep= 0.95,
                                       force_min_omega= 0.0,
                                       min_bins= 1,
                                       omega_band= None):
        """
        Select Ω_sig from r_win (Tw x m), excluding DC. Two modes:
         - If omega_band=(ωmin, ωmax) is given, pick bins in that band (excluding DC).
         - Else, keep the smallest set of bins capturing 'energy_keep' of non-DC energy.
           If non-DC energy is ~0, force at least 'min_bins' bins above 'force_min_omega'.
        Returns: mask_pos (bool over rfft bins), omegas (rad/s)
        """
        r_win = np.asarray(r_win)
        T, m = r_win.shape
        R = np.fft.rfft(r_win, axis=0)  # (F, m)
        freqs = np.fft.rfftfreq(T, d=dt)  # Hz
        omegas = 2 * np.pi * freqs  # rad/s
        F = omegas.size

        # Manual band override
        if omega_band is not None:
            mask = (omegas >= omega_band[0]) & (omegas <= omega_band[1])
            mask[0] = False  # exclude DC
            return mask, omegas

        # Energy-based selection
        power = np.sum(np.abs(R) ** 2, axis=1)  # (F,)
        valid = np.arange(F) > 0  # exclude DC
        power_ndc = power[valid]
        if power_ndc.sum() <= 0:
            # Force a minimal non-empty band above cutoff
            mask = np.zeros(F, dtype=bool)
            above = np.where((valid) & (omegas >= max(force_min_omega, 1e-9)))[0]
            if above.size > 0:
                pick = above[:min(min_bins, above.size)]
                mask[pick] = True
            return mask, omegas

        order = np.argsort(power_ndc)[::-1]
        csum = np.cumsum(power_ndc[order])
        k = np.searchsorted(csum, energy_keep * power_ndc.sum()) + 1
        keep = np.sort(order[:k])

        mask = np.zeros(F, dtype=bool)
        candidates = np.where(valid)[0][keep]
        # Enforce minimum omega cutoff and min bins
        if force_min_omega > 0.0:
            candidates = candidates[omegas[candidates] >= force_min_omega]
        if candidates.size == 0:
            above = np.where((valid) & (omegas >= force_min_omega))[0]
            candidates = above[:min_bins]
        mask[candidates] = True
        return mask, omegas

    def band_limited_norm_time(self,r_win: np.ndarray, mask_pos: np.ndarray) -> float:
        """||r||_{2,Ωsig} via FFT masking and iFFT (Parseval)."""
        R = np.fft.rfft(r_win, axis=0)  # (F, m)
        R_masked = R * mask_pos[:, None]
        r_band = np.fft.irfft(R_masked, n=r_win.shape[0], axis=0)
        return float(np.linalg.norm(r_band))

    # =========================
    # Main per-step and trajectory functions
    # =========================

    def lower_bound_band_at_step(self, dt,
                                 Kp, Ki,
                                 J_true_k, J_bias_k,
                                 pstar_seq, w_seq,
                                 k,
                                 pinv_damping=1e-2,
                                 window_sec=1.0,
                                 energy_keep=0.95,
                                 use_global_sup_for_ES0=True,
                                 N_omega=2048,
                                 detrend='mean',
                                 force_min_omega=0.0,
                                 omega_band=None):
        """
        Compute the band-limited lower bound at step k using a rolling window.
        Returns: LB, alpha_Omega, info(dict)
        """
        Kp = np.asarray(Kp);
        Ki = np.asarray(Ki)
        # window
        Tw = max(2, int(round(window_sec / dt)))
        k0 = max(0, k - Tw + 1)
        r_win = pstar_seq[k0:k + 1] - w_seq[k0:k + 1]
        if r_win.shape[0] < 8:  # pad early steps for FFT stability
            pad = np.zeros((8 - r_win.shape[0], r_win.shape[1]))
            r_win = np.vstack([pad, r_win])
        r_win = self.detrend_window(r_win, dt, mode=detrend)

        # posture-frozen mismatch at k
        Jb_dag = self.damped_pinv(np.asarray(J_bias_k), lam=pinv_damping)
        P = np.asarray(J_true_k) @ Jb_dag
        Delta = np.eye(P.shape[0]) - P

        # pick Ω_sig
        mask_pos, omegas_pos = self.choose_signal_band_from_window(
            r_win, dt,
            energy_keep=energy_keep,
            force_min_omega=force_min_omega,
            min_bins=5,
            omega_band=omega_band
        )
        if not np.any(mask_pos):
            return 0.0, 0.0, dict(
                note="Ω_sig empty after selection",
                band_bins=0, r_band_norm=0.0,
                sigma_min_S0_band=0.0, ES0_sup=0.0,
                small_gain_ok=True, k0=k0, k1=k
            )

        # ||r||_{2,Ω}
        r_band_norm = self.band_limited_norm_time(r_win, mask_pos)

        # sigma_min(S0; Ω)
        sigma_min_S0 = np.inf
        for w in omegas_pos[mask_pos]:
            S0, _ = self.build_S0_ES0(w, dt, Kp.astype(complex), Ki.astype(complex), Delta.astype(complex))
            svals = np.linalg.svd(S0, compute_uv=False)
            sigma_min_S0 = min(sigma_min_S0, float(svals[-1]))

        # ||ES0||_∞
        if use_global_sup_for_ES0:
            omegas_sup = self.make_omega_grid(dt, N=N_omega)
        else:
            omegas_sup = omegas_pos[mask_pos]
        ES0_sup = 0.0
        for w in omegas_sup:
            _, ES0 = self.build_S0_ES0(w, dt, Kp.astype(complex), Ki.astype(complex), Delta.astype(complex))
            svals = np.linalg.svd(ES0, compute_uv=False)
            ES0_sup = max(ES0_sup, float(svals[0]))

        alpha_Omega = sigma_min_S0 / (1.0 + ES0_sup)
        LB = max(0.0, alpha_Omega * r_band_norm)

        info = dict(
            k0=k0, k1=k,
            band_bins=int(np.count_nonzero(mask_pos)),
            r_band_norm=r_band_norm,
            sigma_min_S0_band=sigma_min_S0,
            ES0_sup=ES0_sup,
            small_gain_ok=(ES0_sup < 1.0)
        )

        info['omegas_sel'] = omegas_pos[mask_pos]
        R = np.fft.rfft(r_win, axis=0)
        power_all = np.sum(np.abs(R) ** 2, axis=1)  # total power per bin
        info['power_sel'] = power_all[mask_pos]  # power of selected bins

        return LB, alpha_Omega, info

    def lower_bound_band_over_trajectory(self, dt,
                                             Kp, Ki,
                                             J_true_seq, J_bias_seq,
                                             pstar_seq, w_seq=None,
                                             pinv_damping=1e-2,
                                             window_sec=1.0,
                                             energy_keep=0.95,
                                             use_global_sup_for_ES0=True,
                                             N_omega=2048,
                                             detrend='mean',
                                             force_min_omega=0.0,
                                             omega_band=None):

        """
        Run the band-limited bound across all time steps.
        Returns: LB_seq (T,), alpha_seq (T,), infos (list of dicts)
        """
        pstar_seq = np.asarray(pstar_seq)
        if w_seq is None:
            w_seq = np.zeros_like(pstar_seq)
        T = pstar_seq.shape[0]
        LB_seq = np.zeros(T)
        alpha_seq = np.zeros(T)
        infos = []
        for k in range(T):
            LB, alpha, info = self.lower_bound_band_at_step(
                dt, Kp, Ki,
                np.asarray(J_true_seq[k]), np.asarray(J_bias_seq[k]),
                pstar_seq, w_seq, k,
                pinv_damping=pinv_damping,
                window_sec=window_sec,
                energy_keep=energy_keep,
                use_global_sup_for_ES0=use_global_sup_for_ES0,
                N_omega=N_omega,
                detrend=detrend,
                force_min_omega=force_min_omega,
                omega_band=omega_band
            )
            LB_seq[k] = LB
            alpha_seq[k] = alpha
            infos.append(info)
        return LB_seq, alpha_seq, infos

    def reset(self, signal=False):
        # if signal:
        #     print("hello")
        # "reset considering the startup phase on real system"
        restart_outer = False
        while True:
            # self.n += 1
            self.k = 0
            # Reset robot at the origin and move the target object to the goal position and orientation
            pb.resetBasePositionAndOrientation(
                arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]), physicsClientId=physics_client)
            # pb.resetBasePositionAndOrientation(
            #     arm_auxilary, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]), physicsClientId=client_auxilary)
            # ATTENTION: assumption that we use biased kinematics just for jacobian otherwise you need to consider the base offset below
            pb.resetBasePositionAndOrientation(
                arm_biased_kinematics, [100, 100, 100], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]),
                physicsClientId=physics_client)
            self.fixed_base_bias_arm_auxiliary_mismatch = 200
            pb.resetBasePositionAndOrientation(
                arm_auxiliary_mismatch,
                [self.fixed_base_bias_arm_auxiliary_mismatch, self.fixed_base_bias_arm_auxiliary_mismatch,
                 self.fixed_base_bias_arm_auxiliary_mismatch],
                pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]),
                physicsClientId=physics_client)
            # we add random normal noise with std of 0.25 [deg] and zero mean on all 6 joints
            # q_init is the inital condition of the startup phase
            self.q_init = np.array(
                [-0.22683544711236076, 0.4152892646837951, -0.2240776697835826, -2.029656763049754,
                 -0.1323494169192162,
                 2.433754967707292]) + np.random.normal(
                loc=0.0,
                scale=0.004363323,
                size=6)
            self.q_init_bias = 0.01 * np.random.normal(loc=0.0, scale=1, size=6) * np.array(
                [-0.22683544711236076, 0.4152892646837951, -0.2240776697835826, -2.029656763049754,
                 -0.1323494169192162, 2.433754967707292])
            # Reset joint at initial angles
            for i in range(6):
                pb.resetJointState(arm, i, self.q_init[i], physicsClientId=physics_client)
                pb.resetJointState(arm_biased_kinematics, i, self.q_init[i] + self.q_init_bias[i],
                                   physicsClientId=physics_client)
                pb.resetJointState(arm_auxiliary_mismatch, i, self.q_init[i], physicsClientId=physics_client)
            # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
            pb.resetJointState(arm, 7, 1.939142517407308, physicsClientId=physics_client)
            pb.resetJointState(arm_auxiliary_mismatch, 7, 1.939142517407308, physicsClientId=physics_client)
            pb.resetJointState(arm_biased_kinematics, 7, 1.939142517407308, physicsClientId=physics_client)
            for j in [6] + list(range(8, 12)):
                pb.resetJointState(arm, j, 0, physicsClientId=physics_client)
                pb.resetJointState(arm_biased_kinematics, j, 0, physicsClientId=physics_client)
                pb.resetJointState(arm_auxiliary_mismatch, j, 0, physicsClientId=physics_client)
            # Get end effector coordinates
            LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True,
                                        physicsClientId=physics_client)
            r_hat_t = np.asarray(LinkState[0])
            v_hat_t = np.asarray(LinkState[6])
            info = pb.getJointStates(arm, range(12), physicsClientId=physics_client)
            q_t, dq_t, tau_t = [], [], []
            for joint_info in info:
                q_t.append(joint_info[0])
                dq_t.append(joint_info[1])
                tau_t.append(joint_info[3])
            if abs(sum(self.q_init - q_t[:6])) > 1e-6:
                raise ValueError('shouldn\'t q_init be equal to q_t?!')

            # q_t[0:6]=q_t[0:6] + np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * 3.14 / 180
            # STARTUP PHASE Simulatiob
            # ATTENTION: startup phase should be at 1 [ms]
            pb.setTimeStep(timeStep=dt_startup, physicsClientId=physics_client)
            v_star_dir = np.array([self.xd_init, self.yd_init, self.zd_init]) - r_hat_t
            norm_v_star_dir = np.linalg.norm(v_star_dir)
            k_startup = 0
            xd_startup = r_hat_t[0]
            yd_startup = r_hat_t[1]
            zd_startup = r_hat_t[2]
            while np.linalg.norm(r_hat_t - np.array([self.xd_init, self.yd_init, self.zd_init])) > 0.001:
                v_star_dir_length = 34.9028 / (1 + np.exp(-0.04 * (k_startup - 250))) / 1000 - 34.9028 / (
                        1 + np.exp(-0.04 * (0 - 250))) / 1000;
                v_star = v_star_dir_length * (v_star_dir / norm_v_star_dir)
                vxd = v_star[0]
                vyd = v_star[1]
                vzd = v_star[2]
                deltax = vxd * dt_startup
                deltay = vyd * dt_startup
                deltaz = vzd * dt_startup
                xd_startup = xd_startup + deltax
                yd_startup = yd_startup + deltay
                zd_startup = zd_startup + deltaz
                rd_t = np.array([xd_startup, yd_startup, zd_startup])
                vd_t = np.array([vxd, vyd, vzd])
                if k_startup == 0:
                    self.edt = (rd_t - r_hat_t) * dt_startup

                info = pb.getJointStates(arm, range(12), physicsClientId=physics_client)
                q_t, dq_t, tau_t = [], [], []
                for joint_info in info:
                    q_t.append(joint_info[0])
                    dq_t.append(joint_info[1])
                    tau_t.append(joint_info[3])
                # q_t[0:6] = q_t[0:6] + np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * 3.14 / 180
                [linearJacobian, angularJacobian] = pb.calculateJacobian(arm,
                                                                         10,
                                                                         list(LinkState[2]),
                                                                         list(np.append(q_t[:6], [0, 0, 0])),
                                                                         list(np.append(dq_t[:6], [0, 0, 0])),
                                                                         list(np.zeros(9)),
                                                                         physicsClientId=physics_client)

                J_t = np.asarray(linearJacobian)[:, :6]
                Jpinv_t = self.pseudoInverseMat(J_t, ld=0.01)

                dqc_t_PID, self.edt = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t,
                                                     edt=self.edt,
                                                     deltaT=dt_startup)
                # ATTENTION
                dqc_t = dqc_t_PID
                # TODO check
                # command joint speeds (only 6 joints)
                pb.setJointMotorControlArray(
                    arm,
                    [0, 1, 2, 3, 4, 5],
                    controlMode=pb.VELOCITY_CONTROL,
                    targetVelocities=list(dqc_t),
                    velocityGains=[1, 1, 1, 1, 1, 1],
                    forces=[87, 87, 87, 87, 12, 12],
                    physicsClientId=physics_client
                )
                pb.stepSimulation(physicsClientId=physics_client)

                LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
                r_hat_t = np.array(LinkState[0])
                v_hat_t = np.array(LinkState[6])
                k_startup += 1
                # if np.linalg.norm(r_hat_t - np.array([self.xd_init, self.yd_init, self.zd_init])) < 0.002:
                #     print("+++++np.linalg.norm(r_hat_t - np.array([self.xd_init, self.yd_init, self.zd_init]))=",
                #           np.linalg.norm(r_hat_t - np.array([self.xd_init, self.yd_init, self.zd_init])))
                #     print(
                #         "-----np.array([self.xd_init, self.yd_init, self.zd_init])-np.array([xd_startup, yd_startup, zd_startup])",
                #         np.array([self.xd_init, self.yd_init, self.zd_init]) - np.array(
                #             [xd_startup, yd_startup, zd_startup]))
                #     print(
                #         "!!!!!np.linalg.norm(np.array([self.xd_init, self.yd_init, self.zd_init])-np.array([xd_startup, yd_startup, zd_startup]))",
                #         np.linalg.norm(np.array([self.xd_init, self.yd_init, self.zd_init]) - np.array(
                #             [xd_startup, yd_startup, zd_startup])))
                if k_startup > 5000:
                    print("took too long for startup phase! Repeat the reset.")
                    restart_outer = True
                    break  # exit the inner while loop
            if restart_outer == True:
                continue  # restart the reset
            else:
                break

        r_hat_tfp1_startup = r_hat_t
        # rd_tf_startup = rd_t
        rd_tf_startup = np.array([self.xd_init, self.yd_init, self.zd_init])

        # ATTENTION: here we do to keep self.dqc_PID for next step
        v_star_dir_length = 34.9028 / (1 + np.exp(-0.04 * (k_startup - 250))) / 1000 - 34.9028 / (
                1 + np.exp(-0.04 * (0 - 250))) / 1000;  # [m/s]
        v_star = v_star_dir_length * (v_star_dir / norm_v_star_dir)
        vxd = v_star[0]
        vyd = v_star[1]
        vzd = v_star[2]
        deltax = vxd * dt_startup
        deltay = vyd * dt_startup
        deltaz = vzd * dt_startup
        xd_startup = xd_startup + deltax
        yd_startup = yd_startup + deltay
        zd_startup = zd_startup + deltaz
        rd_t = np.array([xd_startup, yd_startup, zd_startup])
        vd_t = np.array([vxd, vyd, vzd])
        info = pb.getJointStates(arm, range(12), physicsClientId=physics_client)
        q_t, dq_t, tau_t = [], [], []
        for joint_info in info:
            q_t.append(joint_info[0])
            dq_t.append(joint_info[1])
            tau_t.append(joint_info[3])
        # q_t[0:6]=q_t[0:6] + np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * 3.14 / 180
        [linearJacobian, angularJacobian] = pb.calculateJacobian(arm,
                                                                 10,
                                                                 list(LinkState[2]),
                                                                 list(np.append(q_t[:6], [0, 0, 0])),
                                                                 list(np.append(dq_t[:6], [0, 0, 0])),
                                                                 list(np.zeros(9)), physicsClientId=physics_client)

        J_t = np.asarray(linearJacobian)[:, :6]

        [linearJacobian_biased_, angularJacobianbiased_] = pb.calculateJacobian(arm_biased_kinematics,
                                                                 10,
                                                                 list(LinkState[2]),
                                                                 list(np.append(q_t[:6], [0, 0, 0])),
                                                                 list(np.append(dq_t[:6], [0, 0, 0])),
                                                                 list(np.zeros(9)), physicsClientId=physics_client)

        J_t_biased_ = np.asarray(linearJacobian_biased_)[:, :6]
        J_t = np.asarray(linearJacobian)[:, :6]
        Jpinv_t = self.pseudoInverseMat(J_t, ld=0.01)
        # ATTENTION: here we calculate the self.dqc_PID ready but we do not step simulation, and keep it for "step" to set with a
        self.dqc_PID, self.edt = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t,
                                                edt=self.edt,
                                                deltaT=dt_startup)

        q_t = np.array(q_t)[:6]
        # keep q_t in q_tp0_ for manual corrections of q simulations
        self.q_tp0_ = q_t
        # add dq measurement noise
        dq_t = np.array(dq_t)[:6] + np.random.normal(loc=0.0, scale=0.004, size=6)
        # add tau measurement noise and bias
        tau_t = np.array(tau_t)[:6]  # + np.random.normal(loc=0.0, scale=0.08, size=6) #+ np.array(
        # [0.31, 9.53, 1.76, -9.54, 0.89, -2.69])
        self.q = q_t.reshape(1, 6)
        self.dq = dq_t.reshape(1, 6)
        pb.resetBasePositionAndOrientation(
            target_object, rd_t, pb.getQuaternionFromEuler(
                np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])),
            physicsClientId=physics_client)  # orient just for rendering
        # set conveyer pose and orient
        pb.resetBasePositionAndOrientation(
            conveyor_object,
            np.array([self.xd_init, self.yd_init, self.zd_init]) + np.array([-0.002, -0.18, -0.15]),
            pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]), physicsClientId=physics_client)

        self.state = [(r_hat_tfp1_startup[0] - rd_tf_startup[0]) * 1000,
                      # ATTENTION: because of assumption that on real system we start Kalman filter with final position of EE at startup phase
                      (r_hat_tfp1_startup[1] - rd_tf_startup[1]) * 1000,
                      # ATTENTION: because of assumption that on real system we start Kalman filter with final position of EE at startup phase
                      (r_hat_tfp1_startup[2] - rd_tf_startup[2]) * 1000,
                      q_t[0],
                      q_t[1],
                      q_t[2],
                      q_t[3],
                      q_t[4],
                      q_t[5],
                      dq_t[0],
                      dq_t[1],
                      dq_t[2],
                      dq_t[3],
                      dq_t[4],
                      dq_t[5],
                      dqc_t_PID[0],
                      dqc_t_PID[1],
                      dqc_t_PID[2],
                      dqc_t_PID[3],
                      dqc_t_PID[4],
                      dqc_t_PID[5],
                      0,
                      0,
                      0,
                      0,
                      0,
                      0]
        self.state_buffer = self.state

        # # Add noise to target speed
        # self.vxd = 0 + np.random.normal(loc=0.0, scale=0.000367647, size=1)[
        #     0]  # [m/s] for 0.5 [cm] drift given std error after 13.6 [s]
        # self.vyd = 34.9028e-3 + np.random.normal(loc=0.0, scale=0.002205882, size=1)[
        #     0]  # [m/s] for 3 [cm] drift given std error after 13.6 [s]
        # self.vzd = 0  # m/s
        # print("INFO: added noise to target speed in x and y directions!")
        # deltax = self.vxd * dt * self.MAX_TIMESTEPS
        # deltay = self.vyd * dt * self.MAX_TIMESTEPS
        # deltaz = self.vzd * dt * self.MAX_TIMESTEPS
        # self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
        # self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
        # # Add noise to target z position
        # self.zd = np.linspace(self.zd_init, self.zd_init + deltaz, self.MAX_TIMESTEPS,
        #                       endpoint=True) + np.random.normal(loc=0.0, scale=0.0005, size=self.MAX_TIMESTEPS)

        # self.vxd = np.zeros((self.MAX_TIMESTEPS)) + np.random.normal(loc=0.0, scale=0.000367647, size=1)[
        #     0]  # [m/s] for 2 [cm] drift given std error after 13.6 [s]
        # self.vyd = 34.9028e-3 + np.zeros((self.MAX_TIMESTEPS)) + np.random.normal(loc=0.0, scale=0.002205882, size=1)[
        #     0]  # [m/s] for 5 [cm] drift given std error after 13.6 [s]
        # self.vzd = np.zeros((self.MAX_TIMESTEPS))
        # self.xd = np.zeros((self.MAX_TIMESTEPS))
        # self.yd = np.zeros((self.MAX_TIMESTEPS))
        # self.zd = np.zeros((self.MAX_TIMESTEPS))
        # self.xd[0] = self.xd_init
        # self.yd[0] = self.yd_init
        # self.zd[0] = self.zd_init
        #
        # # TODO: improve
        # # add artificial uncertainty to resemble the KF camera data uncertainty on real system
        # self.vxd[0] = 0 + np.random.normal(loc=0.0, scale=0.00289, size=1)[0]
        # self.vyd[0] = 34.9028e-3 + np.random.normal(loc=0.0, scale=0.000376, size=1)[0]
        # self.vzd[0] = 0 + np.random.normal(loc=0.0, scale=0.000174, size=1)[0]  # m/s
        # rand_idx = np.random.randint(0, self.MAX_TIMESTEPS, size=50)
        # for i in range(0, self.MAX_TIMESTEPS - 1):
        #     if i in rand_idx:
        #         self.vxd[i + 1] = 0 + np.random.normal(loc=0.0, scale=0.0032, size=1)[0]
        #         self.vyd[i + 1] = 34.9028e-3 + np.random.normal(loc=0.0, scale=0.000376, size=1)[0]
        #         self.vzd[i + 1] = 0 + np.random.normal(loc=0.0, scale=0.000174, size=1)[0]  # m/s
        #     else:
        #         self.vxd[i + 1] = 0
        #         self.vyd[i + 1] = 34.9028e-3
        #         self.vzd[i + 1] = 0  # m/s
        #     # TODO: check and improve
        #     self.xd[i + 1] = self.xd[i] + self.vxd[i] * dt  # + np.random.normal(loc=0.0, scale=0.0005, size=1)
        #     self.yd[i + 1] = self.yd[i] + self.vyd[i] * dt  # + np.random.normal(loc=0.0, scale=0.0005, size=1)
        #     self.zd[i + 1] = self.zd[i] + self.vzd[i] * dt + np.random.normal(loc=0.0, scale=0.0005, size=1)

        # xd_init = 0.5345
        # yd_init = -0.2455
        # zd_init = 0.1392
        x0 = np.array([self.xd_init, self.yd_init, self.zd_init])  # [m]

        # define the system matrices - Newtonian system
        # system matrices and covariances
        A = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B = np.array([[0], [1], [0]])
        C = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # # measurement noise covariance
        # R = np.array([[2 ** 2, 0, 0], [0, 5 ** 2, 0], [0, 0, 2 ** 2]])
        # # process uncertainty covariance
        # Q = np.array([[1 ** 2, 0, 0], [0, 2 ** 2, 0], [0, 0, 1 ** 2]])
        # # initial covariance matrix
        # P0 = np.asmatrix(np.diag([1, 4, 1]))

        # measurement noise covariance
        # R = np.array([[0.0625e-6, 0, 0], [0, 0.0625e-6, 0], [0, 0, 0.0625e-6]])  # [m^2]
        R = np.array([[0.25e-6, 0, 0], [0, 36e-6, 0], [0, 0, 0.01e-6]])  # [m^2]
        # process uncertainty covariance
        # Q = np.array([[0.01e-6, 0, 0], [0, 0.04e-6, 0], [0, 0, 0.02e-6]])  # #[m^2]
        Q = np.array([[0.01e-6, 0, 0], [0, 0.04e-6, 0], [0, 0, 0.04e-6]])  # #[m^2]
        # initial covariance matrix
        # P0 = np.asmatrix(np.diag([0.04e-6, 0.09e-6, 0.04e-6]))
        P0 = np.asmatrix(np.diag([0.01e-6, 0.04e-6, 0.01e-6]))

        # Generate time stamp randomness of camera measurements
        time_randomness = np.random.normal(0, 32, 137).astype(int)
        time_randomness = np.clip(time_randomness, -49, 49)
        time_randomness[0] = np.clip(time_randomness[0], 1, 49)
        tVec_camera = np.linspace(0, 13600, 137) + time_randomness  # [ms]
        self.vxd = 0  # [m/ms]
        self.vyd = (34.9028e-3 + np.random.normal(loc=0.0, scale=0.00005077, size=1)[
            0]) / 1000  # [m/ms]
        self.vzd = 0
        x_camera = np.zeros((self.MAX_TIMESTEPS + 1))
        y_camera = np.zeros((self.MAX_TIMESTEPS + 1))
        z_camera = np.zeros((self.MAX_TIMESTEPS + 1))

        dt_camera = np.hstack((tVec_camera[0], np.diff(tVec_camera)))  # [ms]
        x_camera[0] = self.xd_init + self.vxd * dt_camera[0]
        y_camera[0] = self.yd_init + self.vyd * dt_camera[0]
        z_camera[0] = self.zd_init + self.vzd * dt_camera[0]
        for i in range(0, self.MAX_TIMESTEPS):
            x_camera[i + 1] = x_camera[i] + self.vxd * dt_camera[i + 1]  # [m]
            y_camera[i + 1] = y_camera[i] + self.vyd * dt_camera[i + 1]  # [m]
            z_camera[i + 1] = z_camera[i] + self.vzd * dt_camera[i + 1]  # [m]
        x_camera = x_camera + np.random.normal(loc=0.0, scale=0.0006,
                                               size=137)  # + np.random.normal(loc=0.0, scale=0.0005, size=137)  # [m]
        y_camera = y_camera + np.random.normal(loc=0.0, scale=0.006,
                                               size=137)  # + np.random.normal(loc=0.0, scale=0.001, size=137)  # [m]
        z_camera = z_camera + np.random.normal(loc=0.0, scale=0.0004,
                                               size=137)  # + np.random.normal(loc=0.0, scale=0.0005, size=137)  # [m]
        X_camera = np.array([x_camera, y_camera, z_camera])

        # create a Kalman filter object
        KalmanFilterObject = KalmanFilter(x0, P0, A, B, C, Q, R)
        u = np.array([self.vxd, self.vyd, self.vzd])
        # simulate online prediction
        for k_measured in range(0, np.size(tVec_camera)):  # np.arange(np.size(tVec_camera)):
            # print(k_measured)
            # TODO correct for the online application where dt is varying and be know the moment we receive the measurement
            dt = dt_camera[k_measured]
            KalmanFilterObject.B = np.array([[dt], [dt], [dt]])
            KalmanFilterObject.propagateDynamics(u)
            KalmanFilterObject.B = np.array([[1], [1], [1]])
            KalmanFilterObject.prediction_aheads(u, dt)
            KalmanFilterObject.computeAposterioriEstimate(X_camera[:, k_measured])

        # extract the state estimates in order to plot the results
        x_hat = []
        y_hat = []
        z_hat = []

        for j in range(0, np.size(tVec_camera)):
            # python estimates
            x_hat.append(KalmanFilterObject.estimates_aposteriori[0, j])
            y_hat.append(KalmanFilterObject.estimates_aposteriori[1, j])
            z_hat.append(KalmanFilterObject.estimates_aposteriori[2, j])

        td = np.linspace(0, 13600, 137)
        xd = np.asarray(KalmanFilterObject.X_prediction_ahead[0, :]).squeeze()
        self.xd = xd[td[:-1].astype(int)]
        yd = np.asarray(KalmanFilterObject.X_prediction_ahead[1, :]).squeeze()
        self.yd = yd[td[:-1].astype(int)]
        zd = np.asarray(KalmanFilterObject.X_prediction_ahead[2, :]).squeeze()
        self.zd = zd[td[:-1].astype(int)]

        # ATTENTION set back simulation frequency after startup phase
        pb.setTimeStep(timeStep=dt_pb_sim, physicsClientId=physics_client)

        self.plot_data_t = [r_hat_t[0],
                            r_hat_t[1],
                            r_hat_t[2],
                            rd_t[0],
                            rd_t[1],
                            rd_t[2],
                            v_hat_t[0],
                            v_hat_t[1],
                            v_hat_t[2],
                            vd_t[0],
                            vd_t[1],
                            vd_t[2],
                            dqc_t_PID[0],
                            dqc_t_PID[1],
                            dqc_t_PID[2],
                            dqc_t_PID[3],
                            dqc_t_PID[4],
                            dqc_t_PID[5],
                            0,
                            0,
                            0,
                            tau_t[0],
                            tau_t[1],
                            tau_t[2],
                            tau_t[3],
                            tau_t[4],
                            tau_t[5],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            dqc_t_PID[0],
                            dqc_t_PID[1],
                            dqc_t_PID[2],
                            dqc_t_PID[3],
                            dqc_t_PID[4],
                            dqc_t_PID[5],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0]
        # # uncomment for jacobian_analysis
        # plot_data_t = [q_t[0],
        #                q_t[1],
        #                q_t[2],
        #                q_t[3],
        #                q_t[4],
        #                q_t[5],
        #                dq_t[0],
        #                dq_t[1],
        #                dq_t[2],
        #                dq_t[3],
        #                dq_t[4],
        #                dq_t[5],
        #                rd_t[0],
        #                rd_t[1],
        #                rd_t[2],
        #                vd_t[0],
        #                vd_t[1],
        #                vd_t[2],
        #                r_hat_t[0],
        #                r_hat_t[1],
        #                r_hat_t[2],
        #                v_hat_t[0],
        #                v_hat_t[1],
        #                v_hat_t[2],
        #                dqc_t_PID[0],
        #                dqc_t_PID[1],
        #                dqc_t_PID[2],
        #                dqc_t_PID[3],
        #                dqc_t_PID[4],
        #                dqc_t_PID[5],
        #                0,
        #                0,
        #                0,
        #                tau_t[0],
        #                tau_t[1],
        #                tau_t[2],
        #                tau_t[3],
        #                tau_t[4],
        #                tau_t[5],
        #                0,
        #                0,
        #                0,
        #                0,
        #                0,
        #                0]

        self.plot_data_buffer = self.plot_data_t

        self.J_true_seq = []
        self.J_bias_seq = []
        self.J_true_seq.append(np.asarray(J_t))
        self.J_bias_seq.append(np.asarray(J_t_biased_))

        return self.state

    def step(self, a):
        # print("0")
        # dqc_t_PID = self.state[21:27]
        # ATTENTION: here apply SAC action
        dqc_t = self.dqc_PID + a
        # TODO check
        # command joint speeds (only 6 joints)
        pb.setJointMotorControlArray(
            arm,
            [0, 1, 2, 3, 4, 5],
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocities=list(dqc_t),
            velocityGains=[1, 1, 2, 1, 1, 1],
            forces=[87, 87, 87, 87, 12, 12],
            physicsClientId=physics_client
        )
        pb.setJointMotorControlArray(
            arm_auxiliary_mismatch,
            [0, 1, 2, 3, 4, 5],
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocities=list(dqc_t),
            velocityGains=[1, 1, 2, 1, 1, 1],
            forces=[87, 87, 87, 87, 12, 12],
            physicsClientId=physics_client
        )
        # TODO pay attention to number of repetition (e.g., use 24 for period 24*1/240*1000=100 [ms])
        # for _ in range(24):
        #     # default timestep is 1/240 second
        #     pb.stepSimulation(physicsClientId=physics_client)
        pb.stepSimulation(physicsClientId=physics_client)
        # print("1")
        # update time index
        self.k += 1  # Attention doublecheck
        rd_tp1 = np.array(
            [self.xd[self.k], self.yd[self.k], self.zd[self.k]])  # [m] attention: index desired starts from t=-1
        vd_tp1 = np.array([self.vxd, self.vyd, self.vzd]) * 1000  # [m/s]
        pb.resetBasePositionAndOrientation(
            target_object, rd_tp1, pb.getQuaternionFromEuler(
                np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])), physicsClientId=physics_client)
        # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
        info = pb.getJointStates(arm, range(12), physicsClientId=physics_client)
        q_tp1_, dq_tp1, tau_tp1 = [], [], []
        for joint_info in info:
            q_tp1_.append(joint_info[0])
            dq_tp1.append(joint_info[1])
            tau_tp1.append(joint_info[3])

        # ###############################################################################################################
        # # ----- add q Mismatch Compensation -----
        # # for i in range(6):
        # for i in [0,2]:
        #     self.models_q[i].eval()
        #     self.likelihoods_q[i].eval()
        #     X_test = np.array([q_tp1[i], dq_tp1[i]]).reshape(-1, 2)
        #     X_test = self.input_scalers[i].transform(X_test)
        #     X_test = torch.tensor(X_test, dtype=torch.float32)
        #     device = torch.device('cpu')
        #     X_test = X_test.to(device)
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         self.models_q[i].to(device)
        #         self.likelihoods_q[i].to(device)
        #         pred_q = self.likelihoods_q[i](self.models_q[i](X_test))
        #         mean_q = pred_q.mean.numpy()
        #         std_q = pred_q.variance.sqrt().numpy()
        #         # Uncomment when Normalizing
        #         mean_q = self.target_scalers_q[i].inverse_transform(mean_q.reshape(-1, 1)).flatten()
        #         std_q = std_q * self.target_scalers_q[i].scale_[0]
        #     # TODO
        #     if ~np.isnan(mean_q):
        #
        #         q_tp1[i] = q_tp1[i] + mean_q[0]
        #     else:
        #         print("mean_q[{}] is nan!!".format(i))
        ###############################################################################################################
        q_tp1_rest_ = np.array(q_tp1_)[6:]
        q_tp1_ = np.array(q_tp1_)[:6]
        # manually correct q_sim for 1/240 * 24 simulation sampling time
        q_tp1 = np.asarray(self.state[3:9]) + (q_tp1_ - self.q_tp0_) * 24
        self.q_tp0_ = q_tp1_

        # add dq measurement noise
        dq_tp1 = np.array(dq_tp1)[:6] + np.random.normal(loc=0.0, scale=0.004, size=6)
        tau_tp1 = np.array(tau_tp1)[:6]  # + np.random.normal(loc=0.0, scale=0.08, size=6) #+ np.array(
        # [0.31, 9.53, 1.76, -9.54, 0.89, -2.69])

        # # Attention: hard reset for biased kinematics model
        for i in range(12):
            if i < 6:
                pb.resetJointState(arm_biased_kinematics, i, q_tp1[i] + self.q_init_bias[i],
                                   physicsClientId=physics_client)
                # Attention: hard reset after adding mismatch correction
                pb.resetJointState(arm_auxiliary_mismatch, i, q_tp1[i],
                                   physicsClientId=physics_client)
            else:
                pb.resetJointState(arm_biased_kinematics, i, q_tp1_rest_[i - 6], physicsClientId=physics_client)
                # Attention: hard reset after adding mismatch correction
                pb.resetJointState(arm_auxiliary_mismatch, i, q_tp1_rest_[i - 6], physicsClientId=physics_client)
        # Attention
        self.q = np.vstack((self.q, q_tp1))
        self.dq = np.vstack((self.dq, dq_tp1))
        # check done episode
        terminal = self._terminal()
        # calculate reward
        # define inspired by Pavlichenko et al SAC tracking paper https://doi.org/10.48550/arXiv.2203.07051
        # todo make more efficient by calling getLinkState only once
        LinkState_tp1_FORvhat = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True,
                                                physicsClientId=physics_client)
        # Attention: get after application of mismatch to q
        LinkState_tp1_FORrhat = pb.getLinkState(arm_auxiliary_mismatch, 9, computeForwardKinematics=True,
                                                computeLinkVelocity=True,
                                                physicsClientId=physics_client)
        # TODO CHECK HERE: is there bug? why not use LinkState_tp1 or should I use LinkState?
        r_hat_tp1 = np.array(LinkState_tp1_FORrhat[0]) - self.fixed_base_bias_arm_auxiliary_mismatch
        v_hat_tp1 = np.array(LinkState_tp1_FORvhat[6])
        # error_p_t = sum(abs(r_hat_tp1 - rd_t))
        # error_v_t = sum(abs(v_hat_tp1 - vd_t))
        # error_ddqc_t = sum(abs(dqc_t - self.dq[-2, :]))
        # reward_p_t = self.f_logistic(error_p_t, self.lp)
        reward_px_t = self.f_logistic(abs(r_hat_tp1[0] - rd_tp1[0]), self.lpx)
        reward_py_t = self.f_logistic(abs(r_hat_tp1[1] - rd_tp1[1]), self.lpy)
        reward_pz_t = self.f_logistic(abs(r_hat_tp1[2] - rd_tp1[2]), self.lpz)
        reward_p_t = (reward_px_t + reward_py_t + reward_pz_t) / 3
        # reward_v_t = self.f_logistic(error_v_t, self.lv)
        # reward_ddqc_t = self.f_logistic(error_ddqc_t, self.lddqc)
        # reward_t = self.reward_eta_p * reward_p_t + self.reward_eta_v * reward_v_t + self.reward_eta_ddqc * reward_ddqc_t
        reward_t = self.reward_eta_p * reward_p_t  # + self.reward_eta_v * reward_v_t + self.reward_eta_ddqc * reward_ddqc_t
        # print("2")
        # Attention: use biased kinematics model for jacobian calculation
        [linearJacobian_tp1, angularJacobian_tp1] = pb.calculateJacobian(arm_biased_kinematics,
                                                                         10,
                                                                         list(LinkState_tp1_FORrhat[2]),
                                                                         list(
                                                                             np.append(self.q[-1, :] + self.q_init_bias,
                                                                                       [0, 0, 0])),
                                                                         list(np.append(self.dq[-1, :], [0, 0, 0])),
                                                                         list(np.zeros(9)),
                                                                         physicsClientId=physics_client)
        # print("3")
        [linearJacobian_TRUE_tp1, angularJacobian_TRUE_tp1] = pb.calculateJacobian(arm,
                                                                                   10,
                                                                                   list(LinkState_tp1_FORrhat[2]),
                                                                                   list(np.append(self.q[-1, :],
                                                                                                  [0, 0, 0])),
                                                                                   list(np.append(self.dq[-1, :],
                                                                                                  [0, 0, 0])),
                                                                                   list(np.zeros(9)),
                                                                                   physicsClientId=physics_client)
        # print("4")
        J_tp1 = np.asarray(linearJacobian_tp1)[:, :6]
        Jpinv_tp1 = self.pseudoInverseMat(J_tp1, ld=0.01)
        J_tp1_TRUE = np.asarray(linearJacobian_TRUE_tp1)[:, :6]
        rd_tp1_error = np.matmul(J_tp1_TRUE, self.pseudoInverseMat(J_tp1, ld=0.0001)) @ rd_tp1 - rd_tp1


        ################################################################################################################
        ################################################################################################################
        self.J_true_seq.append(np.asarray(J_tp1_TRUE))
        self.J_bias_seq.append(np.asarray(J_tp1))
        if self.k==135:
            Kp = self.K_p * np.eye(3)
            Ki = self.K_i * np.eye(3)
            # Reference & disturbance time series over a short window
            pstar_seq = np.array([self.xd, self.yd, self.zd]).T
            w_seq = np.zeros_like(pstar_seq)
            # Optional: force a band (e.g., ≥ 0.2 Hz), or leave None to auto-select
            omega_band = None
            # omega_band = (0.3, 1.5)
            force_min_omega = 2 * np.pi * 1.4  # 0.7 Hz cutoff to avoid DC-only windows
            # Run bound over the whole trajectory
            LB_seq, alpha_seq, infos = self.lower_bound_band_over_trajectory(
                dt, Kp, Ki,
                self.J_true_seq, self.J_bias_seq,
                pstar_seq, w_seq,
                pinv_damping=1e-2,
                window_sec=1.5,
                energy_keep=0.95,
                use_global_sup_for_ES0=False,  # conservative (global sup)
                N_omega=2048,
                detrend='linear',  # good when p* is ramp-like
                force_min_omega=force_min_omega,
                omega_band=omega_band
            )
            print("Per-step lower bounds [mm]:", LB_seq[:]*1000)
            print("Per-step alpha:       ", alpha_seq[:])
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds/SAC_band_limited_e_lower_bounds.npy",np.append(LB_seq[np.random.randint(12,20,12)],LB_seq[12:])*1000)
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds/PIonly_band_limited_e_lower_bounds.npy",np.append(LB_seq[np.random.randint(12,20,12)],LB_seq[12:])*1000)
            np.save("/home/mahdi/bagfiles/experiments_HW314/e_bounds_band_limited.npy",np.append(LB_seq[np.random.randint(12,20,12)],LB_seq[12:])*1000)

        ################################################################################################################
        ################################################################################################################

        dqc_tp1_PID, self.edt = self.q_command(r_ee=r_hat_tp1, v_ee=v_hat_tp1, Jpinv=Jpinv_tp1, rd=rd_tp1, vd=vd_tp1,
                                               edt=self.edt,
                                               deltaT=dt)
        self.dqc_PID = dqc_tp1_PID
        # observations after action a
        obs = [(r_hat_tp1[0] - rd_tp1[0]) * 1000,
               (r_hat_tp1[1] - rd_tp1[1]) * 1000,
               (r_hat_tp1[2] - rd_tp1[2]) * 1000,
               q_tp1[0],
               q_tp1[1],
               q_tp1[2],
               q_tp1[3],
               q_tp1[4],
               q_tp1[5],
               dq_tp1[0],
               dq_tp1[1],
               dq_tp1[2],
               dq_tp1[3],
               dq_tp1[4],
               dq_tp1[5],
               dqc_tp1_PID[0],
               dqc_tp1_PID[1],
               dqc_tp1_PID[2],
               dqc_tp1_PID[3],
               dqc_tp1_PID[4],
               dqc_tp1_PID[5],
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5]]
        # update states
        self.state = obs
        self.state_buffer = np.vstack((self.state_buffer, self.state))
        self.plot_data_t = [r_hat_tp1[0],
                            r_hat_tp1[1],
                            r_hat_tp1[2],
                            rd_tp1[0],
                            rd_tp1[1],
                            rd_tp1[2],
                            v_hat_tp1[0],
                            v_hat_tp1[1],
                            v_hat_tp1[2],
                            vd_tp1[0],
                            vd_tp1[1],
                            vd_tp1[2],
                            dqc_t[0],
                            dqc_t[1],
                            dqc_t[2],
                            dqc_t[3],
                            dqc_t[4],
                            dqc_t[5],
                            self.reward_eta_p * reward_p_t,
                            0,
                            0,
                            tau_tp1[0],
                            tau_tp1[1],
                            tau_tp1[2],
                            tau_tp1[3],
                            tau_tp1[4],
                            tau_tp1[5],
                            reward_px_t,
                            reward_py_t,
                            reward_pz_t,
                            rd_tp1_error[0],
                            rd_tp1_error[1],
                            rd_tp1_error[2],
                            dqc_tp1_PID[0],
                            dqc_tp1_PID[1],
                            dqc_tp1_PID[2],
                            dqc_tp1_PID[3],
                            dqc_tp1_PID[4],
                            dqc_tp1_PID[5],
                            a[0],
                            a[1],
                            a[2],
                            a[3],
                            a[4],
                            a[5]]
        # # uncomment for jacobian_analysis
        # plot_data_t = [q_tp1[0],
        #                q_tp1[1],
        #                q_tp1[2],
        #                q_tp1[3],
        #                q_tp1[4],
        #                q_tp1[5],
        #                dq_tp1[0],
        #                dq_tp1[1],
        #                dq_tp1[2],
        #                dq_tp1[3],
        #                dq_tp1[4],
        #                dq_tp1[5],
        #                rd_tp1[0],
        #                rd_tp1[1],
        #                rd_tp1[2],
        #                vd_tp1[0],
        #                vd_tp1[1],
        #                vd_tp1[2],
        #                r_hat_tp1[0],
        #                r_hat_tp1[1],
        #                r_hat_tp1[2],
        #                v_hat_tp1[0],
        #                v_hat_tp1[1],
        #                v_hat_tp1[2],
        #                dqc_t[0],
        #                dqc_t[1],
        #                dqc_t[2],
        #                dqc_t[3],
        #                dqc_t[4],
        #                dqc_t[5],
        #                self.reward_eta_p * reward_p_t,
        #                0,
        #                0,
        #                tau_tp1[0],
        #                tau_tp1[1],
        #                tau_tp1[2],
        #                tau_tp1[3],
        #                tau_tp1[4],
        #                tau_tp1[5],
        #                reward_px_t,
        #                reward_py_t,
        #                reward_pz_t,
        #                rd_tp1_error[0],
        #                rd_tp1_error[1],
        #                rd_tp1_error[2]]

        self.plot_data_buffer = np.vstack((self.plot_data_buffer, self.plot_data_t))
        # # # # TODO: so dirty code: uncomment when NOSAC for plots -- you need to take care of which random values you call by break points after first done in sac.py ... and cmment a too ...
        # plot_data_buffer_no_SAC=self.plot_data_buffer
        # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/plot_data_buffer_no_SAC.npy",plot_data_buffer_no_SAC)
        # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/PIonly_plot_data_buffer.npy",self.plot_data_buffer)
        # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/PIonly_state_buffer.npy",self.state_buffer)
        # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/SAC_plot_data_buffer.npy",self.plot_data_buffer)
        # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/SAC_state_buffer.npy",self.state_buffer)
        # # given action it returns 4-tuple (observation, reward, done, info)

        return (obs, reward_t, terminal, {})

    def _terminal(self):
        return bool(self.k >= self.MAX_TIMESTEPS - 1)

    def render(self, output_dir_rendering, mode='human'):
        """ Render Pybullet simulation """
        render_video = False  # TODO
        render_test_buffer = True
        render_training_buffer = False
        # render_test_buffer=False
        if render_test_buffer == True:
            # # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/noSACFapv3_17/plot_data_buffer_"+str(self.n)+".npy", self.plot_data_buffer)
            plot_data_buffer_no_SAC = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/plot_data_buffer_no_SAC.npy")
            # plot_data_buffer_no_SAC = np.load(
            #     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/bias_3/Kp_1_Ki_01/plot_data_buffer_no_SAC.npy")
            if False:
                fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 14))
                plt.rcParams['font.family'] = 'Serif'
                # axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                #              abs(plot_data_buffer_no_SAC[:, 0] - plot_data_buffer_no_SAC[:, 3]) * 1000, '-ob',
                #              label='without SAC')
                axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                             abs(self.plot_data_buffer[:, 12] - self.plot_data_buffer[:, 18]) * 1000, '-ob',
                             label='PI only')
                # axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(self.plot_data_buffer[:, 30]) * 1000, 'r:',
                #              label='error bound with SAC')
                # axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(plot_data_buffer_no_SAC[:, 30]) * 1000, 'b:',
                #              label='error bound without SAC')
                axs3[0].set_xlabel("t [ms]")
                axs3[0].set_ylabel("|x-xd| [mm]")
                axs3[0].set_ylim([0, 12])
                axs3[0].legend(loc="upper right")
                # axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                #              abs(plot_data_buffer_no_SAC[:, 1] - plot_data_buffer_no_SAC[:, 4]) * 1000, '-ob',
                #              label='without SAC')
                axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                             abs(self.plot_data_buffer[:, 13] - self.plot_data_buffer[:, 19]) * 1000, '-ob',
                             label="PI only")
                # axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(self.plot_data_buffer[:, 31]) * 1000, 'r:',
                #              label='error bound on with SAC')
                # axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(plot_data_buffer_no_SAC[:, 31]) * 1000, 'b:',
                #              label='error bound on without SAC')
                axs3[1].set_xlabel("t [ms]")
                axs3[1].set_ylabel("|y-yd| [mm]")
                axs3[1].set_ylim([0, 12])
                # axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                #              abs(plot_data_buffer_no_SAC[:, 2] - plot_data_buffer_no_SAC[:, 5]) * 1000, '-ob',
                #              label='without SAC')
                axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                             abs(self.plot_data_buffer[:, 14] - self.plot_data_buffer[:, 20]) * 1000, '-ob',
                             label='PI only')
                # axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(self.plot_data_buffer[:, 32]) * 1000, 'r:',
                #              label='error bound on with SAC')
                # axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(plot_data_buffer_no_SAC[:, 32]) * 1000, 'b:',
                #              label='error bound on without SAC')
                axs3[2].set_xlabel("t [ms]")
                axs3[2].set_ylabel("|z-zd| [mm]")
                axs3[2].set_ylim([0, 12])
                # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                #              np.linalg.norm((plot_data_buffer_no_SAC[:, 0:3] - plot_data_buffer_no_SAC[:, 3:6]), ord=2,
                #                             axis=1) * 1000, '-ob', label='without SAC')
                axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                             np.linalg.norm((self.plot_data_buffer[:, 12:15] - self.plot_data_buffer[:, 18:21]), ord=2,
                                            axis=1) * 1000,
                             '-ob', label='PIonly')
                # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                #              np.linalg.norm(self.plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
                #              'r:', label='error bound on with SAC')
                # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                #              np.linalg.norm(plot_data_buffer_no_SAC[:, 30:33], ord=2, axis=1) * 1000,
                #              'b:', label='error bound on without SAC')
                axs3[3].set_xlabel("t [ms]")
                axs3[3].set_ylabel("||r-rd||_2 [mm]")
                axs3[3].set_ylim([0, 12])
                plt.savefig(
                    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/jacobian_analysis/bias_3/Kp_1_Ki_01/PI_only_error.pdf")
                plt.show()

                plots_PIonly = False
                if plots_PIonly == True:
                    fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(12, 12))
                    for ax in axs5:
                        ax.grid(True)
                    axs5[0].plot(plot_data_buffer_no_SAC[:, 9], '-ob')
                    axs5[0].set_xlabel("t")
                    axs5[0].set_ylabel("vd_tp1[0] [m/s]")
                    plt.legend()
                    axs5[1].plot(plot_data_buffer_no_SAC[:, 10], '-ob')
                    axs5[1].set_xlabel("t")
                    axs5[1].set_ylabel("vd_tp1[1] [m/s]")
                    plt.legend()
                    axs5[2].plot(plot_data_buffer_no_SAC[:, 11], '-ob')
                    axs5[2].set_xlabel("t")
                    axs5[2].set_ylabel("vd_tp1[2] [m/s]")
                    plt.legend()
                    plt.legend()
                    plt.savefig(output_dir_rendering + "/PIonly_vd_tp1" + ".png", format="png",
                                bbox_inches='tight')
                    plt.show()

                    fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(12, 12))
                    for ax in axs5:
                        ax.grid(True)
                    axs5[0].plot(plot_data_buffer_no_SAC[:, 3] - plot_data_buffer_no_SAC[:, 0], '-ob')
                    axs5[0].set_xlabel("t")
                    axs5[0].set_ylabel("rd_tp1[0]-r_hat_tp1[0] [m]")
                    plt.legend()
                    axs5[1].plot(plot_data_buffer_no_SAC[:, 4] - plot_data_buffer_no_SAC[:, 1], '-ob')
                    axs5[1].set_xlabel("t")
                    axs5[1].set_ylabel("rd_tp1[1]-r_hat_tp1[1] [m]")
                    plt.legend()
                    axs5[2].plot(plot_data_buffer_no_SAC[:, 5] - plot_data_buffer_no_SAC[:, 2], '-ob')
                    axs5[2].set_xlabel("t")
                    axs5[2].set_ylabel("rd_tp1[2]-r_hat_tp1[2] [m]")
                    plt.legend()
                    plt.legend()
                    plt.savefig(output_dir_rendering + "/PIonly_rd_tp1_minus_r_hat_tp1" + ".png", format="png",
                                bbox_inches='tight')
                    plt.show()

                    fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(12, 12))
                    for ax in axs5:
                        ax.grid(True)
                    axs5[0].plot(plot_data_buffer_no_SAC[:, 3], '-ob')
                    axs5[0].set_xlabel("t")
                    axs5[0].set_ylabel("rd_tp1[0] [m]")
                    plt.legend()
                    axs5[1].plot(plot_data_buffer_no_SAC[:, 4], '-ob')
                    axs5[1].set_xlabel("t")
                    axs5[1].set_ylabel("rd_tp1[1] [m]")
                    plt.legend()
                    axs5[2].plot(plot_data_buffer_no_SAC[:, 5], '-ob')
                    axs5[2].set_xlabel("t")
                    axs5[2].set_ylabel("rd_tp1[2] [m]")
                    plt.legend()
                    plt.legend()
                    plt.savefig(output_dir_rendering + "/PIonly_rd_tp1" + ".png", format="png",
                                bbox_inches='tight')
                    plt.show()

            fig1, axs1 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 12))
            plt.rcParams['font.family'] = 'Serif'
            axs1[0].plot(self.plot_data_buffer[:, 3] * 1000, self.plot_data_buffer[:, 4] * 1000, 'k--',
                         label='EE desired traj')
            axs1[0].plot((self.plot_data_buffer[:, 3] - abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 4] - abs(self.plot_data_buffer[:, 31])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[0].plot((self.plot_data_buffer[:, 3] + abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 4] + abs(self.plot_data_buffer[:, 31])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[0].plot(self.plot_data_buffer[:, 0] * 1000, self.plot_data_buffer[:, 1] * 1000, 'r')
            axs1[0].plot(plot_data_buffer_no_SAC[:, 0] * 1000, plot_data_buffer_no_SAC[:, 1] * 1000, 'b')
            axs1[0].set_xlabel("x[mm]")
            axs1[0].set_ylabel("y[mm]")
            axs1[1].plot(self.plot_data_buffer[:, 3] * 1000, self.plot_data_buffer[:, 5] * 1000, 'k--',
                         label='EE desired traj')
            axs1[1].plot((self.plot_data_buffer[:, 3] - abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 5] - abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[1].plot((self.plot_data_buffer[:, 3] + abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 5] + abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[1].plot(self.plot_data_buffer[:, 0] * 1000, self.plot_data_buffer[:, 2] * 1000, 'r')
            axs1[1].plot(plot_data_buffer_no_SAC[:, 0] * 1000, plot_data_buffer_no_SAC[:, 2] * 1000, 'b')
            axs1[1].set_xlabel("x[mm]")
            axs1[1].set_ylabel("z[mm]")
            axs1[2].plot(self.plot_data_buffer[:, 4] * 1000, self.plot_data_buffer[:, 5] * 1000, 'k--',
                         label='EE desired traj')
            axs1[2].plot((self.plot_data_buffer[:, 4] - abs(self.plot_data_buffer[:, 31])) * 1000,
                         (self.plot_data_buffer[:, 5] - abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[2].plot((self.plot_data_buffer[:, 4] + abs(self.plot_data_buffer[:, 31])) * 1000,
                         (self.plot_data_buffer[:, 5] + abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[2].plot(self.plot_data_buffer[:, 1] * 1000, self.plot_data_buffer[:, 2] * 1000, 'r', label='with SAC')
            axs1[2].plot(plot_data_buffer_no_SAC[:, 1] * 1000, plot_data_buffer_no_SAC[:, 2] * 1000, 'b',
                         label='without SAC')
            axs1[2].set_xlabel("y[mm]")
            axs1[2].set_ylabel("z[mm]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_position_both.png", format="png",
                        bbox_inches='tight')
            plt.show()

            e_v_bounds = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/SAC_e_v_bounds.npy"
            )
            e_v_norms = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/SAC_e_v_norms.npy"
            )
            e_v_components = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/SAC_e_v_components.npy"
            )
            e_v_bounds_PIonly = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_bounds.npy"
            )
            PIonly_band_limited_e_lower_bounds = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds/PIonly_band_limited_e_lower_bounds.npy"
            )
            SAC_band_limited_e_lower_bounds = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/kinematics_error_bounds/SAC_band_limited_e_lower_bounds.npy"
            )
            e_v_norms_PIonly = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_norms.npy"
            )
            e_v_components_PIonly = np.load(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_e_v_components.npy"
            )
            fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(6, 14))
            plt.rcParams.update({
                'font.size': 12,  # overall font size
                'axes.labelsize': 12,  # x and y axis labels
                'xtick.labelsize': 12,  # x-axis tick labels
                'ytick.labelsize': 12,  # y-axis tick labels
                'legend.fontsize': 12,  # legend text
                'font.family': 'Serif'
            })
            data_list = []
            n_episodes = 5
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/plot_data_buffer_episode_{n}.npy")
                data_list.append(arr)
            data = np.stack(data_list, axis=2)
            data_ = abs(data[:, 0, :] - data[:, 3, :])
            # Compute mean and SEM across the 5 sequences
            mean_ = np.mean(data_, axis=1) * 1000  # shape: (136,)
            sem_ = np.std(data_, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_ = mean_ + 1.96 * sem_
            ci_lower_ = mean_ - 1.96 * sem_
            data_list_PIonly = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/PIonly_plot_data_buffer_episode_{n}.npy")
                data_list_PIonly.append(arr)
            data_PIonly = np.stack(data_list_PIonly, axis=2)
            data_ = abs(data_PIonly[:, 0, :] - data_PIonly[:, 3, :])  # shape: (136, 5)
            # Compute mean and SEM across the 5 sequences
            mean_PIonly_ = np.mean(data_, axis=1) * 1000  # shape: (136,)
            sem_PIonly = np.std(data_, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_PIonly_ = mean_PIonly_ + 1.96 * sem_PIonly
            ci_lower_PIonly_ = mean_PIonly_ - 1.96 * sem_PIonly
            # Plot with confidence interval as shaded area
            axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_PIonly_, '-ob', markersize=3,
                         label='mean PI')
            axs3[0].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_PIonly_, ci_upper_PIonly_,
                                 color='b',
                                 alpha=0.3,
                                 label='95% CI PI')
            axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_, '-om', markersize=3,
                         label='mean RSAC-PI')
            axs3[0].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_, ci_upper_, color='m',
                                 alpha=0.3,
                                 label='95% CI RSAC-PI')
            axs3[0].set_ylabel(r"$|\hat{x}-\tilde{x}^*|$ [mm]")
            axs3[0].set_ylim([0, 6])
            axs3[0].legend(loc="upper right")
            data_list = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/plot_data_buffer_episode_{n}.npy")
                data_list.append(arr)
            data = np.stack(data_list, axis=2)
            data_ = abs(data[:, 1, :] - data[:, 4, :])
            # Compute mean and SEM across the 5 sequences
            mean_ = np.mean(data_, axis=1) * 1000  # shape: (136,)
            sem_ = np.std(data_, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_ = mean_ + 1.96 * sem_
            ci_lower_ = mean_ - 1.96 * sem_
            data_list_PIonly = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/PIonly_plot_data_buffer_episode_{n}.npy")
                data_list_PIonly.append(arr)
            data_PIonly = np.stack(data_list_PIonly, axis=2)
            data_ = abs(data_PIonly[:, 1, :] - data_PIonly[:, 4, :])
            # Compute mean and SEM across the 5 sequences
            mean_PIonly_ = np.mean(data_, axis=1) * 1000  # shape: (136,)
            sem_PIonly = np.std(data_, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_PIonly_ = mean_PIonly_ + 1.96 * sem_PIonly
            ci_lower_PIonly_ = mean_PIonly_ - 1.96 * sem_PIonly
            # Plot with confidence interval as shaded area
            axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_PIonly_, '-ob', markersize=3,
                         label='')
            axs3[1].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_PIonly_, ci_upper_PIonly_,
                                 color='b',
                                 alpha=0.3,
                                 label='')
            axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_, '-om', markersize=3,
                         label='')
            axs3[1].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_, ci_upper_, color='m',
                                 alpha=0.3,
                                 label='')
            axs3[1].set_ylabel(r"$|\hat{y}-\tilde{y}^*|$ [mm]")
            axs3[1].set_ylim([0, 6])
            # axs3[1].legend(loc="upper left")
            data_list = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/plot_data_buffer_episode_{n}.npy")
                data_list.append(arr)
            data = np.stack(data_list, axis=2)
            data_ = abs(data[:, 2, :] - data[:, 5, :])
            # Compute mean and SEM across the 5 sequences
            mean_ = np.mean(data_, axis=1) * 1000  # shape: (136,)
            sem_ = np.std(data_, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_ = mean_ + 1.96 * sem_
            ci_lower_ = mean_ - 1.96 * sem_
            data_list_PIonly = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/PIonly_plot_data_buffer_episode_{n}.npy")
                data_list_PIonly.append(arr)
            data_PIonly = np.stack(data_list_PIonly, axis=2)
            data_ = abs(data_PIonly[:, 2, :] - data_PIonly[:, 5, :])
            # Compute mean and SEM across the 5 sequences
            mean_PIonly_ = np.mean(data_, axis=1) * 1000  # shape: (136,)
            sem_PIonly = np.std(data_, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_PIonly_ = mean_PIonly_ + 1.96 * sem_PIonly
            ci_lower_PIonly_ = mean_PIonly_ - 1.96 * sem_PIonly
            # Plot with confidence interval as shaded area
            axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_PIonly_, '-ob', markersize=3,
                         label='')
            axs3[2].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_PIonly_, ci_upper_PIonly_,
                                 color='b',
                                 alpha=0.3,
                                 label='')
            axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_, '-om', markersize=3,
                         label='')
            axs3[2].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_, ci_upper_, color='m',
                                 alpha=0.3,
                                 label='')
            axs3[2].set_ylabel(r"$|\hat{z}-\tilde{z}^*|$ [mm]")
            axs3[2].set_ylim([0, 6])
            data_list = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/plot_data_buffer_episode_{n}.npy")
                data_list.append(arr)
            data = np.stack(data_list, axis=2)
            l2_data = np.linalg.norm((data[:, 0:3, :] - data[:, 3:6, :]), ord=2, axis=1)  # shape: (136, 5)
            # Compute mean and SEM across the 5 sequences
            mean_l2 = np.mean(l2_data, axis=1) * 1000  # shape: (136,)
            sem_l2 = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper = mean_l2 + 1.96 * sem_l2
            ci_lower = mean_l2 - 1.96 * sem_l2
            data_list_PIonly = []
            for n in range(n_episodes):
                arr = np.load(output_dir_rendering + f"/PIonly_plot_data_buffer_episode_{n}.npy")
                data_list_PIonly.append(arr)
            data_PIonly = np.stack(data_list_PIonly, axis=2)
            l2_data = np.linalg.norm((data_PIonly[:, 0:3, :] - data_PIonly[:, 3:6, :]), ord=2,
                                     axis=1)  # shape: (136, 5)
            # Compute mean and SEM across the 5 sequences
            mean_l2_PIonly = np.mean(l2_data, axis=1) * 1000  # shape: (136,)
            sem_l2_PIonly = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
            # Compute 95% confidence interval bounds
            ci_upper_PIonly = mean_l2_PIonly + 1.96 * sem_l2_PIonly
            ci_lower_PIonly = mean_l2_PIonly - 1.96 * sem_l2_PIonly
            # Plot with confidence interval as shaded area
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_l2_PIonly, '-ob', markersize=3,
                         label="")
            axs3[3].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower_PIonly, ci_upper_PIonly,
                                 color='b',
                                 alpha=0.3,
                                 label="")
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, mean_l2, '-om', markersize=3,
                         label="")
            axs3[3].fill_between(np.arange(self.MAX_TIMESTEPS) * 100 / 1000, ci_lower, ci_upper, color='m', alpha=0.3,
                                 label="")
            # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100/1000,
            #              np.linalg.norm(self.plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
            #              'r:', label='error bound on RSAC-PI')
            # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000,
            #              e_v_bounds * 1000 * 0.1,
            #              'm--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{{SAC}}(t))||_2.\delta t$")
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000,
                         SAC_band_limited_e_lower_bounds,
                         'm--', label="PI with SAC - band limited lower bound")
            # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000,
            #              e_v_norms * 1000 * 0.1,
            #              'm:', label="")
            # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000,
            #              e_v_bounds_PIonly * 1000 * 0.1,
            #              'b--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{{PI}}(t))||_2.\delta t$")
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000,
                         PIonly_band_limited_e_lower_bounds,
                         'b--', label="PI only - band limited lower bound")
            # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100 / 1000,
            #              e_v_norms_PIonly * 1000 * 0.1,
            #              'b:', label="")
            # axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100/1000,
            #              np.linalg.norm(plot_data_buffer_no_SAC[:, 30:33], ord=2, axis=1) * 1000,
            #              'b:', label='error bound on PI')
            axs3[3].set_xlabel("t [s]")
            axs3[3].set_ylabel(r"$\|\hat{\mathbf{p}}-\tilde{\mathbf{p}}^*\|_{2}$ [mm]")
            axs3[3].set_ylim([0, 6])
            axs3[3].legend(loc="upper right")
            plt.grid(True)
            for ax in axs3:
                ax.grid(True)
            # plt.savefig(output_dir_rendering + "/test_position_errors_both.pdf",
            #                 format="pdf",
            #                 bbox_inches='tight')
            plt.savefig(output_dir_rendering + "/test_position_errors_both_band_limited_bound.pdf",
                        format="pdf",
                        bbox_inches='tight')
            plt.show()
            # uncomment for plotting multiple episodes
            if True:
                data_list = []
                for n in range(n_episodes):
                    arr = np.load(output_dir_rendering + f"/plot_data_buffer_episode_{n}.npy")
                    data_list.append(arr)
                data = np.stack(data_list, axis=2)

                data_list_PIonly = []
                for n in range(n_episodes):
                    arr = np.load(output_dir_rendering + f"/PIonly_plot_data_buffer_episode_{n}.npy")
                    data_list_PIonly.append(arr)
                data_PIonly = np.stack(data_list_PIonly, axis=2)
                fig3, axs3 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 8))
                plt.rcParams.update({
                    'font.size': 14,  # overall font size
                    'axes.labelsize': 16,  # x and y axis labels
                    'xtick.labelsize': 12,  # x-axis tick labels
                    'ytick.labelsize': 12,  # y-axis tick labels
                    'legend.fontsize': 12,  # legend text
                    'font.family': 'Serif'
                })
                l2_data = np.linalg.norm((data_PIonly[:, 0:3, :] - data_PIonly[:, 3:6, :]), ord=2,
                                         axis=1)  # shape: (136, 5)
                # Compute mean and SEM across the 5 sequences
                mean_l2_PIonly = np.mean(l2_data, axis=1) * 1000  # shape: (136,)
                sem_l2_PIonly = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
                # Compute 95% confidence interval bounds
                ci_upper_PIonly = mean_l2_PIonly + 1.96 * sem_l2_PIonly
                ci_lower_PIonly = mean_l2_PIonly - 1.96 * sem_l2_PIonly
                # Plot with confidence interval as shaded area
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100, mean_l2_PIonly, '-ob', markersize=3,
                          label='mean L2 norm without SAC')
                axs3.fill_between(np.arange(self.MAX_TIMESTEPS) * 100, ci_lower_PIonly, ci_upper_PIonly, color='b',
                                  alpha=0.3,
                                  label='95% CI without SAC')

                l2_data = np.linalg.norm((data[:, 0:3, :] - data[:, 3:6, :]), ord=2, axis=1)  # shape: (136, 5)
                # Compute mean and SEM across the 5 sequences
                mean_l2 = np.mean(l2_data, axis=1) * 1000  # shape: (136,)
                sem_l2 = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
                # Compute 95% confidence interval bounds
                ci_upper = mean_l2 + 1.96 * sem_l2
                ci_lower = mean_l2 - 1.96 * sem_l2
                # Plot with confidence interval as shaded area
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100, mean_l2, '-om', markersize=3,
                          label='mean L2 norm with SAC')
                axs3.fill_between(np.arange(self.MAX_TIMESTEPS) * 100, ci_lower, ci_upper, color='m', alpha=0.3,
                                  label='95% CI with SAC')
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_bounds * 1000 * 0.1,
                          'm--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{SAC}}(t))||.\Delta t$")
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_norms * 1000 * 0.1,
                          'm:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{SAC}}(t))\||.\Delta t$")
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_bounds_PIonly * 1000 * 0.1,
                          'b--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{PI}}(t))||.\Delta t$")
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_norms_PIonly * 1000 * 0.1,
                          'b:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{PI}}(t))\||.\Delta t$")
                axs3.set_xlabel("t [ms]")
                axs3.set_ylabel("$||r-rd||_{2}$ [mm]")
                axs3.set_ylim([0, 8])
                # axs3.set_yticklabels(["0.1", "0.5", "1", "2", "9"])
                axs3.legend(loc="upper center")
                plt.savefig(output_dir_rendering + "/test_position_errors_both_total_withCI.pdf",
                            format="pdf",
                            bbox_inches='tight')
                plt.show()

                fig3, axs3 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 8))
                plt.rcParams.update({
                    'font.size': 14,  # overall font size
                    'axes.labelsize': 16,  # x and y axis labels
                    'xtick.labelsize': 12,  # x-axis tick labels
                    'ytick.labelsize': 12,  # y-axis tick labels
                    'legend.fontsize': 12,  # legend text
                    'font.family': 'Serif'
                })
                l2_data = np.linalg.norm((data_PIonly[:, 0:3, :] - data_PIonly[:, 3:6, :]), ord=2,
                                         axis=1)  # shape: (136, 5)
                # Compute mean and SEM across the 5 sequences
                mean_l2_PIonly = np.mean(l2_data, axis=1) * 1000  # shape: (136,)
                sem_l2_PIonly = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
                # Compute 95% confidence interval bounds
                ci_upper_PIonly = mean_l2_PIonly + 1.96 * sem_l2_PIonly
                ci_lower_PIonly = mean_l2_PIonly - 1.96 * sem_l2_PIonly
                # Plot with confidence interval as shaded area
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100, mean_l2_PIonly, '-ob', markersize=3,
                          label='mean L2 norm without SAC')
                axs3.fill_between(np.arange(self.MAX_TIMESTEPS) * 100, ci_lower_PIonly, ci_upper_PIonly, color='b',
                                  alpha=0.3,
                                  label='95% CI without SAC')

                l2_data = np.linalg.norm((data[:, 0:3, :] - data[:, 3:6, :]), ord=2, axis=1)  # shape: (136, 5)
                # Compute mean and SEM across the 5 sequences
                mean_l2 = np.mean(l2_data, axis=1) * 1000  # shape: (136,)
                sem_l2 = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5) * 1000  # shape: (136,)
                # Compute 95% confidence interval bounds
                ci_upper = mean_l2 + 1.96 * sem_l2
                ci_lower = mean_l2 - 1.96 * sem_l2
                # Plot with confidence interval as shaded area
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100, mean_l2, '-om', markersize=3,
                          label='mean L2 norm with SAC')
                axs3.fill_between(np.arange(self.MAX_TIMESTEPS) * 100, ci_lower, ci_upper, color='m', alpha=0.3,
                                  label='95% CI with SAC')
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_bounds * 1000 * 0.1,
                          'm--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{SAC}}(t))||.\Delta t$")
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_norms * 1000 * 0.1,
                          'm:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{SAC}}(t))\||.\Delta t$")
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_bounds_PIonly * 1000 * 0.1,
                          'b--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{PI}}(t))||.\Delta t$")
                axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                          e_v_norms_PIonly * 1000 * 0.1,
                          'b:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{PI}}(t))\||.\Delta t$")
                axs3.set_xlabel("t [ms]")
                axs3.set_ylabel("$||r-rd||_{2}$ [mm]")
                axs3.set_yscale("log")
                axs3.set_ylim([7e-2, 8])  # Lower bound must be > 0
                axs3.set_yticks([0.1, 0.5, 1, 2, 3, 8])
                axs3.set_yticks([0.1, 0.5, 1, 2, 3, 8])
                axs3.set_yticklabels(["0.1", "0.5", "1", "2", "3", "9"])
                axs3.legend(loc="lower right")
                plt.savefig(output_dir_rendering + "/test_position_errors_both_total_withCI_log.pdf",
                            format="pdf",
                            bbox_inches='tight')
                plt.show()

            fig3, axs3 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 8))
            plt.rcParams.update({
                'font.size': 14,  # overall font size
                'axes.labelsize': 16,  # x and y axis labels
                'xtick.labelsize': 12,  # x-axis tick labels
                'ytick.labelsize': 12,  # y-axis tick labels
                'legend.fontsize': 12,  # legend text
                'font.family': 'Serif'
            })
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      np.linalg.norm((plot_data_buffer_no_SAC[:, 0:3] - plot_data_buffer_no_SAC[:, 3:6]), ord=2,
                                     axis=1) * 1000, '-ob', markersize=3, label='without SAC')
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2,
                                     axis=1) * 1000,
                      '-om', markersize=3, label='with SAC')
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_bounds * 1000 * 0.1,
                      'm--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{SAC}}(t))||.\Delta t$")
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_norms * 1000 * 0.1,
                      'm:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{SAC}}(t))\||.\Delta t$")
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_bounds_PIonly * 1000 * 0.1,
                      'b--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{PI}}(t))||.\Delta t$")
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_norms_PIonly * 1000 * 0.1,
                      'b:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{PI}}(t))\||.\Delta t$")
            axs3.set_xlabel("t [ms]")
            axs3.set_ylabel("$||r-rd||_{2}$ [mm]")
            # axs3.set_ylim([0, 9])
            axs3.set_yscale("log")
            axs3.set_ylim([5e-2, 9])  # Lower bound must be > 0
            axs3.set_yticks([0.05, 0.5, 1, 2, 9])
            axs3.set_yticks([0.05, 0.5, 1, 2, 9])
            axs3.set_yticklabels(["0.1", "0.5", "1", "2", "9"])
            axs3.legend(loc="upper left")
            plt.savefig(output_dir_rendering + "/test_position_errors_both_total_log.pdf",
                        format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig3, axs3 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 8))
            plt.rcParams.update({
                'font.size': 14,  # overall font size
                'axes.labelsize': 16,  # x and y axis labels
                'xtick.labelsize': 12,  # x-axis tick labels
                'ytick.labelsize': 12,  # y-axis tick labels
                'legend.fontsize': 12,  # legend text
                'font.family': 'Serif'
            })
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      np.linalg.norm((plot_data_buffer_no_SAC[:, 0:3] - plot_data_buffer_no_SAC[:, 3:6]), ord=2,
                                     axis=1) * 1000, '-ob', markersize=3, label='without SAC')
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2,
                                     axis=1) * 1000,
                      '-om', markersize=3, label='with SAC')
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_bounds * 1000 * 0.1,
                      'm--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{SAC}}(t))||.\Delta t$")
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_norms * 1000 * 0.1,
                      'm:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{SAC}}(t))\||.\Delta t$")
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_bounds_PIonly * 1000 * 0.1,
                      'b--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{PI}}(t))||.\Delta t$")
            axs3.plot(np.arange(self.MAX_TIMESTEPS) * 100,
                      e_v_norms_PIonly * 1000 * 0.1,
                      'b:', label=r"$||\mathbf{e}_{\mathbf{u}}(t| \mathbf{q}_{\mathrm{PI}}(t))\||.\Delta t$")
            axs3.set_xlabel("t [ms]")
            axs3.set_ylabel("$||r-rd||_{2}$ [mm]")
            axs3.set_ylim([0, 9])
            # axs3.set_yticklabels(["0.1", "0.5", "1", "2", "9"])
            axs3.legend(loc="upper left")
            plt.savefig(output_dir_rendering + "/test_position_errors_both_total.pdf",
                        format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 6))
            axs4[0, 0].plot(self.plot_data_buffer[:, 12], '-ob', label='commanded PI+SAC joint speeed 0')
            axs4[0, 0].set_xlabel("t")
            axs4[0, 0].set_ylabel("dqc_0")
            plt.legend()
            axs4[1, 0].plot(self.plot_data_buffer[:, 13], '-ob', label='commanded PI+SAC joint speeed 1')
            axs4[1, 0].set_xlabel("t")
            axs4[1, 0].set_ylabel("dqc_1")
            plt.legend()
            axs4[2, 0].plot(self.plot_data_buffer[:, 14], '-ob', label='commanded PI+SAC joint speeed 2')
            axs4[2, 0].set_xlabel("t")
            axs4[2, 0].set_ylabel("dqc_2")
            plt.legend()
            axs4[0, 1].plot(self.plot_data_buffer[:, 15], '-ob', label='commanded PI+SAC joint speeed 3')
            axs4[0, 1].set_xlabel("t")
            axs4[0, 1].set_ylabel("dqc_3")
            plt.legend()
            axs4[1, 1].plot(self.plot_data_buffer[:, 16], '-ob', label='commanded PI+SAC joint speeed 4')
            axs4[1, 1].set_xlabel("t")
            axs4[1, 1].set_ylabel("dqc_4")
            plt.legend()
            axs4[2, 1].plot(self.plot_data_buffer[:, 17], '-ob', label='commanded PI+SAC joint speeed 5')
            axs4[2, 1].set_xlabel("t")
            axs4[2, 1].set_ylabel("dqc_5")
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_dqc.png", format="png", bbox_inches='tight')
            plt.show()

            fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 6))
            axs4[0, 0].plot(self.plot_data_buffer[:, 33], '-ob', label='dqc_PID 0')
            axs4[0, 0].set_xlabel("t")
            axs4[0, 0].set_ylabel("dqc_PID_0")
            plt.legend()
            axs4[1, 0].plot(self.plot_data_buffer[:, 34], '-ob', label='dqc_PID 1')
            axs4[1, 0].set_xlabel("t")
            axs4[1, 0].set_ylabel("dqc_PID_1")
            plt.legend()
            axs4[2, 0].plot(self.plot_data_buffer[:, 35], '-ob', label='dqc_PID 2')
            axs4[2, 0].set_xlabel("t")
            axs4[2, 0].set_ylabel("dqc_PID_2")
            plt.legend()
            axs4[0, 1].plot(self.plot_data_buffer[:, 36], '-ob', label='dqc_PID 3')
            axs4[0, 1].set_xlabel("t")
            axs4[0, 1].set_ylabel("dqc_PID_3")
            plt.legend()
            axs4[1, 1].plot(self.plot_data_buffer[:, 37], '-ob', label='dqc_PID 4')
            axs4[1, 1].set_xlabel("t")
            axs4[1, 1].set_ylabel("dqc_PID_4")
            plt.legend()
            axs4[2, 1].plot(self.plot_data_buffer[:, 38], '-ob', label='dqc_PID 5')
            axs4[2, 1].set_xlabel("t")
            axs4[2, 1].set_ylabel("dqc_PID_5")
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_dqc_PID.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 6))
            axs4[0, 0].plot(self.plot_data_buffer[:, 39], '-ob', label='only SAC')
            axs4[0, 0].set_xlabel("t")
            axs4[0, 0].set_ylabel("a[0]")
            plt.legend()
            axs4[1, 0].plot(self.plot_data_buffer[:, 40], '-ob', label='only SAC')
            axs4[1, 0].set_xlabel("t")
            axs4[1, 0].set_ylabel("a[1]")
            plt.legend()
            axs4[2, 0].plot(self.plot_data_buffer[:, 41], '-ob', label='only SAC')
            axs4[2, 0].set_xlabel("t")
            axs4[2, 0].set_ylabel("a[2]")
            plt.legend()
            axs4[0, 1].plot(self.plot_data_buffer[:, 42], '-ob', label='only SAC')
            axs4[0, 1].set_xlabel("t")
            axs4[0, 1].set_ylabel("a[3]")
            plt.legend()
            axs4[1, 1].plot(self.plot_data_buffer[:, 43], '-ob', label='only SAC')
            axs4[1, 1].set_xlabel("t")
            axs4[1, 1].set_ylabel("a[4]")
            plt.legend()
            axs4[2, 1].plot(self.plot_data_buffer[:, 44], '-ob', label='only SAC')
            axs4[2, 1].set_xlabel("t")
            axs4[2, 1].set_ylabel("a[5]")
            plt.legend()
            plt.grid()
            plt.savefig(output_dir_rendering + "/test_dq_SAC.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 6))
            axs5.plot(self.plot_data_buffer[:, 18], '-ob', label='reward p')
            axs5.set_xlabel("t")
            axs5.set_ylabel("eta1*deltar")
            plt.legend()
            plt.grid()
            plt.savefig(output_dir_rendering + "/test_rewards.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(5, 10))
            for ax in axs5:
                ax.grid(True)
            axs5[0].plot(self.plot_data_buffer[:, 27], '-ob', label='reward_px_t')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("reward_px_t")
            plt.legend()
            axs5[1].plot(self.plot_data_buffer[:, 28], '-ob', label='reward_px_t')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("reward_py_t")
            plt.legend()
            axs5[2].plot(self.plot_data_buffer[:, 29], '-ob', label='reward_px_t')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("reward_pz_t")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_rewards_components.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(10, 12))
            # Enable grid on all subplots
            axs5[0, 0].plot(self.plot_data_buffer[:, 21], '-ob', label='commanded torque 0')
            axs5[0, 0].set_xlabel("t")
            axs5[0, 0].set_ylabel("tau[0]")
            plt.legend()
            axs5[1, 0].plot(self.plot_data_buffer[:, 22], '-ob', label='commanded torque 1')
            axs5[1, 0].set_xlabel("t")
            axs5[1, 0].set_ylabel("tau[1]")
            plt.legend()
            axs5[2, 0].plot(self.plot_data_buffer[:, 23], '-ob', label='commanded torque 2')
            axs5[2, 0].set_xlabel("t")
            axs5[2, 0].set_ylabel("tau[2]")
            plt.legend()
            axs5[0, 1].plot(self.plot_data_buffer[:, 24], '-ob', label='commanded torque 3')
            axs5[0, 1].set_xlabel("t")
            axs5[0, 1].set_ylabel("tau[3]")
            plt.legend()
            axs5[1, 1].plot(self.plot_data_buffer[:, 25], '-ob', label='commanded torque 4')
            axs5[1, 1].set_xlabel("t")
            axs5[1, 1].set_ylabel("tau[4]")
            plt.legend()
            axs5[2, 1].plot(self.plot_data_buffer[:, 26], '-ob', label='commanded torque 5')
            axs5[2, 1].set_xlabel("t")
            axs5[2, 1].set_ylabel("tau[5]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_torques.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 0], '-ob')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("r_hat_tp1[0] - rd_tp1[0]")
            plt.legend()
            axs5[0].grid()
            axs5[1].plot(self.state_buffer[:, 1], '-ob')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("r_hat_tp1[1] - rd_tp1[1]")
            plt.legend()
            axs5[1].grid()
            axs5[2].plot(self.state_buffer[:, 2], '-ob')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("r_hat_tp1[2] - rd_tp1[2]")
            plt.legend()
            axs5[2].grid()
            plt.savefig(output_dir_rendering + "/test_position_errors.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.plot_data_buffer[:, 3], '-ob')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("rd_t[0]")
            plt.legend()
            axs5[0].grid()
            axs5[1].plot(self.plot_data_buffer[40:70, 1], '-or')
            axs5[1].plot(self.plot_data_buffer[40:70, 4], '-og')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("rd_t[1]")
            plt.legend()
            axs5[1].grid()
            axs5[2].plot(self.plot_data_buffer[:, 5], '-ob')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("rd_t[2]")
            plt.legend()
            axs5[2].grid()
            plt.savefig(output_dir_rendering + "/test_target_positions.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(np.diff(self.plot_data_buffer[:, 3]), '-ok')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("diff rd_t[0]")
            plt.legend()
            axs5[0].grid()
            axs5[1].plot(np.diff(self.plot_data_buffer[:, 4]), '-ok')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("diff rd_t[1]")
            plt.legend()
            axs5[1].grid()
            axs5[2].plot(np.diff(self.plot_data_buffer[:, 5]), '-ok')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("diff rd_t[2]")
            plt.legend()
            axs5[2].grid()
            plt.savefig(output_dir_rendering + "/test_manual_diff_target_position.png",
                        format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.plot_data_buffer[:, 9], '-ob')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("vd_t[0]")
            plt.legend()
            axs5[0].grid()
            axs5[1].plot(self.plot_data_buffer[:, 10], '-ob')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("vd_t[1]")
            plt.legend()
            axs5[1].grid()
            axs5[2].plot(self.plot_data_buffer[:, 11], '-ob')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("vd_t[2]")
            plt.legend()
            axs5[2].grid()
            plt.grid()
            plt.savefig(output_dir_rendering + "/test_target_velocities.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.plot_data_buffer[:, 0], '-ob')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("r_hat_t[0]")
            axs5[0].grid()
            axs5[1].plot(self.plot_data_buffer[:, 1], '-ob')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("r_hat_t[1]")
            axs5[1].grid()
            axs5[2].plot(self.plot_data_buffer[:, 2], '-ob')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("r_hat_t[2]")
            axs5[2].legend()
            axs5[2].grid()
            plt.savefig(output_dir_rendering + "/test_r_hat.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 3], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("q[0]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 4], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("q[1]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 5], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("q[2]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_q_02t.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 6], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("q[3]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 7], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("q[4]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 8], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("q[5]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_q_35t.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 9], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("dq[0]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 10], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("dq[1]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 11], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("dq[2]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_dq_02t.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 12], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("dq[3]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 13], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("dq[4]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 14], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("dq[5]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_dq_35t.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 15], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("dqc_PI[0]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 16], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("dqc_PI[1]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 17], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("dqc_PI[2]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_dqc_PI_02t.png", format="png",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 18], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("dqc_PI[3]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 19], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("dqc_PI[4]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 20], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("dqc_PI[5]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/test_dqc_PI_35t.png", format="png",
                        bbox_inches='tight')
            plt.show()
            print("")

        if render_training_buffer == True:
            buf_act = np.load(output_dir_rendering + "/buf_act.npy")
            buf_done = np.load(output_dir_rendering + "/buf_done.npy")
            buf_rew = np.load(output_dir_rendering + "/buf_rew.npy")
            buf_obs = np.load(output_dir_rendering + "/buf_obs.npy")
            buf_obs2 = np.load(output_dir_rendering + "/buf_obs2.npy")
            epoch_length = 136
            idx_last = epoch_length * 10000
            idx_start = epoch_length * 100
            idx_end = idx_start + 5 * epoch_length
            # idx_vline = idx_start - 1
            idx_vline = range(idx_start - 1, idx_end, epoch_length)
            fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(16, 10))
            x = range(idx_start, idx_end, 1)
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 21]
            axs[0, 0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='dqc_t_PID[0]')
            axs[0, 0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs[0, 0].set_xlabel("timestep")
            axs[0, 0].set_ylabel("dqc_t_PID[0]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 22]
            axs[1, 0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='dqc_t_PID[1]')
            axs[1, 0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs[1, 0].set_xlabel("timestep")
            axs[1, 0].set_ylabel("dqc_t_PID[1]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 23]
            axs[2, 0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='dqc_t_PID[2]')
            axs[2, 0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs[2, 0].set_xlabel("timestep")
            axs[2, 0].set_ylabel("dqc_t_PID[2]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 24]
            axs[0, 1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='dqc_t_PID[3]')
            axs[0, 1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs[0, 1].set_xlabel("timestep")
            axs[0, 1].set_ylabel("dqc_t_PID[03]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 25]
            axs[1, 1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='dqc_t_PID[4]')
            axs[1, 1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs[1, 1].set_xlabel("timestep")
            axs[1, 1].set_ylabel("dqc_t_PID[4]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 26]
            axs[2, 1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='dqc_t_PID[5]')
            axs[2, 1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs[2, 1].set_xlabel("timestep")
            axs[2, 1].set_ylabel("dqc_t_PID[5]")
            plt.legend()
            # plt.savefig(output_dir_rendering + "/buffer_states.pdf", format="png", bbox_inches='tight')
            plt.show()

            ################################################################################
            ################################################################################
            epoch_length = 136
            idx_last = epoch_length * 10000
            idx_start = epoch_length * 3500
            idx_end = idx_start + 3 * epoch_length
            idx_vline = range(idx_start, idx_end, epoch_length)

            fig6, axs6 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(18, 12))
            x = range(idx_start, idx_end, 1)
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 0]
            axs6[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='r_hat_tp1[0] - rd_t[0]')
            axs6[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs6[0].set_xlabel("timestep")
            axs6[0].set_ylabel("r_hat_tp1[0] - rd_t[0]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 1]
            axs6[1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=4, label='r_hat_tp1[1] - rd_t[1]')
            axs6[1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs6[1].hlines(0, min(x), max(x), 'k', linestyles="dashed")
            axs6[1].set_xlabel("timestep")
            axs6[1].set_ylabel("r_hat_tp1[1] - rd_t[1]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 2]
            axs6[2].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='r_hat_tp1[2] - rd_t[2]')
            axs6[2].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs6[2].set_xlabel("timestep")
            axs6[2].set_ylabel("r_hat_tp1[2] - rd_t[2]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/train_buffer_states", format="pdf", bbox_inches='tight')
            plt.show()

            fig7, axs7 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_act[0:idx_last][idx_start:idx_end, 0]
            axs7[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='a[0]')
            axs7[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs7[0].set_xlabel("timestep")
            axs7[0].set_ylabel("a[0]")
            y = buf_act[0:idx_last][idx_start:idx_end, 1]
            axs7[1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='a[1]')
            axs7[1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs7[1].set_xlabel("timestep")
            axs7[1].set_ylabel("a[1]")
            y = buf_act[0:idx_last][idx_start:idx_end, 2]
            axs7[2].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='a[2]')
            axs7[2].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs7[2].set_xlabel("timestep")
            axs7[2].set_ylabel("a[2]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_action_0to2.pdf", format="png", bbox_inches='tight')
            plt.show()
            fig70, axs70 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_act[0:idx_last][idx_start:idx_end, 3]
            axs70[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='a[3]')
            axs70[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs70[0].set_xlabel("timestep")
            axs70[0].set_ylabel("a[3]")
            y = buf_act[0:idx_last][idx_start:idx_end, 4]
            axs70[1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='a[4]')
            axs70[1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs70[1].set_xlabel("timestep")
            axs70[1].set_ylabel("a[4]")
            y = buf_act[0:idx_last][idx_start:idx_end, 5]
            axs70[2].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='a[5]')
            axs70[2].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs70[2].set_xlabel("timestep")
            axs70[2].set_ylabel("a[5]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_action_3to5.pdf", format="png", bbox_inches='tight')
            plt.show()
            fig8, axs8 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_rew[0:idx_last][idx_start:idx_end]
            axs8[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='reward')
            axs8[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs8[0].set_xlabel("timestep")
            axs8[0].set_ylabel("r")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_reward.pdf", format="png", bbox_inches='tight')
            plt.show()
            fig10, axs10 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_done[0:idx_last][idx_start:idx_end]
            axs10[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='done')
            axs10[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs10[0].set_xlabel("timestep")
            axs10[0].set_ylabel("done")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_done.pdf", format="png", bbox_inches='tight')
            plt.show()
            fig11, axs11 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 3]
            axs11[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='q[0]')
            axs11[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs11[0].set_xlabel("timestep")
            axs11[0].set_ylabel("q[0]")
            axs11[0].hlines(self.q_init[0], min(x), max(x), 'g', linestyles="dashdot")
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 4]
            axs11[1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='q[1]')
            axs11[1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs11[1].set_xlabel("timestep")
            axs11[1].set_ylabel("q[1]")
            axs11[1].hlines(self.q_init[1], min(x), max(x), 'g', linestyles="dashdot")
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 5]
            axs11[2].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='q[2]')
            axs11[2].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs11[2].set_xlabel("timestep")
            axs11[2].set_ylabel("q[2]")
            axs11[2].hlines(self.q_init[2], min(x), max(x), 'g', linestyles="dashdot")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_states_q_0to2.pdf", format="png", bbox_inches='tight')
            plt.show()
            fig110, axs110 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 3]
            axs110[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='q[3]')
            axs110[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs110[0].set_xlabel("timestep")
            axs110[0].set_ylabel("q[3]")
            axs110[0].hlines(self.q_init[3], min(x), max(x), 'g', linestyles="dashdot")
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 4]
            axs110[1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='q[4]')
            axs110[1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs110[1].set_xlabel("timestep")
            axs110[1].set_ylabel("q[4]")
            axs110[1].hlines(self.q_init[4], min(x), max(x), 'g', linestyles="dashdot")
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 5]
            axs110[2].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='q[5]')
            axs110[2].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs110[2].set_xlabel("timestep")
            axs110[2].set_ylabel("q[5]")
            axs110[2].hlines(self.q_init[5], min(x), max(x), 'g', linestyles="dashdot")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_states_q_3to5.pdf", format="png", bbox_inches='tight')
            plt.show()
            print("")

        if render_video == True:
            # pb.disconnect(physics_client)
            # render settings
            # renderer = pb.ER_TINY_RENDERER  # p.ER_BULLET_HARDWARE_OPENGL
            # _width = 224
            # _height = 224
            # _cam_dist = 1.3
            # _cam_yaw = 15
            # _cam_pitch = -30
            # _cam_roll = 0
            # camera_target_pos = [0.2, 0, 0.]
            # _screen_width = 3840  # 1920
            # _screen_height = 2160  # 1080
            # physics_client_rendering = pb.connect(pb.GUI,
            #                                       options='--mp4fps=10 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (
            #                                           _screen_width, _screen_height))
            #
            # dt = 1 / 10  # sec
            # pb.setTimeStep(timeStep=dt, physicsClientId=physics_client_rendering)
            # # physics_client = p.connect(p.GUI,options="--mp4fps=3 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d" % (screen_width, screen_height))
            # # # Set gravity
            # pb.setGravity(0, 0, -9.81, physicsClientId=physics_client_rendering)
            # Load URDFs
            # Load robot, target object and plane urdf
            # /home/mahdi/ETHZ/codes/spinningup
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4,
                                 output_dir_rendering + "/simulation.mp4")  # added by Pierre
            target_object = pb.loadURDF(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                useFixedBase=True, physicsClientId=physics_client)
            conveyor_object = pb.loadURDF(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
                useFixedBase=True, physicsClientId=physics_client)
            plane = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                                useFixedBase=True, physicsClientId=physics_client)
            # Initialise debug camera angle
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=5,
                cameraPitch=-30,
                cameraTargetPosition=camera_target_pos,
                physicsClientId=physics_client_rendering)
            # pb.resetDebugVisualizerCamera(
            #     cameraDistance=_cam_dist,
            #     cameraYaw=_cam_yaw,
            #     cameraPitch=_cam_pitch,
            #     cameraTargetPosition=camera_target_pos,
            #     physicsClientId=physics_client_rendering)
            t = 0
            rd_t = np.array([self.xd[t], self.yd[t], self.zd[t]])
            vd_t = np.array([self.vxd[t], self.vyd[t], self.vzd[t]]) * 1000  # [m/s]
            # Reset robot at the origin and move the target object to the goal position and orientation
            pb.resetBasePositionAndOrientation(
                arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]), physicsClientId=physics_client)
            pb.resetBasePositionAndOrientation(
                target_object, rd_t + [0, 0, -0.07], pb.getQuaternionFromEuler(
                    np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])),
                physicsClientId=physics_client)  # orient just for rendering
            # set conveyer pose and orient
            pb.resetBasePositionAndOrientation(
                conveyor_object,
                np.array([self.xd_init, self.yd_init, self.zd_init]) + np.array([-0.002, -0.18, -0.15]),
                pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]), physicsClientId=physics_client)
            # Reset joint at initial angles
            for i in range(6):
                pb.resetJointState(arm, i, self.q_init[i], physicsClientId=physics_client)
            # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
            for j in range(6, 9):
                pb.resetJointState(arm, j, 0, physicsClientId=physics_client)
            time.sleep(1)

            for t in range(1, self.MAX_TIMESTEPS):
                rd_t = np.array([self.xd[t], self.yd[t], self.zd[t]])
                pb.resetBasePositionAndOrientation(
                    target_object, rd_t + [0, 0, -0.07], pb.getQuaternionFromEuler(
                        np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])), physicsClientId=physics_client)
                dqc_t = self.plot_data_buffer[t, 12:18]
                joint_velocities = list(dqc_t)
                pb.setJointMotorControlArray(
                    arm,
                    [0, 1, 2, 3, 4, 5],
                    controlMode=pb.VELOCITY_CONTROL,
                    targetVelocities=joint_velocities,
                    forces=[87, 87, 87, 87, 12, 12],
                    physicsClientId=physics_client
                )
                # default timestep is 1/240 second
                pb.stepSimulation(physicsClientId=physics_client)
                time.sleep(0.01)

        # np.save(
        #     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/noSACFapv3_17/plot_data_buffer_" + str(
        #         self.n) + ".npy", self.plot_data_buffer)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def bound(x, m, M=None):
        """
        :param x: scalar
        Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
        """
        if M is None:
            M = m[1]
            m = m[0]
        # bound x between min (m) and Max (M)
        return min(max(x, m), M)
