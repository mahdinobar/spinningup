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

sys.path.append('/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch')
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
# #                      "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/simulation.mp4")
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
# /cluster/home/mnobar/code/spinningup
arm = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
                  useFixedBase=True, physicsClientId=physics_client)

# # Create the second physics client
# client_auxilary = pb.connect(pb.DIRECT)  # Use p.GUI for visualization
# pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_auxilary)
# pb.loadURDF("plane.urdf", physicsClientId=client_auxilary)
# arm_auxilary = pb.loadURDF(
#     "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
#     useFixedBase=True, physicsClientId=client_auxilary)
# # # default timestep is 1/240 second (search fixedTimeStep)
# pb.setTimeStep(timeStep=dt_pb_sim, physicsClientId=client_auxilary)
# # # Set gravity
# pb.setGravity(0, 0, -9.81, physicsClientId=client_auxilary)

arm_biased_kinematics = pb.loadURDF(
    "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_1.urdf",
    useFixedBase=True, physicsClientId=physics_client)
# arm_biased_kinematics = pb.loadURDF(
#     "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_biased_kinematics_3.urdf",
#     useFixedBase=True)

# import os
# import rospkg
# import subprocess
# rospack = rospkg.RosPack()
# xacro_filename = os.path.join("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep2/robots/panda/panda.urdf.xacro")
# urdf_filename = os.path.join("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep2/robots/panda/panda.urdf")
# urdf = open(urdf_filename, "w")
#
# # Recompile the URDF to make sure it's up to date
# subprocess.call(['rosrun', 'xacro', 'xacro.py', xacro_filename], stdout=urdf)
#
#
# arm2 = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep2/robots/panda/panda.urdf.xacro",
#                   useFixedBase=True)
target_object = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                            useFixedBase=True, physicsClientId=physics_client)
conveyor_object = pb.loadURDF(
    "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
    useFixedBase=True, physicsClientId=physics_client)
plane = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
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
        self.lpx = 1000
        self.lpy = 1000
        self.lpz = 1000
        self.lv = 10
        self.lddqc = 1
        self.reward_eta_p = 1
        self.reward_eta_v = 0
        self.reward_eta_ddqc = 0
        # TODO: User defined linear position gain
        self.K_p = 0.1
        self.K_i = 0.01
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
        #     checkpoint_q = torch.load('/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/mismatch_learning/gp_model_q{}.pth'.format(str(joint_number)))
        #     model_q.load_state_dict(checkpoint_q['model_state_dict'])
        #     likelihood_q.load_state_dict(checkpoint_q['likelihood_state_dict'])
        #     input_scaler = checkpoint_q['input_scaler']
        #     target_scaler_q = checkpoint_q['target_scaler']
        #     self.model_q.eval()
        #     self.likelihood_q.eval()
        #     # Repeat for dq
        #     likelihood_dq = gpytorch.likelihoods.GaussianLikelihood()
        #     model_dq = GPModel(train_x_shape=(1, input_dim), likelihood=likelihood_dq)
        #     checkpoint_dq = torch.load('/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/mismatch_learning/gp_model_dq{}.pth'.format(str(joint_number)))
        #     model_dq.load_state_dict(checkpoint_dq['model_state_dict'])
        #     likelihood_dq.load_state_dict(checkpoint_dq['likelihood_state_dict'])
        #     target_scaler_dq = checkpoint_dq['target_scaler']
        #     self.model_dq.eval()
        #     self.likelihood_dq.eval()

        # # Initialize lists to hold models and scalers for all joints
        # self.input_scalers = []
        # self.target_scalers_q = []
        # self.target_scalers_dq = []
        # self.models_q = []
        # self.likelihoods_q = []
        # self.models_dq = []
        # self.likelihoods_dq = []
        # # GP_dir = "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/mismatch_learning/trainOnSAC1and2PI1and2and3_testOnSAC3_trackingPhaseOnly/"
        # GP_dir = "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/mismatch_learning/trainOnSAC1and2and3_testOnSAC3_trackingPhaseOnly/"
        # GP_input_dim=2
        # for joint_number in range(6):
        #     # Load scalers
        #     input_scaler = joblib.load(GP_dir+f'input_scaler{joint_number}.pkl')
        #     target_scaler_q = joblib.load(GP_dir+f'target_scaler_q{joint_number}.pkl')
        #     target_scaler_dq = joblib.load(GP_dir+f'target_scaler_dq{joint_number}.pkl')
        #
        #     # Instantiate and load model for q
        #     likelihood_q = gpytorch.likelihoods.GaussianLikelihood()
        #     model_q = GPModel(train_x_shape=(1, GP_input_dim), likelihood=likelihood_q)
        #     checkpoint_q = torch.load(
        #         GP_dir+f'gp_model_q{joint_number}.pth')
        #     model_q.load_state_dict(checkpoint_q['model_state_dict'])
        #     likelihood_q.load_state_dict(checkpoint_q['likelihood_state_dict'])
        #     input_scaler = checkpoint_q['input_scaler']  # overwrite with trained one
        #     target_scaler_q = checkpoint_q['target_scaler']
        #
        #     model_q.eval()
        #     likelihood_q.eval()
        #
        #     # Instantiate and load model for dq
        #     likelihood_dq = gpytorch.likelihoods.GaussianLikelihood()
        #     model_dq = GPModel(train_x_shape=(1, GP_input_dim), likelihood=likelihood_dq)
        #     checkpoint_dq = torch.load(
        #         GP_dir+f'gp_model_dq{joint_number}.pth')
        #     model_dq.load_state_dict(checkpoint_dq['model_state_dict'])
        #     likelihood_dq.load_state_dict(checkpoint_dq['likelihood_state_dict'])
        #     target_scaler_dq = checkpoint_dq['target_scaler']
        #
        #     model_dq.eval()
        #     likelihood_dq.eval()
        #
        #     # Append to lists
        #     self.input_scalers.append(input_scaler)
        #     self.target_scalers_q.append(target_scaler_q)
        #     self.target_scalers_dq.append(target_scaler_dq)
        #     self.models_q.append(model_q)
        #     self.likelihoods_q.append(likelihood_q)
        #     self.models_dq.append(model_dq)
        #     self.likelihoods_dq.append(likelihood_dq)

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
            # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
            pb.resetJointState(arm, 7, 1.939142517407308, physicsClientId=physics_client)
            pb.resetJointState(arm_biased_kinematics, 7, 1.939142517407308, physicsClientId=physics_client)
            for j in [6] + list(range(8, 12)):
                pb.resetJointState(arm, j, 0, physicsClientId=physics_client)
                pb.resetJointState(arm_biased_kinematics, j, 0, physicsClientId=physics_client)
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
                                                                         list(np.zeros(9)), physicsClientId=physics_client)

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
                    break #exit the inner while loop
            if restart_outer==True:
                continue #restart the reset
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
        Jpinv_t = self.pseudoInverseMat(J_t, ld=0.01)
        # ATTENTION: here we calculate the self.dqc_PID ready but we do not step simulation, and keep it for "step" to set with a
        self.dqc_PID, self.edt = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t,
                                                edt=self.edt,
                                                deltaT=dt_startup)

        q_t = np.array(q_t)[:6]
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
        R = np.array([[0.0625e-6, 0, 0], [0, 0.0625e-6, 0], [0, 0, 0.0625e-6]])  # [m^2]
        # process uncertainty covariance
        Q = np.array([[0.01e-6, 0, 0], [0, 0.04e-6, 0], [0, 0, 0.02e-6]])  # #[m^2]
        # initial covariance matrix
        P0 = np.asmatrix(np.diag([0.04e-6, 0.09e-6, 0.04e-6]))

        # Generate time stamp randomness of camera measurements
        time_randomness = np.random.normal(0, 32, 137).astype(int)
        time_randomness = np.clip(time_randomness, -49, 49)
        time_randomness[0] = np.clip(time_randomness[0], 1, 49)
        tVec_camera = np.linspace(0, 13600, 137) + time_randomness  # [ms]
        # self.vxd = (np.random.normal(loc=0.0, scale=0.000367647, size=1)[
        #     0]) / 1000  # [m/ms] for 2 [cm] drift given std error after 13.6 [s]
        # self.vyd = (34.9028e-3 + np.random.normal(loc=0.0, scale=0.002205882, size=1)[
        #     0]) / 1000  # [m/ms] for 5 [cm] drift given std error after 13.6 [s]
        # self.vzd = 0
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
        x_camera = x_camera + np.random.normal(loc=0.0, scale=0.0005, size=137) # [m]
        y_camera = y_camera + np.random.normal(loc=0.0, scale=0.001, size=137)  # [m]
        z_camera = z_camera + np.random.normal(loc=0.0, scale=0.0005, size=137) # [m]
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

        plot_data_t = [r_hat_t[0],
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
        self.plot_data_buffer = plot_data_t
        return self.state

    def step(self, a):
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
        # TODO pay attention to number of repetition (e.g., use 24 for period 24*1/240*1000=100 [ms])
        for _ in range(24):
            # default timestep is 1/240 second
            pb.stepSimulation(physicsClientId=physics_client)

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
        q_tp1, dq_tp1, tau_tp1 = [], [], []
        for joint_info in info:
            q_tp1.append(joint_info[0])
            dq_tp1.append(joint_info[1])
            tau_tp1.append(joint_info[3])

            # # Attention: hard reset for biased kinematics model
        for i in range(12):
            if i < 6:
                pb.resetJointState(arm_biased_kinematics, i, q_tp1[i] + self.q_init_bias[i],
                                   physicsClientId=physics_client)
            else:
                pb.resetJointState(arm_biased_kinematics, i, q_tp1[i], physicsClientId=physics_client)

        q_tp1 = np.array(q_tp1)[:6]
        # add dq measurement noise
        dq_tp1 = np.array(dq_tp1)[:6] + np.random.normal(loc=0.0, scale=0.004, size=6)

        # #########################################################################
        # # ----- q and dq Mismatch Compensation -----
        # for i in range(6):
        #     self.models_q[i].eval()
        #     self.models_dq[i].eval()
        #     self.likelihoods_q[i].eval()
        #     self.likelihoods_dq[i].eval()
        #     X_test = np.array([q_tp1[i], dq_tp1[i]]).reshape(-1, 2)
        #     X_test = self.input_scalers[i].transform(X_test)
        #     X_test = torch.tensor(X_test, dtype=torch.float32)
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         pred_q = self.likelihoods_q[i](self.models_q[i](X_test))
        #         pred_dq = self.likelihoods_dq[i](self.models_dq[i](X_test))
        #         mean_q = pred_q.mean.numpy()
        #         mean_dq = pred_dq.mean.numpy()
        #         std_q = pred_q.variance.sqrt().numpy()
        #         std_dq = pred_dq.variance.sqrt().numpy()
        #         # Uncomment when Normalizing
        #         mean_q = self.target_scalers_q[i].inverse_transform(mean_q.reshape(-1, 1)).flatten()
        #         std_q = std_q * self.target_scalers_q[i].scale_[0]  # only scale, don't shift
        #         mean_dq = self.target_scalers_dq[i].inverse_transform(mean_dq.reshape(-1, 1)).flatten()
        #         std_dq = std_dq * self.target_scalers_dq[i].scale_[0]  # only scale, don't shift
        #     # TODO
        #     if ~np.isnan(mean_q):
        #         q_tp1[i] = q_tp1[i] + mean_q
        #     else:
        #         print("mean_q[{}] is nan!".format(i))
        #     if ~np.isnan(mean_dq):
        #         dq_tp1[i] = dq_tp1[i] + mean_dq
        #     else:
        #         print("mean_dq[{}] is nan!".format(i))
        # #########################################################################

        # add tau measurement noise and bias
        tau_tp1 = np.array(tau_tp1)[:6]  # + np.random.normal(loc=0.0, scale=0.08, size=6) #+ np.array(
        # [0.31, 9.53, 1.76, -9.54, 0.89, -2.69])
        self.q = np.vstack((self.q, q_tp1))  # Attention
        self.dq = np.vstack((self.dq, dq_tp1))  # Attention
        # check done episode
        terminal = self._terminal()
        # calculate reward
        # define inspired by Pavlichenko et al SAC tracking paper https://doi.org/10.48550/arXiv.2203.07051
        # todo make more efficient by calling getLinkState only once
        LinkState_tp1 = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True,
                                        physicsClientId=physics_client)
        # TODO CHECK HERE: is there bug? why not use LinkState_tp1 or should I use LinkState?
        r_hat_tp1 = np.array(LinkState_tp1[0])
        v_hat_tp1 = np.array(LinkState_tp1[6])
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

        # Attention: use biased kinematics model for jacobian calculation
        [linearJacobian_tp1, angularJacobian_tp1] = pb.calculateJacobian(arm_biased_kinematics,
                                                                         10,
                                                                         list(LinkState_tp1[2]),
                                                                         list(
                                                                             np.append(self.q[-1, :] + self.q_init_bias,
                                                                                       [0, 0, 0])),
                                                                         list(np.append(self.dq[-1, :], [0, 0, 0])),
                                                                         list(np.zeros(9)),
                                                                         physicsClientId=physics_client)

        [linearJacobian_TRUE_tp1, angularJacobian_TRUE_tp1] = pb.calculateJacobian(arm,
                                                                                   10,
                                                                                   list(LinkState_tp1[2]),
                                                                                   list(np.append(self.q[-1, :],
                                                                                                  [0, 0, 0])),
                                                                                   list(np.append(self.dq[-1, :],
                                                                                                  [0, 0, 0])),
                                                                                   list(np.zeros(9)),
                                                                                   physicsClientId=physics_client)
        J_tp1 = np.asarray(linearJacobian_tp1)[:, :6]
        Jpinv_tp1 = self.pseudoInverseMat(J_tp1, ld=0.01)
        J_tp1_TRUE = np.asarray(linearJacobian_TRUE_tp1)[:, :6]
        rd_tp1_error = np.matmul(J_tp1_TRUE, self.pseudoInverseMat(J_tp1, ld=0.0001)) @ rd_tp1 - rd_tp1
        dqc_tp1_PID, self.edt = self.q_command(r_ee=r_hat_tp1, v_ee=v_hat_tp1, Jpinv=Jpinv_tp1, rd=rd_tp1, vd=vd_tp1,
                                               edt=self.edt,
                                               deltaT=dt)
        self.dqc_PID = dqc_tp1_PID
        # observations after applying the action a
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
        plot_data_t = [r_hat_tp1[0],
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
        self.plot_data_buffer = np.vstack((self.plot_data_buffer, plot_data_t))
        # # # # TODO: so dirty code: uncomment when NOSAC for plots -- you need to take care of which random values you call by break points after first done in sac.py ... and cmment a too ...
        # plot_data_buffer_no_SAC=self.plot_data_buffer
        # np.save("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/Fep_HW_284/plot_data_buffer_no_SAC.npy",plot_data_buffer_no_SAC)
        # given action it returns 4-tuple (observation, reward, done, info)
        return (obs, reward_t, terminal, {})

    def _terminal(self):
        return bool(self.k >= self.MAX_TIMESTEPS - 1)

    def render(self, output_dir_rendering, mode='human'):
        """ Render Pybullet simulation """
        render_video = False  # TODO
        render_test_buffer = True
        render_training_buffer = False
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
            # /cluster/home/mnobar/code/spinningup
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4,
                                 output_dir_rendering + "/simulation.mp4")  # added by Pierre
            target_object = pb.loadURDF(
                "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                useFixedBase=True, physicsClientId=physics_client)
            conveyor_object = pb.loadURDF(
                "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
                useFixedBase=True, physicsClientId=physics_client)
            plane = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
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
        #     "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/noSACFapv3_17/plot_data_buffer_" + str(
        #         self.n) + ".npy", self.plot_data_buffer)
        # render_test_buffer=False
        if render_test_buffer == True:
            # # np.save("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/noSACFapv3_17/plot_data_buffer_"+str(self.n)+".npy", self.plot_data_buffer)
            plot_data_buffer_no_SAC = np.load(
                "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/Fep_HW_284/plot_data_buffer_no_SAC_correctKF.npy")
            plots_PIonly = True
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

            # fig5, axs5 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 6))
            # t = np.linspace(0, 135, 136) / 10
            # axs5.plot(t, self.plot_data_buffer[:, 1] * 1000, '-ob', MarkerSize=3, label="r_hat_tp1[1], [mm]")
            # axs5.plot(t, self.plot_data_buffer[:, 4] * 1000, '-og', MarkerSize=3, label="rd_tp1[1], [mm]")
            # axs5.set_xlabel("t")
            # axs5.set_ylabel("y")
            # plt.legend()
            # plt.grid()
            # axs5.set_xlim([0, 1.8])
            # axs5.set_ylim([-250, -180])
            # plt.savefig(output_dir_rendering + "/tmp_checcking_KF" + str(self.n) + ".png", format="png",
            #             bbox_inches='tight')
            # plt.show()

            fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 14))
            plt.rcParams['font.family'] = 'Serif'
            axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         abs(plot_data_buffer_no_SAC[:, 0] - plot_data_buffer_no_SAC[:, 3]) * 1000, '-ob',
                         label='without SAC')
            axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         abs(self.plot_data_buffer[:, 0] - self.plot_data_buffer[:, 3]) * 1000, '-or', label='with SAC')
            axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(self.plot_data_buffer[:, 30]) * 1000, 'r:',
                         label='error bound with SAC')
            axs3[0].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(plot_data_buffer_no_SAC[:, 30]) * 1000, 'b:',
                         label='error bound without SAC')
            axs3[0].set_xlabel("t [ms]")
            axs3[0].set_ylabel("|x-xd| [mm]")
            axs3[0].set_ylim([0, 12])
            axs3[0].legend(loc="upper right")
            axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         abs(plot_data_buffer_no_SAC[:, 1] - plot_data_buffer_no_SAC[:, 4]) * 1000, '-ob',
                         label='without SAC')
            axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         abs(self.plot_data_buffer[:, 1] - self.plot_data_buffer[:, 4]) * 1000, '-or', label='with SAC')
            axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(self.plot_data_buffer[:, 31]) * 1000, 'r:',
                         label='error bound on with SAC')
            axs3[1].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(plot_data_buffer_no_SAC[:, 31]) * 1000, 'b:',
                         label='error bound on without SAC')
            axs3[1].set_xlabel("t [ms]")
            axs3[1].set_ylabel("|y-yd| [mm]")
            axs3[1].set_ylim([0, 12])
            axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         abs(plot_data_buffer_no_SAC[:, 2] - plot_data_buffer_no_SAC[:, 5]) * 1000, '-ob',
                         label='without SAC')
            axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         abs(self.plot_data_buffer[:, 2] - self.plot_data_buffer[:, 5]) * 1000, '-or', label='with SAC')
            axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(self.plot_data_buffer[:, 32]) * 1000, 'r:',
                         label='error bound on with SAC')
            axs3[2].plot(np.arange(self.MAX_TIMESTEPS) * 100, abs(plot_data_buffer_no_SAC[:, 32]) * 1000, 'b:',
                         label='error bound on without SAC')
            axs3[2].set_xlabel("t [ms]")
            axs3[2].set_ylabel("|z-zd| [mm]")
            axs3[2].set_ylim([0, 12])
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         np.linalg.norm((plot_data_buffer_no_SAC[:, 0:3] - plot_data_buffer_no_SAC[:, 3:6]), ord=2,
                                        axis=1) * 1000, '-ob', label='without SAC')
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2,
                                        axis=1) * 1000,
                         '-or', label='with SAC')
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         np.linalg.norm(self.plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
                         'r:', label='error bound on with SAC')
            axs3[3].plot(np.arange(self.MAX_TIMESTEPS) * 100,
                         np.linalg.norm(plot_data_buffer_no_SAC[:, 30:33], ord=2, axis=1) * 1000,
                         'b:', label='error bound on without SAC')
            axs3[3].set_xlabel("t [ms]")
            axs3[3].set_ylabel("||r-rd||_2 [mm]")
            axs3[3].set_ylim([0, 12])
            plt.savefig(output_dir_rendering + "/test_position_errors_both.pdf",
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
            axs5[1].plot(self.plot_data_buffer[:, 4], '-ob')
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
            plt.savefig(output_dir_rendering + "/test_q_02t" + str(self.n) + ".png", format="png",
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
            plt.savefig(output_dir_rendering + "/test_q_35t" + str(self.n) + ".png", format="png",
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
            plt.savefig(output_dir_rendering + "/test_dq_02t" + str(self.n) + ".png", format="png",
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
            plt.savefig(output_dir_rendering + "/test_dq_35t" + str(self.n) + ".png", format="png",
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
            plt.savefig(output_dir_rendering + "/test_dqc_PI_02t" + str(self.n) + ".png", format="png",
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
            plt.savefig(output_dir_rendering + "/test_dqc_PI_35t" + str(self.n) + ".png", format="png",
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
