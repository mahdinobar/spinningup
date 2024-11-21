"""Two-link RR Planar Manipulator Tracking Task"""
import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
from scipy.integrate import solve_ivp
import math
import pybullet as pb
import pybullet_data
import matplotlib.pyplot as plt
import time

__copyright__ = "Copyright 2024, IfA https://control.ee.ethz.ch/"
__credits__ = ["Mahdi Nobar"]
__author__ = "Mahdi Nobar from ETH Zurich <mnobar@ethz.ch>"

# Connect to physics client
physics_client = pb.connect(pb.DIRECT)

# TDOO ATTENTION how you choose dt
dt = 1e-3  # sec

# TODO: improve code for STARTUP_exist
STARTUP_exist = False
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
# pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4,
#                      "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/simulation.mp4")
# # Initialise debug camera angle
# pb.resetDebugVisualizerCamera(
#     cameraDistance=1.2,
#     cameraYaw=5,
#     cameraPitch=-30,
#     cameraTargetPosition=camera_target_pos,
#     physicsClientId=physics_client)

# # default timestep is 1/240 second (search fixedTimeStep)
pb.setTimeStep(timeStep=dt, physicsClientId=physics_client)
# # Set gravity
pb.setGravity(0, 0, -9.81, physicsClientId=physics_client)
# Load URDFs
# Load robot, target object and plane urdf
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
arm = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
                  useFixedBase=True)
arm_biased_kinematics = pb.loadURDF(
    "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_1.urdf",
    useFixedBase=True)
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
                            useFixedBase=True)
conveyor_object = pb.loadURDF(
    "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
    useFixedBase=True)
plane = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                    useFixedBase=True)


class FepEnv(core.Env):
    """
    Two-link planar arm with two revolut joints (based on simplified models at book "A Mathematical Introduction to
Robotic Manipulation" by Murry et al.
    """

    def __init__(self):
        seed = 1
        self.n = 0
        # reset seed(here is where seed is reset to count 0)
        np.random.seed(seed)
        self.seed(seed=seed)
        # TODO: reward params
        self.lp = 800
        self.lv = 10
        self.lddqc = 1
        self.reward_eta_p = 1
        self.reward_eta_v = 0
        self.reward_eta_ddqc = 0
        # TODO: User defined linear position gain
        self.K_p = 5
        self.K_i = 0.5
        self.K_d = 0
        self.korque_noise_max = 0.  # TODO
        self.viewer = None
        self.state = None
        self.state_buffer = None
        self.k = 0
        # self.xd_init = 0.43086
        # self.yd_init = -0.07530
        # self.zd_init = 0.17432
        # Attention: this is the tf final of startup phase
        self.xd_init = 0.5345
        self.yd_init = -0.2455
        self.zd_init = 0.1392

        # TODO correct q_init
        # self.q_init = np.deg2rad(np.array([-23.1218, 3.6854, 13.0462, -148.512, -8.7462, 150.2532]))
        # TDOO ATTENTION how you choose MAX_TIMESTEPS
        self.MAX_TIMESTEPS = 13600  # maximum timesteps per episode
        self.vxd = 0  # m/s
        self.vyd = 34.9028e-3  # m/s
        self.vzd = 0  # m/s
        deltax = self.vxd * dt * self.MAX_TIMESTEPS
        deltay = self.vyd * dt * self.MAX_TIMESTEPS
        deltaz = self.vzd * dt * self.MAX_TIMESTEPS
        self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
        self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
        self.zd = np.linspace(self.zd_init, self.zd_init + deltaz, self.MAX_TIMESTEPS, endpoint=True)
        # TODO Attention: just the dimension of the observation space is enforced. The data here is not used. If you need to enforce them then modify the code.
        # Attention just 6 DOF is simulated (7th DOF is disabled)
        high_s = np.array([0.2, 0.2, 0.2,
                           1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                           2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100,
                           87, 87, 87, 87, 12, 12,
                           2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100])
        low_s = -high_s
        self.observation_space = spaces.Box(low=low_s, high=high_s, dtype=np.float32)
        # Attention just 6 DOF is simulated (7th DOF is disabled)
        # Attention: limits of SAC actions
        high_a = 0.2 * np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100,
                                 2.6100])  # TODO Attention: limits should be the same otherwise modify sac code
        # high_a = 0.05 * np.array([2.1750, 2.1750, 2.1750])  # TODO Attention: limits should be the same otherwise modify sac code
        low_a = -high_a
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        # output_dir_rendering = "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/"

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

    def q_command(self, r_ee, v_ee, Jpinv, rd, vd, e, dt):
        """
        PID Traj Tracking Feedback Controller
        Inputs:
            r_ee          : current end effector position
            rd       : desired end effector position
            vd       : desired end effector velocity
            Jpinv : current pseudo inverse jacobian matrix
        Output: joint-space velocity command of the robot.
        """
        e_t = (rd - r_ee)
        e = np.vstack((e, e_t.reshape(1, 3)))
        v_command = vd + self.K_p * e_t + self.K_i * np.sum(e[1:, :], 0) * dt + self.K_d * (vd - v_ee)
        dqc = np.dot(Jpinv, v_command)
        return dqc, e

    def f_logistic(self, x, l):
        H = 2
        return H / (math.e ** (x * l) + math.e ** (-x * l))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if STARTUP_exist==True:
            "reset considering the startup phase on real system"
            self.n += 1
            self.k = 0

            # Reset robot at the origin and move the target object to the goal position and orientation
            pb.resetBasePositionAndOrientation(
                arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
            pb.resetBasePositionAndOrientation(
                arm_biased_kinematics, [100, 100, 100], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))

            q_init_noise = False
            if q_init_noise == True:
                # we add random normal noise with std of 0.5 degrees and zero mean on all 6 joints
                # q_init is the inital condition of the startup phase
                self.q_init = np.array(
                    [-0.22683544711236076, 0.4152892646837951, -0.2240776697835826, -2.029656763049754,
                     -0.1323494169192162,
                     2.433754967707292]) + np.random.normal(
                    loc=0.0,
                    scale=0.00872,
                    size=6)
            else:
                self.q_init = np.array(
                    [-0.22683544711236076, 0.4152892646837951, -0.2240776697835826, -2.029656763049754,
                     -0.1323494169192162,
                     2.433754967707292])
            # Reset joint at initial angles
            for i in range(6):
                pb.resetJointState(arm, i, self.q_init[i])
                pb.resetJointState(arm_biased_kinematics, i, self.q_init[i])
            # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
            pb.resetJointState(arm, 7, 1.939142517407308)
            pb.resetJointState(arm_biased_kinematics, 7, 1.939142517407308)
            for j in [6] + list(range(8, 12)):
                pb.resetJointState(arm, j, 0)
                pb.resetJointState(arm_biased_kinematics, j, 0)
            # Get end effector coordinates
            LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
            r_hat_t = np.asarray(LinkState[0])
            v_hat_t = np.asarray(LinkState[6])
            info = pb.getJointStates(arm, range(12))
            q_t, dq_t, tau_t = [], [], []
            for joint_info in info:
                q_t.append(joint_info[0])
                dq_t.append(joint_info[1])
                tau_t.append(joint_info[3])
            # q_t = np.array(q_t)[:6]
            # dq_t = np.array(dq_t)[:6]
            if abs(sum(self.q_init - q_t[:6])) > 1e-6:
                raise ValueError('shouldn\'t q_init be equal to q_t?!')

            # STARTUP PHASE
            # ATTENTION: startup phase should be at 1 [ms]
            pb.setTimeStep(timeStep=1e-3, physicsClientId=physics_client)
            v_star_dir = np.array([self.xd_init, self.yd_init, self.zd_init]) - r_hat_t
            norm_v_star_dir = np.linalg.norm(v_star_dir)
            k_startup = 0
            while np.linalg.norm(r_hat_t - np.array([self.xd_init, self.yd_init, self.zd_init])) > 0.002:
                v_star_dir_length = 34.9028 / (1 + np.exp(-0.04 * (k_startup - 250))) / 1000 - 34.9028 / (
                        1 + np.exp(-0.04 * (0 - 250))) / 1000;
                v_star = v_star_dir_length * (v_star_dir / norm_v_star_dir)
                self.vxd = v_star[0]
                self.vyd = v_star[1]
                self.vzd = v_star[2]

                deltax = self.vxd * 1e-3
                deltay = self.vyd * 1e-3
                deltaz = self.vzd * 1e-3

                self.xd = r_hat_t[0] + deltax
                self.yd = r_hat_t[1] + deltay
                self.zd = r_hat_t[2] + deltaz

                rd_t = np.array([self.xd, self.yd, self.zd])
                vd_t = np.array([self.vxd, self.vyd, self.vzd])

                if k_startup == 0:
                    e = rd_t - r_hat_t
                    self.e = e.reshape(1, 3)

                [linearJacobian, angularJacobian] = pb.calculateJacobian(arm_biased_kinematics,
                                                                         10,
                                                                         list(LinkState[2]),
                                                                         list(np.append(q_t[:6], [0, 0, 0])),
                                                                         list(np.append(dq_t[:6], [0, 0, 0])),
                                                                         list(np.zeros(9)))

                J_t = np.asarray(linearJacobian)[:, :6]
                Jpinv_t = self.pseudoInverseMat(J_t, ld=0.01)

                dqc_t_PID, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t,
                                                   e=self.e,
                                                   dt=1e-3)
                # ATTENTION: here apply SAC action
                dqc_t = dqc_t_PID
                # TODO check
                # command joint speeds (only 6 joints)
                pb.setJointMotorControlArray(
                    arm,
                    [0, 1, 2, 3, 4, 5],
                    controlMode=pb.VELOCITY_CONTROL,
                    targetVelocities=list(dqc_t),
                    forces=[87, 87, 87, 87, 12, 12]
                )
                pb.stepSimulation(physicsClientId=physics_client)
                # # Attention: hard reset for biased kinematics model
                for i in range(12):
                    pb.resetJointState(arm_biased_kinematics, i, q_t[i])
                LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
                # TODO CHECK HERE: is there bug? why not use LinkState_tp1 or should I use LinkState?
                r_hat_t = np.array(LinkState[0])
                v_hat_t = np.array(LinkState[6])

                k_startup += 1
                if k_startup > 5000:
                    raise ValueError("took too long for startup phase!")

            # ATTENTION set back simulation frequency after startup phase
            pb.setTimeStep(timeStep=dt, physicsClientId=physics_client)
            # TODO pay attention
            noisy_target = True
            if noisy_target == True:
                self.vxd = 0 + np.random.normal(loc=0.0, scale=0.00077, size=1)[0]  # m/s
                self.vyd = 34.9028e-3 + np.random.normal(loc=0.0, scale=0.00231, size=1)[0]  # m/s
                print("self.vxd=", self.vxd)
                print("self.vyd=", self.vyd)
                self.vzd = 0  # m/s
                deltax = self.vxd * dt * self.MAX_TIMESTEPS
                deltay = self.vyd * dt * self.MAX_TIMESTEPS
                deltaz = self.vzd * dt * self.MAX_TIMESTEPS
                self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
                self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
                self.zd = np.linspace(self.zd_init, self.zd_init + deltaz, self.MAX_TIMESTEPS,
                                      endpoint=True) + np.random.normal(loc=0.0, scale=0.001, size=self.MAX_TIMESTEPS)
            rd_t = np.array([self.xd[self.k], self.yd[self.k], self.zd[self.k]])
            vd_t = np.array([self.vxd, self.vyd, self.vzd])
            # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
            info = pb.getJointStates(arm, range(12))
            q_t, dq_t, tau_t = [], [], []
            for joint_info in info:
                q_t.append(joint_info[0])
                dq_t.append(joint_info[1])
                tau_t.append(joint_info[3])
            q_t = np.array(q_t)[:6]
            dq_t = np.array(dq_t)[:6]
            tau_t = np.array(tau_t)[:6]
            e0 = rd_t - r_hat_t
            self.e = e0.reshape(1, 3)
            self.q = q_t.reshape(1, 6)
            self.dq = dq_t.reshape(1, 6)
            a = np.zeros(6)  # TODO check

            pb.resetBasePositionAndOrientation(
                target_object, rd_t, pb.getQuaternionFromEuler(
                    np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])))  # orient just for rendering
            # set conveyer pose and orient
            pb.resetBasePositionAndOrientation(
                conveyor_object,
                np.array([self.xd_init, self.yd_init, self.zd_init]) + np.array([-0.002, -0.18, -0.15]),
                pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]))

            self.state = [r_hat_t[0] - rd_t[0],
                          r_hat_t[1] - rd_t[1],
                          r_hat_t[2] - rd_t[2],
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
                          tau_t[0],
                          tau_t[1],
                          tau_t[2],
                          tau_t[3],
                          tau_t[4],
                          tau_t[5],
                          dqc_t_PID[0],
                          dqc_t_PID[1],
                          dqc_t_PID[2],
                          dqc_t_PID[3],
                          dqc_t_PID[4],
                          dqc_t_PID[5]]
            self.state_buffer = self.state
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
                           q_t[0],
                           q_t[1],
                           q_t[2],
                           q_t[3],
                           q_t[4],
                           q_t[5],
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
                           a[0],
                           a[1],
                           a[2],
                           a[3],
                           a[4],
                           a[5]]
            self.plot_data_buffer = plot_data_t
            return self._get_ob()

        elif STARTUP_exist == False:
            # # randomize true model parameter in every episode
            # self.LINK_MASS_2_TRUE = 1.1 + np.random.normal(loc=0.0, scale=0.01, size=1)
            # # randomize PID gains
            # self.K_p = 2 + np.random.normal(loc=0.0, scale=1, size=1)
            # self.K_i = 0.5 + np.random.normal(loc=0.0, scale=0.5, size=1)
            # self.K_d = 0.1 + np.random.normal(loc=0.0, scale=0.1, size=1)
            # at time t=0
            self.n += 1
            self.k = 0
            # TODO pay attention
            noisy_target = True
            if noisy_target == True:
                self.vxd = 0 + np.random.normal(loc=0.0, scale=0.00077, size=1)[0]  # m/s
                self.vyd = 34.9028e-3 + np.random.normal(loc=0.0, scale=0.00231, size=1)[0]  # m/s
                print("self.vxd=", self.vxd)
                print("self.vyd=", self.vyd)
                self.vzd = 0  # m/s
                deltax = self.vxd * dt * self.MAX_TIMESTEPS
                deltay = self.vyd * dt * self.MAX_TIMESTEPS
                deltaz = self.vzd * dt * self.MAX_TIMESTEPS
                self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
                self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
                self.zd = np.linspace(self.zd_init, self.zd_init + deltaz, self.MAX_TIMESTEPS,
                                      endpoint=True) + np.random.normal(loc=0.0, scale=0.001, size=self.MAX_TIMESTEPS)
            rd_t = np.array([self.xd[self.k], self.yd[self.k], self.zd[self.k]])
            vd_t = np.array([self.vxd, self.vyd, self.vzd])
            # Reset robot at the origin and move the target object to the goal position and orientation
            pb.resetBasePositionAndOrientation(
                arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
            pb.resetBasePositionAndOrientation(
                arm_biased_kinematics, [100, 100, 100], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
            pb.resetBasePositionAndOrientation(
                target_object, rd_t, pb.getQuaternionFromEuler(
                    np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])))  # orient just for rendering
            # set conveyer pose and orient
            pb.resetBasePositionAndOrientation(
                conveyor_object,
                np.array([self.xd_init, self.yd_init, self.zd_init]) + np.array([-0.002, -0.18, -0.15]),
                pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]))
            # np.array(
            #     pb.calculateInverseKinematics(arm, 9, list(rd_t), solver=0, residualThreshold=1e-6, maxNumIterations=1000)[
            #     0:6])
            q_init_noise = False
            if q_init_noise == True:
                # we add random normal noise with std of 0.5 degrees and zero mean on all 6 joints
                self.q_init = np.array(
                    [-0.204535,
                     0.401101,
                     -0.201406,
                     -2.055715,
                     -0.127495,
                     2.418523]) + np.random.normal(
                    loc=0.0,
                    scale=0.0174,
                    size=6)
                # self.q_init = np.array(
                #     [-0.38198187, 1.32720032, -0.17534288, -0.3604967, -0.16008594, 0.4936846]) + np.random.normal(
                #     loc=0.0,
                #     scale=0.0174,
                #     size=6)
                print("self.q_init=", self.q_init)
                # self.q_init = np.array(
                #     [-0.38198187, 1.32720032, -0.17534288, -0.3604967, -0.16008594, 0.4936846]) + np.random.normal(
                #     loc=0.0,
                #     scale=0.0087266,
                #     size=6)
                # self.q_init = np.array(
                #     [-0.44282133, -0.27180934, 0.17985816, -2.65595454, -0.16388257, 2.47417267]) + np.random.normal(
                #     loc=0.0,
                #     scale=0.02,
                #     size=6)
                # self.q_init = np.array(
                #     [-0.42529795, 0.11298615, 0.20446317, -2.52843438, -0.15231932, 2.63230466]) + np.random.normal(loc=0.0,
                #                                                                                                     scale=0.02,
                #                                                                                                     size=6)
            else:
                self.q_init = np.array([-0.20251,
                                        0.368896,
                                        -0.204403,
                                        -2.06126,
                                        -0.131167,
                                        2.4026])

            # Reset joint at initial angles
            for i in range(6):
                pb.resetJointState(arm, i, self.q_init[i])
                pb.resetJointState(arm_biased_kinematics, i, self.q_init[i])
            # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
            pb.resetJointState(arm, 7, 1.946597)
            pb.resetJointState(arm_biased_kinematics, 7, 1.946597)
            for j in [6] + list(range(8, 12)):
                pb.resetJointState(arm, j, 0)
                pb.resetJointState(arm_biased_kinematics, j, 0)
            # Get end effector coordinates
            # LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
            # r_hat_t = np.array(LinkState[0])
            # v_hat_t = np.array(LinkState[6])
            # get values at center of two fingers links 9 and 10 with indices 8 and 9 (assume COM has zero local offset from finger links origin)
            # r_hat_t = (np.asarray(pb.getLinkState(arm, 8, computeForwardKinematics=True, computeLinkVelocity=True)[0])+np.asarray(pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)[0]))/2
            # v_hat_t = (np.asarray(pb.getLinkState(arm, 8, computeForwardKinematics=True, computeLinkVelocity=True)[6])+np.asarray(pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)[6]))/2
            LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
            r_hat_t = np.asarray(LinkState[0])
            v_hat_t = np.asarray(LinkState[6])
            info = pb.getJointStates(arm, range(12))
            q_t, dq_t, tau_t = [], [], []
            for joint_info in info:
                q_t.append(joint_info[0])
                dq_t.append(joint_info[1])
                tau_t.append(joint_info[3])
            q_t = np.array(q_t)[:6]

            if abs(sum(self.q_init - q_t)) > 1e-6:
                raise ValueError('shouldn\'t q_init be equal to q_t?!')
            modified_initial_condition = True
            if modified_initial_condition == False:
                dq_t = np.array(dq_t)[:6]
                tau_t = np.array(tau_t)[:6]  # CHECK!!!!!!!!!!!!!!!!!!!!¨ if tau_t is not 0 what is it and why?
                dqc_t = np.zeros(6)  # TODO check
                dqc_t_PID = np.zeros(6)  # TODO check
                a = np.zeros(6)  # TODO check
                self.q = q_t.reshape(1, 6)
                self.dq = dq_t.reshape(1, 6)
                e0 = rd_t - r_hat_t
                self.e = e0.reshape(1, 3)
            elif modified_initial_condition == True:
                dq_t = np.array(dq_t)[:6]
                a = np.zeros(6)  # TODO check
                e0 = rd_t - r_hat_t
                self.e = e0.reshape(1, 3)
                self.q = q_t.reshape(1, 6)
                self.dq = dq_t.reshape(1, 6)
                [linearJacobian, angularJacobian] = pb.calculateJacobian(arm_biased_kinematics,
                                                                         9,
                                                                         list(LinkState[2]),
                                                                         list(np.append(self.q[-1, :], [0, 0, 0])),
                                                                         list(np.append(self.dq[-1, :], [0, 0, 0])),
                                                                         list(np.zeros(9)))

                J_t = np.asarray(linearJacobian)[:, :6]
                Jpinv_t = self.pseudoInverseMat(J_t, ld=0.01)
                dqc_t_PID, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t,
                                                   e=self.e,
                                                   dt=dt)
                pb.setJointMotorControlArray(
                    arm,
                    [0, 1, 2, 3, 4, 5],
                    controlMode=pb.VELOCITY_CONTROL,
                    targetVelocities=list(dqc_t_PID),
                    forces=[87, 87, 87, 87, 12, 12]
                )
                # imposes setPhysicsEngineParameter timestep (default is 240 Hz)
                pb.stepSimulation(physicsClientId=physics_client)
                # Get end effector coordinates
                LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
                r_hat_t = np.array(LinkState[0])
                v_hat_t = np.array(LinkState[6])
                info = pb.getJointStates(arm, range(12))
                q_t, dq_t, tau_t = [], [], []
                for joint_info in info:
                    q_t.append(joint_info[0])
                    dq_t.append(joint_info[1])
                    tau_t.append(joint_info[3])
                # # Attention: hard reset for biased kinematics model
                for i in range(12):
                    pb.resetJointState(arm_biased_kinematics, i, q_t[i])
                q_t = np.array(q_t)[:6]
                dq_t = np.array(dq_t)[:6]
                tau_t = np.array(tau_t)[:6]
                e0 = rd_t - r_hat_t
                self.e = e0.reshape(1, 3)
                self.q = q_t.reshape(1, 6)
                self.dq = dq_t.reshape(1, 6)

            self.state = [r_hat_t[0] - rd_t[0],
                          r_hat_t[1] - rd_t[1],
                          r_hat_t[2] - rd_t[2],
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
                          tau_t[0],
                          tau_t[1],
                          tau_t[2],
                          tau_t[3],
                          tau_t[4],
                          tau_t[5],
                          dqc_t_PID[0],
                          dqc_t_PID[1],
                          dqc_t_PID[2],
                          dqc_t_PID[3],
                          dqc_t_PID[4],
                          dqc_t_PID[5]]
            self.state_buffer = self.state
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
                           q_t[0],
                           q_t[1],
                           q_t[2],
                           q_t[3],
                           q_t[4],
                           q_t[5],
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
                           a[0],
                           a[1],
                           a[2],
                           a[3],
                           a[4],
                           a[5]]
            self.plot_data_buffer = plot_data_t
            return self._get_ob()

    def step(self, a):
        # update time index
        self.k += 1  # Attention doublecheck
        rd_t = np.array(
            [self.xd[self.k], self.yd[self.k], self.zd[self.k]])  # attention: index desired starts from t=-1
        pb.resetBasePositionAndOrientation(
            target_object, rd_t, pb.getQuaternionFromEuler(
                np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])))
        vd_t = np.array([self.vxd, self.vyd, self.vzd])
        LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
        r_hat_t = np.array(LinkState[0])
        v_hat_t = np.array(LinkState[6])
        # TODO check objVelocities in jacobian input
        # Attention: use biased kinematics model for jacobian calculation
        [linearJacobian, angularJacobian] = pb.calculateJacobian(arm_biased_kinematics,
                                                                 10,
                                                                 list(LinkState[2]),
                                                                 list(np.append(self.q[-1, :], [0, 0, 0])),
                                                                 list(np.append(self.dq[-1, :], [0, 0, 0])),
                                                                 list(np.zeros(9)))
        # [linearJacobian_TRUE, angularJacobian_TRUE] = pb.calculateJacobian(arm,
        #                                                                    9,
        #                                                                    list(LinkState[2]),
        #                                                                    list(np.append(self.q[-1, :], [0])),
        #                                                                    list(np.append(self.dq[-1, :], [0])),
        #                                                                    list(np.zeros(7)))
        [linearJacobian_TRUE, angularJacobian_TRUE] = pb.calculateJacobian(arm,
                                                                           10,
                                                                           list(LinkState[2]),
                                                                           list(np.append(self.q[-1, :], [0, 0, 0])),
                                                                           list(np.append(self.dq[-1, :], [0, 0, 0])),
                                                                           list(np.zeros(9)))
        J_t = np.asarray(linearJacobian)[:, :6]
        Jpinv_t = self.pseudoInverseMat(J_t, ld=0.01)

        J_t_TRUE = np.asarray(linearJacobian_TRUE)[:, :6]
        Jpinv_t_TRUE = self.pseudoInverseMat(J_t_TRUE, ld=0.01)
        # U, S, Vh = np.linalg.svd(Jpinv_t, full_matrices=True)
        # TODO: do you need to monitor this during training to avoid singularity??
        # rel_sing= S.max() / S.min()
        # v1=np.linalg.svd(J_t, full_matrices=True)[2][0,:]
        # v1_TRUE = np.linalg.svd(J_t_TRUE, full_matrices=True)[2][0, :]
        # A=np.multiply(v1_TRUE.reshape(1,6), v1.reshape(6,1))
        # # I = np.multiply(v1_TRUE.reshape(1, 6), v1_TRUE.reshape(6, 1))
        # sings=np.linalg.svd(A, full_matrices=True)[1]
        # print("sings=", sings,"\n")
        # u1_TRUE=np.linalg.svd(J_t_TRUE, full_matrices=True)[0][:, 0]
        # u1 = np.linalg.svd(J_t, full_matrices=True)[0][:, 0]
        # A_u = np.multiply(u1_TRUE.reshape(1, 3), u1.reshape(3, 1))
        # sings_u = np.linalg.svd(A_u, full_matrices=True)[1]
        # print("sings_u=", sings_u,"\n")
        # v13=np.linalg.svd(J_t, full_matrices=True)[2][0:3,:]
        # v13_TRUE = np.linalg.svd(J_t_TRUE, full_matrices=True)[2][0:3, :]
        # C = np.matmul(v13_TRUE.T, v13)
        # D = np.matmul(v13_TRUE.T, v13_TRUE)
        # div_metric=np.linalg.svd(np.matmul(v13_TRUE, v13.T))
        # check_metric=np.linalg.svd(np.matmul(v13_TRUE, v13_TRUE.T))
        # check_metric_2=np.linalg.svd(np.matmul(v13, v13.T))
        # print("div_metric[1]=", div_metric[1],"\n")

        rd_t_error = np.matmul(J_t_TRUE, self.pseudoInverseMat(J_t, ld=0.0001)) @ rd_t - rd_t

        dqc_t_PID, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t, e=self.e,
                                           dt=dt)
        # ATTENTION: here apply SAC action
        dqc_t = dqc_t_PID + a
        # TODO check
        # command joint speeds (only 6 joints)
        pb.setJointMotorControlArray(
            arm,
            [0, 1, 2, 3, 4, 5],
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocities=list(dqc_t),
            forces=[87, 87, 87, 87, 12, 12]
        )

        # pb.setJointMotorControlArray(
        #     arm_biased_kinematics,
        #     [0, 1, 2, 3, 4, 5],
        #     controlMode=pb.VELOCITY_CONTROL,
        #     targetVelocities=list(dqc_t),
        #     forces=[87, 87, 87, 87, 12, 12]
        # )
        # default timestep is 1/240 second
        pb.stepSimulation(physicsClientId=physics_client)
        # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
        info = pb.getJointStates(arm, range(12))
        q_tp1, dq_tp1, tau_t = [], [], []
        for joint_info in info:
            q_tp1.append(joint_info[0])
            dq_tp1.append(joint_info[1])
            tau_t.append(joint_info[3])
        # # Attention: hard reset for biased kinematics model
        for i in range(12):
            pb.resetJointState(arm_biased_kinematics, i, q_tp1[i])
        q_tp1 = np.array(q_tp1)[:6]
        dq_tp1 = np.array(dq_tp1)[:6]
        tau_t = np.array(tau_t)[:6]
        self.q = np.vstack((self.q, q_tp1))  # Attention
        self.dq = np.vstack((self.dq, dq_tp1))  # Attention
        # check done episode
        terminal = self._terminal()
        # calculate reward
        # define inspired by Pavlichenko et al SAC tracking paper https://doi.org/10.48550/arXiv.2203.07051
        # todo make more efficient by calling getLinkState only once
        LinkState_tp1 = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
        # TODO CHECK HERE: is there bug? why not use LinkState_tp1 or should I use LinkState?
        r_hat_tp1 = np.array(LinkState_tp1[0])
        v_hat_tp1 = np.array(LinkState_tp1[6])
        # error_p_t = sum(abs(r_hat_tp1 - rd_t))
        error_v_t = sum(abs(v_hat_tp1 - vd_t))
        error_ddqc_t = sum(abs(dqc_t - self.dq[-2, :]))
        # reward_p_t = self.f_logistic(error_p_t, self.lp)
        reward_px_t = self.f_logistic(abs(r_hat_tp1[0] - rd_t[0]), self.lp)
        reward_py_t = self.f_logistic(abs(r_hat_tp1[1] - rd_t[1]), self.lp)
        reward_pz_t = self.f_logistic(abs(r_hat_tp1[2] - rd_t[2]), 400)
        reward_p_t = (reward_px_t + reward_py_t + reward_pz_t) / 3
        reward_v_t = self.f_logistic(error_v_t, self.lv)
        reward_ddqc_t = self.f_logistic(error_ddqc_t, self.lddqc)
        # reward_t = self.reward_eta_p * reward_p_t + self.reward_eta_v * reward_v_t + self.reward_eta_ddqc * reward_ddqc_t
        reward_t = self.reward_eta_p * reward_p_t + self.reward_eta_v * reward_v_t + self.reward_eta_ddqc * reward_ddqc_t
        # collect observations(after you apply action)
        # manual correction
        if self.k < self.MAX_TIMESTEPS - 1:
            next_p_idx = self.k + 1
        else:
            next_p_idx = self.k  # TODO correct
        # TODO double check concept

        obs = [r_hat_tp1[0] - rd_t[0],
               r_hat_tp1[1] - rd_t[1],
               r_hat_tp1[2] - rd_t[2],
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
               tau_t[0],
               tau_t[1],
               tau_t[2],
               tau_t[3],
               tau_t[4],
               tau_t[5],
               dqc_t_PID[0],
               dqc_t_PID[1],
               dqc_t_PID[2],
               dqc_t_PID[3],
               dqc_t_PID[4],
               dqc_t_PID[5]]
        # update states
        self.state = obs
        self.state_buffer = np.vstack((self.state_buffer, self.state))
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
                       dqc_t[0],
                       dqc_t[1],
                       dqc_t[2],
                       dqc_t[3],
                       dqc_t[4],
                       dqc_t[5],
                       self.reward_eta_p * reward_p_t,
                       self.reward_eta_v * reward_v_t,
                       self.reward_eta_ddqc * reward_ddqc_t,
                       tau_t[0],
                       tau_t[1],
                       tau_t[2],
                       tau_t[3],
                       tau_t[4],
                       tau_t[5],
                       reward_px_t,
                       reward_py_t,
                       reward_pz_t,
                       rd_t_error[0],
                       rd_t_error[1],
                       rd_t_error[2],
                       dqc_t_PID[0],
                       dqc_t_PID[1],
                       dqc_t_PID[2],
                       dqc_t_PID[3],
                       dqc_t_PID[4],
                       dqc_t_PID[5],
                       a[0],
                       a[1],
                       a[2],
                       a[3],
                       a[4],
                       a[5]]
        self.plot_data_buffer = np.vstack((self.plot_data_buffer, plot_data_t))
        # if self.plot_data_buffer.__len__()==136:
        #     print("self.plot_data_buffer.__len__()=",self.plot_data_buffer.__len__())
        #     fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 12))
        #     plt.rcParams['font.family'] = 'Serif'
        #     axs3[0].plot(abs(self.plot_data_buffer[:, 0] - self.plot_data_buffer[:, 3]) * 1000, 'k', label='no SAC')
        #     axs3[0].plot(abs(self.plot_data_buffer[:, 30]) * 1000, 'm:', label='no SAC')
        #     axs3[0].set_xlabel("t")
        #     axs3[0].set_ylabel("|x-xd| [mm]")
        #     plt.legend()
        #     axs3[1].plot(abs(self.plot_data_buffer[:, 1] - self.plot_data_buffer[:, 4]) * 1000, 'k', label='no SAC')
        #     axs3[1].plot(abs(self.plot_data_buffer[:, 31]) * 1000, 'm:', label='error bound')
        #     axs3[1].set_xlabel("t")
        #     axs3[1].set_ylabel("|y-yd| [mm]")
        #     plt.legend()
        #     axs3[2].plot(abs(self.plot_data_buffer[:, 2] - self.plot_data_buffer[:, 5]) * 1000, 'r', label='no SAC')
        #     axs3[2].plot(abs(self.plot_data_buffer[:, 32]) * 1000, 'm:', label='error bound')
        #     axs3[2].set_xlabel("t")
        #     axs3[2].set_ylabel("|z-zd| [mm]")
        #     plt.legend()
        #     axs3[3].plot(
        #         np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2, axis=1) * 1000,
        #         'k', label='no SAC')
        #     axs3[3].plot(
        #         np.linalg.norm(self.plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
        #         'm:', label='error bound')
        #     axs3[3].set_xlabel("t")
        #     axs3[3].set_ylabel("||r-rd||_2 [mm]")
        #     # axs3[3].set_ylim([0, 10])
        #     # axs3[3].set_yscale('log')
        #     plt.legend()
        #     plt.show()
        #     print("")
        # # # # # TODO: so dirty code: uncomment when NOSAC for plots -- you need to take care of which random values you call by break points after first done in sac.py ... and cmment a too ...
        # plot_data_buffer_no_SAC=self.plot_data_buffer
        # np.save("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/Fep_HW_0/plot_data_buffer_no_SAC.npy",plot_data_buffer_no_SAC)
        # given action it returns 4-tuple (observation, reward, done, info)
        return (self._get_ob(), reward_t, terminal, {})

    def _get_ob(self):  # TODO is state=observation a reasonable assumption?
        s = self.state
        return s

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
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4,
                                 output_dir_rendering + "/simulation.mp4")  # added by Pierre
            target_object = pb.loadURDF(
                "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                useFixedBase=True)
            conveyor_object = pb.loadURDF(
                "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
                useFixedBase=True)
            plane = pb.loadURDF("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                                useFixedBase=True)
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
            vd_t = np.array([self.vxd, self.vyd, self.vzd])
            # Reset robot at the origin and move the target object to the goal position and orientation
            pb.resetBasePositionAndOrientation(
                arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
            pb.resetBasePositionAndOrientation(
                target_object, rd_t + [0, 0, -0.07], pb.getQuaternionFromEuler(
                    np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])))  # orient just for rendering
            # set conveyer pose and orient
            pb.resetBasePositionAndOrientation(
                conveyor_object,
                np.array([self.xd_init, self.yd_init, self.zd_init]) + np.array([-0.002, -0.18, -0.15]),
                pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]))
            # Reset joint at initial angles
            for i in range(6):
                pb.resetJointState(arm, i, self.q_init[i])
            # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
            for j in range(6, 9):
                pb.resetJointState(arm, j, 0)
            time.sleep(1)

            for t in range(1, self.MAX_TIMESTEPS):
                rd_t = np.array([self.xd[t], self.yd[t], self.zd[t]])
                pb.resetBasePositionAndOrientation(
                    target_object, rd_t + [0, 0, -0.07], pb.getQuaternionFromEuler(
                        np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])))
                dqc_t = self.plot_data_buffer[t, 12:18]
                joint_velocities = list(dqc_t)
                pb.setJointMotorControlArray(
                    arm,
                    [0, 1, 2, 3, 4, 5],
                    controlMode=pb.VELOCITY_CONTROL,
                    targetVelocities=joint_velocities,
                    forces=[87, 87, 87, 87, 12, 12]
                )
                # default timestep is 1/240 second
                pb.stepSimulation(physicsClientId=physics_client)
                time.sleep(0.01)
        # np.save(
        #     "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/noSACFapv3_17/plot_data_buffer_" + str(
        #         self.n) + ".npy", self.plot_data_buffer)
        # render_test_buffer=False
        if render_test_buffer == True:
            # np.save("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/noSACFapv3_17/plot_data_buffer_"+str(self.n)+".npy", self.plot_data_buffer)
            plot_data_buffer_no_SAC = np.load(
                "/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/Fep_HW_0/plot_data_buffer_no_SAC.npy")
            fig1, axs1 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(7, 14))
            axs1[0].plot(self.plot_data_buffer[:, 3] * 1000, self.plot_data_buffer[:, 4] * 1000, 'r--',
                         label='EE desired traj')
            axs1[0].plot((self.plot_data_buffer[:, 3] - abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 4] - abs(self.plot_data_buffer[:, 31])) * 1000, 'm:',
                         label='jacobian uncertainty')

            axs1[0].plot((self.plot_data_buffer[:, 3] + abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 4] + abs(self.plot_data_buffer[:, 31])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[0].plot(self.plot_data_buffer[:, 0] * 1000, self.plot_data_buffer[:, 1] * 1000, 'k',
                         label='EE position - with SAC')
            axs1[0].set_xlabel("x[mm]")
            axs1[0].set_ylabel("y[mm]")
            axs1[1].plot(self.plot_data_buffer[:, 3] * 1000, self.plot_data_buffer[:, 5] * 1000, 'r--',
                         label='EE desired traj')
            axs1[1].plot((self.plot_data_buffer[:, 3] - abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 5] - abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[1].plot((self.plot_data_buffer[:, 3] + abs(self.plot_data_buffer[:, 30])) * 1000,
                         (self.plot_data_buffer[:, 5] + abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[1].plot(self.plot_data_buffer[:, 0] * 1000, self.plot_data_buffer[:, 2] * 1000, 'k',
                         label='EE position - with SAC')
            axs1[1].set_xlabel("x[mm]")
            axs1[1].set_ylabel("z[mm]")
            axs1[2].plot(self.plot_data_buffer[:, 4] * 1000, self.plot_data_buffer[:, 5] * 1000, 'r--',
                         label='EE desired traj')
            axs1[2].plot((self.plot_data_buffer[:, 4] - abs(self.plot_data_buffer[:, 31])) * 1000,
                         (self.plot_data_buffer[:, 5] - abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[2].plot((self.plot_data_buffer[:, 4] + abs(self.plot_data_buffer[:, 31])) * 1000,
                         (self.plot_data_buffer[:, 5] + abs(self.plot_data_buffer[:, 32])) * 1000, 'm:',
                         label='jacobian uncertainty')
            axs1[2].plot(self.plot_data_buffer[:, 1] * 1000, self.plot_data_buffer[:, 2] * 1000, 'k',
                         label='EE position - with SAC')
            axs1[2].set_xlabel("y[mm]")
            axs1[2].set_ylabel("z[mm]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/position_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
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
            plt.savefig(output_dir_rendering + "/position_both_" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig2, axs2 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(7, 14))
            axs2[0].plot(self.plot_data_buffer[:, 9], self.plot_data_buffer[:, 10], 'r--', label='EE desired traj',
                         marker=".",
                         markersize=30)
            axs2[0].plot(self.plot_data_buffer[:, 6], self.plot_data_buffer[:, 7], 'k',
                         label='EE position - PID only controller')
            axs2[0].set_xlabel("vx")
            axs2[0].set_ylabel("vy")
            axs2[1].plot(self.plot_data_buffer[:, 9], self.plot_data_buffer[:, 11], 'r--', label='EE desired traj',
                         marker=".",
                         markersize=30)
            axs2[1].plot(self.plot_data_buffer[:, 6], self.plot_data_buffer[:, 8], 'k',
                         label='EE position - PID only controller')
            axs2[1].set_xlabel("vx")
            axs2[1].set_ylabel("vz")
            axs2[2].plot(self.plot_data_buffer[:, 10], self.plot_data_buffer[:, 11], 'r--', label='EE desired velocity',
                         marker=".",
                         markersize=30)
            axs2[2].plot(self.plot_data_buffer[:, 7], self.plot_data_buffer[:, 8], 'k',
                         label='EE velocity - PID only controller')
            axs2[2].set_xlabel("vy")
            axs2[2].set_ylabel("vz")
            plt.legend()
            plt.savefig(output_dir_rendering + "/velocity_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
            plt.show()

            fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(6, 8))
            axs3[0].plot(abs(self.plot_data_buffer[:, 0] - self.plot_data_buffer[:, 3]) * 1000, 'b', label='x error')
            axs3[0].set_xlabel("t")
            axs3[0].set_ylabel("|x-xd| [mm]")
            plt.legend()
            axs3[1].plot(abs(self.plot_data_buffer[:, 1] - self.plot_data_buffer[:, 4]) * 1000, 'b', label='y error')
            axs3[1].set_xlabel("t")
            axs3[1].set_ylabel("|y-yd| [mm]")
            plt.legend()
            axs3[2].plot(abs(self.plot_data_buffer[:, 2] - self.plot_data_buffer[:, 5]) * 1000, 'b', label='z error')
            axs3[2].set_xlabel("t")
            axs3[2].set_ylabel("|z-zd| [mm]")
            plt.legend()
            axs3[3].plot(
                np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2, axis=1) * 1000,
                'b',
                label='Euclidean error')
            axs3[3].set_xlabel("t")
            axs3[3].set_ylabel("||r-rd||_2 [mm]")
            # axs3[3].set_ylim([0, 10])
            # axs3[3].set_yscale('log')
            plt.legend()
            plt.savefig(output_dir_rendering + "/position_errors_" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 12))
            plt.rcParams['font.family'] = 'Serif'
            axs3[0].plot(abs(plot_data_buffer_no_SAC[:, 0] - plot_data_buffer_no_SAC[:, 3]) * 1000, 'b',
                         label='without SAC')
            axs3[0].plot(abs(self.plot_data_buffer[:, 0] - self.plot_data_buffer[:, 3]) * 1000, 'r', label='with SAC')
            axs3[0].plot(abs(self.plot_data_buffer[:, 30]) * 1000, 'm:', label='with SAC')
            axs3[0].set_xlabel("t")
            axs3[0].set_ylabel("|x-xd| [mm]")
            plt.legend()
            axs3[1].plot(abs(plot_data_buffer_no_SAC[:, 1] - plot_data_buffer_no_SAC[:, 4]) * 1000, 'b',
                         label='without SAC')
            axs3[1].plot(abs(self.plot_data_buffer[:, 1] - self.plot_data_buffer[:, 4]) * 1000, 'r', label='with SAC')
            axs3[1].plot(abs(self.plot_data_buffer[:, 31]) * 1000, 'm:', label='error bound')
            axs3[1].set_xlabel("t")
            axs3[1].set_ylabel("|y-yd| [mm]")
            plt.legend()
            axs3[2].plot(abs(plot_data_buffer_no_SAC[:, 2] - plot_data_buffer_no_SAC[:, 5]) * 1000, 'b',
                         label='without SAC')
            axs3[2].plot(abs(self.plot_data_buffer[:, 2] - self.plot_data_buffer[:, 5]) * 1000, 'r', label='with SAC')
            axs3[2].plot(abs(self.plot_data_buffer[:, 32]) * 1000, 'm:', label='error bound')
            axs3[2].set_xlabel("t")
            axs3[2].set_ylabel("|z-zd| [mm]")
            plt.legend()
            axs3[3].plot(
                np.linalg.norm((plot_data_buffer_no_SAC[:, 0:3] - plot_data_buffer_no_SAC[:, 3:6]), ord=2,
                               axis=1) * 1000, 'b', label='without SAC')
            axs3[3].plot(
                np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2, axis=1) * 1000,
                'r', label='with SAC')
            axs3[3].plot(
                np.linalg.norm(self.plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
                'm:', label='error bound')
            axs3[3].set_xlabel("t")
            axs3[3].set_ylabel("||r-rd||_2 [mm]")
            # axs3[3].set_ylim([0, 10])
            # axs3[3].set_yscale('log')
            plt.legend()
            plt.savefig(output_dir_rendering + "/position_errors_both_" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 6))
            axs4[0, 0].plot(self.plot_data_buffer[:, 12], 'b', label='commanded SAC joint speeed 0')
            axs4[0, 0].set_xlabel("t")
            axs4[0, 0].set_ylabel("dqc_0")
            plt.legend()
            axs4[1, 0].plot(self.plot_data_buffer[:, 13], 'b', label='commanded SAC joint speeed 1')
            axs4[1, 0].set_xlabel("t")
            axs4[1, 0].set_ylabel("dqc_1")
            plt.legend()
            axs4[2, 0].plot(self.plot_data_buffer[:, 14], 'b', label='commanded SAC joint speeed 2')
            axs4[2, 0].set_xlabel("t")
            axs4[2, 0].set_ylabel("dqc_2")
            plt.legend()
            axs4[0, 1].plot(self.plot_data_buffer[:, 15], 'b', label='commanded SAC joint speeed 3')
            axs4[0, 1].set_xlabel("t")
            axs4[0, 1].set_ylabel("dqc_3")
            plt.legend()
            axs4[1, 1].plot(self.plot_data_buffer[:, 16], 'b', label='commanded SAC joint speeed 4')
            axs4[1, 1].set_xlabel("t")
            axs4[1, 1].set_ylabel("dqc_4")
            plt.legend()
            axs4[2, 1].plot(self.plot_data_buffer[:, 17], 'b', label='commanded SAC joint speeed 5')
            axs4[2, 1].set_xlabel("t")
            axs4[2, 1].set_ylabel("dqc_5")
            plt.legend()
            plt.savefig(output_dir_rendering + "/dqc_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
            plt.show()

            fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 6))
            axs4[0, 0].plot(self.plot_data_buffer[:, 33], 'b', label='dqc_PID 0')
            axs4[0, 0].set_xlabel("t")
            axs4[0, 0].set_ylabel("dqc_PID_0")
            plt.legend()
            axs4[1, 0].plot(self.plot_data_buffer[:, 34], 'b', label='dqc_PID 1')
            axs4[1, 0].set_xlabel("t")
            axs4[1, 0].set_ylabel("dqc_PID_1")
            plt.legend()
            axs4[2, 0].plot(self.plot_data_buffer[:, 35], 'b', label='dqc_PID 2')
            axs4[2, 0].set_xlabel("t")
            axs4[2, 0].set_ylabel("dqc_PID_2")
            plt.legend()
            axs4[0, 1].plot(self.plot_data_buffer[:, 36], 'b', label='dqc_PID 3')
            axs4[0, 1].set_xlabel("t")
            axs4[0, 1].set_ylabel("dqc_PID_3")
            plt.legend()
            axs4[1, 1].plot(self.plot_data_buffer[:, 37], 'b', label='dqc_PID 4')
            axs4[1, 1].set_xlabel("t")
            axs4[1, 1].set_ylabel("dqc_PID_4")
            plt.legend()
            axs4[2, 1].plot(self.plot_data_buffer[:, 38], 'b', label='dqc_PID 5')
            axs4[2, 1].set_xlabel("t")
            axs4[2, 1].set_ylabel("dqc_PID_5")
            plt.legend()
            plt.savefig(output_dir_rendering + "/dqc_PID_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
            plt.show()

            fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 6))
            axs4[0, 0].plot(self.plot_data_buffer[:, 39], 'ob', label='only SAC')
            axs4[0, 0].set_xlabel("t")
            axs4[0, 0].set_ylabel("a_0")
            plt.legend()
            axs4[1, 0].plot(self.plot_data_buffer[:, 40], 'ob', label='only SAC')
            axs4[1, 0].set_xlabel("t")
            axs4[1, 0].set_ylabel("a_1")
            plt.legend()
            axs4[2, 0].plot(self.plot_data_buffer[:, 41], 'ob', label='only SAC')
            axs4[2, 0].set_xlabel("t")
            axs4[2, 0].set_ylabel("a_2")
            plt.legend()
            axs4[0, 1].plot(self.plot_data_buffer[:, 42], 'ob', label='only SAC')
            axs4[0, 1].set_xlabel("t")
            axs4[0, 1].set_ylabel("a_3")
            plt.legend()
            axs4[1, 1].plot(self.plot_data_buffer[:, 43], 'ob', label='only SAC')
            axs4[1, 1].set_xlabel("t")
            axs4[1, 1].set_ylabel("a_4")
            plt.legend()
            axs4[2, 1].plot(self.plot_data_buffer[:, 44], 'ob', label='only SAC')
            axs4[2, 1].set_xlabel("t")
            axs4[2, 1].set_ylabel("a_5")
            plt.legend()
            plt.savefig(output_dir_rendering + "/dq_SAC_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(7, 14))
            axs5[0].plot(self.plot_data_buffer[:, 18], 'b', label='reward p')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("eta1*deltar")
            plt.legend()
            axs5[1].plot(self.plot_data_buffer[:, 19], 'b', label='reward v')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("eta2*deltav")
            plt.legend()
            axs5[2].plot(self.plot_data_buffer[:, 20], 'b', label='reward ddqc')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("eta3*ddqc")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/rewards_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(5, 10))
            axs5[0].plot(self.plot_data_buffer[:, 27], 'b', label='reward_px_t')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("reward_px_t")
            plt.legend()
            axs5[1].plot(self.plot_data_buffer[:, 28], 'b', label='reward_px_t')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("reward_py_t")
            plt.legend()
            axs5[2].plot(self.plot_data_buffer[:, 29], 'b', label='reward_px_t')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("reward_pz_t")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/rewards_position_" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(10, 12))
            axs5[0, 0].plot(self.plot_data_buffer[:, 21], 'b', label='commanded torque 0')
            axs5[0, 0].set_xlabel("t")
            axs5[0, 0].set_ylabel("tau_0")
            plt.legend()
            axs5[1, 0].plot(self.plot_data_buffer[:, 22], 'b', label='commanded torque 1')
            axs5[1, 0].set_xlabel("t")
            axs5[1, 0].set_ylabel("tau_1")
            plt.legend()
            axs5[2, 0].plot(self.plot_data_buffer[:, 23], 'b', label='commanded torque 2')
            axs5[2, 0].set_xlabel("t")
            axs5[2, 0].set_ylabel("tau_2")
            plt.legend()
            axs5[0, 1].plot(self.plot_data_buffer[:, 24], 'b', label='commanded torque 3')
            axs5[0, 1].set_xlabel("t")
            axs5[0, 1].set_ylabel("tau_3")
            plt.legend()
            axs5[1, 1].plot(self.plot_data_buffer[:, 25], 'b', label='commanded torque 4')
            axs5[1, 1].set_xlabel("t")
            axs5[1, 1].set_ylabel("tau_4")
            plt.legend()
            axs5[2, 1].plot(self.plot_data_buffer[:, 26], 'b', label='commanded torque 5')
            axs5[2, 1].set_xlabel("t")
            axs5[2, 1].set_ylabel("tau_5")
            plt.legend()
            plt.savefig(output_dir_rendering + "/tau_" + str(self.n) + ".pdf", format="pdf", bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 0], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("r_hat_tp1[0] - rd_t[0]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 1], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("r_hat_tp1[1] - rd_t[1]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 2], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("r_hat_tp1[2] - rd_t[2]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/e_t" + str(self.n) + ".pdf", format="pdf",
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
            plt.legend()
            plt.savefig(output_dir_rendering + "/q_02t" + str(self.n) + ".pdf", format="pdf",
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
            plt.savefig(output_dir_rendering + "/q_35t" + str(self.n) + ".pdf", format="pdf",
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
            plt.savefig(output_dir_rendering + "/dq_02t" + str(self.n) + ".pdf", format="pdf",
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
            plt.savefig(output_dir_rendering + "/dq_35t" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 15], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("tau[0]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 16], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("tau[1]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 17], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("tau[2]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/tau_02t" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 18], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("tau[3]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 19], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("tau[4]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 20], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("tau[5]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/tau_35t" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 21], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("dqc_PI[0]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 22], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("dqc_PI[1]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 23], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("dqc_PI[2]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/dqc_PI_02t" + str(self.n) + ".pdf", format="pdf",
                        bbox_inches='tight')
            plt.show()

            fig5, axs5 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 10))
            axs5[0].plot(self.state_buffer[:, 24], '-om')
            axs5[0].set_xlabel("t")
            axs5[0].set_ylabel("dqc_PI[3]")
            plt.legend()
            axs5[1].plot(self.state_buffer[:, 25], '-om')
            axs5[1].set_xlabel("t")
            axs5[1].set_ylabel("dqc_PI[4]")
            plt.legend()
            axs5[2].plot(self.state_buffer[:, 26], '-om')
            axs5[2].set_xlabel("t")
            axs5[2].set_ylabel("dqc_PI[5]")
            plt.legend()
            plt.legend()
            plt.savefig(output_dir_rendering + "/dqc_PI_35t" + str(self.n) + ".pdf", format="pdf",
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
            plt.savefig(output_dir_rendering + "/buffer_states.pdf", format="pdf", bbox_inches='tight')
            plt.show()
            fig6, axs6 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            x = range(idx_start, idx_end, 1)
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 0]
            axs6[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='r_hat_tp1[0] - rd_t[0]')
            axs6[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs6[0].set_xlabel("timestep")
            axs6[0].set_ylabel("r_hat_tp1[0] - rd_t[0]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 1]
            axs6[1].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='r_hat_tp1[1] - rd_t[1]')
            axs6[1].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs6[1].set_xlabel("timestep")
            axs6[1].set_ylabel("r_hat_tp1[1] - rd_t[1]")
            plt.legend()
            y = buf_obs[0:idx_last, :][idx_start:idx_end, 2]
            axs6[2].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='r_hat_tp1[2] - rd_t[2]')
            axs6[2].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs6[2].set_xlabel("timestep")
            axs6[2].set_ylabel("r_hat_tp1[2] - rd_t[2]")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_states.pdf", format="pdf", bbox_inches='tight')
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
            plt.savefig(output_dir_rendering + "/buffer_action_0to2.pdf", format="pdf", bbox_inches='tight')
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
            plt.savefig(output_dir_rendering + "/buffer_action_3to5.pdf", format="pdf", bbox_inches='tight')
            plt.show()
            fig8, axs8 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_rew[0:idx_last][idx_start:idx_end]
            axs8[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='reward')
            axs8[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs8[0].set_xlabel("timestep")
            axs8[0].set_ylabel("r")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_reward.pdf", format="pdf", bbox_inches='tight')
            plt.show()
            fig10, axs10 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(16, 10))
            y = buf_done[0:idx_last][idx_start:idx_end]
            axs10[0].plot(x, y, 'b', linewidth=0.08, marker=".", markersize=2, label='done')
            axs10[0].vlines(idx_vline, min(y), max(y), 'r', linestyles="dashed")
            axs10[0].set_xlabel("timestep")
            axs10[0].set_ylabel("done")
            plt.legend()
            plt.savefig(output_dir_rendering + "/buffer_done.pdf", format="pdf", bbox_inches='tight')
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
            plt.savefig(output_dir_rendering + "/buffer_states_q_0to2.pdf", format="pdf", bbox_inches='tight')
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
            plt.savefig(output_dir_rendering + "/buffer_states_q_3to5.pdf", format="pdf", bbox_inches='tight')
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
