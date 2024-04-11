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

physics_client = pb.connect(pb.DIRECT)
# Connect to physics client
dt = 1 / 10  # sec
pb.setTimeStep(timeStep=dt, physicsClientId=physics_client)
# physics_client = p.connect(p.GUI,options="--mp4fps=3 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d" % (screen_width, screen_height))
# # Set gravity
pb.setGravity(0, 0, -9.81, physicsClientId=physics_client)
# Load URDFs
# Load robot, target object and plane urdf
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep/panda.urdf",
                  useFixedBase=True)
target_object = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                            useFixedBase=True)
conveyor_object = pb.loadURDF(
    "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
    useFixedBase=True)
plane = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                    useFixedBase=True)


class FepEnv(core.Env):
    """
    Two-link planar arm with two revolut joints (based on simplified models at book "A Mathematical Introduction to
Robotic Manipulation" by Murry et al.
    """

    def __init__(self):
        seed = 1
        np.random.seed(seed)
        self.seed(seed=seed)
        # TODO: reward params
        self.lp = 1
        self.lv = 10
        self.lddqc = 10
        self.reward_eta_p = 0.7
        self.reward_eta_v = 0.2
        self.reward_eta_ddqc = 0.3
        # TODO: User defined linear position gain
        self.K_p = 2
        self.K_i = 0.5
        self.K_d = 0.1
        self.torque_noise_max = 0.  # TODO
        self.viewer = None
        self.state = None
        self.state_buffer = None
        self.t = 0
        # self.xd_init = 0.43086
        # self.yd_init = -0.07530
        # self.zd_init = 0.17432
        # Attention: update init locations to alway match with q_init
        self.xd_init = 0.45039319
        self.yd_init = -0.09860044
        self.zd_init = 0.17521834
        # TODO correct q_init
        # self.q_init = np.deg2rad(np.array([-23.1218, 3.6854, 13.0462, -148.512, -8.7462, 150.2532]))
        self.MAX_TIMESTEPS = 100  # maximum timesteps per episode
        self.vxd = 0.005  # m/s
        self.vyd = 0.05  # m/s
        self.vzd = 0  # m/s
        deltax = self.vxd * dt * self.MAX_TIMESTEPS
        deltay = self.vyd * dt * self.MAX_TIMESTEPS
        deltaz = self.vzd * dt * self.MAX_TIMESTEPS
        self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
        self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
        self.zd = np.linspace(self.zd_init, self.zd_init + deltaz, self.MAX_TIMESTEPS, endpoint=True)
        # TODO Attention: just the dimension of the observation space is enforced. The data here is not used. If you need to enforce them then modify the code.
        # Attention just 6 DOF is simulated (7th DOF is disabled)
        # high_s = np.array([0.2, 0.2, 0.2,
        #                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        #                    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100,
        #                    87, 87, 87, 87, 12, 12,
        #                    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100])
        high_s = np.array([0.2, 0.2, 0.2,
                           1.5, 
                           2.1750, 
                           87, 
                           2.1750])
        low_s = -high_s
        self.observation_space = spaces.Box(low=low_s, high=high_s, dtype=np.float32)
        # Attention just 6 DOF is simulated (7th DOF is disabled)
        # Attention: limits of SAC actions
        # high_a = 0.05 * np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100,
        #                           2.6100])  # TODO Attention: limits should be the same otherwise modify sac code
        high_a = 0.9 * np.array([2.1750])  # TODO Attention: limits should be the same otherwise modify sac code
        low_a = -high_a
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        self.output_dir_rendering = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/"

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
        # # randomize true model parameter in every episode
        # self.LINK_MASS_2_TRUE = 1.1 + np.random.normal(loc=0.0, scale=0.01, size=1)
        # at time t=0
        self.t = 0
        rd_t = np.array([self.xd[self.t], self.yd[self.t], self.zd[self.t]])
        vd_t = np.array([self.vxd, self.vyd, self.vzd])
        # Reset robot at the origin and move the target object to the goal position and orientation
        pb.resetBasePositionAndOrientation(
            arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
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
        self.q_init = np.array([-0.42529795, 0.11298615, 0.20446317, -2.52843438, -0.15231932, 2.63230466])
        # Reset joint at initial angles
        for i in range(6):
            pb.resetJointState(arm, i, self.q_init[i])
        # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
        for j in range(6, 10):
            pb.resetJointState(arm, j, 0)
        # Get end effector coordinates
        LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
        r_hat_t = np.array(LinkState[0])
        v_hat_t = np.array(LinkState[6])
        info = pb.getJointStates(arm, range(7))
        q_t, dq_t, tau_t = [], [], []
        for joint_info in info:
            q_t.append(joint_info[0])
            dq_t.append(joint_info[1])
            tau_t.append(joint_info[3])
        q_t = np.array(q_t)[:6]
        if abs(sum(self.q_init - q_t)) > 1e-6:
            raise ValueError('shouldn\'t q_init be equal to q_t?!')
        dq_t = np.array(dq_t)[:6]
        tau_t = np.array(tau_t)[:6]  # CHECK!!!!!!!!!!!!!!!!!!!!¨ if tau_t is not 0 what is it and why?
        dqc_t = np.zeros(6)  # TODO check
        self.q = q_t.reshape(1, 6)
        self.dq = dq_t.reshape(1, 6)
        e0 = rd_t - r_hat_t
        self.e = e0.reshape(1, 3)
        self.state = [r_hat_t[0] - rd_t[0],
                      r_hat_t[1] - rd_t[1],
                      r_hat_t[2] - rd_t[2],
                      q_t[0],
                      dq_t[0],
                      tau_t[0],
                      dqc_t[0]]
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
                       0]
        self.plot_data_buffer = plot_data_t
        return self._get_ob()

    def step(self, a):
        # update time index
        self.t += 1  # Attention doublecheck
        rd_t = np.array(
            [self.xd[self.t], self.yd[self.t], self.zd[self.t]])  # attention: index desired starts from t=-1
        pb.resetBasePositionAndOrientation(
            target_object, rd_t, pb.getQuaternionFromEuler(
                np.array([-np.pi, 0, 0]) + np.array([np.pi / 2, 0, 0])))
        vd_t = np.array([self.vxd, self.vyd, self.vzd])
        LinkState = pb.getLinkState(arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
        r_hat_t = np.array(LinkState[0])
        v_hat_t = np.array(LinkState[6])
        # TODO check objVelocities in jacobian input
        [linearJacobian, angularJacobian] = pb.calculateJacobian(arm,
                                                                 9,
                                                                 list(LinkState[2]),
                                                                 list(np.append(self.q[-1, :], [0, 0, 0])),
                                                                 list(np.append(self.dq[-1, :], [0, 0, 0])),
                                                                 list(np.zeros(9)))
        J_t = np.asarray(linearJacobian)[:, :6]
        Jpinv_t = self.pseudoInverseMat(J_t, ld=0.1)  # TODO: check pseudo-inverse damping coefficient
        dqc_t, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t, e=self.e,
                                       dt=dt)
        # inject SAC action
        dqc_t[0] = dqc_t[0] + a
        # TODO check
        # command joint speeds (only 6 joints)
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
        # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
        info = pb.getJointStates(arm, range(7))
        q_tp1, dq_tp1, tau_t = [], [], []
        for joint_info in info:
            q_tp1.append(joint_info[0])
            dq_tp1.append(joint_info[1])
            tau_t.append(joint_info[3])
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
        r_hat_tp1 = np.array(LinkState[0])
        v_hat_tp1 = np.array(LinkState[6])
        error_p_t = sum(abs(r_hat_tp1 - vd_t))
        error_v_t = sum(abs(v_hat_tp1 - vd_t))
        error_ddqc_t = 0#(abs(dqc_t[0] - self.state[12]))
        reward_p_t = self.f_logistic(error_p_t, self.lp)
        reward_v_t = 0*self.f_logistic(error_v_t, self.lv)
        reward_ddqc_t = 0*self.f_logistic(error_ddqc_t, self.lddqc)
        reward_t = self.reward_eta_p * reward_p_t + self.reward_eta_v * reward_v_t + self.reward_eta_ddqc * reward_ddqc_t
        # collect observations(after you apply action)
        # TODO double check concept
        # obs = [r_hat_t[0] - rd_t[0],
        #        r_hat_t[1] - rd_t[1],
        #        r_hat_t[2] - rd_t[2],
        #        q_tp1[0],
        #        q_tp1[1],
        #        q_tp1[2],
        #        q_tp1[3],
        #        q_tp1[4],
        #        q_tp1[5],
        #        dq_tp1[0],
        #        dq_tp1[1],
        #        dq_tp1[2],
        #        dq_tp1[3],
        #        dq_tp1[4],
        #        dq_tp1[5],
        #        tau_t[0],
        #        tau_t[1],
        #        tau_t[2],
        #        tau_t[3],
        #        tau_t[4],
        #        tau_t[5],
        #        dqc_t[0],
        #        dqc_t[1],
        #        dqc_t[2],
        #        dqc_t[3],
        #        dqc_t[4],
        #        dqc_t[5]]
        obs = [r_hat_t[0] - rd_t[0],
               r_hat_t[1] - rd_t[1],
               r_hat_t[2] - rd_t[2],
               q_tp1[0],
               dq_tp1[0],
               tau_t[0],
               dqc_t[0]]
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
                       self.reward_eta_ddqc * reward_ddqc_t]
        self.plot_data_buffer = np.vstack((self.plot_data_buffer, plot_data_t))

        # given action it returns 4-tuple (observation, reward, done, info)
        return (self._get_ob(), reward_t, terminal, {})

    def _get_ob(self):  # TODO is state=observation a reasonable assumption?
        s = self.state
        return s

    def _terminal(self):
        return bool(self.t >= self.MAX_TIMESTEPS - 1)

    def render(self, mode='human'):
        """ Render Pybullet simulation """
        render_video = False  # for fast debuging
        if render_video == True:
            pb.disconnect(physics_client)
            # render settings
            renderer = pb.ER_TINY_RENDERER  # p.ER_BULLET_HARDWARE_OPENGL
            _width = 224
            _height = 224
            _cam_dist = 1.3
            _cam_yaw = 15
            _cam_pitch = -40
            _cam_roll = 0
            camera_target_pos = [0.2, 0, 0.1]
            _screen_width = 3840  # 1920
            _screen_height = 2160  # 1080
            physics_client_rendering = pb.connect(pb.GUI,
                                                  options='--mp4fps=5 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (
                                                      _screen_width, _screen_height))
            dt = 1 / 10  # sec
            pb.setTimeStep(timeStep=dt, physicsClientId=physics_client_rendering)
            # physics_client = p.connect(p.GUI,options="--mp4fps=3 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d" % (screen_width, screen_height))
            # # Set gravity
            pb.setGravity(0, 0, -9.81, physicsClientId=physics_client_rendering)
            # Load URDFs
            # Load robot, target object and plane urdf
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4,
                                 self.output_dir_rendering + "/simulation.mp4")  # added by Pierre
            arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep/panda.urdf",
                              useFixedBase=True)
            target_object = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                                        useFixedBase=True)
            conveyor_object = pb.loadURDF(
                "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
                useFixedBase=True)
            plane = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                                useFixedBase=True)
            # Initialise debug camera angle
            pb.resetDebugVisualizerCamera(
                cameraDistance=_cam_dist,
                cameraYaw=_cam_yaw,
                cameraPitch=_cam_pitch,
                cameraTargetPosition=camera_target_pos,
                physicsClientId=physics_client_rendering)
            t = 0
            rd_t = np.array([self.xd[t], self.yd[t], self.zd[t]])
            vd_t = np.array([self.vxd, self.vyd, self.vzd])
            # Reset robot at the origin and move the target object to the goal position and orientation
            pb.resetBasePositionAndOrientation(
                arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
            pb.resetBasePositionAndOrientation(
                target_object, rd_t, pb.getQuaternionFromEuler(
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
                    target_object, rd_t, pb.getQuaternionFromEuler(
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

        fig1, axs1 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(7, 14))
        axs1[0].plot(self.plot_data_buffer[:, 3], self.plot_data_buffer[:, 4], 'r--', label='EE desired traj')
        axs1[0].plot(self.plot_data_buffer[:, 0], self.plot_data_buffer[:, 1], 'k',
                     label='EE position - PID only controller')
        axs1[0].set_xlabel("x")
        axs1[0].set_ylabel("y")
        axs1[1].plot(self.plot_data_buffer[:, 3], self.plot_data_buffer[:, 5], 'r--', label='EE desired traj')
        axs1[1].plot(self.plot_data_buffer[:, 0], self.plot_data_buffer[:, 2], 'k',
                     label='EE position - PID only controller')
        axs1[1].set_xlabel("x")
        axs1[1].set_ylabel("z")
        axs1[2].plot(self.plot_data_buffer[:, 4], self.plot_data_buffer[:, 5], 'r--', label='EE desired traj')
        axs1[2].plot(self.plot_data_buffer[:, 1], self.plot_data_buffer[:, 2], 'k',
                     label='EE position - PID only controller')
        axs1[2].set_xlabel("y")
        axs1[2].set_ylabel("z")
        plt.legend()
        plt.savefig(self.output_dir_rendering + "/position.pdf", format="pdf", bbox_inches='tight')
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
        plt.savefig(self.output_dir_rendering + "/velocity.pdf", format="pdf", bbox_inches='tight')
        plt.show()

        fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 12))
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
            np.linalg.norm((self.plot_data_buffer[:, 0:3] - self.plot_data_buffer[:, 3:6]), ord=2, axis=1) * 1000, 'b',
            label='Euclidean error')
        axs3[3].set_xlabel("t")
        axs3[3].set_ylabel("||r-rd||_2 [mm]")
        plt.legend()
        plt.savefig(self.output_dir_rendering + "/position_errors.pdf", format="pdf", bbox_inches='tight')
        plt.show()

        fig4, axs4 = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(12, 10))
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
        plt.savefig(self.output_dir_rendering + "/dqc.pdf", format="pdf", bbox_inches='tight')
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
        plt.savefig(self.output_dir_rendering + "/rewards.pdf", format="pdf", bbox_inches='tight')
        plt.show()

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
