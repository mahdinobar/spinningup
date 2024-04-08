"""Two-link RR Planar Manipulator Tracking Task"""
import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
from scipy.integrate import solve_ivp
import math
import pybullet as pb
import pybullet_data

__copyright__ = "Copyright 2024, IfA https://control.ee.ethz.ch/"
__credits__ = ["Mahdi Nobar"]
__author__ = "Mahdi Nobar ETH Zurich <mnobar@ethz.ch>"


class FepEnv(core.Env):
    """
    Two-link planar arm with two revolut joints (based on simplified models at book "A Mathematical Introduction to
Robotic Manipulation" by Murry et al.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):
        seed = 1
        np.random.seed(seed)
        self.seed(seed=seed)
        # TODO: reward params
        self.lp = 100
        self.lv = 100
        self.reward_eta_p = 0.7
        self.reward_eta_v = 0.3
        # TODO: User defined linear position gain
        self.K_p = 10
        self.K_i = 5
        self.K_d = 5
        self.torque_noise_max = 0.  # TODO
        self.viewer = None
        self.state = None
        self.state_buffer = None
        self.t = 0
        self.xd_init = 0.43086
        self.yd_init = -0.07530
        self.zd_init = 0.17432
        # TODO correct q_init
        self.q_init = np.deg2rad(np.array([-23.1218, 3.6854, 13.0462, -148.512, -8.7462, 150.2532]))
        self.dt = 1 / 10  # sec
        self.MAX_TIMESTEPS = 100  # maximum timesteps per episode
        self.vxd = 0.005  # m/s
        self.vyd = 0.05  # m/s
        self.vzd = 0  # m/s
        deltax = self.vxd * self.dt * self.MAX_TIMESTEPS
        deltay = self.vyd * self.dt * self.MAX_TIMESTEPS
        deltaz = self.vzd * self.dt * self.MAX_TIMESTEPS
        self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
        self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
        self.zd = np.linspace(self.zd_init, self.zd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)
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
        high_a = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100,
                           2.6100])  # TODO Attention: limits should be the same otherwise modify sac code
        low_a = -high_a
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        # Connect to physics client
        self.physics_client = pb.connect(pb.DIRECT)
        # physics_client = p.connect(p.GUI,options="--mp4fps=3 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d" % (screen_width, screen_height))
        # Load URDFs
        self.create_world()

    def create_world(self):
        """ Setup camera and load URDFs"""
        # # Set gravity
        pb.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        # Load robot, target object and plane urdf
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep/panda.urdf",
                              useFixedBase=True)
        self.target_object = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/sphere.urdf",
                                        useFixedBase=True)
        self.conveyor_object = pb.loadURDF(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
            useFixedBase=True)
        self.plane = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",useFixedBase=True)

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
            pinvA = np.linalg.lstsq((np.matmul(A, A.T) + ld * ld * np.eye(m, m)).T, A)[0].T
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
        qc = np.dot(Jpinv, v_command)
        return qc, e

    def f_logistic(self, x, l):
        return 2 / (math.e ** (x * l) + math.e ** (-x * l))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # # randomize true model parameter in every episode
        # self.LINK_MASS_2_TRUE = 1.1 + np.random.normal(loc=0.0, scale=0.01, size=1)
        # at time t=0
        self.t = 0
        rd_t = np.array([self.xd[self.t], self.yd[self.t], self.zd[self.t]])
        # Reset robot at the origin and move the target object to the goal position and orientation
        pb.resetBasePositionAndOrientation(
            self.arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        pb.resetBasePositionAndOrientation(
            self.target_object, rd_t, pb.getQuaternionFromEuler(np.array([-np.pi, 0, 0])+np.array([np.pi/2 , 0, 0]))) #orient just for rendering
        # set conveyer pose and orient
        pb.resetBasePositionAndOrientation(
            self.conveyor_object,
            np.array([self.xd_init, self.yd_init, self.zd_init]) + np.array([-0.002, -0.18, -0.15]),
            pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]))
        # Reset joint at initial angles
        for i in range(6):
            pb.resetJointState(self.arm, i, self.q_init[i])
        # In Pybullet, gripper halves are controlled separately+we also deactivated the 7th joint too
        for j in range(6, 9):
            pb.resetJointState(self.arm, j, 0)
        # Get end effector coordinates
        r_hat_t = np.array(pb.getLinkState(self.arm, 9, computeForwardKinematics=True)[0])
        info = pb.getJointStates(self.arm, range(7))
        q_t, dq_t, tau_t = [], [], []
        for joint_info in info:
            q_t.append(joint_info[0])
            dq_t.append(joint_info[1])
            tau_t.append(joint_info[3])
        q_t = np.array(q_t)[:6]
        dq_t = np.array(dq_t)[:6]
        tau_t = np.array(tau_t)[:6]  # CHECK!!!!!!!!!!!!!!!!!!!!¨ if tau_t is not 0 what is it and why?
        dqc_t = np.zeros(6)  # TODO check
        self.q = q_t.reshape(1, 6)

        e0 = rd_t - r_hat_t
        self.e = e0.reshape(1, 3)
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
                      dqc_t[0],
                      dqc_t[1],
                      dqc_t[2],
                      dqc_t[3],
                      dqc_t[4],
                      dqc_t[5]]
        self.state_buffer = self.state
        return self._get_ob()

    def step(self, a):
        # update time index
        self.t += 1  # Attention doublecheck
        rd_t = np.array([self.xd[self.t], self.yd[self.t], self.zd[self.t]])  # attention: index desired starts from t=-1
        vd_t = np.array([self.vxd, self.vyd, self.vzd])
        LinkState=pb.getLinkState(self.arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
        r_hat_t = np.array(LinkState[0])
        v_hat_t = np.array(LinkState[6])
        # TODO check objVelocities in jacobian input
        [linearJacobian, angularJacobian] = pb.calculateJacobian(self.arm, 7, list(LinkState[2]), list(np.append(self.q[-1, :],[0,0,0])),
                                                                 list(np.zeros(9)), list(np.zeros(9)))
        J_t = np.asarray(linearJacobian)[:,:6]
        Jpinv_t = self.pseudoInverseMat(J_t,ld=0.01)  # TODO: check pseudo-inverse damping coefficient
        dqc_t, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t, e=self.e,
                                       dt=self.dt)
        # TODO check
        # command joint speeds (only 6 joints)
        joint_velocities = list(dqc_t)
        pb.setJointMotorControlArray(
            self.arm,
            [0, 1, 2, 3, 4, 5],
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
            forces=[87, 87, 87, 87, 12, 12]
        )
        # default timestep is 1/240 second
        pb.setTimeStep(timeStep=self.dt,physicsClientId=self.physics_client)
        pb.stepSimulation(physicsClientId=self.physics_client)
        # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
        info = pb.getJointStates(self.arm, range(7))
        q_tp1, dq_tp1, tau_t = [], [], []
        for joint_info in info:
            q_tp1.append(joint_info[0])
            dq_tp1.append(joint_info[1])
            tau_t.append(joint_info[3])
        q_tp1 = np.array(q_tp1)[:6]
        dq_tp1 = np.array(dq_tp1)[:6]
        tau_t = np.array(tau_t)[:6]
        self.q = np.vstack((self.q, q_tp1)) #Attention
        # collect observations(after you apply action)
        # TODO double check concept
        obs = [r_hat_t[0] - rd_t[0],
               r_hat_t[1] - rd_t[1],
               r_hat_t[2] - rd_t[2],
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
               dqc_t[0],
               dqc_t[1],
               dqc_t[2],
               dqc_t[3],
               dqc_t[4],
               dqc_t[5]]
        # update states
        self.state = obs
        self.state_buffer = np.vstack((self.state_buffer, self.state))
        # check done episode
        terminal = self._terminal()
        # calculate reward
        # define inspired by Pavlichenko et al SAC tracking paper https://doi.org/10.48550/arXiv.2203.07051
        # todo make more efficient by calling getLinkState only once
        LinkState_tp1=pb.getLinkState(self.arm, 9, computeForwardKinematics=True, computeLinkVelocity=True)
        r_hat_tp1 = np.array(LinkState[0])
        v_hat_tp1 = np.array(LinkState[6])
        error_p_t = sum(abs(r_hat_tp1 - vd_t))
        error_v_t = sum(abs(v_hat_tp1 - vd_t))
        reward_p_t = self.f_logistic(error_p_t, self.lp)
        reward_v_t = self.f_logistic(error_v_t, self.lv)
        reward_t = self.reward_eta_p * reward_p_t + self.reward_eta_v * reward_v_t
        # given action it returns 4-tuple (observation, reward, done, info)
        return (self._get_ob(), reward_t, terminal, {})

    def _get_ob(self):  # TODO is state=observation a reasonable assumption?
        s = self.state
        return s

    def _terminal(self):
        return bool(self.t >= self.MAX_TIMESTEPS - 1)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None: return None
        # x-y coordinates are vice versa for rendering
        p1 = [self.LINK_LENGTH_1 * np.sin(s[2]), self.LINK_LENGTH_1 * np.cos(s[2])]
        p2 = [p1[1] + self.LINK_LENGTH_2 * np.sin(s[2] + s[3]), p1[0] + self.LINK_LENGTH_2 * np.cos(s[2] + s[3])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[2], s[3]]  # TODO check compatible with rendering
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 0), (2.2, 0))
        self.viewer.draw_line((0, -2.2), (0, 2.2))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0, 1)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.7, .7, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
