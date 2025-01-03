"""Two-link RR Planar Manipulator Tracking Task"""
import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
from scipy.integrate import solve_ivp
import math

__copyright__ = "Copyright 2024, IfA https://control.ee.ethz.ch/"
__credits__ = ["Mahdi Nobar"]
__author__ = "Mahdi Nobar ETH Zurich <mnobar@ethz.ch>"


class TworrEnv(core.Env):
    """
    Two-link planar arm with two revolut joints (based on simplified models at book "A Mathematical Introduction to
Robotic Manipulation" by Murry et al.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):
        # TODO: reward params
        self.lp = 100
        self.lv = 100
        # self.ljerk = 100
        self.reward_eta_p = 0.7
        self.reward_eta_v = 0.15
        # self.reward_eta_jerk = 0.15
        # TODO: User defined linear position gain
        self.K_p = 10
        self.K_i = 5
        self.K_d = 5

        self.LINK_LENGTH_1 = 1.  # [m]
        self.LINK_LENGTH_2 = 1.  # [m]
        self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.  #: [kg] imperfect mass of link 2
        self.LINK_MASS_2_TRUE = 1.1  #: [kg] TRUE mass of link 2
        self.LINK_COM_POS_1 = 0.5  # r1: [m] distance of the center of mass of link 1 wrt joint
        self.LINK_COM_POS_2 = 0.5  # r2: [m] distance of the center of mass of link 2 wrt joint
        self.torque_noise_max = 0.  # TODO
        self.viewer = None
        self.state = None
        self.state_buffer = None
        self.t = 0
        seed = 1
        np.random.seed(seed)
        self.seed(seed=seed)

        self.xd_init = 1.5
        self.yd_init = 0.4
        self.dt = 1 / 10  # sec
        self.MAX_TIMESTEPS = 100  # maximum timesteps per episode
        self.vxd = -0.01  # m/s
        self.vyd = 0.05  # m/s
        deltax = self.vxd * self.dt * self.MAX_TIMESTEPS
        deltay = self.vyd * self.dt * self.MAX_TIMESTEPS
        self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS, endpoint=True)
        self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS, endpoint=True)

        # TODO Attention: just the dimension of the observation space is enforced. The data here is not used. If you need to enforce them then modify the code.
        high_s = np.array([0.2, 0.2, 1.5, 1.5, 2, 2, 18, 5, 2, 2])
        low_s = np.array([-0.2, -0.2, -1.5, -1.5, -2, -2, -18, -5, -2, -2])
        self.observation_space = spaces.Box(low=low_s, high=high_s, dtype=np.float32)
        high_a = np.array([1, 1])  # TODO Attention: limits should be the same otherwise modify sac code
        low_a = np.array([-1, -1])
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)

    def two_link_forward_kinematics(self, q):
        """Compute the forward kinematics.  Returns the base-coordinate Cartesian
        position of End-effector for a given joint angle vector.  Optional
        parameters self.LINK_LENGTH_1 and self.LINK_LENGTH_2 are the link lengths.  The base is assumed to be at the
        origin.

        :param q: two-element list or ndarray with [q1, q2] joint angles
        :param self.LINK_LENGTH_1: length of link 1
        :param self.LINK_LENGTH_2: length of link 2
        :return: two-element ndarrays with [x,y] locations of End-Effector
        """
        x = self.LINK_LENGTH_1 * np.cos(q[0]) + self.LINK_LENGTH_2 * np.cos(q[0] + q[1])
        y = self.LINK_LENGTH_1 * np.sin(q[0]) + self.LINK_LENGTH_2 * np.sin(q[0] + q[1])
        return [x, y]

    def two_link_inverse_kinematics(self, x, y):
        """Compute two inverse kinematics solutions for a target end position.  The
        target is a Cartesian position vector (two-element ndarray) in world
        coordinates, and the result vectors are joint angles as ndarrays [q0, q1].
        If the target is out of reach, returns the closest pose.

        :param target: two-element list or ndarray with [x1, y] target position
        :param self.LINK_LENGTH_1: optional proximal link length
        :param self.LINK_LENGTH_2: optional distal link length
        :return: tuple (solution1, solution2) of two-element ndarrays with q1, q2 angles
        """
        # find the position of the point in polar coordinates
        r = np.min((np.sqrt(x ** 2 + y ** 2), self.LINK_LENGTH_1 + self.LINK_LENGTH_2))
        # phi is the angle of target point w.r.t. -Y axis, same origin as arm
        phi = np.arctan2(y, x)
        alpha = np.arccos((self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (
                    2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2))
        beta = np.arccos((r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r))
        if self.t == 68:
            print(
                "(self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2)=",
                (self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (
                            2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2))
            print("alpha=", alpha)
            print("(r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r)=",
                  (r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r))
            print("beta=", beta)
        soln1 = np.array((phi - beta, np.pi - alpha))
        soln2 = np.array((phi + beta, np.pi + alpha))
        return soln1, soln2

    def two_link_inverse_dynamics(self, th, dth, ddth):
        """Compute two inverse dynamics solutions for a target end position.
        :param th:
        :param dth:
        :param ddth:
        :return: tau1, tau2
        """
        Iz1 = self.LINK_MASS_1 * self.LINK_LENGTH_1 ** 2 / 12
        Iz2 = self.LINK_MASS_2 * self.LINK_LENGTH_2 ** 2 / 12
        alpha = Iz1 + Iz2 + self.LINK_MASS_1 * self.LINK_COM_POS_1 ** 2 + self.LINK_MASS_2 * (
                    self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2)
        beta = self.LINK_MASS_2 * self.LINK_LENGTH_1 * self.LINK_COM_POS_2
        sigma = Iz2 + self.LINK_MASS_2 * self.LINK_COM_POS_2 ** 2
        tau1 = (alpha + 2 * beta * np.cos(th[1])) * ddth[0] + (sigma + beta * np.cos(th[1])) * ddth[1] + (
                -beta * np.sin(th[1]) * dth[1]) * dth[1] + (-beta * np.sin(th[1]) * (dth[0] + dth[1]) * dth[1])
        tau2 = (sigma + beta * np.cos(th[1])) * ddth[0] + sigma * ddth[1] + (beta * np.sin(th[1]) * dth[0] * dth[0])
        return tau1, tau2

    def two_link_forward_dynamics(self, tau1, tau2, s_init):
        """Compute two inverse dynamics solutions for a target end position.
        :param s_init: [th1_init,dth1_init,th2_init,dth2_init]
        :param th:
        :param dth:
        :param ddth:
        :param self.LINK_MASS_1:
        :param self.LINK_MASS_2_TRUE:
        :param self.LINK_LENGTH_1:
        :param self.LINK_LENGTH_2:
        :return: th1, dth1, th2, dth2 (second column)
        """

        # Define derivative function
        def f(t, s):
            Iz1 = self.LINK_MASS_1 * self.LINK_LENGTH_1 ** 2 / 12
            Iz2 = self.LINK_MASS_2_TRUE * self.LINK_LENGTH_2 ** 2 / 12
            alpha = Iz1 + Iz2 + self.LINK_MASS_1 * self.LINK_COM_POS_1 ** 2 + self.LINK_MASS_2_TRUE * (
                        self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2)
            beta = self.LINK_MASS_2_TRUE * self.LINK_LENGTH_1 * self.LINK_COM_POS_2
            sigma = Iz2 + self.LINK_MASS_2_TRUE * self.LINK_COM_POS_2 ** 2
            d11 = alpha + 2 * beta * np.cos(s[2])
            d21 = sigma + beta * np.cos(s[2])
            d12 = d21
            d22 = sigma
            c = -beta * np.sin(s[2])
            dsdt = [s[1], d12 * d22 / (d12 * d21 - d11 * d22) * (
                    tau2 / d22 - tau1 / d12 - c / d22 * s[1] ** 2 + (c + c) / d12 * s[1] * s[3] + c / d12 * s[
                3] ** 2),
                    s[3], d11 * d21 / (d11 * d22 - d12 * d21) * (
                            tau2 / d21 - tau1 / d11 - c / d21 * s[1] ** 2 + (c + c) / d11 * s[1] * s[3] + c / d11 *
                            s[3] ** 2)]
            return dsdt

        # Define time spans, initial values, and constants
        tspan = np.linspace(0, 1, 100)
        # Solve differential equation
        sol = solve_ivp(lambda t, s: f(t, s),
                        [tspan[0], tspan[-1]], s_init, t_eval=tspan, rtol=1e-5)
        return sol.y[:, 1]

    def two_link_jacobian(self, q, ld=0.1):
        J = np.array([[-self.LINK_LENGTH_1 * np.sin(q[0]) - self.LINK_LENGTH_2 * np.sin(q[0] + q[1]),
                       -self.LINK_LENGTH_2 * np.sin(q[0] + q[1])],
                      [self.LINK_LENGTH_1 * np.cos(q[0]) + self.LINK_LENGTH_2 * np.cos(q[0] + q[1]),
                       +self.LINK_LENGTH_2 * np.cos(q[0] + q[1])]])

        def pseudoInverseMat(A, ld):
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

        return pseudoInverseMat(J, ld), J

    def two_link_inverse_kinematics_joint_speeds(self, dxdt, dydt, q1, q2):
        """
        given joint positions and end effector cartesian speed, using kinematics, returns joint speeds
        """
        alpha = self.LINK_LENGTH_1 * np.sin(q1) + self.LINK_LENGTH_2 * np.sin(q1 + q2)
        beta = self.LINK_LENGTH_2 * np.cos(q1 + q2)
        gama = -self.LINK_LENGTH_1 * np.cos(q1) - self.LINK_LENGTH_2 * np.cos(q1 + q2)
        dq1dt = (1 / alpha) * (-dxdt - self.LINK_LENGTH_2 * dydt / beta * np.sin(q1 + q2)) / (
                1 + self.LINK_LENGTH_2 * gama * np.sin(q1 + q2) / (alpha * beta))
        dq2dt = (dydt + gama * dq1dt) / beta
        return np.array([dq1dt, dq2dt])

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
        # TODO: User defined pseudo-inverse damping coefficient
        # ld = 0.1
        # Jpinv=two_link_jacobian(q_hat_soln1, ld)
        e_t = (rd - r_ee)
        e = np.vstack((e, e_t.reshape(1, 2)))
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
        rd_t = np.array([self.xd[self.t], self.yd[self.t]])
        r_hat_t = rd_t + np.array([np.random.normal(loc=0.0, scale=0.1 * np.abs(self.vxd) * self.dt, size=1),
                                   np.random.normal(loc=0.0, scale=0.1 * np.abs(self.vyd) * self.dt, size=1)]).reshape(
            1, 2)
        r_hat_t = r_hat_t.squeeze()
        vd_t = np.array([self.vxd, self.vyd])
        v_hat_t = vd_t + np.array([np.random.normal(loc=0.0, scale=0.01 * np.abs(self.vxd), size=1),
                                   np.random.normal(loc=0.0, scale=0.01 * np.abs(self.vyd), size=1)]).reshape(1, 2)
        v_hat_t = v_hat_t.squeeze()
        q_hat_soln1, q_hat_soln2 = self.two_link_inverse_kinematics(r_hat_t[0], r_hat_t[1])
        q_t = q_hat_soln1
        dq_t = self.two_link_inverse_kinematics_joint_speeds(v_hat_t[0], v_hat_t[1], q_t[0], q_t[1])
        self.q = q_t.reshape(1, 2)
        self.dq = dq_t.reshape(1, 2)
        Jpinv_t, J_t = self.two_link_jacobian(q_t, ld=0.01)
        self.r_hat = r_hat_t.reshape(1, 2)
        e0 = rd_t - r_hat_t
        self.e = e0.reshape(1, 2)
        dqc_t, e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t, e=self.e, dt=self.dt)
        # Attention: for simple observer ASSUME q(-1)=q(0) and dq(-1)=dq(0)
        qc_t = dqc_t * self.dt + self.q[-1, :]
        ddqc_t = (dqc_t - self.dq[-1, :]) / self.dt
        self.qc = qc_t.reshape(1, 2)
        self.dqc = dqc_t.reshape(1, 2)
        self.ddqc = ddqc_t.reshape(1, 2)
        tau1_hat, tau2_hat = self.two_link_inverse_dynamics(qc_t, dqc_t, ddqc_t)
        s_init = np.array([q_t[0], dq_t[0], q_t[1], dq_t[1]])
        q_FD = self.two_link_forward_dynamics(tau1_hat, tau2_hat,
                                              s_init)
        self.q = np.vstack((self.q, np.array([q_FD[0], q_FD[2]])))
        self.dq = np.vstack((self.dq, np.array([q_FD[1], q_FD[3]])))

        self.state = [r_hat_t[0] - rd_t[0],
                      r_hat_t[1] - rd_t[1],
                      q_FD[0],
                      q_FD[2],
                      q_FD[1],
                      q_FD[2],
                      tau1_hat,
                      tau2_hat,
                      dqc_t[0],
                      dqc_t[1]]
        self.state_buffer = self.state
        plot_data_t = [r_hat_t[0],
                       r_hat_t[1],
                       rd_t[0],
                       rd_t[1],
                       v_hat_t[0],
                       v_hat_t[1],
                       vd_t[0],
                       vd_t[1]]
        self.plot_data_buffer = plot_data_t
        return self._get_ob()

    def step(self, a):
        # update time index
        self.t += 1  # Attention doublecheck
        rd_t = np.array([self.xd[self.t], self.yd[self.t]])  # attention: index desired starts from t=-1
        vd_t = np.array([self.vxd, self.vyd])
        q_t = self.q[-1, :]
        dq_t = self.dq[-1, :]
        r_hat_t = self.two_link_forward_kinematics(q_t)
        Jpinv_t, J_t = self.two_link_jacobian(q_t, ld=0.01)
        v_hat_t = (r_hat_t - self.r_hat[-1, :]) / self.dt
        self.r_hat = np.vstack((self.r_hat, r_hat_t))
        dqc_t, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd_t, e=self.e,
                                       dt=self.dt)
        # inject SAC action
        dqc_t = dqc_t + a
        qc_t = dqc_t * self.dt + self.q[-2, :]  # TODO is this observer(taking q(t-1) for integration) sufficient?
        ddqc_t = (dqc_t - self.dq[-2, :]) / self.dt
        self.qc = np.vstack((self.qc, qc_t))
        self.dqc = np.vstack((self.dqc, dqc_t))
        self.ddqc = np.vstack((self.ddqc, ddqc_t))
        tau1_hat, tau2_hat = self.two_link_inverse_dynamics(qc_t, dqc_t, ddqc_t)
        s_init = np.array([q_t[0], dq_t[0], q_t[1], dq_t[1]])
        # t = time.time()
        # TODO HERE WHY TAKES LONG??
        q_FD = self.two_link_forward_dynamics(tau1_hat, tau2_hat,
                                              s_init)  # attention: forward dynamics robot has correct m2 value
        # print("FD elapsed time=", time.time() - t)
        self.q = np.vstack((self.q, np.array([q_FD[0], q_FD[2]])))
        self.dq = np.vstack((self.dq, np.array([q_FD[1], q_FD[3]])))

        x_FK, y_FK = self.two_link_forward_kinematics(np.array([q_FD[0], q_FD[2]]))

        # collect observations(after you apply action)
        # TODO double check concept
        obs = np.array([x_FK - rd_t[0],
                        y_FK - rd_t[1],
                        q_FD[0],
                        q_FD[2],
                        q_FD[1],
                        q_FD[3],
                        tau1_hat,
                        tau2_hat,
                        dqc_t[0],
                        dqc_t[1]])
        # update states
        self.state = obs
        self.state_buffer = np.vstack((self.state_buffer, self.state))
        plot_data_t = [r_hat_t[0],
                       r_hat_t[1],
                       rd_t[0],
                       rd_t[1],
                       v_hat_t[0],
                       v_hat_t[1],
                       vd_t[0],
                       vd_t[1]]
        self.plot_data_buffer = np.vstack((self.plot_data_buffer, plot_data_t))
        # check done episode
        terminal = self._terminal()
        # calculate reward
        # reward = 1. if np.sqrt(obs[0] ** 2 + obs[1] ** 2) < 0.01 else 0. 
        # reward_t = 100. if np.sqrt(obs[0] ** 2 + obs[1] ** 2) < 0.001 else - 10000*(obs[0] ** 2 + obs[1] ** 2)
        # define inspired by Pavlichenko et al SAC tracking paper https://doi.org/10.48550/arXiv.2203.07051
        error_p_t = sum(abs(obs[0:2]))
        v_hat_after = np.array(self.two_link_forward_kinematics(np.array([q_FD[0], q_FD[2]]))) - np.array(
            r_hat_t) / self.dt
        error_v_t = sum(abs(v_hat_after - vd_t))
        # jerk_level_t = np.abs(self.state_buffer[-1, 6] - self.state_buffer[-2, 6]) + np.abs(
        #     self.state_buffer[-1, 7] - self.state_buffer[-2, 7])
        # reward_jerk_t = self.f_logistic(jerk_level_t, self.ljerk)
        reward_p_t = self.f_logistic(error_p_t, self.lp)
        reward_v_t = self.f_logistic(error_v_t, self.lv)
        reward_t = self.reward_eta_p * reward_p_t + self.reward_eta_v * reward_v_t #+ self.reward_eta_jerk * reward_jerk_t
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
