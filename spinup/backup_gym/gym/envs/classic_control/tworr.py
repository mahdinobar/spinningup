"""Two-link RR Planar Manipulator Tracking Task"""
import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
from scipy.integrate import solve_ivp

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
        'video.frames_per_second' : 15
    }

    def __init__(self):
        self.LINK_LENGTH_1 = 1.  # [m]
        self.LINK_LENGTH_2 = 1.  # [m]
        self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.  #: [kg] imperfect mass of link 2
        self.LINK_MASS_2_TRUE = 1.05  #: [kg] TRUE mass of link 2
        self.LINK_COM_POS_1 = 0.5  #r1: [m] distance of the center of mass of link 1 wrt joint
        self.LINK_COM_POS_2 = 0.5  #r2: [m] distance of the center of mass of link 2 wrt joint
        self.torque_noise_max = 0.  # TODO
        self.viewer = None
        self.state = None
        self.state_buffer= None
        self.t = 0
        self.seed()

        self.xd_init = 1.5
        self.yd_init = 0.4
        self.dt = 1 / 10  # sec
        self.MAX_TIMESTEPS = 100 # maximum timesteps per episode
        self.vxd = -0.01  # m/s
        self.vyd = 0.05  # m/s
        deltax = self.vxd * self.dt * self.MAX_TIMESTEPS
        deltay = self.vyd * self.dt * self.MAX_TIMESTEPS
        self.xd = np.linspace(self.xd_init, self.xd_init + deltax, self.MAX_TIMESTEPS + 1, endpoint=True)
        self.yd = np.linspace(self.yd_init, self.yd_init + deltay, self.MAX_TIMESTEPS + 1, endpoint=True)

        # TODO CHECK
        high_s = np.array([0.2,  0.2,  1.5,  1.5,  0.5,  0.5,  18,  5])
        low_s = np.array([-0.2, -0.2, -1.5, -1.5, -0.5, -0.5, -18, -5])
        self.observation_space = spaces.Box(low=low_s, high=high_s, dtype=np.float32)
        high_a = np.array([2, 2])
        low_a  = np.array([-2, -2])
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)



    def two_link_forward_kinematics(self,q):
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

    def two_link_inverse_kinematics(self,x, y):
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
        r = np.min((np.sqrt(x ** 2 + y ** 2), self.LINK_LENGTH_1+self.LINK_LENGTH_2))
        # phi is the angle of target point w.r.t. -Y axis, same origin as arm
        phi = np.arctan2(y, x)
        alpha = np.arccos((self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2))
        beta = np.arccos((r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r))
        if self.t ==68:
            print("(self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2)=",(self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2))
            print("alpha=",alpha)
            print("(r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r)=",(r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r))
            print("beta=",beta)
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
        alpha = Iz1 + Iz2 + self.LINK_MASS_1 * self.LINK_COM_POS_1 ** 2 + self.LINK_MASS_2 * (self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2)
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
            alpha = Iz1 + Iz2 + self.LINK_MASS_1 * self.LINK_COM_POS_1 ** 2 + self.LINK_MASS_2_TRUE * (self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2)
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
        return sol.y[:,1]

    def two_link_jacobian(self, q, ld=0.1):
        J=np.array([[-self.LINK_LENGTH_1*np.sin(q[0])-self.LINK_LENGTH_2*np.sin(q[0]+q[1]), -self.LINK_LENGTH_2*np.sin(q[0]+q[1])],
                    [self.LINK_LENGTH_1*np.cos(q[0])+self.LINK_LENGTH_2*np.cos(q[0]+q[1]), +self.LINK_LENGTH_2*np.cos(q[0]+q[1])]])
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
        return pseudoInverseMat(J,ld), J

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
        # TODO: User defined linear position gain
        K_p = 40
        K_i = 10
        K_d = 10
        # TODO: User defined pseudo-inverse damping coefficient
        # ld = 0.1
        # Jpinv=two_link_jacobian(q_hat_soln1, ld)
        e_t = (rd - r_ee)
        e = np.vstack((e, e_t.reshape(1, 2)))
        v_command = vd + K_p * e_t + K_i * np.sum(e[1:, :], 0) * dt + K_d * (vd - v_ee)
        qc = np.dot(Jpinv, v_command)
        return qc, e

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # initialize at t=-1
        rd_minus1 = np.array([self.xd[0], self.yd[0]])
        v_minus1 = np.array([self.vxd, self.vyd])
        ree_minus1 = rd_minus1
        q_hat_soln1, q_hat_soln2 = self.two_link_inverse_kinematics(rd_minus1[0], rd_minus1[1])
        q_minus1 = q_hat_soln1
        e_minus1 = rd_minus1 - ree_minus1
        dq_t_minus1 = self.two_link_inverse_kinematics_joint_speeds(v_minus1[0], v_minus1[1], q_minus1[0], q_minus1[1])
        # starting from t=-1
        self.e = e_minus1.reshape(1, 2)
        self.q = q_minus1.reshape(1, 2)
        self.dq = dq_t_minus1.reshape(1, 2)
        self.qc = np.zeros(2).reshape(1, 2)  # attention on presumptions on qc
        self.dqc = np.zeros(2).reshape(1, 2)
        self.ddqc = np.zeros(2).reshape(1, 2)
        self.r_hat = rd_minus1.reshape(1, 2)  # attention: trivial assumption(?)

        self.t = 0
        
        rd_t = np.array([self.xd[self.t + 1], self.yd[self.t + 1]])  # attention: index desired starts from t=-1
        vd = np.array([self.vxd, self.vyd])
        # at time t=0
        q_t=self.q[-1,:]
        dq_t = self.dq[-1, :]
        self.q = np.vstack((self.q, q_t))
        self.dq = np.vstack((self.dq, dq_t))

        r_hat_t = self.two_link_forward_kinematics(q_t)
        Jpinv_t, J_t = self.two_link_jacobian(q_t, ld=0.01)
        v_hat_t = (r_hat_t - self.r_hat[-1, :]) / self.dt
        self.r_hat=np.vstack((self.r_hat,r_hat_t))
        ree0 = rd_t
        e0 = rd_t - ree0
        self.e = np.vstack((self.e, e0.reshape(1, 2)))
        dqc_t, e=self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd, e=self.e, dt=self.dt)
        qc_t = dqc_t*self.dt+self.q[-2,:] #TODO is this observer(taking q(t-1) for integration) sufficient?
        ddqc_t =  (dqc_t-self.dq[-2,:])/self.dt

        self.qc = np.vstack((self.qc, qc_t))
        self.dqc = np.vstack((self.dqc, dqc_t))
        self.ddqc = np.vstack((self.ddqc, ddqc_t))

        tau1_hat, tau2_hat = self.two_link_inverse_dynamics(q_t, dqc_t, ddqc_t)

        self.state = [r_hat_t[0]-rd_t[0],
                      r_hat_t[1]-rd_t[1],
                      q_t[0],
                      q_t[1],
                      dq_t[0],
                      dq_t[1],
                      tau1_hat,
                      tau2_hat]
        self.state_buffer = self.state
        return self._get_ob()

    def step(self,a):
        # update time index
        self.t += 1 #Attention doublecheck
        rd_t = np.array([self.xd[self.t + 1], self.yd[self.t + 1]])  # attention: index desired starts from t=-1
        vd = np.array([self.vxd, self.vyd])
        q_t = self.q[-1, :]
        dq_t = self.dq[-1, :]
        r_hat_t = self.two_link_forward_kinematics(q_t)
        Jpinv_t, J_t = self.two_link_jacobian(q_t, ld=0.01)
        v_hat_t = (r_hat_t - self.r_hat[-1, :]) / self.dt
        self.r_hat = np.vstack((self.r_hat, r_hat_t))
        dqc_t, self.e = self.q_command(r_ee=r_hat_t, v_ee=v_hat_t, Jpinv=Jpinv_t, rd=rd_t, vd=vd, e=self.e, dt=self.dt)
        qc_t = dqc_t * self.dt + self.q[-2, :]  # TODO is this observer(taking q(t-1) for integration) sufficient?
        ddqc_t = (dqc_t - self.dq[-2, :]) / self.dt

        self.qc = np.vstack((self.qc, qc_t))
        self.dqc = np.vstack((self.dqc, dqc_t))
        self.ddqc = np.vstack((self.ddqc, ddqc_t))

        tau1_hat, tau2_hat = self.two_link_inverse_dynamics(q_t, dqc_t, ddqc_t)

        # s_init=[th1_init,dth1_init,th2_init,dth2_init]
        s_init = np.array([q_t[0], dq_t[0], q_t[1], dq_t[1]])
        # t = time.time()
        tau1 = tau1_hat + a[0]
        tau2 = tau2_hat + a[1]
        # TODO HERE WHY TAKES LONG??
        q_FD = self.two_link_forward_dynamics(tau1, tau2,
                                         s_init)  # attention: forward dynamics robot has correct m2 value
        # print("FD elapsed time=", time.time() - t)
        self.q = np.vstack((self.q, np.array([q_FD[0], q_FD[2]])))
        self.dq = np.vstack((self.dq, np.array([q_FD[1], q_FD[3]])))

        x_FK, y_FK = self.two_link_forward_kinematics(np.array([q_FD[0], q_FD[2]]))

        # collect observations(after you apply action)
        # TODO double check concept
        obs = np.array([x_FK-rd_t[0],
                        y_FK-rd_t[1],
                        q_FD[0],
                        q_FD[2],
                        q_FD[1],
                        q_FD[3],
                        tau1,
                        tau2])
        # update states
        self.state = obs
        self.state_buffer = np.vstack((self.state_buffer, self.state))
        # check done episode
        terminal = self._terminal()
        # calculate reward
        reward = 1. if np.sqrt(obs[0] ** 2 + obs[1] ** 2) < 0.01 else 0. # TODO double check concept indexing timestep
        # given action it returns 4-tuple (observation, reward, done, info)
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self): #TODO is state=observation a reasonable assumption?
        s = self.state
        return s

    def _terminal(self):
        return bool(self.t >= self.MAX_TIMESTEPS-1)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None
        # x-y coordinates are vice versa for rendering
        p1 = [self.LINK_LENGTH_1 * np.sin(s[2]), self.LINK_LENGTH_1 * np.cos(s[2])]
        p2 = [p1[1] + self.LINK_LENGTH_2 * np.sin(s[2] + s[3]), p1[0] + self.LINK_LENGTH_2 * np.cos(s[2] + s[3])]
        
        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[2], s[3]] # TODO check compatible with rendering
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 0), (2.2, 0))
        self.viewer.draw_line((0, -2.2), (0, 2.2))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,0, 1)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.7, .7, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

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
