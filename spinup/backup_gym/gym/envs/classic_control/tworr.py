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
        self.dt = 0.01  # [s]
        self.LINK_LENGTH_1 = 1.  # [m]
        self.LINK_LENGTH_2 = 1.  # [m]
        self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.  #: [kg] mass of link 2
        self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
        self.torque_noise_max = 0.  # TODO
        self.MAX_TIMESTEPS = 100  # maximum timesteps per episode
        
        self.viewer = None
        # states=[xt, yt, th1, th2, dth1, dth2, th1_hat, th2_hat, dth1_hat, dth2_hat]
        high_s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        low_s = -high_s
        self.observation_space = spaces.Box(low=low_s, high=high_s, dtype=np.float32)
        high_a = np.array([0.1, 0.1])
        low_a = -high_a
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        self.state = None
        self.state_buffer= None
        self.t = 0
        self.seed()
        self.xd = np.linspace(1.2,1.3, self.MAX_TIMESTEPS, endpoint=True)
        self.yd = np.linspace(1, 1.8, self.MAX_TIMESTEPS, endpoint=True)

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
        return x, y

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
        r = np.sqrt(x ** 2 + y ** 2)
        # phi is the angle of target point w.r.t. -Y axis, same origin as arm
        phi = np.arctan2(y, x)
        alpha = np.arccos((self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2 - r ** 2) / (2 * self.LINK_LENGTH_1 * self.LINK_LENGTH_2))
        beta = np.arccos((r ** 2 + self.LINK_LENGTH_1 ** 2 - self.LINK_LENGTH_2 ** 2) / (2 * self.LINK_LENGTH_1 * r))
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
        :param self.LINK_MASS_2:
        :param self.LINK_LENGTH_1:
        :param self.LINK_LENGTH_2:
        :return: th1, dth1, th2, dth2
        """
        # Define derivative function
        def f(t, s):
            Iz1 = self.LINK_MASS_1 * self.LINK_LENGTH_1 ** 2 / 12
            Iz2 = self.LINK_MASS_2 * self.LINK_LENGTH_2 ** 2 / 12
            alpha = Iz1 + Iz2 + self.LINK_MASS_1 * self.LINK_COM_POS_1 ** 2 + self.LINK_MASS_2 * (self.LINK_LENGTH_1 ** 2 + self.LINK_LENGTH_2 ** 2)
            beta = self.LINK_MASS_2 * self.LINK_LENGTH_1 * self.LINK_COM_POS_2
            sigma = Iz2 + self.LINK_MASS_2 * self.LINK_COM_POS_2 ** 2
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
        tspan = np.linspace(0, 1, 2)
        # Solve differential equation
        sol = solve_ivp(lambda t, s: f(t, s),
                        [tspan[0], tspan[-1]], s_init, t_eval=tspan, rtol=1e-5)
        # Plot states
        state_plotter(sol.t, sol.y, 1)
        return sol.y

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # The reset method should return a tuple of the initial observation and some auxiliary information.
        # states=(xd,yd,q1,q2,dq1,dq2,qd1,qd2,dqd1,dqd2,tau1_hat,tau2_hat)
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(12,)) #TODO correct initialization
        self.state_buffer=self.state #initialize episode state buffer
        self.t = 0
        return self._get_ob(), {}

    def step(self, a):
        if self.t==0:
            s_t = self.state_buffer[:]  # states at t
            s_tm1 = s_t  # TODO check. states at t-1
        else:
            s_t = self.state_buffer[-1, :]  # states at t
            s_tm1 = self.state_buffer[-2, :] #states at t-1
        # TOODO 
        # Add noise to the force action 
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)
        # choose a sample target desired position to test IK
        xd = self.xd[self.t]
        yd = self.yd[self.t]
        q_d = self.two_link_inverse_kinematics(xd, yd)[0] #attention: only return and use first solution of IK
        # TODO correct for observation:
        dqd_t = np.array([(q_d[0]-s_tm1[6])/self.dt, (q_d[1]-s_tm1[7])/self.dt])
        ddqd_t = np.array([(dqd_t[0]-s_tm1[8])/self.dt, (dqd_t[1]-s_tm1[9])/self.dt])
        tau1_hat, tau2_hat = self.two_link_inverse_dynamics(q_d, dqd_t, ddqd_t)
        s_init = [s[2], s[4], s[3], s[5]]
        q_FD = self.two_link_forward_dynamics(tau1_hat+a[0], tau2_hat+a[1], s_init)
        # collect observations
        obs=np.array([xd,yd,q_FD[0,1],q_FD[2,1],q_FD[1,1],q_FD[3,1],q_d[0],q_d[1],dqd_t[0],dqd_t[1],tau1_hat,tau2_hat])
        # update states
        self.state=obs
        self.state_buffer=np.vstack((self.state_buffer,self.state))
        # update time index
        self.t += 1
        # check done episode
        terminal = self._terminal()

        # calculate reward
        x_FK, y_FK = self.two_link_forward_kinematics(np.array([q_FD[0,1],q_FD[2,1]]))
        reward = 1. if  np.sqrt((x_FK-xd)**2+(y_FK-yd)**2)<0.001 else 0.
        # given action it returns 4-tuple (observation, reward, done, info)
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self): #TODO is state=observation a reasonable assumption?
        s = self.state
        return s

    def _terminal(self):
        return bool(self.t >= self.MAX_TIMESTEPS)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None
        p1 = [self.LINK_LENGTH_1 * np.cos(s[2]), self.LINK_LENGTH_1 * np.sin(s[2])] 
        p2 = [p1[0] + self.LINK_LENGTH_2 * np.cos(s[2] + s[3]), p1[1] + self.LINK_LENGTH_2 * np.sin(s[2] + q[3])]
        
        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[2], s[3]] # TODO check compatible with rendering
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
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
