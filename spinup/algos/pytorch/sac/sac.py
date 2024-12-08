from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger

import matplotlib.pyplot as plt


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, sample_mode=1, sequence_length=1):
        if sample_mode == 1:
            idxs = np.random.randint(0, self.size, size=batch_size)
        elif sample_mode == 2:
            # idxs = np.random.randint(sequence_length - 1, self.size, size=1) - np.arange(0, sequence_length, 1)
            idxs = np.random.randint(0, self.size // batch_size, 1) * batch_size + np.random.randint(0,
                                                                                                    batch_size - sequence_length,
                                                                                                    1) - np.arange(0,
                                                                                                                   sequence_length,
                                                                                                                   1)
        elif sample_mode == 3:
            idxs = np.array([item for item in range(self.size - batch_size, self.size, 1)])
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha_init=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, initial_actions="random", save_buffer=False, sample_mode=1, automatic_entropy_tuning=False):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound! (Attention: correct here if bounds of action dimensions are different)
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        if automatic_entropy_tuning==True:
            # get updated alpha
            alpha = log_alpha.exp()
        else:
            alpha = alpha_init

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup (mean-squared Bellman error (MSBE))
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        if automatic_entropy_tuning==True:
            # get updated alpha
            alpha = log_alpha.exp()
        else:
            alpha = alpha_init

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())
        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    if automatic_entropy_tuning is True:
        device = torch.device("cpu")
        target_entropy = -0.1*(ac.pi.mu_layer.out_features)
        # log_alpha=torch.zeros(1, requires_grad=True, device=device)
        log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True, device=device)
        alpha_optimizer = Adam([log_alpha], lr=0.0005)
        alpha = log_alpha.exp()

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        if automatic_entropy_tuning is True:
            o = data['obs']
            pi, logp_pi = ac.pi(o)
            alpha_optimizer.zero_grad()
            alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean() #equation 18 of SAC paper
            # print("entropy_actor_estimate=",logp_pi.mean().item())
            alpha_loss.backward()
            # Check the gradient of log_alpha
            # if log_alpha.grad is not None:
            #     print(f"Gradient of log_alpha: {log_alpha.grad.mean().item()}")
            alpha_optimizer.step()
            alpha = log_alpha.exp()
            # alpha_info = dict(Alpha=alpha.detach().numpy())
            # logger.store(LossAlpha=alpha_loss.item(), **alpha_info)
            logger.store(LossAlpha=alpha_loss.item(), Alpha=alpha.detach().numpy())
        else:
            # alpha_info = dict(Alpha=alpha.detach().numpy())
            # logger.store(LossAlpha=alpha_loss.item(), **alpha_info)
            logger.store(LossAlpha=0, Alpha=alpha_init)
        # Unfreeze Q-networks so you can optimize it at next (DDPG, SAC, ...) step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)  # here we log AverageTestEpRet to progress.txt
            env.reset()  # need reset here to properly use Pybullet engine

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # print("in SAC main loop, t =",t)
        # print("epoch=(t+1)//steps_per_epoch=", (t + 1) // steps_per_epoch)
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t >= start_steps:
            a = get_action(o)
        else:
            if initial_actions == "random":
                a = env.action_space.sample()
            elif initial_actions == "zero":
                a = np.zeros(env.action_space.shape[0])

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state: at gym: The keyword argument "max_episode_steps" will ensure that GridWorld environments that are instantiated via gymnasium.make will be wrapped in a TimeLimit wrapper (see the wrapper documentation for more information). A done signal will then be produced if the agent has reached the target or 300 steps have been executed in the current episode. To distinguish truncation and termination, you can check info["TimeLimit.truncated"])
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # uncomment for debugging plots#############################################################################
            if (False):
                fig1, axs1 = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(7, 14))
                axs1[0].plot(env.env.plot_data_buffer[:, 3] * 1000, env.env.plot_data_buffer[:, 4] * 1000, 'r--',
                             label='EE desired traj')
                axs1[0].plot((env.env.plot_data_buffer[:, 3] - abs(env.env.plot_data_buffer[:, 30])) * 1000,
                             (env.env.plot_data_buffer[:, 4] - abs(env.env.plot_data_buffer[:, 31])) * 1000, 'm:',
                             label='jacobian uncertainty')

                axs1[0].plot((env.env.plot_data_buffer[:, 3] + abs(env.env.plot_data_buffer[:, 30])) * 1000,
                             (env.env.plot_data_buffer[:, 4] + abs(env.env.plot_data_buffer[:, 31])) * 1000, 'm:',
                             label='jacobian uncertainty')
                axs1[0].plot(env.env.plot_data_buffer[:, 0] * 1000, env.env.plot_data_buffer[:, 1] * 1000, 'k',
                             label='EE position - with SAC')
                axs1[0].set_xlabel("x[mm]")
                axs1[0].set_ylabel("y[mm]")
                axs1[1].plot(env.env.plot_data_buffer[:, 3] * 1000, env.env.plot_data_buffer[:, 5] * 1000, 'r--',
                             label='EE desired traj')
                axs1[1].plot((env.env.plot_data_buffer[:, 3] - abs(env.env.plot_data_buffer[:, 30])) * 1000,
                             (env.env.plot_data_buffer[:, 5] - abs(env.env.plot_data_buffer[:, 32])) * 1000, 'm:',
                             label='jacobian uncertainty')
                axs1[1].plot((env.env.plot_data_buffer[:, 3] + abs(env.env.plot_data_buffer[:, 30])) * 1000,
                             (env.env.plot_data_buffer[:, 5] + abs(env.env.plot_data_buffer[:, 32])) * 1000, 'm:',
                             label='jacobian uncertainty')
                axs1[1].plot(env.env.plot_data_buffer[:, 0] * 1000, env.env.plot_data_buffer[:, 2] * 1000, 'k',
                             label='EE position - with SAC')
                axs1[1].set_xlabel("x[mm]")
                axs1[1].set_ylabel("z[mm]")
                axs1[2].plot(env.env.plot_data_buffer[:, 4] * 1000, env.env.plot_data_buffer[:, 5] * 1000, 'r--',
                             label='EE desired traj')
                axs1[2].plot((env.env.plot_data_buffer[:, 4] - abs(env.env.plot_data_buffer[:, 31])) * 1000,
                             (env.env.plot_data_buffer[:, 5] - abs(env.env.plot_data_buffer[:, 32])) * 1000, 'm:',
                             label='jacobian uncertainty')
                axs1[2].plot((env.env.plot_data_buffer[:, 4] + abs(env.env.plot_data_buffer[:, 31])) * 1000,
                             (env.env.plot_data_buffer[:, 5] + abs(env.env.plot_data_buffer[:, 32])) * 1000, 'm:',
                             label='jacobian uncertainty')
                axs1[2].plot(env.env.plot_data_buffer[:, 1] * 1000, env.env.plot_data_buffer[:, 2] * 1000, 'k',
                             label='EE position - with SAC')
                axs1[2].set_xlabel("y[mm]")
                axs1[2].set_ylabel("z[mm]")
                plt.legend()
                plt.show()

                fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(6, 8))
                axs3[0].plot(abs(env.env.plot_data_buffer[:, 0] - env.env.plot_data_buffer[:, 3]) * 1000, 'b',
                             label='x error')
                axs3[0].plot(abs(env.env.plot_data_buffer[:, 30]) * 1000, 'm:', label='error bound')
                axs3[0].set_xlabel("t")
                axs3[0].set_ylabel("|x-xd| [mm]")
                plt.legend()
                axs3[1].plot(abs(env.env.plot_data_buffer[:, 1] - env.env.plot_data_buffer[:, 4]) * 1000, 'b',
                             label='y error')
                axs3[1].plot(abs(env.env.plot_data_buffer[:, 31]) * 1000, 'm:', label='error bound')
                axs3[1].set_xlabel("t")
                axs3[1].set_ylabel("|y-yd| [mm]")
                plt.legend()
                axs3[2].plot(abs(env.env.plot_data_buffer[:, 2] - env.env.plot_data_buffer[:, 5]) * 1000, 'b',
                             label='z error')
                axs3[2].plot(abs(env.env.plot_data_buffer[:, 32]) * 1000, 'm:', label='error bound')
                axs3[2].set_xlabel("t")
                axs3[2].set_ylabel("|z-zd| [mm]")
                plt.legend()
                axs3[3].plot(
                    np.linalg.norm((env.env.plot_data_buffer[:, 0:3] - env.env.plot_data_buffer[:, 3:6]), ord=2,
                                   axis=1) * 1000,
                    'b',
                    label='Euclidean error')
                axs3[3].plot(
                    np.linalg.norm(env.env.plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
                    'm:', label='error bound')
                axs3[3].set_xlabel("t")
                axs3[3].set_ylabel("||r-rd||_2 [mm]")
                plt.legend()
                plt.show()
                print("---ERROR=",np.mean(np.linalg.norm(env.env.plot_data_buffer[:, 30:33], ord=2, axis=1))*1000)
            ################################################################################################
            env.env.plot_data_buffer
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling (gradient descent on Q and pi networks and eventually polyak update the target q networks)
        if t >= update_after and (t + 1) % update_every == 0:
            if sample_mode == 1:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size, sample_mode)
                    update(data=batch)
            elif sample_mode == 2:
                sequence_length = 20
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size, sample_mode, sequence_length)
                    update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.(At test time, to see how well the policy exploits what it has learned, we remove stochasticity and use the mean action instead of a sample from the distribution. This tends to improve performance over the original stochastic policy.)
            test_agent()  # TODO double check

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            if not (t >= update_after and (t + 1) % update_every == 0):
                q_info = dict(Q1Vals=np.zeros(1), Q2Vals=np.zeros(1))
                logger.store(**q_info, LogPi=0, LossPi=0, LossQ=0, LossAlpha=0, Alpha=0)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    if save_buffer == True:
        np.save(logger_kwargs["output_dir"] + "/buf_act.npy", replay_buffer.act_buf)
        np.save(logger_kwargs["output_dir"] + "/buf_done.npy", replay_buffer.done_buf)
        np.save(logger_kwargs["output_dir"] + "/buf_rew.npy", replay_buffer.rew_buf)
        np.save(logger_kwargs["output_dir"] + "/buf_obs.npy", replay_buffer.obs_buf)
        np.save(logger_kwargs["output_dir"] + "/buf_obs2.npy", replay_buffer.obs2_buf)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
