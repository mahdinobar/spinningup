
import gym
from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils import plot
DEBUG=True
if __name__ == '__main__':
    # train
    # env_fn = lambda: gym.make('Pendulum-v0')
    # logger_kwargs = dict(output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs', exp_name='test_0_1')
    # sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=1000, epochs=50, replay_size=100000, gamma=0.99, polyak=0.995, lr=0.001, alpha=0.2, batch_size=100, start_steps=1000, update_after=200, update_every=50, num_test_episodes=10, max_ep_len=1000, logger_kwargs=logger_kwargs, save_freq=1)
    if DEBUG:
        # visualize output
        _, get_action = load_policy_and_env('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs')
        env_fn = gym.make('Pendulum-v0')
        run_policy(env_fn, get_action,num_episodes=5)

