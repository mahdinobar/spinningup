
import gym
import numpy as np

from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils import plot
import os

TRAIN=False
DEBUG=True
if __name__ == '__main__':
    if TRAIN:
        # train
        env_fn = lambda: gym.make('Pendulum-v0')
        exp_name="logs"
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/'+exp_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)
        sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=1000, epochs=50, replay_size=100000, gamma=0.99, polyak=0.995, lr=0.001, alpha=0.2, batch_size=100, start_steps=1000, update_after=200, update_every=50, num_test_episodes=10, max_ep_len=1000, logger_kwargs=logger_kwargs, save_freq=1)
    if DEBUG:
        from gym.wrappers.monitoring import VideoRecorder
        VideoRecorder(env_fn,'/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs')
        # visualize output
        _, get_action = load_policy_and_env('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs')
        env_fn = gym.make('Pendulum-v0')
        run_policy(env_fn, get_action,num_episodes=3)

