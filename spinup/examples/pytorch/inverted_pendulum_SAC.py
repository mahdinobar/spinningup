
import gym
import numpy as np

from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils import plot
import os

TRAIN=True
DEBUG=False
env_fn = lambda: gym.make('Tworr-v0')
exp_name = "Tworrv0_0"
if __name__ == '__main__':
    if TRAIN:
        # train
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)
        sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=500, epochs=2, replay_size=10000, gamma=0.99, polyak=0.995, lr=0.001, alpha=0.2, batch_size=100, start_steps=1000, update_after=200, update_every=50, num_test_episodes=10, max_ep_len=100, logger_kwargs=logger_kwargs, save_freq=1)
    if DEBUG:
        env_fn = gym.make('Pendulum-v0')
        # from gym.wrappers.monitoring.video_recorder import VideoRecorder
        # VideoRecorder(env_fn,'/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/video.mp4', enabled=True)
        # visualize output
        _, get_action = load_policy_and_env('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name)
        run_policy(env_fn, get_action,num_episodes=3)
