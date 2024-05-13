
import gym
import numpy as np

from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
import os

TRAIN=1
env_fn = lambda: gym.make('Fep-v0')
exp_name = "Fepv0_16"
if __name__ == '__main__':
    if TRAIN:
        # train
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)
        sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=100, epochs=30, replay_size=1000000, gamma=0.99, polyak=0.995,
            lr=0.001, alpha_init=0.001, batch_size=100, start_steps=10000, update_after=10000, update_every=100, num_test_episodes=2,
            max_ep_len=np.inf, logger_kwargs=logger_kwargs, save_freq=1, initial_actions="random", save_buffer=True, sample_mode = 1, automatic_entropy_tuning=True)
    else:
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
        env_loaded, get_action = load_policy_and_env(output_dir, deterministic=True)
        env=env_fn()
        run_policy(env, get_action,num_episodes=1, output_dir=output_dir)
