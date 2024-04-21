
import gym
import numpy as np

from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
import os

TRAIN=1
env_fn = lambda: gym.make('Fep-v0')
exp_name = "Fepv0_8_1"
if __name__ == '__main__':
    if TRAIN:
        # train
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)
        sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=100, epochs=300, replay_size=1000000, gamma=0.99, polyak=0.995,
            lr=0.005, alpha=0.2, batch_size=100, start_steps=10000, update_after=100, update_every=100, num_test_episodes=2,
            max_ep_len=np.inf, logger_kwargs=logger_kwargs, save_freq=1, initial_actions="zero", save_buffer=True, sample_mode = 1)
    else:
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
        env_loaded, get_action = load_policy_and_env(output_dir)
        env=env_fn()
        run_policy(env, get_action,num_episodes=1, output_dir=output_dir)
