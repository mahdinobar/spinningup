
import gym
import numpy as np
from tensorflow_core.python.ops.metrics_impl import false_negatives

from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
import os

TRAIN=0
env_fn = lambda: gym.make('Fep-v0')
exp_name = "Fep_HW_228_b"
Euler_server=False
XPS_laptop=True
if __name__ == '__main__':
    if Euler_server==True:
        output_dir='/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/'+exp_name
    elif XPS_laptop==True:
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
    if TRAIN:
        # train
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)
        sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=136, epochs=3, replay_size=1360000, gamma=0.99, polyak=0.995,
            lr=0.001, alpha_init=2.721, batch_size=136, start_steps=136, update_after=136, update_every=136, num_test_episodes=2,
            max_ep_len=np.inf, logger_kwargs=logger_kwargs, save_freq=1, initial_actions="random", save_buffer=True, sample_mode = 1, automatic_entropy_tuning=True)
    else:
        env_loaded, get_action = load_policy_and_env(output_dir, deterministic=True)
        env=env_fn()
        run_policy(env, get_action,num_episodes=1, output_dir=output_dir)
