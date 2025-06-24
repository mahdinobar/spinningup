
import gym
import numpy as np
from tensorflow_core.python.ops.metrics_impl import false_negatives

from spinup import sac_pytorch as sac
import spinup.algos.pytorch.sac.core as core
from spinup.algos.pytorch.sac.sac import load_checkpoint
from spinup.utils.test_policy import load_policy_and_env, run_policy
import os

TRAIN=0
env_fn = lambda: gym.make('Fep-v0')
exp_name = "Fep_HW_309"
exp_name_checkpoint = "Fep_HW_301"
Euler_server=False
XPS_laptop=True
if __name__ == '__main__':
    if Euler_server==True:
        output_dir='/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/'+exp_name
        checkpoint_dir='/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/'+exp_name_checkpoint
    elif XPS_laptop==True:
        output_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name
        checkpoint_dir='/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/'+exp_name_checkpoint
    if TRAIN:
        # train
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)
        # to save checkpoint sac
        sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=136, epochs=4000, replay_size=1360000, gamma=0.99, polyak=0.995,
            lr=0.001, alpha_init=0.1, batch_size=136, start_steps=13600, update_after=13600, update_every=136, num_test_episodes=2,
            max_ep_len=np.inf, logger_kwargs=logger_kwargs, save_freq=1, initial_actions="random", save_buffer=True, sample_mode = 1, automatic_entropy_tuning=True, save_checkpoint_switch=True)

        # # to load from checkpoint sac
        # sac(env_fn, ac_kwargs={}, seed=0, steps_per_epoch=136, epochs=40, replay_size=1360000, gamma=0.99, polyak=0.995,
        #         lr=0.001, alpha_init=0.1, batch_size=136, start_steps=0, update_after=0, update_every=136, num_test_episodes=2,
        #         max_ep_len=np.inf, logger_kwargs=logger_kwargs, save_freq=1, initial_actions="random", save_buffer=True, sample_mode = 1, automatic_entropy_tuning=True, load_checkpoint_switch=True, checkpoint_dir=checkpoint_dir)
    else:
        env_loaded, get_action = load_policy_and_env(output_dir, deterministic=True)
        env=env_fn()
        run_policy(env, get_action,num_episodes=5, output_dir=output_dir)
