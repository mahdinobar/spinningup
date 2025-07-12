import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import matplotlib.pyplot as plt
import numpy as np


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)
    # make function for producing an action given a single state
    def get_action(x):
        # start_time = time.time()
        with torch.no_grad():
            # comment for libtorch Cpp save
            x = torch.as_tensor(x,dtype=torch.float32)
            action = model.act(x, deterministic)

            # # uncomment for libtorch Cpp save
            # x = torch.as_tensor(x, dtype=torch.double)
            # action = model.act(x, deterministic=True)

            # end_time=time.time()
            # print("dt=", (end_time - start_time)*1000 , " [ms]\n")

            # trace_script_module = torch.jit.trace(model, x)
            # trace_script_module.save("/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/Fep_HW_37/pyt_save/tracedModel.pt")
            # model2=torch.jit.script(model.pi)
            # # # uncomment to save model of actor for libtorch
            # traced_model_Cpp=torch.jit.trace(model.pi, x.reshape(1,27)) #ATTENTION to set correctly dimension of state space here
            # traced_model_Cpp.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_312/traced_model_Cpp_Fep_HW_312_double.pt")
        return action

    # ac.act(torch.as_tensor(o, dtype=torch.float32),
    #        deterministic)

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, output_dir=""):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    # logger = EpochLogger(output_dir=output_dir, resume=False)
    logger = EpochLogger()
    o, d, ep_ret, ep_len, n, r = env.reset(), False, 0, 0, 0, 0

    while n < num_episodes:
        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):

            # os.makedirs("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds", exist_ok=True)
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_q.npy",
            #         env.env.plot_data_buffer[:, :6])
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_dq.npy",
            #         env.env.plot_data_buffer[:, 6:12])
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_rd.npy",
            #         env.env.plot_data_buffer[:, 12:15])
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_drd.npy",
            #         env.env.plot_data_buffer[:, 15:18])
            # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_309/kinematics_error_bounds/PIonly_r.npy",
            #         env.env.plot_data_buffer[:, 18:21])
            if num_episodes ==1:
                env.render(output_dir)
            elif num_episodes > 1:
                # np.save(output_dir + "/plot_data_buffer_episode_{}".format(str(n)), env.unwrapped.plot_data_buffer)
                # np.save(output_dir + "/PIonly_plot_data_buffer_episode_{}".format(str(n)), env.unwrapped.plot_data_buffer)
                if n==num_episodes-1:
                    env.render(output_dir)
            n += 1
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str,
                        default="/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/Tworrv0_10")
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath,
                                          args.itr if args.itr >= 0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
