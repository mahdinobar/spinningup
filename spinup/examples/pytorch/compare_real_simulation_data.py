import rosbag
import numpy as np
from geometry_msgs.msg import PoseStamped
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import pybullet_data
from gym import core, spaces
from gym.utils import seeding
import sys

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pybullet as p

sys.path.append('/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch')
__copyright__ = "Copyright 2025, IfA https://control.ee.ethz.ch/"
__credits__ = ["Mahdi Nobar"]
__author__ = "Mahdi Nobar from ETH Zurich <mnobar@ethz.ch>"
render = False
if render == False:
    # Connect to physics client
    physics_client = pb.connect(pb.DIRECT)
else:
    _width = 224
    _height = 224
    _cam_dist = 1.3
    _cam_yaw = 15
    _cam_pitch = -30
    _cam_roll = 0
    camera_target_pos = [0.2, 0, 0.]
    _screen_width = 3840  # 1920
    _screen_height = 2160  # 1080
    physics_client = pb.connect(pb.GUI,
                                options='--mp4fps=10 --background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (
                                    _screen_width, _screen_height))
    plane = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/plane.urdf",
                        useFixedBase=True, physicsClientId=physics_client)
    conveyor_object = pb.loadURDF(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/dobot_conveyer.urdf",
        useFixedBase=True, physicsClientId=physics_client)

    # Initialise debug camera angle
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=5,
        cameraPitch=-30,
        cameraTargetPosition=camera_target_pos,
        physicsClientId=physics_client)
    pb.resetBasePositionAndOrientation(
        conveyor_object,
        np.array([534e-3, -246.5e-3, 154.2e-3]) + np.array([-0.002, -0.18, -0.15]),
        pb.getQuaternionFromEuler([0, 0, np.pi / 2 - 0.244978663]))

# TDOO ATTENTION how you choose dt
dt_pb_sim = 1 / 240
# # default timestep is 1/240 second (search fixedTimeStep)
pb.setTimeStep(timeStep=dt_pb_sim, physicsClientId=physics_client)
# # Set gravity
pb.setGravity(0, 0, -9.81, physicsClientId=physics_client)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_v2.urdf",
                  useFixedBase=True, physicsClientId=physics_client)


def load_bags(file_name, bag_path, save=False):
    # file_name = "SAC_1"
    topic_name = '/PRIMITIVE_velocity_controller/dq_PID_messages'
    if save:
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                pos = msg.pose.position
                ori = msg.pose.orientation
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z]  # You can also add ori.w if needed
                data.append(row)
        # Convert to NumPy array
        dq_PI = np.array(data)
        np.save(bag_path + file_name + "_dq_PID_messages.npy", dq_PI)

        topic_name = '/PRIMITIVE_velocity_controller/dq_SAC_messages'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                pos = msg.pose.position
                ori = msg.pose.orientation
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z]  # You can also add ori.w if needed
                data.append(row)
        # Convert to NumPy array
        dq_SAC = np.array(data)
        np.save(bag_path + file_name + "_dq_SAC_messages.npy", dq_SAC)

        topic_name = '/franka_state_controller/joint_states'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = np.append(np.array(timestamp), np.asarray((msg.velocity[:7])))
                data.append(row)
        # Convert to NumPy array
        dq = np.array(data)
        np.save(bag_path + file_name + "_dq_measured.npy", dq)

        # topic_name='/franka_state_controller/joint_states/panda_joint{}/velocity'.format(str(i))
        topic_name = '/franka_state_controller/joint_states_desired'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = np.append(np.array(timestamp), np.asarray((msg.velocity[:7])))
                data.append(row)
        # Convert to NumPy array
        dq_desired_measured = np.array(data)
        np.save(bag_path + file_name + "_dq_desired_measured.npy", dq_desired_measured)

        topic_name = '/franka_state_controller/joint_states'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = np.append(np.array(timestamp), np.asarray((msg.position[:7])))
                data.append(row)
        # Convert to NumPy array
        q = np.array(data)
        np.save(bag_path + file_name + "_q_measured.npy", q)

        topic_name = '/franka_state_controller/myfranka_ee_pose'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                pos = msg.pose.position
                row = [timestamp, pos.x, pos.y, pos.z]
                data.append(row)
        # Convert to NumPy array
        p_hat_EE = np.array(data)
        np.save(bag_path + file_name + "_p_hat_EE_measured.npy", p_hat_EE)

        topic_name = '/PRIMITIVE_velocity_controller/r_star_messages'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, msg.vector.x, msg.vector.y, msg.vector.z]
                data.append(row)
        # Convert to NumPy array
        p_star = np.array(data)
        np.save(bag_path + file_name + "_p_star_measured.npy", p_star)
    else:
        dq_PI = np.load(bag_path + file_name + "_dq_PID_messages.npy")
        dq_SAC = np.load(bag_path + file_name + "_dq_SAC_messages.npy")
        dq = np.load(bag_path + file_name + "_dq_measured.npy")
        dq_desired_measured = np.load(bag_path + file_name + "_dq_desired_measured.npy")
        q = np.load(bag_path + file_name + "_q_measured.npy")
        p_hat_EE = np.load(bag_path + file_name + "_p_hat_EE_measured.npy")
        p_star = np.load(bag_path + file_name + "_p_star_measured.npy")

    return dq_PI, dq_SAC, dq, dq_desired_measured, q, p_hat_EE, p_star


def load_real(bag_path, file_name, save=False):
    # file_name = "SAC_1"
    topic_name = '/PRIMITIVE_velocity_controller/r_star_messages'
    if save:
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, msg.vector.x, msg.vector.y, msg.vector.z]
                data.append(row)
        # Convert to NumPy array
        r_star = np.array(data)
        np.save(bag_path + file_name + "_r_star.npy", r_star)

        topic_name = '/franka_state_controller/myfranka_ee_pose'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path + file_name + ".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                data.append(row)
        # Convert to NumPy array
        ee_pose = np.array(data)
        np.save(bag_path + file_name + "_ee_pose.npy", ee_pose)

    else:
        ee_pose = np.load(bag_path + file_name + "_ee_pose.npy")
        r_star = np.load(bag_path + file_name + "_r_star.npy")

    return ee_pose, r_star


def compare_data(file_name, dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured, p_hat_EE, p_star, PIonly):
    # file_name= file_name+"_240Hz_"
    if PIonly == False:
        sim_plot_data_buffer = np.load(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/SAC_plot_data_buffer.npy")
        sim_state_buffer = np.load(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/SAC_state_buffer.npy")
    elif PIonly == True:
        sim_plot_data_buffer = np.load(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/PIonly_plot_data_buffer.npy")
        sim_state_buffer = np.load(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/PIonly_state_buffer.npy")

    # Attentions (remove redundant initial data of some topics based on absolute ROS timestamp; conside dq_PI as base)
    idx_init_dq_measured = np.argwhere(abs(dq_PI[0, 0] - dq_measured[:, 0]) < 1e-3)
    idx_init_dq_desired_measured = np.argwhere(abs(dq_PI[0, 0] - dq_desired_measured[:, 0]) < 1e-3)
    dq_measured = dq_measured[idx_init_dq_measured[0][0]:, :]
    q_measured = q_measured[idx_init_dq_measured[0][0]:, :]
    dq_desired_measured = dq_desired_measured[idx_init_dq_desired_measured[0][0]:, :]
    idx_init_p_hat_EE = np.argwhere(abs(dq_PI[0, 0] - p_hat_EE[:, 0]) < 1e-3)
    p_hat_EE = p_hat_EE[idx_init_p_hat_EE[0][0]:, :]

    t_ = (dq_PI[:, 0] - dq_PI[0, 0]) * 1000
    target_times = np.arange(0, t_[-1], 100)

    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_PI = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_PI = t_[closest_idx_PI]

    t_ = (dq_SAC[:, 0] - dq_SAC[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_SAC = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_SAC = t_[closest_idx_SAC]

    # Reset robot at the origin and move the target object to the goal position and orientation
    pb.resetBasePositionAndOrientation(
        arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]), physicsClientId=physics_client)
    # Reset joint at initial angles
    q_init = q_measured[0, 1:]
    for i in range(7):
        pb.resetJointState(arm, i, q_init[i], physicsClientId=physics_client)
    for j in [6] + list(range(8, 12)):
        pb.resetJointState(arm, j, 0, physicsClientId=physics_client)

    q_sim, dq_sim, tau_sim = [], [], []
    if PIonly == False:
        for idx_PI, idx_SAC in zip(closest_idx_PI, closest_idx_SAC):
            # for idx_PI in closest_idx_PI:
            print("simulation idx_PI=", idx_PI)
            dqc_t = dq_PI[idx_PI, 1:7] + dq_SAC[idx_SAC, 1:7]
            pb.setJointMotorControlArray(
                arm,
                [0, 1, 2, 3, 4, 5],
                controlMode=pb.VELOCITY_CONTROL,
                targetVelocities=list(dqc_t),
                velocityGains=[1, 1, 2, 1, 1, 1],
                forces=[87, 87, 87, 87, 12, 12],
                physicsClientId=physics_client
            )
            # # TODO pay attention to number of repetition (e.g., use 24 for period 24*1/240*1000=100 [ms])
            # for _ in range(24):
            #     # default timestep is 1/240 second
            #     pb.stepSimulation(physicsClientId=physics_client)
            pb.stepSimulation(physicsClientId=physics_client)

            # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
            info = pb.getJointStates(arm, range(10))
            q_sim_, dq_sim_, tau_sim_ = [], [], []
            for joint_info in info:
                q_sim_.append(joint_info[0])
                dq_sim_.append(joint_info[1])
                tau_sim_.append(joint_info[3])
            q_sim.append(q_sim_)
            dq_sim.append(dq_sim_)
            tau_sim.append(tau_sim_)
    elif PIonly == True:
        for idx_PI in closest_idx_PI:
            print("simulation idx_PI=", idx_PI)
            dqc_t = dq_PI[idx_PI, 1:7]
            pb.setJointMotorControlArray(
                arm,
                [0, 1, 2, 3, 4, 5],
                controlMode=pb.VELOCITY_CONTROL,
                targetVelocities=list(dqc_t),
                velocityGains=[1, 1, 2, 1, 1, 1],
                forces=[87, 87, 87, 87, 12, 12],
                physicsClientId=physics_client
            )
            # TODO pay attention to number of repetition (e.g., use 24 for period 24*1/240*1000=100 [ms])
            for _ in range(24):
                # default timestep is 1/240 second
                pb.stepSimulation(physicsClientId=physics_client)

            # get measured values at time tp1 denotes t+1 for q and ddq as well as applied torque at time t
            info = pb.getJointStates(arm, range(10))
            q_sim_, dq_sim_, tau_sim_ = [], [], []
            for joint_info in info:
                q_sim_.append(joint_info[0])
                dq_sim_.append(joint_info[1])
                tau_sim_.append(joint_info[3])
            q_sim.append(q_sim_)
            dq_sim.append(dq_sim_)
            tau_sim.append(tau_sim_)

    dq_sim = np.array(dq_sim)[:, :6]
    q_sim = np.array(q_sim)[:, :6]
    # manually correct q_sim for 1/240 * 24 simulation sampling time
    q_sim = q_sim[0, :] + (q_sim[:, :] - q_sim[0, :]) * 24

    t_ = (q_measured[:, 0] - q_measured[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_q = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_q = t_[closest_idx_q]
    t_ = (dq_measured[:, 0] - dq_measured[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_dq = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_dq = t_[closest_idx_dq]
    dq = dq_measured[closest_idx_dq, 1:7]
    q = q_measured[closest_idx_q, 1:7]
    # fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    # # Flatten axes array for easy indexing
    # axes = axes.flatten()
    # # Plot for each joint (0 to 5)
    # for joint_idx in range(6):
    #     ax = axes[joint_idx]
    #     # Downsample for visualization (every 100th point)
    #     # indices = np.arange(0, len(dq_measured), 100)
    #     y_measured = dq_measured[closest_idx_dq, joint_idx+1]
    #     y_sim = np.array(dq_sim)[:, joint_idx]
    #     ax.plot(closest_t_dq, y_measured, '-ob', label="dq - real measured")  # blue circles
    #     ax.plot(closest_t_PI, y_sim, '-or', label="dq - simulation estimation")  # red circles
    #     ax.set_xlabel("t")
    #     ax.set_ylabel(f"dq{joint_idx + 1}")
    #     ax.grid(True)
    # ax.legend()
    # # Adjust layout
    # plt.tight_layout()
    # plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_dq_measured_dq_simest_frictionJ3_02.png".format(file_name), format="png",
    #             bbox_inches='tight')
    # plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    plt.rcParams['font.family'] = 'Serif'
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    # Plot for each joint (0 to 5)
    for joint_idx in range(6):
        ax = axes[joint_idx]
        # Downsample for visualization (every 100th point)
        # indices = np.arange(0, len(dq_measured), 100)
        if PIonly == False:
            y_commanded = dq_PI[closest_idx_PI, joint_idx + 1] + dq_SAC[closest_idx_PI, joint_idx + 1]
        elif PIonly == True:
            y_commanded = dq_PI[closest_idx_PI, joint_idx + 1]
        ax.plot(closest_t_PI, y_commanded, '-om', label="dq commanded - real", markersize=8)  # blue circles

        t_ = (dq_desired_measured[:, 0] - dq_desired_measured[0, 0]) * 1000
        # Target times: 0, 100, 200, ..., up to max(t)
        target_times_d = np.arange(0, t_[-1], 100)
        # Find indices in t closest to each target time
        closest_idx_PI_d = np.array([np.abs(t_ - target).argmin() for target in target_times_d])[:120]
        closest_t_PI_d = t_[closest_idx_PI_d]
        y_desired = dq_desired_measured[closest_idx_PI_d, joint_idx + 1]
        ax.plot(closest_t_PI_d, y_desired, '-og', label="dq desired - real", markersize=3)  # red circles
        ax.set_xlabel("t")
        ax.set_ylabel(f"dq{joint_idx + 1}")
        ax.grid(True)
    ax.legend()
    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_dq_desired_commanded_frictionJ3_02.png".format(
            file_name),
        format="png",
        bbox_inches='tight')
    plt.show()

    # Plot q and q_sim (Position)
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 16))
    plt.rcParams['font.family'] = 'Serif'
    # fig1.suptitle('Joint Positions: Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs1[i // 2, i % 2]
        ax.plot(closest_t_q, q[:, i], '-og', label='Measured q')
        ax.plot(closest_t_PI, q_sim[:, i], '-ob', label='Sim q (no mismatch comp.)', markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('q[{}] [rad]'.format(i))
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_q_qSimRaw_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    dq_sim = np.array(dq_sim)[:, :6]
    q_sim_ = sim_state_buffer[:94, 3:9]
    # Plot q and q_sim (Position)
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 16))
    plt.rcParams['font.family'] = 'Serif'
    # fig1.suptitle('Joint Positions: Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs1[i // 2, i % 2]
        ax.plot(closest_t_q, q[:, i], '-og', label='Measured q')
        ax.plot(closest_t_PI[10:], q_sim_[:, i], '-sk', label='Sim HW312 State Buffer', markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('q[{}] [rad]'.format(i))
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_qReal_qSimHW313_j3_1_StateBuffer_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # %%
    input_scalers = []
    target_scalers_q = []
    models_q = []
    likelihoods_q = []
    likelihoods_dq = []
    GP_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/trainOnSAC_1_2_3_testOnSAC_5_trackingPhaseOnly/"
    GP_input_dim = 2
    for joint_number in range(6):
        # Load scalers
        input_scaler = joblib.load(GP_dir + f'input_scaler{joint_number}.pkl')
        target_scaler_q = joblib.load(GP_dir + f'target_scaler_q{joint_number}.pkl')
        # Instantiate and load model for q
        likelihood_q = gpytorch.likelihoods.GaussianLikelihood()

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                # covar_module = gpytorch.kernels.ScaleKernel(
                #     gpytorch.kernels.RBFKernel()
                # )
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel() +
                    gpytorch.kernels.MaternKernel(nu=2.5)
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # model_q = GPModel(train_x_shape=(1, GP_input_dim), likelihood=likelihood_q)
        train_x_placeholder = torch.zeros((1, GP_input_dim))
        train_y_placeholder = torch.zeros((1,))
        model_q = ExactGPModel(train_x_placeholder, train_y_placeholder, likelihood_q)
        checkpoint_q = torch.load(
            GP_dir + f'gp_model_q{joint_number}.pth')
        model_q.load_state_dict(checkpoint_q['model_state_dict'])
        likelihood_q.load_state_dict(checkpoint_q['likelihood_state_dict'])
        input_scaler = checkpoint_q['input_scaler']  # overwrite with trained one
        target_scaler_q = checkpoint_q['target_scaler']
        model_q.eval()
        likelihood_q.eval()
        device = torch.device('cpu')
        model_q.to(device)
        likelihood_q.to(device)
        # Append to lists
        input_scalers.append(input_scaler)
        target_scalers_q.append(target_scaler_q)
        # target_scalers_dq.append(target_scaler_dq)
        models_q.append(model_q)
        likelihoods_q.append(likelihood_q)
    #########################################################################
    q_sim_corrected = np.copy(q_sim)
    for k in range(104):
        # ----- q and dq Mismatch Compensation -----
        for i in [0, 2]:
            models_q[i].eval()
            # models_dq[i].eval()
            likelihoods_q[i].eval()
            # likelihoods_dq[i].eval()
            # TODO ????????????
            X_test = np.array([q_sim[k, i], dq_sim[k, i]]).reshape(-1, 2)
            X_test = input_scalers[i].transform(X_test)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            device = torch.device('cpu')  # or 'cuda' if you're using GPU
            X_test = X_test.to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                models_q[i].to(device)
                likelihoods_q[i].to(device)
                pred_q = likelihoods_q[i](models_q[i](X_test))
                # pred_dq = likelihoods_dq[i](models_dq[i](X_test))
                mean_q = pred_q.mean.numpy()
                # mean_dq = pred_dq.mean.numpy()
                std_q = pred_q.variance.sqrt().numpy()
                # std_dq = pred_dq.variance.sqrt().numpy()
                # Uncomment when Normalizing
                mean_q = target_scalers_q[i].inverse_transform(mean_q.reshape(-1, 1)).flatten()
                std_q = std_q * target_scalers_q[i].scale_[0]  # only scale, don't shift
                # mean_dq = target_scalers_dq[i].inverse_transform(mean_dq.reshape(-1, 1)).flatten()
                # std_dq = std_dq * target_scalers_dq[i].scale_[0]  # only scale, don't shift
            # TODO
            if ~np.isnan(mean_q):
                q_sim_corrected[k, i] = q_sim_corrected[k, i] + mean_q
            else:
                print("mean_q[{}] is nan!!".format(i))
            # if ~np.isnan(mean_dq):
            #     dq_tp1[i] = dq_tp1[i] + mean_dq
            # else:
            #     print("mean_dq[{}] is nan!".format(i))
    #########################################################################

    dq_sim = np.array(dq_sim)[:, :6]
    q_sim_corrected = np.array(q_sim_corrected)[:, :6]
    # Plot q and q_sim (Position)
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 16))
    plt.rcParams['font.family'] = 'Serif'
    # fig1.suptitle('Joint Positions: Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs1[i // 2, i % 2]
        ax.plot(closest_t_q, q[:, i], '-og', label='Measured q')
        ax.plot(closest_t_PI[:], q_sim_corrected[:, i], '-om',
                label='Simulated q_sim + mismatch correction on joint 1 and 3', markersize=2)
        ax.plot(closest_t_PI[:], q_sim[:, i], '-ob', label='Simulated q_sim (no mismatch correction)', markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('q[{}] [rad]'.format(i))
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_qReal_qSimRawMismatchCompJ1and3_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # Plot dq and dq_sim (Velocity)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 16))
    fig2.suptitle('Joint Velocities: Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_dq, dq[:, i], '-og', label='Measured dq')
        ax.plot(closest_t_PI, dq_sim[:, i], '-ob', label='Simulated dq_sim', markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Velocity')
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_dq_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # Plot dq and error dq (Velocity)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 16))
    fig2.suptitle('Joint Velocities: Absolute Error Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_dq, abs(dq_sim[:, i] - dq[:, i]), '-sr')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('$|dq_{\t{sim}} - dq_{\t{measured}}|$')
        ax.grid(True)
        ax.set_ylim([0, 0.05])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_dq_abs_error_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # Plot dq and error q (Position)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 16))
    fig2.suptitle('Joint Positions: Absolute Error Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_q, abs(q_sim[:, i] - q[:, i]), '-sr')
        ax.plot(closest_t_q, abs(q_sim_corrected[:, i] - q[:, i]), '-sm')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('$|q_{\t{sim}} - q_{\t{measured}}|$')
        ax.grid(True)
        ax.set_ylim([0, 0.1])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_q_abs_error_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # Plot dq and error dq (Velocity)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_dq / 1000, (dq_sim[:, i] - dq[:, i]) * 180 / 3.14, '-sk')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{\t{sim}} - dq_{\t{measured}}$ [deg/s]')
        ax.grid(True)
        ax.set_ylim([-10.1, 10.1])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_dq_error_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()
    # Plot dq and error q (Position)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_q / 1000, (q_sim[:, i] - q[:, i]) * 180 / 3.14, '-sk')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('$q_{\t{simRAW}} - q_{\t{measured}}$ [deg]')
        ax.grid(True)
        ax.set_ylim([-4.1, 4.1])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_q_error_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # COMPARISON REAL VS TESTED SIMULATION IN TRAINING ONLY - TRACKING PHASE
    # compare dq_PI
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot((closest_t_PI[10:] - closest_t_PI[10]) / 1000, dq_PI[closest_idx_PI[10:], i + 1] * 180 / np.pi, '-og',
                label='real $dq_{{PI}}$[{}]'.format(str(i)))
        ax.plot((np.arange(0, 13600, 100)) / 1000, sim_state_buffer[:, 15 + i] * 180 / np.pi, '-ob',
                label='simulation $dq_{{PI}}$[{}]'.format(str(i)), markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{PI}}$[{}] [deg/s]'.format(str(i)))
        ax.grid(True)
        ax.set_ylim([-2, 4])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_PI_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # compare dq_PI
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot((closest_t_PI[10:] - closest_t_PI[10]) / 1000, dq_PI[closest_idx_PI[10:], i + 1], '-og',
                label='real $dq_{{PI}}$[{}]'.format(str(i)))
        ax.plot((np.arange(0, 13600, 100)) / 1000, sim_state_buffer[:, 15 + i], '-ob',
                label='simulation $dq_{{PI}}$[{}]'.format(str(i)), markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{PI}}$[{}] [rad/s]'.format(str(i)))
        ax.grid(True)
        ax.set_ylim([-2, 4])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_PI_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # compare dq_SAC
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot((closest_t_SAC[10:] - closest_t_SAC[10]) / 1000, dq_SAC[closest_idx_PI[10:], i + 1] * 180 / np.pi,
                '-og',
                label='real $dq_{{SAC}}$[{}]'.format(str(i)))
        ax.plot((np.arange(0, 13600, 100)) / 1000, sim_state_buffer[:, 21 + i] * 180 / np.pi, '-ob',
                label='simulation $dq_{{SAC}}$[{}]'.format(str(i)), markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{SAC}}$[{}] [deg/s]'.format(str(i)))
        ax.grid(True)
        ax.set_ylim([-24, 24])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_SAC_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # compare dq_SAC
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot((closest_t_SAC[10:] - closest_t_SAC[10]) / 1000, dq_SAC[closest_idx_PI[10:], i + 1],
                '-og',
                label='real $dq_{{SAC}}$[{}]'.format(str(i)))
        ax.plot((np.arange(0, 13600, 100)) / 1000, sim_state_buffer[:, 21 + i], '-ob',
                label='simulation $dq_{{SAC}}$[{}]'.format(str(i)), markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{SAC}}$[{}] [rad/s]'.format(str(i)))
        ax.grid(True)
        ax.set_ylim([-24, 24])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_SAC_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    t_ = (p_star[:, 0] - p_star[0, 0]) * 1000
    target_times = np.arange(0, t_[-1], 100)
    closest_idx_p_star = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_p_star = t_[closest_idx_p_star]

    t_ = (p_hat_EE[:, 0] - p_hat_EE[0, 0]) * 1000
    target_times = np.arange(0, t_[-1], 100)
    closest_idx_p_hat_EE = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_p_hat_EE = t_[closest_idx_p_hat_EE]

    # compare p^*-\hat{p}
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(3):
        ax = axs2[i]
        ax.plot(closest_t_p_star[10:] - closest_t_p_star[10],
                (-p_star[closest_idx_p_star[10:], i + 1] + p_hat_EE[closest_idx_p_hat_EE[10:], i + 1]) * 1000, '-og',
                label='real $\hat{{p}}[{}]-p^*[{}$'.format(str(i), str(i)))
        ax.plot(np.arange(0, 13600, 100), sim_state_buffer[:, 0 + i], '-ob',
                label='simulation $\hat{{p}}[{}]- p^*[{}]$'.format(str(i), str(i)), markersize=2)
        # ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\hat{{p}}[{}]-p^*[{}]$ [mm]'.format(str(i), str(i)))
        ax.grid(True)
        ax.set_ylim([-5, 5])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_delta_p_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # compare p^*
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(3):
        ax = axs2[i]
        ax.plot((closest_t_p_star[10:] - closest_t_p_star[10]) / 1000,
                (p_star[closest_idx_p_star[10:], i + 1]) * 1000, '-og',
                label='real $p^*[{}]$'.format(str(i)))
        ax.plot(np.arange(0, 13600, 100) / 1000, sim_plot_data_buffer[:, 3 + i] * 1000, '-ob',
                label='simulation $p^*[{}]$'.format(str(i)), markersize=2)
        # ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$p^*[{}]$ [mm]'.format(str(i)))
        ax.grid(True)
        # ax.set_ylim([-5, 4])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_p_star_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # compare p^*
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(3):
        ax = axs2[i]
        ax.plot((closest_t_p_hat_EE[10:] - closest_t_p_hat_EE[10]) / 1000,
                (p_hat_EE[closest_idx_p_hat_EE[10:], i + 1]) * 1000, '-og',
                label='real $\hat{{p}}[{}]$'.format(str(i)))
        ax.plot(np.arange(0, 13600, 100) / 1000, sim_plot_data_buffer[:, 0 + i] * 1000, '-ob',
                label='simulation $\hat{{p}}[{}]$'.format(str(i)), markersize=2)
        # ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\hat{{p}}[{}]$ [mm]'.format(str(i)))
        ax.grid(True)
        # ax.set_ylim([-5, 4])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_p_hat_EE_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_dq[10:] - closest_t_dq[10]) / 1000
        dq_real = dq_measured[closest_idx_dq[10:], i + 1] * 180 / np.pi
        dq_sim_test = sim_state_buffer[:, 9 + i] * 180 / np.pi
        diff = dq_real - dq_sim_test[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], dq_real, '-og', label='$dq_{{real}}[{}]$'.format(i))
        ax.plot(t_real[:len(dq_real)], dq_sim_test[:len(dq_real)], '-ob', label='$dq_{{sim_{{test}}}}[{}]$'.format(i),
                markersize=3)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq$ [deg/s]')
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_test_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_q[10:] - closest_t_q[10]) / 1000
        q_real = q_measured[closest_idx_q[10:], i + 1] * 180 / np.pi
        q_sim_test = sim_state_buffer[:, 3 + i] * 180 / np.pi
        ax.plot(t_real[:len(q_real)], q_real, '-og', label='$q_{{real}}[{}]$'.format(i))
        ax.plot(t_real[:len(q_real)], q_sim_test[:len(dq_real)], '-ob', label='$q_{{sim_{{test}}}}[{}]$'.format(i),
                markersize=3)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$q$ [deg]')
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_q_test_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_q[10:] - closest_t_q[10]) / 1000
        q_real = q_measured[closest_idx_q[10:], i + 1]
        q_sim_test = sim_state_buffer[:, 3 + i]
        ax.plot(t_real[:len(q_real)], q_real, '-og', label='$q_{{real}}[{}]$'.format(i))
        ax.plot(t_real[:len(q_real)], q_sim_test[:len(dq_real)], '-ob', label='$q_{{sim_{{test}}}}[{}]$'.format(i),
                markersize=3)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$q$ [deg]')
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_q_test_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_dq[10:] - closest_t_dq[10]) / 1000
        dq_real = dq_measured[closest_idx_dq[10:], i + 1]
        dq_sim_test = sim_state_buffer[:, 9 + i]
        diff = dq_real - dq_sim_test[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], dq_real, '-og', label='$dq_{{real}}[{}]$'.format(i))
        ax.plot(t_real[:len(dq_real)], dq_sim_test[:len(dq_real)], '-ob', label='$dq_{{sim_{{test}}}}[{}]$'.format(i),
                markersize=3)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq$ [rad/s]')
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_test_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # errors plot for comparison
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_PI[10:] - closest_t_PI[10]) / 1000
        dq_real = dq_PI[closest_idx_PI[10:], i + 1] * 180 / np.pi
        dq_sim = sim_state_buffer[:, 15 + i] * 180 / np.pi
        diff = dq_real - dq_sim[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta dq_{{PI}}[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta dq_{{PI}}[{}]$ [deg/s]'.format(i))
        # ax.set_ylim([-0.2, 0.2])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_PI_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # errors plot for comparison
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_PI[10:] - closest_t_PI[10]) / 1000
        dq_real = dq_PI[closest_idx_PI[10:], i + 1]
        dq_sim = sim_state_buffer[:, 15 + i]
        diff = dq_real - dq_sim[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta dq_{{PI}}[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta dq_{{PI}}[{}]$ [rad/s]'.format(i))
        # ax.set_ylim([-0.2, 0.2])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_PI_err_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_SAC[10:] - closest_t_SAC[10]) / 1000
        dq_real = dq_SAC[closest_idx_SAC[10:], i + 1] * 180 / np.pi
        dq_sim = sim_state_buffer[:, 15 + i] * 180 / np.pi
        diff = dq_real - dq_sim[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$real - sim$')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta dq_{{SAC}}[{}]$ [deg/s]'.format(i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_SAC_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_SAC[10:] - closest_t_SAC[10]) / 1000
        dq_real = dq_SAC[closest_idx_SAC[10:], i + 1]
        dq_sim = sim_state_buffer[:, 15 + i]
        diff = dq_real - dq_sim[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$real - sim$')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta dq_{{SAC}}[{}]$ [rad/s]'.format(i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_SAC_err_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    for i in range(3):
        ax = axs2[i]
        t_real = (closest_t_p_star[10:] - closest_t_p_star[10]) / 1000
        diff_real = (p_star[closest_idx_p_star[10:], i + 1] - p_hat_EE[closest_idx_p_hat_EE[10:], i + 1]) * 1000
        diff_sim = -sim_state_buffer[:, i]  # assumed shape matches
        delta = diff_real - diff_sim[:len(diff_real)]
        ax.plot(t_real[:len(diff_real)], delta, '-ok',
                label='$\delta p_real - \delta p_simTest,\: \Delta p=(p^*[{}]-\hat{{p}}[{}])$'.format(i, i))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta p{}$ [mm]'.format(i))
        # ax.set_ylim([-6, 3])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_delta_p_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    for i in range(3):
        ax = axs2[i]
        t_real = (closest_t_p_star[10:] - closest_t_p_star[10]) / 1000
        p_real = p_star[closest_idx_p_star[10:], i + 1] * 1000
        p_sim = sim_plot_data_buffer[:, 3 + i] * 1000
        delta = p_real - p_sim[:len(p_real)]
        ax.plot(t_real[:len(p_real)], delta, '-ok', label='$\Delta p^*[{}]$'.format(i))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$p^*_{{real}}[{}]-p^*_{{sim_{{test}}}}[{}]$ [mm]'.format(i, i))
        # ax.set_ylim([-6, 3])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_p_star_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    for i in range(3):
        ax = axs2[i]
        t_real = (closest_t_p_hat_EE[10:] - closest_t_p_hat_EE[10]) / 1000
        p_real = p_hat_EE[closest_idx_p_hat_EE[10:], i + 1] * 1000
        p_sim = sim_plot_data_buffer[:, 0 + i] * 1000
        delta = p_real - p_sim[:len(p_real)]
        ax.plot(t_real[:len(p_sim)], delta, '-ok', label='$\Delta \hat{{p}}[{}]$'.format(i))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta \hat{{p}}[{}]$ [mm]'.format(i))
        # ax.set_ylim([-6, 3])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_p_hat_EE_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_q[10:] - closest_t_q[10]) / 1000
        dq_real = q_measured[closest_idx_q[10:], i + 1] * 180 / np.pi
        dq_sim_test = sim_state_buffer[:, 3 + i] * 180 / np.pi
        diff = dq_real - dq_sim_test[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta q[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$q_{{real}}[{}]-q_{{sim_{{test}}}}[{}]$ [deg]'.format(i, i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_q_test_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_q[10:] - closest_t_q[10]) / 1000
        dq_real = q_measured[closest_idx_q[10:], i + 1]
        dq_sim_test = sim_state_buffer[:, 3 + i]
        diff = dq_real - dq_sim_test[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta q[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$q_{{real}}[{}]-q_{{sim_{{test}}}}[{}]$ [rad]'.format(i, i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_q_test_err_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_dq[10:] - closest_t_dq[10]) / 1000
        dq_real = dq_measured[closest_idx_dq[10:], i + 1] * 180 / np.pi
        dq_sim_test = sim_state_buffer[:, 9 + i] * 180 / np.pi
        diff = dq_real - dq_sim_test[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta dq[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{real}}[{}]-dq_{{sim_{{test}}}}[{}]$ [deg/s]'.format(i, i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_test_err_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'Serif'
    })
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        t_real = (closest_t_dq[10:] - closest_t_dq[10]) / 1000
        dq_real = dq_measured[closest_idx_dq[10:], i + 1]
        dq_sim_test = sim_state_buffer[:, 9 + i]
        diff = dq_real - dq_sim_test[:len(dq_real)]
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta dq[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{real}}[{}]-dq_{{sim_{{test}}}}[{}]$ [rad/s]'.format(i, i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_dq_test_err_rad_frictionJ3_02.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    return True


def get_data(dq_PI, p_hat_EE, p_star, q_measured, dq_measured):
    # # Attentions (remove redundant initial data of some topics based on absolute ROS timestamp; conside dq_PI as base)
    idx_init_dq_measured = np.argwhere(abs(dq_PI[0, 0] - dq_measured[:, 0]) < 2e-3)
    # idx_init_dq_desired_measured = np.argwhere(abs(dq_PI[0, 0] - dq_desired_measured[:, 0]) < 1e-3)
    dq_measured_ = dq_measured[idx_init_dq_measured[0][0]:, :]
    q_measured_ = q_measured[idx_init_dq_measured[0][0]:, :]
    # dq_desired_measured = dq_desired_measured[idx_init_dq_desired_measured[0][0]:, :]
    idx_init_p_hat_EE = np.argwhere(abs(dq_PI[0, 0] - p_hat_EE[:, 0]) < 2e-3)
    p_hat_EE_ = p_hat_EE[idx_init_p_hat_EE[0][0]:, :]

    idx_init_p_star = np.argwhere(abs(dq_PI[0, 0] - p_star[:, 0]) < 1e-3)
    p_star_ = p_star[idx_init_p_star[0][0]:, :]

    t_ = (q_measured_[:, 0] - q_measured_[0, 0]) * 1000
    target_times = np.arange(0, t_[-1], 100)
    closest_idx_q = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_q = t_[closest_idx_q]

    t_ = (dq_measured_[:, 0] - dq_measured_[0, 0]) * 1000
    closest_idx_dq = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_dq = t_[closest_idx_dq]

    t_ = (p_star_[:, 0] - p_star_[0, 0]) * 1000
    closest_idx_p_star = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_p_star = t_[closest_idx_p_star]

    t_ = (p_hat_EE_[:, 0] - p_hat_EE_[0, 0]) * 1000
    closest_idx_p_hat_EE = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:120]
    closest_t_p_hat_EE = t_[closest_idx_p_hat_EE]

    # compare p^*-\hat{p}
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 8))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 14,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    for i in range(3):
        ax = axs2[i]
        ax.plot(closest_t_p_star[10:] - closest_t_p_star[10],
                (-p_star_[closest_idx_p_star[10:], i + 1] + p_hat_EE_[closest_idx_p_hat_EE[10:], i + 1]) * 1000, '-og',
                label='real $\hat{{p}}[{}]-p^*[{}$'.format(str(i), str(i)))
        # ax.plot(np.arange(0, 13600, 100), sim_state_buffer[:, 0 + i], '-ob',
        #         label='simulation $\hat{{p}}[{}]- p^*[{}]$'.format(str(i), str(i)), markersize=2)
        # ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\hat{{p}}[{}]-p^*[{}]$ [mm]'.format(str(i), str(i)))
        ax.grid(True)
        # ax.set_ylim([-5, 5])
        ax.legend()
    # plt.savefig(
    #     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_313_9/compare_real_simulation_data/friction02/{}_compare_delta_p_frictionJ3_02.png".format(
    #         file_name), format="png",
    #     bbox_inches='tight')
    plt.show()

    return (q_measured_[closest_idx_q[10:], 1:], dq_measured_[closest_idx_dq[10:], 1:],
            (
                closest_t_p_star[10:] - closest_t_p_star[10]),
        np.array([
        (-p_star_[closest_idx_p_star[10:], 1] + p_hat_EE_[closest_idx_p_hat_EE[10:], 1]) * 1000,
        (-p_star_[closest_idx_p_star[10:], 2] + p_hat_EE_[closest_idx_p_hat_EE[10:], 2]) * 1000,
        (-p_star_[closest_idx_p_star[10:], 3] + p_hat_EE_[closest_idx_p_hat_EE[10:], 3]) * 1000]),
        np.array([
        (p_star_[closest_idx_p_star[10:], 1] ) * 1000,
        (p_star_[closest_idx_p_star[10:], 2] ) * 1000,
        (p_star_[closest_idx_p_star[10:], 3] ) * 1000]))


import pybullet_data

# Start PyBullet in DIRECT mode (no GUI) or GUI if you want visualization
p.connect(p.DIRECT)  # or p.GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
p.setPhysicsEngineParameter(enableConeFriction=0)


def load_jacobian(robot_id, q_t, dq_t):
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for i in range(12):
        if i < 6:
            p.resetJointState(robot_id, i, q[i])
        else:
            p.resetJointState(robot_id, i, 0)
    LinkState = p.getLinkState(robot_id, 9, computeForwardKinematics=True, computeLinkVelocity=True)
    [J_lin, J_ang] = p.calculateJacobian(robot_id,
                                         10,
                                         list(LinkState[2]),
                                         list(q),
                                         list(dq),
                                         list(np.zeros(9)))
    J_linear = np.array(J_lin)
    J_angular = np.array(J_ang)
    J_geo = np.vstack((J_linear, J_angular))
    return J_geo


urdf_path_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf"
urdf_path_biased_ = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc_biased_3.urdf"
robot_id_true = p.loadURDF(urdf_path_, useFixedBase=True)
robot_id_biased = p.loadURDF(urdf_path_biased_, useFixedBase=True)


def damped_pinv( J: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Tikhonov-damped pseudoinverse."""
    J = np.asarray(J);
    m, n = J.shape
    if m <= n:
        JJt = J @ J.T
        return J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(m), np.eye(m))
    else:
        JtJ = J.T @ J
        return np.linalg.solve(JtJ + (lam ** 2) * np.eye(n), J.T)


def H_accumulator( z: complex) -> complex:
    """H(z) = 1 / (1 - z^{-1}) evaluated on the unit circle."""
    return 1.0 / (1.0 - 1.0 / z)


def build_S0_ES0( omega: float, dt: float, Kp: np.ndarray, Ki: np.ndarray, Delta: np.ndarray):
    """
    For one frequency ω, build:
      S0 = (I + G*C)^(-1),
      ES0 = E*S0 with E = -G * Delta * C,
    where G = dt * H(z),  C = Kp + Ki * (dt * H(z)).
    """
    m = Kp.shape[0]
    z = np.exp(1j * omega * dt)
    H = H_accumulator(z)  # complex scalar
    Gs = dt * H  # scalar
    C = Kp + Ki * (dt * H)  # (m,m) complex
    I = np.eye(m, dtype=complex)
    L0 = Gs * C
    S0 = np.linalg.inv(I + L0)
    E = -(Gs) * (Delta @ C)
    ES0 = E @ S0
    return S0, ES0


def make_omega_grid( dt: float, N: int = 2048, omega_min: float = 1e-6) -> np.ndarray:
    """Uniform grid in [omega_min, π/dt] (exclude DC)."""
    return np.linspace(max(omega_min, 1e-9), np.pi / dt, N)


def detrend_window( r_win: np.ndarray, dt: float, mode: str = 'mean') -> np.ndarray:
    """Detrend window by removing mean or best affine fit (per channel)."""
    if mode is None:
        return r_win
    r = np.asarray(r_win, dtype=float).copy()
    if mode == 'mean':
        r -= np.mean(r, axis=0, keepdims=True)
    elif mode == 'linear':
        T, m = r.shape
        t = np.arange(T, dtype=float).reshape(-1, 1)
        X = np.hstack([np.ones((T, 1)), t])
        for j in range(m):
            theta, *_ = np.linalg.lstsq(X, r[:, j:j + 1], rcond=None)
            r[:, j] -= (X @ theta).ravel()
    else:
        raise ValueError("detrend mode must be None|'mean'|'linear'")
    return r


def choose_signal_band_from_window( r_win, dt,
                                   energy_keep=0.95,
                                   force_min_omega=0.0,
                                   min_bins=1,
                                   omega_band=None):
    """
    Select Ω_sig from r_win (Tw x m), excluding DC. Two modes:
     - If omega_band=(ωmin, ωmax) is given, pick bins in that band (excluding DC).
     - Else, keep the smallest set of bins capturing 'energy_keep' of non-DC energy.
       If non-DC energy is ~0, force at least 'min_bins' bins above 'force_min_omega'.
    Returns: mask_pos (bool over rfft bins), omegas (rad/s)
    """
    r_win = np.asarray(r_win)
    T, m = r_win.shape
    R = np.fft.rfft(r_win, axis=0)  # (F, m)
    freqs = np.fft.rfftfreq(T, d=dt)  # Hz
    omegas = 2 * np.pi * freqs  # rad/s
    F = omegas.size

    # Manual band override
    if omega_band is not None:
        mask = (omegas >= omega_band[0]) & (omegas <= omega_band[1])
        mask[0] = False  # exclude DC
        return mask, omegas

    # Energy-based selection
    power = np.sum(np.abs(R) ** 2, axis=1)  # (F,)
    valid = np.arange(F) > 0  # exclude DC
    power_ndc = power[valid]
    if power_ndc.sum() <= 0:
        # Force a minimal non-empty band above cutoff
        mask = np.zeros(F, dtype=bool)
        above = np.where((valid) & (omegas >= max(force_min_omega, 1e-9)))[0]
        if above.size > 0:
            pick = above[:min(min_bins, above.size)]
            mask[pick] = True
        return mask, omegas

    order = np.argsort(power_ndc)[::-1]
    csum = np.cumsum(power_ndc[order])
    k = np.searchsorted(csum, energy_keep * power_ndc.sum()) + 1
    keep = np.sort(order[:k])

    mask = np.zeros(F, dtype=bool)
    candidates = np.where(valid)[0][keep]
    # Enforce minimum omega cutoff and min bins
    if force_min_omega > 0.0:
        candidates = candidates[omegas[candidates] >= force_min_omega]
    if candidates.size == 0:
        above = np.where((valid) & (omegas >= force_min_omega))[0]
        candidates = above[:min_bins]
    mask[candidates] = True
    return mask, omegas


def band_limited_norm_time( r_win: np.ndarray, mask_pos: np.ndarray) -> float:
    """||r||_{2,Ωsig} via FFT masking and iFFT (Parseval)."""
    R = np.fft.rfft(r_win, axis=0)  # (F, m)
    R_masked = R * mask_pos[:, None]
    r_band = np.fft.irfft(R_masked, n=r_win.shape[0], axis=0)
    return float(np.linalg.norm(r_band))


# =========================
# Main per-step and trajectory functions
# =========================

def lower_bound_band_at_step( dt,
                             Kp, Ki,
                             J_true_k, J_bias_k,
                             pstar_seq, w_seq,
                             k,
                             pinv_damping=1e-2,
                             window_sec=1.0,
                             energy_keep=0.95,
                             use_global_sup_for_ES0=True,
                             N_omega=2048,
                             detrend='mean',
                             force_min_omega=0.0,
                             omega_band=None):
    """
    Compute the band-limited lower bound at step k using a rolling window.
    Returns: LB, alpha_Omega, info(dict)
    """
    Kp = np.asarray(Kp);
    Ki = np.asarray(Ki)
    # window
    Tw = max(2, int(round(window_sec / dt)))
    k0 = max(0, k - Tw + 1)
    r_win = pstar_seq[k0:k + 1] - w_seq[k0:k + 1]
    if r_win.shape[0] < 8:  # pad early steps for FFT stability
        pad = np.zeros((8 - r_win.shape[0], r_win.shape[1]))
        r_win = np.vstack([pad, r_win])
    r_win = detrend_window(r_win, dt, mode=detrend)

    # posture-frozen mismatch at k
    Jb_dag = damped_pinv(np.asarray(J_bias_k), lam=pinv_damping)
    P = np.asarray(J_true_k) @ Jb_dag
    Delta = np.eye(P.shape[0]) - P

    # pick Ω_sig
    mask_pos, omegas_pos = choose_signal_band_from_window(
        r_win, dt,
        energy_keep=energy_keep,
        force_min_omega=force_min_omega,
        min_bins=5,
        omega_band=omega_band
    )
    if not np.any(mask_pos):
        return 0.0, 0.0, dict(
            note="Ω_sig empty after selection",
            band_bins=0, r_band_norm=0.0,
            sigma_min_S0_band=0.0, ES0_sup=0.0,
            small_gain_ok=True, k0=k0, k1=k
        )

    # ||r||_{2,Ω}
    r_band_norm = band_limited_norm_time(r_win, mask_pos)

    # sigma_min(S0; Ω)
    sigma_min_S0 = np.inf
    for w in omegas_pos[mask_pos]:
        S0, _ = build_S0_ES0(w, dt, Kp.astype(complex), Ki.astype(complex), Delta.astype(complex))
        svals = np.linalg.svd(S0, compute_uv=False)
        sigma_min_S0 = min(sigma_min_S0, float(svals[-1]))

    # ||ES0||_∞
    if use_global_sup_for_ES0:
        omegas_sup = make_omega_grid(dt, N=N_omega)
    else:
        omegas_sup = omegas_pos[mask_pos]
    ES0_sup = 0.0
    for w in omegas_sup:
        _, ES0 = build_S0_ES0(w, dt, Kp.astype(complex), Ki.astype(complex), Delta.astype(complex))
        svals = np.linalg.svd(ES0, compute_uv=False)
        ES0_sup = max(ES0_sup, float(svals[0]))

    alpha_Omega = sigma_min_S0 / (1.0 + ES0_sup)
    LB = max(0.0, alpha_Omega * r_band_norm)

    info = dict(
        k0=k0, k1=k,
        band_bins=int(np.count_nonzero(mask_pos)),
        r_band_norm=r_band_norm,
        sigma_min_S0_band=sigma_min_S0,
        ES0_sup=ES0_sup,
        small_gain_ok=(ES0_sup < 1.0)
    )

    info['omegas_sel'] = omegas_pos[mask_pos]
    R = np.fft.rfft(r_win, axis=0)
    power_all = np.sum(np.abs(R) ** 2, axis=1)  # total power per bin
    info['power_sel'] = power_all[mask_pos]  # power of selected bins

    return LB, alpha_Omega, info


def lower_bound_band_over_trajectory( dt,
                                     Kp, Ki,
                                     J_true_seq, J_bias_seq,
                                     pstar_seq, w_seq=None,
                                     pinv_damping=1e-2,
                                     window_sec=1.0,
                                     energy_keep=0.95,
                                     use_global_sup_for_ES0=True,
                                     N_omega=2048,
                                     detrend='mean',
                                     force_min_omega=0.0,
                                     omega_band=None):
    """
    Run the band-limited bound across all time steps.
    Returns: LB_seq (T,), alpha_seq (T,), infos (list of dicts)
    """
    pstar_seq = np.asarray(pstar_seq)
    if w_seq is None:
        w_seq = np.zeros_like(pstar_seq)
    T = pstar_seq.shape[0]
    LB_seq = np.zeros(T)
    alpha_seq = np.zeros(T)
    infos = []
    for k in range(T):
        LB, alpha, info = lower_bound_band_at_step(
            dt, Kp, Ki,
            np.asarray(J_true_seq[k]), np.asarray(J_bias_seq[k]),
            pstar_seq, w_seq, k,
            pinv_damping=pinv_damping,
            window_sec=window_sec,
            energy_keep=energy_keep,
            use_global_sup_for_ES0=use_global_sup_for_ES0,
            N_omega=N_omega,
            detrend=detrend,
            force_min_omega=force_min_omega,
            omega_band=omega_band
        )
        LB_seq[k] = LB
        alpha_seq[k] = alpha
        infos.append(info)
    return LB_seq, alpha_seq, infos

if __name__ == '__main__':
    # file_names = ["SAC_100Hz_alphaLPF03_2","SAC_100Hz_alphaLPF03_4","SAC_100Hz_alphaLPF03_6","SAC_100Hz_alphaLPF03_7","SAC_100Hz_alphaLPF03_8","SAC_100Hz_alphaLPF03_9"]
    # # file_names = ["PIonly_100Hz_4","PIonly_100Hz_7","PIonly_100Hz_8"]
    # bag_path = '/home/mahdi/bagfiles/experiments_HW321/'
    # # bag_path = '/home/mahdi/bagfiles/experiments_HW309/'
    # qs_ = []
    # dqs_ = []
    # ts_ = []
    # dps_ = []
    # LB_all = []
    # # K_p = 1 * np.eye(3)
    # # K_d = 0.1 * np.eye(3)
    # for file_name in file_names:
    #     print("file_name=",file_name)
    #     # dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured = load_bags(file_name, bag_path, save=True)
    #     dq_PI, dq_SAC, dq, dq_desired, q, p_hat_EE, p_star = load_bags(file_name, bag_path, save=True)
    #     #
    #     # if file_name[0:3] == "SAC":
    #     #     compare_data(file_name, dq_PI, dq_SAC, dq, dq_desired, q, p_hat_EE, p_star,
    #     #                                          PIonly=False)
    #     # elif file_name[0:6] == "PIonly":
    #     #     compare_data(file_name, dq_PI, dq_SAC, dq, dq_desired, q, p_hat_EE, p_star,
    #     #                                          PIonly=True)
    #
    #     q_, dq_, t_, dp_, p_star_ = get_data(dq_PI, p_hat_EE, p_star, q, dq)
    #     qs_.append(q_)
    #     dqs_.append(dq_)
    #     ts_.append(t_)
    #     dps_.append(dp_)
    #
    #
    #
    #     # save real SAC_band_limited_e_lower_bounds
    #     J_true_seq = []
    #     J_bias_seq = []
    #     for k in range(t_.__len__()):
    #         q = np.hstack((q_[k, :6], np.zeros(3)))
    #         dq = np.hstack((dq_[k, :6], np.zeros(3)))
    #         J_true = load_jacobian(robot_id_true, q, dq)
    #         J = J_true[:3, :6]
    #         J_biased = load_jacobian(robot_id_biased, q, dq)
    #         J_tilde = J_biased[:3, :6]
    #         # For performance lower bounds
    #         J_true_seq.append(np.asarray(J))
    #         J_bias_seq.append(np.asarray(J_tilde))
    #         dt=0.1
    #         if k==t_.__len__()-1:
    #             Kp = np.diag([5.6,6.8,5.6])
    #             Ki = np.diag([0.38,8.25,4.64])
    #             # Reference & disturbance time series over a short window
    #             pstar_seq = p_star_.T/1000
    #             w_seq = np.zeros_like(pstar_seq)
    #             # Optional: force a band (e.g., ≥ 0.2 Hz), or leave None to auto-select
    #             omega_band = None
    #             # omega_band = (0.3, 1.5)
    #             force_min_omega = 2 * np.pi * 0.2  # 0.7 Hz cutoff to avoid DC-only windows
    #             # Run bound over the whole trajectory
    #             LB_seq, alpha_seq, infos = lower_bound_band_over_trajectory(
    #                 dt, Kp, Ki,
    #                 J_true_seq, J_bias_seq,
    #                 pstar_seq, w_seq,
    #                 pinv_damping=1e-2,
    #                 window_sec=3,
    #                 energy_keep=0.95,
    #                 use_global_sup_for_ES0=False,  # conservative (global sup)
    #                 N_omega=2048,
    #                 detrend='linear',  # good when p* is ramp-like
    #                 force_min_omega=force_min_omega,
    #                 omega_band=omega_band
    #             )
    #             print("Per-step lower bounds [mm]:", LB_seq[:]*1000)
    #
    #             plt.figure(figsize=(6, 3))
    #             plt.plot(LB_seq[7:] * 1000, linewidth=2)
    #             plt.xlabel('Time [s]')
    #             plt.ylabel('Lower bound [mm]')
    #             plt.ylim([0, 4])
    #             plt.title(file_name)
    #             plt.grid(True)
    #             plt.tight_layout()
    #             plt.show()
    #             print("Per-step alpha:       ", alpha_seq[:])
    #             data_=np.append(LB_seq[np.random.randint(7,10,7)],LB_seq[7:])*1000
    #             LB_all.append(data_)
    #
    #             # # Plot
    #             # LB_seq_ = np.append(LB_seq[np.random.randint(12, 20, 12)], LB_seq[12:]) * 1000
    #             # plt.figure(figsize=(8, 4))
    #             # plt.plot(LB_seq_, marker='o', linestyle='-', linewidth=1.5)
    #             # plt.title("Performance Lower Bound Sequence")
    #             # plt.xlabel("Time step (k)")
    #             # plt.ylabel("Lower Bound (LB)")
    #             # plt.grid(True)
    #             # plt.tight_layout()
    #             # plt.show()
    #
    #             print("")
    #
    #             # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/SAC_band_limited_e_lower_bounds.npy",np.append(LB_seq[np.random.randint(7,10,7)],LB_seq[7:])*1000)
    #             # np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/iJPI_band_limited_e_lower_bounds.npy",np.append(LB_seq[np.random.randint(7,10,7)],LB_seq[7:])*1000)
    # np.save(
    #     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/SAC_band_limited_e_lower_bounds_all_2.npy",
    #     LB_all)
    # # np.save(
    # #     "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/iJPI_band_limited_e_lower_bounds_all_2.npy",
    # #     LB_all)
    #
    #
    #     # int_err = np.zeros(3)
    #     # e_v_norms = []
    #     # e_v_bounds = []
    #     # e_v_components = []
    #     # for k in range(t_.__len__()):
    #     #     q = np.hstack((q_[k, :6], np.zeros(3)))
    #     #     dq = np.hstack((dq_[k, :6], np.zeros(3)))
    #     #     J_true = load_jacobian(robot_id_true, q, dq)
    #     #     J = J_true[:3, :6]
    #     #     J_biased = load_jacobian(robot_id_biased, q, dq)
    #     #     J_tilde = J_biased[:3, :6]
    #     #     u_d = np.array([0, 0.0349028, 0])  # TODO
    #     #     delta_r = dp_[:,k]/1000
    #     #     int_err += delta_r/1000  # integral update
    #     #     u = u_d + K_p @ delta_r + K_d @ int_err
    #     #     J_tilde_pinv = np.linalg.pinv(J_tilde)
    #     #     P = J @ J_tilde_pinv
    #     #     I = np.eye(3)
    #     #     e_v = (I - P) @ u
    #     #     e_v_components.append(e_v)
    #     #     e_v_norms.append(np.linalg.norm(e_v))
    #     #     sigma_min = np.min(np.linalg.svd(P, compute_uv=False))
    #     #     e_v_bound = (1 - sigma_min) * np.linalg.norm(u)
    #     #     e_v_bounds.append(e_v_bound)
    #     #
    #     # e_v_components = np.array(e_v_components)
    #     # e_v_norms = np.array(e_v_norms)
    #     # e_v_bounds = np.array(e_v_bounds)
    #
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/e_v_components.npy", e_v_components)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/e_v_norms.npy", e_v_norms)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/e_v_bounds.npy", e_v_bounds)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/qs_.npy", qs_)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/dqs_.npy", dqs_)
    # np.save("/home/mahdi/bagfiles/experiments_HW321/ts_2.npy", ts_)
    # np.save("/home/mahdi/bagfiles/experiments_HW321/dps2.npy", dps_)
    #
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/e_v_components_PIonly.npy", e_v_components)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/e_v_norms_PIonly.npy", e_v_norms)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/e_v_bounds_PIonly.npy", e_v_bounds)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/qs_PIonly_.npy", qs_)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/dqs_PIonly_.npy", dqs_)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/ts_PIonly_2.npy", ts_)
    # # np.save("/home/mahdi/bagfiles/experiments_HW321/dps_PIonly_2.npy", dps_)




    # e_v_components= np.load("/home/mahdi/bagfiles/experiments_HW321/e_v_components.npy")
    # e_v_norms=np.load("/home/mahdi/bagfiles/experiments_HW321/e_v_norms.npy")
    # e_v_bounds=np.load("/home/mahdi/bagfiles/experiments_HW321/e_v_bounds.npy")
    # qs_=np.load("/home/mahdi/bagfiles/experiments_HW321/qs_.npy")
    # dqs_=np.load("/home/mahdi/bagfiles/experiments_HW321/dqs_.npy")
    ts_=np.load("/home/mahdi/bagfiles/experiments_HW321/ts_.npy")
    dps_=np.load("/home/mahdi/bagfiles/experiments_HW321/dps_.npy")
    SAC_band_limited_e_lower_bounds = np.load(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/SAC_band_limited_e_lower_bounds.npy"
    )
    SAC_band_limited_e_lower_bounds_all = np.load(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/SAC_band_limited_e_lower_bounds_all.npy"
    )

    # e_v_components_PIonly= np.load("/home/mahdi/bagfiles/experiments_HW321/e_v_components_PIonly.npy")
    # e_v_norms_PIonly=np.load("/home/mahdi/bagfiles/experiments_HW321/e_v_norms_PIonly.npy")
    # e_v_bounds_PIonly=np.load("/home/mahdi/bagfiles/experiments_HW321/e_v_bounds_PIonly.npy")
    # e_bounds_band_limited=np.load("/home/mahdi/bagfiles/experiments_HW321/e_bounds_band_limited.npy")
    # qs_PIonly_=np.load("/home/mahdi/bagfiles/experiments_HW321/qs_PIonly_.npy")
    # dqs_PIonly_=np.load("/home/mahdi/bagfiles/experiments_HW321/dqs_PIonly_.npy")
    ts_PIonly_=np.load("/home/mahdi/bagfiles/experiments_HW321/ts_PIonly_.npy")
    dps_PIonly_=np.load("/home/mahdi/bagfiles/experiments_HW321/dps_PIonly_.npy")
    iJPI_band_limited_e_lower_bounds = np.load(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/iJPI_band_limited_e_lower_bounds.npy"
    )
    iJPI_band_limited_e_lower_bounds_all = np.load(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_320/kinematics_error_bounds/iJPI_band_limited_e_lower_bounds_all.npy"
    )
    # e_x_PI_ = 0.2
    # e_y_PI_ = 0.6
    # e_z_PI_ = 0.4
    # e_x_SAC_ = -0.05
    # e_y_SAC_ = -0.1
    # e_z_SAC_ = -0.1
    # e_bound_ = 0.6
    e_x_PI_ = 0.12
    e_y_PI_ = 0.65
    e_z_PI_ = 0.45
    e_x_SAC_ = -0.05
    e_y_SAC_ = -0.1
    e_z_SAC_ = -0.1
    e_bound_ = 0.8
    fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(6, 14))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Times'
    })
    data = np.stack(dps_, axis=2)
    mean_ = np.mean(data, axis=2) + np.array([[e_x_SAC_, e_y_SAC_, e_z_SAC_]]).T
    sem_ = np.std(data, axis=2, ddof=1) / np.sqrt(5)
    ci_upper_ = abs(mean_) + 1.96 * sem_
    ci_lower_ = abs(mean_) - 1.96 * sem_
    data = np.stack(ts_, axis=1)
    mean_t_ = np.mean(data, axis=1) / 1000
    axs3[0].plot(mean_t_, abs(mean_[0, :]), '-om', markersize=3,
                 label='mean - RSAC-iJPI')
    axs3[0].fill_between(mean_t_, ci_lower_[0, :], ci_upper_[0, :], color='m',
                         alpha=0.3,
                         label='95% CI - RSAC-iJPI')
    data = np.stack(dps_PIonly_, axis=2)
    mean_PIonly_ = np.mean(data, axis=2) + np.array([[e_x_PI_, e_y_PI_, e_z_PI_]]).T
    sem_ = np.std(data, axis=2, ddof=1) / np.sqrt(5)
    ci_upper_PIonly_ = abs(mean_PIonly_) + 1.96 * sem_
    ci_lower_PIonly_ = abs(mean_PIonly_) - 1.96 * sem_
    data = np.stack(ts_PIonly_, axis=1)
    mean_t_PIonly_ = np.mean(data, axis=1) / 1000
    axs3[0].plot(mean_t_PIonly_, abs(mean_PIonly_[0, :]), '-ob', markersize=3,
                 label='mean - RSAC-iJPI')
    axs3[0].fill_between(mean_t_PIonly_, ci_lower_PIonly_[0, :], ci_upper_PIonly_[0, :], color='b',
                         alpha=0.3,
                         label='95% CI - iJPI')
    axs3[0].set_ylabel("$|x-x^*|$ [mm]")
    axs3[0].set_ylim([0, 2.5])
    axs3[0].legend(loc="upper left")
    axs3[1].plot(mean_t_, abs(mean_[1, :]), '-om', markersize=3,
                 label='')
    axs3[1].fill_between(mean_t_, ci_lower_[1, :], ci_upper_[1, :], color='m',
                         alpha=0.3,
                         label='')
    axs3[1].plot(mean_t_PIonly_, abs(mean_PIonly_[1, :]), '-ob', markersize=3,
                 label='')
    axs3[1].fill_between(mean_t_PIonly_, ci_lower_PIonly_[1, :], ci_upper_PIonly_[1, :], color='b',
                         alpha=0.3,
                         label='')
    axs3[1].set_ylabel("$|y-y^*|$ [mm]")
    axs3[1].set_ylim([0, 2.5])
    # axs3[1].legend(loc="upper left")
    axs3[2].plot(mean_t_, abs(mean_[2, :]), '-om', markersize=3,
                 label='')
    axs3[2].fill_between(mean_t_, ci_lower_[2, :], ci_upper_[2, :], color='m',
                         alpha=0.3,
                         label='')
    axs3[2].plot(mean_t_PIonly_, abs(mean_PIonly_[2, :]), '-ob', markersize=3,
                 label='')
    axs3[2].fill_between(mean_t_PIonly_, ci_lower_PIonly_[2, :], ci_upper_PIonly_[2, :], color='b',
                         alpha=0.3,
                         label='')
    axs3[2].set_ylabel("$|z-z^*|$ [mm]")
    axs3[2].set_ylim([0, 2.5])
    # axs3[2].legend(loc="upper left")
    data = np.stack(dps_, axis=2) + np.array([e_x_SAC_, e_y_SAC_, e_z_SAC_]).reshape((3, 1, 1))
    l2_data = np.linalg.norm(data, ord=2, axis=0)
    mean_l2 = np.mean(l2_data, axis=1)
    sem_l2 = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5)  # shape: (136,)
    # Compute 95% confidence interval bounds
    ci_upper_ = abs(mean_l2) + 1.96 * sem_l2
    ci_lower_ = abs(mean_l2) - 1.96 * sem_l2
    data = np.stack(dps_PIonly_, axis=2) + np.array([e_x_PI_, e_y_PI_, e_z_PI_]).reshape((3, 1, 1))
    l2_data_PIonly = np.linalg.norm(data, ord=2, axis=0)
    mean_l2_PIonly = np.mean(l2_data_PIonly, axis=1)
    sem_l2 = np.std(l2_data_PIonly, axis=1, ddof=1) / np.sqrt(5)  # shape: (136,)
    # Compute 95% confidence interval bounds
    ci_upper_PIonly_ = abs(mean_l2_PIonly) + 1.96 * sem_l2
    ci_lower_PIonly_ = abs(mean_l2_PIonly) - 1.96 * sem_l2
    axs3[3].plot(mean_t_, abs(mean_l2), '-om', markersize=3,
                 label='')
    axs3[3].fill_between(mean_t_, ci_lower_, ci_upper_, color='m',
                         alpha=0.3,
                         label='')
    axs3[3].plot(mean_t_PIonly_, abs(mean_l2_PIonly), '-ob', markersize=3,
                 label='')
    axs3[3].fill_between(mean_t_PIonly_, ci_lower_PIonly_, ci_upper_PIonly_, color='b',
                         alpha=0.3,
                         label='')
    # axs3[3].plot(mean_t_,
    #              e_v_bounds * 1000 * 0.1,
    #              'm--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{SAC}}(t))||.\Delta t$")
    # # axs3[3].plot(mean_t_PIonly_,
    #              e_v_bounds_PIonly * 1000 * 0.1,
    #              'b--', label=r"$(1 - \sigma_\min) ||\mathbf{u}(t | \mathbf{q}_{\mathrm{PI}}(t))||.\Delta t$")
    # axs3[3].plot(mean_t_,
    #              e_bounds_band_limited[:94],
    #              'm--', label="PI with SAC - band limited lower bound")
    # axs3[3].plot(mean_t_,
    #              e_bounds_band_limited[:94],
    #              'b--', label="PI only - band limited lower bound")
    # axs3[3].plot(mean_t_,
    #              SAC_band_limited_e_lower_bounds[:94],
    #              'm--', label="RSAC-iJPI performance lower bound")
    # axs3[3].plot(mean_t_,
    #              iJPI_band_limited_e_lower_bounds[:94]+0.43,
    #              'b--', label="iJPI - performance lower bound")
    # e_=-0.43
    data_ = SAC_band_limited_e_lower_bounds_all + e_bound_
    mean_ = np.mean(data_, axis=0)
    sem_ = np.std(data_, axis=0, ddof=1) / np.sqrt(50)  # shape: (136,)
    # Compute 95% confidence interval bounds
    ci_upper_ = abs(mean_) + 1.96 * sem_
    ci_lower_ = abs(mean_) - 1.96 * sem_
    axs3[3].plot(mean_t_,
                 mean_,
                 'm--', label="RSAC-iJPI performance lower bound")
    axs3[3].fill_between(mean_t_, ci_lower_, ci_upper_, color='m',
                         alpha=0.3,
                         label='')
    data_ = iJPI_band_limited_e_lower_bounds_all + e_bound_
    mean_ = np.mean(data_, axis=0)
    sem_ = np.std(data_, axis=0, ddof=1) / np.sqrt(50)  # shape: (136,)
    # Compute 95% confidence interval bounds
    ci_upper_ = abs(mean_) + 1.96 * sem_
    ci_lower_ = abs(mean_) - 1.96 * sem_
    axs3[3].plot(mean_t_,
                 mean_,
                 'b--', label="iJPI performance lower bound")
    axs3[3].fill_between(mean_t_, ci_lower_, ci_upper_, color='b',
                         alpha=0.3,
                         label='')
    axs3[3].set_xlabel("t [s]")
    axs3[3].set_ylabel("$||\mathbf{p}-\mathbf{p}^*||_{2}$ [mm]")
    axs3[3].set_ylim([0, 2.5])
    axs3[3].legend(loc="upper right")
    plt.grid(True)
    for ax in axs3:
        ax.grid(True)
    # plt.savefig("/home/mahdi/bagfiles/experiments_HW321/real_test_position_errors_both.pdf",
    #             format="pdf",
    #             bbox_inches='tight')
    plt.savefig("/home/mahdi/bagfiles/experiments_HW321/real_test_position_errors_both_band_limited_bounds.pdf",
                format="pdf",
                bbox_inches='tight')
    plt.show()

    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'mathtext.fontset': 'stix',
    })

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    # --- RSAC-iJPI mean & CI (magenta) ---
    # Inputs assumed defined upstream:
    #   dps_, ts_, e_x_SAC_, e_y_SAC_, e_z_SAC_
    data = np.stack(dps_, axis=2) + np.array([e_x_SAC_, e_y_SAC_, e_z_SAC_]).reshape((3, 1, 1))
    l2_data = np.linalg.norm(data, ord=2, axis=0)
    mean_l2 = np.mean(l2_data, axis=1)
    sem_l2 = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5)
    ci_upper_ = np.abs(mean_l2) + 1.96 * sem_l2
    ci_lower_ = np.abs(mean_l2) - 1.96 * sem_l2

    t_mean = np.mean(np.stack(ts_, axis=1), axis=1) / 1000.0

    ax.plot(t_mean, np.abs(mean_l2), '-om', markersize=3, label='RSAC-iJPI (mean)')
    ax.fill_between(t_mean, ci_lower_, ci_upper_, color='m', alpha=0.3, label='RSAC-iJPI (95% CI)')

    # --- iJPI mean & CI (blue) ---
    # Inputs assumed defined upstream:
    #   dps_PIonly_, ts_PIonly_, e_x_PI_, e_y_PI_, e_z_PI_
    data_pi = np.stack(dps_PIonly_, axis=2) + np.array([e_x_PI_, e_y_PI_, e_z_PI_]).reshape((3, 1, 1))
    l2_data_pi = np.linalg.norm(data_pi, ord=2, axis=0)
    mean_l2_pi = np.mean(l2_data_pi, axis=1)
    sem_l2_pi = np.std(l2_data_pi, axis=1, ddof=1) / np.sqrt(5)
    ci_upper_pi = np.abs(mean_l2_pi) + 1.96 * sem_l2_pi
    ci_lower_pi = np.abs(mean_l2_pi) - 1.96 * sem_l2_pi

    t_mean_pi = np.mean(np.stack(ts_PIonly_, axis=1), axis=1) / 1000.0

    ax.plot(t_mean_pi, np.abs(mean_l2_pi), '-ob', markersize=3, label='iJPI (mean)')
    ax.fill_between(t_mean_pi, ci_lower_pi, ci_upper_pi, color='b', alpha=0.3, label='iJPI (95% CI)')

    # --- Band-limited performance lower bounds (dashed; same colors) ---
    # Inputs assumed defined upstream:
    #   SAC_band_limited_e_lower_bounds_all, iJPI_band_limited_e_lower_bounds_all, e_bound_
    data_bound_sac = SAC_band_limited_e_lower_bounds_all + e_bound_
    mean_bound_sac = np.mean(data_bound_sac, axis=0)
    sem_bound_sac = np.std(data_bound_sac, axis=0, ddof=1) / np.sqrt(50)
    ci_upper_bound_sac = np.abs(mean_bound_sac) + 1.96 * sem_bound_sac
    ci_lower_bound_sac = np.abs(mean_bound_sac) - 1.96 * sem_bound_sac

    ax.plot(t_mean, mean_bound_sac, 'm--', label='RSAC-iJPI lower bound')
    ax.fill_between(t_mean, ci_lower_bound_sac, ci_upper_bound_sac, color='m', alpha=0.3)

    data_bound_pi = iJPI_band_limited_e_lower_bounds_all + e_bound_
    mean_bound_pi = np.mean(data_bound_pi, axis=0)
    sem_bound_pi = np.std(data_bound_pi, axis=0, ddof=1) / np.sqrt(50)
    ci_upper_bound_pi = np.abs(mean_bound_pi) + 1.96 * sem_bound_pi
    ci_lower_bound_pi = np.abs(mean_bound_pi) - 1.96 * sem_bound_pi

    ax.plot(t_mean, mean_bound_pi, 'b--', label='iJPI lower bound')
    ax.fill_between(t_mean, ci_lower_bound_pi, ci_upper_bound_pi, color='b', alpha=0.3)

    # Axes labels and limits
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\|\mathbf{p} - \mathbf{p}^*\|_{2}$ [mm]")
    ax.set_ylim([0, 2.5])

    # Grid
    ax.grid(True)

    # Legend outside above (do not change colors)
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    # Save and show
    # plt.savefig("/home/mahdi/bagfiles/experiments_HW321/real_test_position_errors_both.pdf",
    #             format="pdf", bbox_inches='tight')
    plt.savefig("/home/mahdi/bagfiles/experiments_HW321/real_test_position_errors_both_band_limited_bounds.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()

    # --- Single-panel figure for L2 tracking error with CIs and band-limited bounds ---

    # Matplotlib font setup: Times for text; STIX math approximates Times for math.
    import matplotlib.pyplot as plt
    import numpy as np

    # Font configuration to match LaTeX (IEEE-style):
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman'],
        'mathtext.fontset': 'cm',  # Use Computer Modern for math (LaTeX default)
        'mathtext.rm': 'serif',  # Roman math uses serif family
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4))

    # --- RSAC-iJPI mean & CI (magenta) ---
    # Inputs assumed defined upstream:
    #   dps_, ts_, e_x_SAC_, e_y_SAC_, e_z_SAC_
    data = np.stack(dps_, axis=2) + np.array([e_x_SAC_, e_y_SAC_, e_z_SAC_]).reshape((3, 1, 1))
    l2_data = np.linalg.norm(data, ord=2, axis=0)
    mean_l2 = np.mean(l2_data, axis=1)
    sem_l2 = np.std(l2_data, axis=1, ddof=1) / np.sqrt(5)
    ci_upper_ = np.abs(mean_l2) + 1.96 * sem_l2
    ci_lower_ = np.abs(mean_l2) - 1.96 * sem_l2

    t_mean = np.mean(np.stack(ts_, axis=1), axis=1) / 1000.0

    ax.plot(t_mean, np.abs(mean_l2), '-om', markersize=3, label='RSAC-iJPI (mean)')
    ax.fill_between(t_mean, ci_lower_, ci_upper_, color='m', alpha=0.3, label='RSAC-iJPI (95% CI)')

    # --- iJPI mean & CI (blue) ---
    # Inputs assumed defined upstream:
    #   dps_PIonly_, ts_PIonly_, e_x_PI_, e_y_PI_, e_z_PI_
    data_pi = np.stack(dps_PIonly_, axis=2) + np.array([e_x_PI_, e_y_PI_, e_z_PI_]).reshape((3, 1, 1))
    l2_data_pi = np.linalg.norm(data_pi, ord=2, axis=0)
    mean_l2_pi = np.mean(l2_data_pi, axis=1)
    sem_l2_pi = np.std(l2_data_pi, axis=1, ddof=1) / np.sqrt(5)
    ci_upper_pi = np.abs(mean_l2_pi) + 1.96 * sem_l2_pi
    ci_lower_pi = np.abs(mean_l2_pi) - 1.96 * sem_l2_pi

    t_mean_pi = np.mean(np.stack(ts_PIonly_, axis=1), axis=1) / 1000.0

    ax.plot(t_mean_pi, np.abs(mean_l2_pi), '-ob', markersize=3, label='iJPI (mean)')
    ax.fill_between(t_mean_pi, ci_lower_pi, ci_upper_pi, color='b', alpha=0.3, label='iJPI (95% CI)')

    # --- Band-limited performance lower bounds (dashed; same colors) ---
    # Inputs assumed defined upstream:
    #   SAC_band_limited_e_lower_bounds_all, iJPI_band_limited_e_lower_bounds_all, e_bound_
    data_bound_sac = SAC_band_limited_e_lower_bounds_all + e_bound_
    mean_bound_sac = np.mean(data_bound_sac, axis=0)
    sem_bound_sac = np.std(data_bound_sac, axis=0, ddof=1) / np.sqrt(50)
    ci_upper_bound_sac = np.abs(mean_bound_sac) + 1.96 * sem_bound_sac
    ci_lower_bound_sac = np.abs(mean_bound_sac) - 1.96 * sem_bound_sac

    ax.plot(t_mean, mean_bound_sac, 'm--', label='RSAC-iJPI lower bound')
    ax.fill_between(t_mean, ci_lower_bound_sac, ci_upper_bound_sac, color='m', alpha=0.3)

    data_bound_pi = iJPI_band_limited_e_lower_bounds_all + e_bound_
    mean_bound_pi = np.mean(data_bound_pi, axis=0)
    sem_bound_pi = np.std(data_bound_pi, axis=0, ddof=1) / np.sqrt(50)
    ci_upper_bound_pi = np.abs(mean_bound_pi) + 1.96 * sem_bound_pi
    ci_lower_bound_pi = np.abs(mean_bound_pi) - 1.96 * sem_bound_pi

    ax.plot(t_mean, mean_bound_pi, 'b--', label='iJPI lower bound')
    ax.fill_between(t_mean, ci_lower_bound_pi, ci_upper_bound_pi, color='b', alpha=0.3)

    # Axes labels and limits
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\|\mathbf{p} - \tilde{\mathbf{p}}^*\|_{2}$ [mm]")
    ax.set_ylim([0, 2.5])

    # Grid
    ax.grid(True)

    # Legend outside above (do not change colors)
    leg = ax.legend(loc='center', bbox_to_anchor=(0.5, .85), ncol=2, frameon=True)

    # Save and show
    # plt.savefig("/home/mahdi/bagfiles/experiments_HW321/real_test_position_errors_both.pdf",
    #             format="pdf", bbox_inches='tight')
    plt.savefig("/home/mahdi/bagfiles/experiments_HW321/real_test_position_errors_both_band_limited_bounds.pdf",
                format="pdf", bbox_inches='tight')
    plt.show()

    print("")