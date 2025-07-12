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
arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
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
    if PIonly==False:
        sim_plot_data_buffer=np.load(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/SAC_plot_data_buffer.npy")
        sim_state_buffer= np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/SAC_state_buffer.npy")
    elif PIonly==True:
        sim_plot_data_buffer=np.load(
            "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/PIonly_plot_data_buffer.npy")
        sim_state_buffer= np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/PIonly_state_buffer.npy"        )

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
    closest_idx_PI = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:104]
    closest_t_PI = t_[closest_idx_PI]

    t_ = (dq_SAC[:, 0] - dq_SAC[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_SAC = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:104]
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

    t_ = (q_measured[:, 0] - q_measured[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_q = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:104]
    closest_t_q = t_[closest_idx_q]

    t_ = (dq_measured[:, 0] - dq_measured[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    # Find indices in t closest to each target time
    closest_idx_dq = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:104]
    closest_t_dq = t_[closest_idx_dq]
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
    # plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_dq_measured_dq_simest.png".format(file_name), format="png",
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
        closest_idx_PI_d = np.array([np.abs(t_ - target).argmin() for target in target_times_d])[:104]
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_dq_desired_commanded.png".format(
            file_name),
        format="png",
        bbox_inches='tight')
    plt.show()


    dq = dq_measured[closest_idx_dq, 1:7]
    q = q_measured[closest_idx_q, 1:7]
    dq_sim = np.array(dq_sim)[:, :6]
    q_sim = np.array(q_sim)[:, :6]
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_q_qSimRaw.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    dq = dq_measured[closest_idx_dq, 1:7]
    q = q_measured[closest_idx_q, 1:7]
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_qReal_qSimHW312StateBuffer.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()


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
    q_sim_corrected=q_sim
    for k in range(104):
        # ----- q and dq Mismatch Compensation -----
        for i in range(6):
            models_q[i].eval()
            # models_dq[i].eval()
            likelihoods_q[i].eval()
            # likelihoods_dq[i].eval()
            # TODO ????????????
            X_test = np.array([q_sim[k,i], dq_sim[k,i]]).reshape(-1, 2)
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
                q_sim_corrected[k,i] = q_sim_corrected[k,i] + mean_q
            else:
                print("mean_q[{}] is nan!!".format(i))
            # if ~np.isnan(mean_dq):
            #     dq_tp1[i] = dq_tp1[i] + mean_dq
            # else:
            #     print("mean_dq[{}] is nan!".format(i))
    #########################################################################
    dq = dq_measured[closest_idx_dq, 1:7]
    q = q_measured[closest_idx_q, 1:7]
    dq_sim = np.array(dq_sim)[:, :6]
    q_sim_corrected = np.array(q_sim_corrected)[:, :6]
    # Plot q and q_sim (Position)
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 16))
    plt.rcParams['font.family'] = 'Serif'
    # fig1.suptitle('Joint Positions: Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs1[i // 2, i % 2]
        ax.plot(closest_t_q, q[:, i], '-og', label='Measured q')
        ax.plot(closest_t_PI[10:], q_sim_corrected[10:, i], '-or', label='Simulated q_sim + mismatch correction', markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('q[{}] [rad]'.format(i))
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_qReal_qSimMismatchComp.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_dq.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_dq_abs_error.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # Plot dq and error q (Position)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 16))
    fig2.suptitle('Joint Positions: Absolute Error Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_q, abs(q_sim[:, i] - q[:, i]), '-sr')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('$|q_{\t{sim}} - q_{\t{measured}}|$')
        ax.grid(True)
        ax.set_ylim([0, 0.1])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_q_abs_error.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_dq_error.png".format(
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
        ax.set_ylabel('$q_{\t{sim}} - q_{\t{measured}}$ [deg]')
        ax.grid(True)
        ax.set_ylim([-4.1, 4.1])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_q_error.png".format(
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
        ax.plot((closest_t_PI[10:] - closest_t_PI[10])/1000, dq_PI[closest_idx_PI[10:], i + 1] * 180 / np.pi, '-og',
                label='real $dq_{{PI}}$[{}]'.format(str(i)))
        ax.plot((np.arange(0, 13600, 100))/1000, sim_state_buffer[:, 15 + i] * 180 / np.pi, '-ob',
                label='simulation $dq_{{PI}}$[{}]'.format(str(i)), markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq_{{PI}}$[{}] [deg/s]'.format(str(i)))
        ax.grid(True)
        ax.set_ylim([-2, 4])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_dq_PI.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_dq_SAC.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()


    t_ = (p_star[:, 0] - p_star[0, 0]) * 1000
    target_times = np.arange(0, t_[-1], 100)
    closest_idx_p_star = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:104]
    closest_t_p_star = t_[closest_idx_p_star]

    t_ = (p_hat_EE[:, 0] - p_hat_EE[0, 0]) * 1000
    target_times = np.arange(0, t_[-1], 100)
    closest_idx_p_hat_EE = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])[:104]
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
                (p_star[closest_idx_p_star[10:], i + 1] - p_hat_EE[closest_idx_p_hat_EE[10:], i + 1]) * 1000, '-og',
                label='real $p^*[{}]-\hat{{p}}^*[{}]$'.format(str(i), str(i)))
        ax.plot(np.arange(0, 13600, 100), -sim_state_buffer[:, 0 + i], '-ob',
                label='simulation $p^*[{}]-\hat{{p}}[{}]$'.format(str(i), str(i)), markersize=2)
        # ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$p^*[{}]-\hat{{p}}[{}]$ [mm]'.format(str(i), str(i)))
        ax.grid(True)
        ax.set_ylim([-5, 4])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_delta_p.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_p_star.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_p_hat_EE.png".format(
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
        ax.plot(t_real[:len(dq_real)], dq_sim_test[:len(dq_real)], '-ob', label='$dq_{{sim_{{test}}}}[{}]$'.format(i),markersize=3)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$dq$ [deg/s]')
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_dq_test.png".format(
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
        ax.plot(t_real[:len(q_real)], q_sim_test[:len(dq_real)], '-ob', label='$q_{{sim_{{test}}}}[{}]$'.format(i),markersize=3)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$q$ [deg]')
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_q_test.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_dq_PI_err.png".format(
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
        ax.plot(t_real[:len(dq_real)], diff, '-ok', label='$\Delta dq_{{SAC}}[{}]$'.format(i))
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta dq_{{SAC}}[{}]$ [deg/s]'.format(i))
        # ax.set_ylim([-25, 25])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_dq_SAC_err.png".format(
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
                label='$\Delta (p^*[{}]-\hat{{p}}[{}])$'.format(i, i))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta (p^*[{}]-\hat{{p}}[{}])$ [mm]'.format(i, i))
        # ax.set_ylim([-6, 3])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_delta_p_err.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    for i in range(3):
        ax = axs2[i]
        t_real = (closest_t_p_star[10:] - closest_t_p_star[10]) / 1000
        p_real = p_star[closest_idx_p_star[10:], i + 1] * 1000
        p_sim = sim_plot_data_buffer[:, 3 + i] * 1000
        delta = p_real- p_sim[:len(p_real)]
        ax.plot(t_real[:len(p_real)], delta, '-ok', label='$\Delta p^*[{}]$'.format(i))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$p^*_{{real}}[{}]-p^*_{{sim_{{test}}}}[{}]$ [mm]'.format(i,i))
        # ax.set_ylim([-6, 3])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_p_star_err.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 18))
    for i in range(3):
        ax = axs2[i]
        t_real = (closest_t_p_hat_EE[10:] - closest_t_p_hat_EE[10]) / 1000
        p_real = p_hat_EE[closest_idx_p_hat_EE[10:], i + 1] * 1000
        p_sim = sim_plot_data_buffer[:, 0 + i] * 1000
        delta = p_real- p_sim[:len(p_real)]
        ax.plot(t_real[:len(p_sim)], delta, '-ok', label='$\Delta \hat{{p}}[{}]$'.format(i))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$\Delta \hat{{p}}[{}]$ [mm]'.format(i))
        # ax.set_ylim([-6, 3])
        ax.grid(True)
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_p_hat_EE_err.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_q_test_err.png".format(
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/compare_real_simulation_data/Fep_HW_312/{}_compare_dq_test_err.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    return True


if __name__ == '__main__':
    # file_names = ["SAC_1","SAC_2","SAC_3","SAC_4","SAC_5", "PIonly_1", "PIonly_2", "PIonly_3", "PIonly_4", "PIonly_5"]
    # file_names = ["PIonly_4"]
    file_names = ["SAC_1"]
    for file_name in file_names:
        bag_path = '/home/mahdi/bagfiles/experiments_HW312/'
        # dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured = load_bags(file_name, bag_path, save=True)
        dq_PI, dq_SAC, dq, dq_desired, q, p_hat_EE, p_star  = load_bags(file_name, bag_path, save=True)

        if file_name[0:3] == "SAC":
            compare_data(file_name, dq_PI, dq_SAC, dq, dq_desired, q, p_hat_EE, p_star,
                                                 PIonly=False)
        elif file_name[0:6] == "PIonly":
            compare_data(file_name, dq_PI, dq_SAC, dq, dq_desired, q, p_hat_EE, p_star,
                                                 PIonly=True)


