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
render=False
if render==False:
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
        np.array([534e-3,-246.5e-3,154.2e-3]) + np.array([-0.002, -0.18, -0.15]),
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




def load_bags(file_name,bag_path, save=False):
    # file_name = "SAC_1"
    topic_name='/PRIMITIVE_velocity_controller/dq_PID_messages'
    if save:
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path+file_name+".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                pos = msg.pose.position
                ori = msg.pose.orientation
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z]  # You can also add ori.w if needed
                data.append(row)
        # Convert to NumPy array
        dq_PI = np.array(data)
        np.save(bag_path+file_name+"_dq_PID_messages.npy",dq_PI)

        topic_name='/PRIMITIVE_velocity_controller/dq_SAC_messages'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path+file_name+".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                pos = msg.pose.position
                ori = msg.pose.orientation
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z]  # You can also add ori.w if needed
                data.append(row)
        # Convert to NumPy array
        dq_SAC = np.array(data)
        np.save(bag_path+file_name+"_dq_SAC_messages.npy",dq_SAC)


        topic_name='/franka_state_controller/joint_states'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path+file_name+".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row=np.append(np.array(timestamp),np.asarray((msg.velocity[:7])))
                data.append(row)
        # Convert to NumPy array
        dq_measured = np.array(data)
        np.save(bag_path+file_name+"_dq_measured.npy",dq_measured)

        # topic_name='/franka_state_controller/joint_states/panda_joint{}/velocity'.format(str(i))
        topic_name='/franka_state_controller/joint_states_desired'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path+file_name+".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row=np.append(np.array(timestamp),np.asarray((msg.velocity[:7])))
                data.append(row)
        # Convert to NumPy array
        dq_desired_measured = np.array(data)
        np.save(bag_path+file_name+"_dq_desired_measured.npy",dq_desired_measured)


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
        q_measured = np.array(data)
        np.save(bag_path+file_name+"_q_measured.npy",dq_measured)
    else:
        dq_PI = np.load(bag_path + file_name + "_dq_PID_messages.npy")
        dq_SAC = np.load(bag_path + file_name + "_dq_SAC_messages.npy")
        dq_measured = np.load(bag_path + file_name + "_dq_measured.npy")
        dq_desired_measured = np.load(bag_path + file_name + "_dq_desired_measured.npy")
        q_measured = np.load(bag_path + file_name + "_q_measured.npy")

    return dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured

def load_real(bag_path, file_name, save=False):
    # file_name = "SAC_1"
    topic_name='/PRIMITIVE_velocity_controller/r_star_messages'
    if save:
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path+file_name+".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, msg.vector.x, msg.vector.y, msg.vector.z]
                data.append(row)
        # Convert to NumPy array
        r_star = np.array(data)
        np.save(bag_path+file_name+"_r_star.npy",r_star)

        topic_name='/franka_state_controller/myfranka_ee_pose'
        # List to store data
        data = []
        # Open the bag
        with rosbag.Bag(bag_path+file_name+".bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                timestamp = t.secs + t.nsecs * 1e-9
                row = [timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                data.append(row)
        # Convert to NumPy array
        ee_pose = np.array(data)
        np.save(bag_path+file_name+"_ee_pose.npy",ee_pose)

    else:
        ee_pose = np.load(bag_path + file_name + "_ee_pose.npy")
        r_star = np.load(bag_path + file_name + "_r_star.npy")

    return ee_pose, r_star


def retrieve_data(file_name, dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured, PIonly):

    # file_name= file_name+"_240Hz_"

    # Attentions
    idx_init_dq_measured=np.argwhere(abs(dq_PI[0, 0] - dq_measured[:, 0]) < 1e-3)
    idx_init_dq_desired_measured=np.argwhere(abs(dq_PI[0, 0] - dq_desired_measured[:, 0]) < 1e-3)
    dq_measured=dq_measured[idx_init_dq_measured[0][0]:, :]
    q_measured=q_measured[idx_init_dq_measured[0][0]:, :]
    dq_desired_measured=dq_desired_measured[idx_init_dq_desired_measured[0][0]:, :]

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
    q_init = q_measured[0,1:]
    for i in range(7):
        pb.resetJointState(arm, i, q_init[i], physicsClientId=physics_client)
    for j in [6] + list(range(8, 12)):
        pb.resetJointState(arm, j, 0, physicsClientId=physics_client)

    q_sim, dq_sim, tau_sim = [], [], []
    if PIonly==False:
        for idx_PI, idx_SAC in zip(closest_idx_PI, closest_idx_SAC):
        # for idx_PI in closest_idx_PI:
            print("simulation idx_PI=",idx_PI)
            dqc_t=dq_PI[idx_PI,1:7]+dq_SAC[idx_SAC,1:7]
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
    elif PIonly==True:
            for idx_PI in closest_idx_PI:
                print("simulation idx_PI=",idx_PI)
                dqc_t=dq_PI[idx_PI,1:7]
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
    # plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_dq_measured_dq_simest.png".format(file_name), format="png",
    #             bbox_inches='tight')
    # plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    # Plot for each joint (0 to 5)
    for joint_idx in range(6):
        ax = axes[joint_idx]
        # Downsample for visualization (every 100th point)
        # indices = np.arange(0, len(dq_measured), 100)
        if PIonly==False:
            y_commanded = dq_PI[closest_idx_PI, joint_idx+1] + dq_SAC[closest_idx_PI, joint_idx+1]
        elif PIonly==True:
            y_commanded = dq_PI[closest_idx_PI, joint_idx+1]
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
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_dq_desired_commanded.png".format(file_name),
        format="png",
        bbox_inches='tight')
    plt.show()


    dq = dq_measured[closest_idx_dq,1:7]
    q = q_measured[closest_idx_q,1:7]
    dq_sim = np.array(dq_sim)[:,:6]
    q_sim = np.array(q_sim)[:,:6]
    # Plot q and q_sim (Position)
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 8))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    fig1.suptitle('Joint Positions: Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs1[i // 2, i % 2]
        ax.plot(closest_t_q, q[:, i], '-og', label='Measured q')
        ax.plot(closest_t_PI, q_sim[:, i], '-ob', label='Simulated q_sim', markersize=2)
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Position')
        ax.grid(True)
        ax.legend()
    plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_q.png".format(file_name), format="png",
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
    plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_dq.png".format(file_name), format="png",
                bbox_inches='tight')
    plt.show()

    # Plot dq and error dq (Velocity)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 16))
    fig2.suptitle('Joint Velocities: Absolute Error Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_dq, abs(dq_sim[:, i] - dq[:, i])*180/np.pi, '-sr')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('$|dq_{\t{sim}} - dq_{\t{measured}}|$ [deg/s]')
        ax.grid(True)
        ax.set_ylim([0, 3])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_dq_abs_error.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()

    # Plot dq and error q (Position)
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 16))
    fig2.suptitle('Joint Positions: Absolute Error Measured vs Simulated', fontsize=16)
    for i in range(6):
        ax = axs2[i // 2, i % 2]
        ax.plot(closest_t_q, abs(q_sim[:, i] - q[:, i])*180/np.pi, '-sr')
        ax.set_title(f'Joint {i + 1}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('$|q_{\t{sim}} - q_{\t{measured}}|$ [deg]')
        ax.grid(True)
        ax.set_ylim([0, 5])
        ax.legend()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_q_abs_error.png".format(
            file_name), format="png",
        bbox_inches='tight')
    plt.show()


    return q, dq, q_sim, dq_sim

def GP_mismatch_learning(file_names, base_path_extracted_data, plot_dir):

    q_list = []
    dq_list = []
    q_sim_list = []
    dq_sim_list = []

    for file_name in file_names:
        q = np.load(f"{base_path_extracted_data}{file_name}_q.npy")  # shape: (T, DOF)
        dq = np.load(f"{base_path_extracted_data}{file_name}_dq.npy")  # shape: (T, DOF)
        q_sim = np.load(f"{base_path_extracted_data}{file_name}_q_sim.npy")  # shape: (T, DOF)
        dq_sim = np.load(f"{base_path_extracted_data}{file_name}_dq_sim.npy")  # shape: (T, DOF)

        arg_tracking_phase_init = np.argmax(abs(np.diff(dq[:, 0])))+1
        arg_tracking_phase_length=90
        q_list.append(q[arg_tracking_phase_init:arg_tracking_phase_init+arg_tracking_phase_length,:])
        dq_list.append(dq[arg_tracking_phase_init:arg_tracking_phase_init+arg_tracking_phase_length,:])
        q_sim_list.append(q_sim[arg_tracking_phase_init:arg_tracking_phase_init+arg_tracking_phase_length,:])
        dq_sim_list.append(dq_sim[arg_tracking_phase_init:arg_tracking_phase_init+arg_tracking_phase_length,:])

    for joint_idx in range(6):
        # Stack into arrays of shape (3, T, DOF)
        q = np.stack(q_list, axis=0)[:,:,joint_idx]
        dq = np.stack(dq_list, axis=0)[:,:,joint_idx]
        q_sim = np.stack(q_sim_list, axis=0)[:,:,joint_idx]
        dq_sim = np.stack(dq_sim_list, axis=0)[:,:,joint_idx]

        # ----- STEP 1: Load Your Data -----
        # Let's say you have data from 3 iterations
        # q.shape = (3, T, DOF), assuming DOF = 1 for simplicity here

        train_ids = [0, 1, 2]
        test_id = 3
        GP_input_dim=2
        # Concatenate across time
        X_train = np.concatenate([q_sim[train_ids], dq_sim[train_ids]], axis=-1).reshape(-1, 2)
        X_test = np.concatenate([q_sim[test_id], dq_sim[test_id]], axis=-1).reshape(-1, 2)

        y_train_q = (q[train_ids] - q_sim[train_ids]).reshape(-1)
        y_train_dq = (dq[train_ids] - dq_sim[train_ids]).reshape(-1)

        # y_test_q = (q[test_id] - q_sim[test_id]).reshape(-1)
        # y_test_dq = (dq[test_id] - dq_sim[test_id]).reshape(-1)
        # Normalize inputs
        input_scaler = StandardScaler()
        X_train = input_scaler.fit_transform(X_train)
        X_test = input_scaler.transform(X_test)
        # Normalize targets
        target_scaler_q = StandardScaler()
        target_scaler_dq = StandardScaler()
        y_train_q = target_scaler_q.fit_transform(y_train_q.reshape(-1, 1)).flatten()
        y_train_dq = target_scaler_dq.fit_transform(y_train_dq.reshape(-1, 1)).flatten()
        # Convert to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train_q = torch.tensor(y_train_q, dtype=torch.float32)
        y_train_dq = torch.tensor(y_train_dq, dtype=torch.float32)


        # ----- STEP 2: Define GP Model -----
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                # self.covar_module = gpytorch.kernels.ScaleKernel(
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


        def train_gp(train_x, train_y, noise_covar=None):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = torch.tensor(0.05)
            if noise_covar is not None:
                likelihood.noise_covar.noise = torch.tensor(noise_covar)
            model = ExactGPModel(train_x, train_y, likelihood)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iter = 100
            losses = []
            for i in range(training_iter):
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-1):  # Try 1e-2 or even 1e-1
                    output = model(train_x)
                    loss = -mll(output, train_y)
                # output = model(train_x)
                # loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            plt.figure(figsize=(6, 8))
            plt.rcParams['font.family'] = 'Serif'
            # --- q ---
            plt.plot(losses, label="losses", color="b", linewidth=1,
                     marker="o",
                     markersize=4)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.legend()
            plt.show()

            return model, likelihood


        # Train GPs
        model_q, likelihood_q = train_gp(X_train, y_train_q, noise_covar=1e-2)
        model_dq, likelihood_dq = train_gp(X_train, y_train_dq, noise_covar=1e-2)

        # ----- STEP 3: Make Predictions -----
        model_q.eval()
        model_dq.eval()
        likelihood_q.eval()
        likelihood_dq.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_q = likelihood_q(model_q(X_test))
            pred_dq = likelihood_dq(model_dq(X_test))

            mean_q = pred_q.mean.numpy()
            mean_dq = pred_dq.mean.numpy()
            std_q = pred_q.variance.sqrt().numpy()
            std_dq = pred_dq.variance.sqrt().numpy()

            # Uncomment when Normalizing
            mean_q = target_scaler_q.inverse_transform(mean_q.reshape(-1, 1)).flatten()
            std_q = std_q * target_scaler_q.scale_[0]  # only scale, don't shift
            mean_dq = target_scaler_dq.inverse_transform(mean_dq.reshape(-1, 1)).flatten()
            std_dq = std_dq * target_scaler_dq.scale_[0]  # only scale, don't shift

        # ----- STEP 4: Apply Corrections -----
        q_sim_test = q_sim[test_id].reshape(-1)
        dq_sim_test = dq_sim[test_id].reshape(-1)

        q_corrected = q_sim_test + mean_q
        dq_corrected = dq_sim_test + mean_dq
        # q_corrected = q_sim_test + target_scaler_q.inverse_transform(mean_q.reshape(-1, 1)).flatten()
        # dq_corrected = dq_sim_test + target_scaler_dq.inverse_transform(mean_dq.reshape(-1, 1)).flatten()

        # ----- STEP 5: Plotting -----
        timesteps = np.arange(q_sim_test.shape[0])
        q_real = q[test_id].reshape(-1)
        dq_real = dq[test_id].reshape(-1)
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({
            'font.size': 14,  # overall font size
            'axes.labelsize': 16,  # x and y axis labels
            'xtick.labelsize': 12,  # x-axis tick labels
            'ytick.labelsize': 12,  # y-axis tick labels
            'legend.fontsize': 12,  # legend text
            'font.family': 'Serif'
        })        # --- q ---
        plt.subplot(2, 2, 1)
        plt.plot(timesteps, np.rad2deg(q_real), label="Real q[{}]".format(str(joint_idx)), color="g", linewidth=1, marker="o",
                 markersize=4)
        plt.plot(timesteps, np.rad2deg(q_sim_test), label="Simulated q[{}]".format(str(joint_idx)), color="b", linewidth=1,
                 marker="o", markersize=2)
        plt.plot(timesteps, np.rad2deg(q_corrected), label="Corrected q (GP)", color="m", linewidth=1, marker="o", markersize=2)
        plt.fill_between(timesteps, np.rad2deg(q_corrected - std_q), np.rad2deg(q_corrected + std_q), color="m", alpha=0.15,
                         label="Uncertainty")
        plt.title("Joint {} Positions".format(str(joint_idx+1)))
        plt.xlabel("k")
        plt.ylabel("q [deg]")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(timesteps, np.rad2deg(abs(q_real - q_corrected)),
                 label="$|q_{{real}}[{}]-q_{{corrected}}[{}]|$".format(str(joint_idx), str(joint_idx)), color="m",
                 linewidth=1, marker="o", markersize=4)
        plt.plot(timesteps, np.rad2deg(abs(q_real - q_sim_test)),
                 label="$|q_{{real}}[{}]-q_{{sim}}[{}]|$".format(str(joint_idx), str(joint_idx)), color="b",
                 linewidth=1, marker="o", markersize=2)
        plt.fill_between(timesteps, np.clip(np.rad2deg(abs(q_real - q_corrected) - std_q), a_min=0, a_max=None),
                         np.clip(np.rad2deg(abs(q_real - q_corrected) + std_q), a_min=0, a_max=None), color="m", alpha=0.15,
                         label="Uncertainty")
        plt.title("Joint {} Position Errors".format(str(joint_idx+1)))
        plt.xlabel("k")
        plt.ylabel("q abs error [deg]")
        plt.legend()
        # --- dq ---
        plt.subplot(2, 2, 2)
        plt.plot(timesteps, np.rad2deg(dq_real), label="Real dq[{}]".format(str(joint_idx)), color="g", linewidth=1, marker="o",
                 markersize=4)
        plt.plot(timesteps, np.rad2deg(dq_sim_test), label="Simulated dq[{}]".format(str(joint_idx)), color="b", linewidth=1,
                 marker="o", markersize=2)
        plt.plot(timesteps, np.rad2deg(dq_corrected), label="Corrected dq (GP)", color="m", linewidth=1, marker="o", markersize=2)
        plt.fill_between(timesteps, np.rad2deg(dq_corrected - std_dq), np.rad2deg(dq_corrected + std_dq), color="m", alpha=0.15,
                         label="Uncertainty")
        plt.title("Joint {} Speeds".format(str(joint_idx+1)))
        plt.xlabel("k")
        plt.ylabel("dq [deg/s]")
        plt.legend()
        plt.tight_layout()
        plt.subplot(2, 2, 4)
        plt.plot(timesteps, np.rad2deg(abs(dq_real - dq_corrected)),
                 label="$|dq_{{real}}[{}]-dq_{{corrected}}[{}]|$".format(str(joint_idx), str(joint_idx)),
                 color="m", linewidth=1, marker="o", markersize=4)
        plt.plot(timesteps, np.rad2deg(abs(dq_real - dq_sim_test)),
                 label="$|dq_{{real}}[{}]-dq_{{sim}}[{}]|$".format(str(joint_idx), str(joint_idx)), color="b",
                 linewidth=1, marker="o", markersize=2)
        plt.fill_between(timesteps, np.clip(np.rad2deg(abs(dq_real - dq_corrected) - std_dq), a_min=0, a_max=None),
                         np.clip(np.rad2deg(abs(dq_real - dq_corrected) + std_dq), a_min=0, a_max=None), color="m", alpha=0.15,
                         label="Uncertainty")
        plt.title("Joint {} Speed Errors".format(str(joint_idx+1)))
        plt.xlabel("k")
        plt.ylabel("dq abs error [deg/s]")
        plt.legend()
        plt.savefig(
            plot_dir+"joint_{}_normalized_GP.pdf".format(
                str(joint_idx)), format="pdf",
            bbox_inches='tight')
        plt.show()

        joblib.dump(input_scaler, plot_dir+'input_scaler{}.pkl'.format(str(joint_idx)))
        joblib.dump(target_scaler_q, plot_dir+'target_scaler_q{}.pkl'.format(str(joint_idx)))
        joblib.dump(target_scaler_dq, plot_dir+'target_scaler_dq{}.pkl'.format(str(joint_idx)))
        # Save q model
        # torch.save({
        #     'model_state_dict': model_q.state_dict(),
        #     'likelihood_state_dict': likelihood_q.state_dict(),
        #     'input_scaler': input_scaler,
        #     'target_scaler': target_scaler_q,
        # }, plot_dir+'gp_model_q{}.pth'.format(str(joint_idx)))

        torch.save({
            'model_state_dict': model_q.state_dict(),
            'likelihood_state_dict': likelihood_q.state_dict(),
            'input_scaler': input_scaler,
            'target_scaler': target_scaler_q,
            'train_x_shape': (1, GP_input_dim),  # ← save this!
        }, plot_dir + f'gp_model_q{joint_idx}.pth')

        # Save dq model
        torch.save({
            'model_state_dict': model_dq.state_dict(),
            'likelihood_state_dict': likelihood_dq.state_dict(),
            'input_scaler': input_scaler,
            'target_scaler': target_scaler_dq,
        }, plot_dir+'gp_model_dq{}.pth'.format(str(joint_idx)))

    print("")


if __name__ == '__main__':
    # file_names = ["PIonly_1"]
    # for file_name in file_names:
    #     bag_path = '/home/mahdi/bagfiles/experiments_HW309/dqPIandSAC_command_update_100Hz/'
    #     dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured = load_bags(file_name,bag_path, save=True)
    #
    #     if file_name[0:3]=="SAC":
    #         q, dq, q_sim, dq_sim = retrieve_data(file_name, dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured, PIonly=False)
    #     elif file_name[0:6]=="PIonly":
    #         q, dq, q_sim, dq_sim = retrieve_data(file_name, dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured, PIonly=True)
    #
    #     np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_q.npy".format(file_name),q)
    #     np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_dq.npy".format(file_name),dq)
    #     np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_q_sim.npy".format(file_name),q_sim)
    #     np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/{}_dq_sim.npy".format(file_name),dq_sim)
    #
    #
    # ####################################################################################################################
    file_names = ["SAC_1", "SAC_2", "SAC_3", "SAC_5"]
    # file_names = ["SAC_1", "SAC_2", "SAC_3", "PIonly_1", "PIonly_2", "PIonly_3"]
    base_path_extracted_data = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/"
    plot_dir = "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/extracted_data/Fep_HW_309/dqPIandSAC_command_update_100Hz/trainOnSAC_1_2_3_testOnSAC_5_trackingPhaseOnly/"
    GP_mismatch_learning(file_names, base_path_extracted_data, plot_dir)


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Path to your bag file
    bag_path = '/home/mahdi/bagfiles/experiments_HW274/'
    # plot real system tracking errors
    ee_pose, r_star = load_real(bag_path, "SAC_Kp01Ki001_1", save=True)
    # Attentions
    idx_init_ee_pose=np.argwhere(abs(r_star[0, 0] - ee_pose[:, 0]) < 1e-3)
    ee_pose=ee_pose[idx_init_ee_pose[0][0]:, :]

    t_ = (r_star[:,0]-r_star[0,0])*1000
    target_times = np.arange(1100, t_[-1], 100)[:100]
    # Find indices in t closest to each target time
    closest_idx_r_star = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])
    closest_t_r_star = t_[closest_idx_r_star[:100]]

    t_ = (ee_pose[:,0]-ee_pose[0,0])*1000
    # Find indices in t closest to each target time
    closest_idx_ee_pose = np.array([np.abs(t_ - target_).argmin() for target_ in target_times])
    closest_idx_ee_pose = closest_idx_ee_pose[closest_idx_ee_pose<t_.shape[0]]
    closest_t_ee_pose = t_[closest_idx_ee_pose[:100]]

    ee_pose_PIonly, r_star_PIonly = load_real(bag_path,"PIonly_Kp01Ki001_1", save=True)
    # Attentions
    # idx_init_ee_pose_PIonly=np.argwhere(abs(r_star_PIonly[0, 0] - ee_pose_PIonly[:, 0]) < 1e-3)
    # ee_pose_PIonly=ee_pose_PIonly[idx_init_ee_pose_PIonly[0][0]:, :]
    idx_init_r_star_PIonly=np.argwhere(abs(ee_pose_PIonly[0, 0] - r_star_PIonly[:, 0]) < 1e-3)
    r_star_PIonly=r_star_PIonly[idx_init_r_star_PIonly[0][0]:, :]

    t_ = (r_star_PIonly[:,0]-r_star_PIonly[0,0])*1000
    target_times_PIonly = np.arange(1100, t_[-1], 100)[:100]
    # Find indices in t closest to each target time
    closest_idx_r_star_PIonly = np.array([np.abs(t_ - target_).argmin() for target_ in target_times_PIonly])
    closest_t_r_star_PIonly = t_[closest_idx_r_star_PIonly[:100]]

    t_ = (ee_pose_PIonly[:,0]-ee_pose_PIonly[0,0])*1000
    # Find indices in t closest to each target time
    closest_idx_ee_pose_PIonly = np.array([np.abs(t_ - target_).argmin() for target_ in target_times_PIonly])
    closest_idx_ee_pose_PIonly = closest_idx_ee_pose_PIonly[closest_idx_ee_pose_PIonly<t_.shape[0]]
    closest_t_ee_pose_PIonly = t_[closest_idx_ee_pose_PIonly[:100]]

    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 16))
    plt.rcParams.update({
        'font.size': 14,  # overall font size
        'axes.labelsize': 16,  # x and y axis labels
        'xtick.labelsize': 12,  # x-axis tick labels
        'ytick.labelsize': 12,  # y-axis tick labels
        'legend.fontsize': 12,  # legend text
        'font.family': 'Serif'
    })
    axs3[0].plot(target_times_PIonly,
                 abs(ee_pose_PIonly[closest_idx_ee_pose_PIonly, 1] - r_star_PIonly[
                     closest_idx_r_star_PIonly, 1]) * 1000, '-ob',
                 label='without SAC')
    axs3[0].plot(target_times,
                 abs(ee_pose[closest_idx_ee_pose, 1] - r_star[closest_idx_r_star, 1]) * 1000, '-om', label='with SAC')
    axs3[0].set_xlabel("t [ms]")
    axs3[0].set_ylabel("|x-xd| [mm]")
    axs3[0].set_ylim([0, 3.2])
    axs3[0].legend(loc="upper right")
    axs3[1].plot(target_times_PIonly,
                 abs(ee_pose_PIonly[closest_idx_ee_pose_PIonly, 2] - r_star_PIonly[
                     closest_idx_r_star_PIonly, 2]) * 1000, '-ob',
                 label='without SAC')
    axs3[1].plot(target_times,
                 abs(ee_pose[closest_idx_ee_pose, 2] - r_star[closest_idx_r_star, 2]) * 1000, '-om', label='with SAC')
    axs3[1].set_xlabel("t [ms]")
    axs3[1].set_ylabel("|y-yd| [mm]")
    axs3[1].set_ylim([0, 3.2])
    axs3[2].plot(target_times_PIonly,
                 abs(ee_pose_PIonly[closest_idx_ee_pose_PIonly, 3] - r_star_PIonly[
                     closest_idx_r_star_PIonly, 3]) * 1000, '-ob',
                 label='without SAC')
    axs3[2].plot(target_times,
                 abs(ee_pose[closest_idx_ee_pose, 3] - r_star[closest_idx_r_star, 3]) * 1000, '-om', label='with SAC')
    axs3[2].set_xlabel("t [ms]")
    axs3[2].set_ylabel("|z-zd| [mm]")
    axs3[2].set_ylim([0, 3.2])
    axs3[3].plot(target_times_PIonly,
                 np.linalg.norm(
                     (ee_pose_PIonly[closest_idx_ee_pose_PIonly, 1:4] - r_star_PIonly[closest_idx_r_star_PIonly, 1:4]),
                     ord=2,
                     axis=1) * 1000, '-ob', label='without SAC')
    axs3[3].plot(target_times,
                 np.linalg.norm((ee_pose[closest_idx_ee_pose, 1:4] - r_star[closest_idx_r_star, 1:4]), ord=2,
                                axis=1) * 1000,
                 '-om', label='with SAC')
    axs3[3].set_xlabel("t [ms]")
    axs3[3].set_ylabel("||r-rd||_2 [mm]")
    axs3[3].set_ylim([0, 5])
    plt.savefig(bag_path + "/real_position_errors_both_SAC_Kp01Ki001_1_PIonly_Kp01Ki001_1.pdf", format="pdf",
                bbox_inches='tight')
    plt.show()

    print("")
