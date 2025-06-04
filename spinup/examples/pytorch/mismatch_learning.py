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

from numpy import asarray

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




def load_bags(file_name, save=False):
    # Path to your bag file
    bag_path = '/home/mahdi/bagfiles/experiments_HW274/'
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

if __name__ == '__main__':
    file_name = "SAC_1"
    dq_PI, dq_SAC, dq_measured, dq_desired_measured, q_measured = load_bags(file_name, save=False)

    file_name= file_name+"_240Hz_"

    idx_init_dq_measured=np.argwhere(abs(dq_PI[0, 0] - dq_measured[:, 0]) < 1e-3)
    idx_init_dq_desired_measured=np.argwhere(abs(dq_PI[0, 0] - dq_desired_measured[:, 0]) < 1e-3)
    dq_measured=dq_measured[idx_init_dq_measured[0][0]:, :]
    dq_desired_measured=dq_desired_measured[idx_init_dq_desired_measured[0][0]:, :]

    t = (dq_PI[:, 0] - dq_PI[0, 0]) * 1000
    # Target times: 0, 100, 200, ..., up to max(t)
    target_times = np.arange(0, t[-1], 4.166666667)
    # Find indices in t closest to each target time
    closest_indices = np.array([np.abs(t - target).argmin() for target in target_times])[:2480]
    closest_times = t[closest_indices]

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
    for idx in closest_indices:
        print("idx=",idx)
        dqc_t=dq_PI[idx,1:7]+dq_SAC[idx,1:7]
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
        for _ in range(1):
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

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    # Plot for each joint (0 to 5)
    for joint_idx in range(6):
        ax = axes[joint_idx]
        # Downsample for visualization (every 100th point)
        # indices = np.arange(0, len(dq_measured), 100)
        y_measured = dq_measured[closest_indices, joint_idx+1]
        y_sim = np.array(q_sim)[:, joint_idx]
        ax.plot(closest_times, y_measured, '-ob', label="dq - real measured")  # blue circles
        ax.plot(closest_times, y_sim, '-or', label="dq - simulation estimation")  # red circles
        ax.set_xlabel("t")
        ax.set_ylabel(f"dq{joint_idx + 1}")
        ax.grid(True)
    ax.legend()
    # Adjust layout
    plt.tight_layout()
    plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/{}_dq_measured_dq_simest.png".format(file_name), format="png",
                bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    # Plot for each joint (0 to 5)
    for joint_idx in range(6):
        ax = axes[joint_idx]
        # Downsample for visualization (every 100th point)
        # indices = np.arange(0, len(dq_measured), 100)
        y_measured = dq_desired_measured[closest_indices, joint_idx+1]
        y_sim = np.array(q_sim)[:, joint_idx]
        ax.plot(closest_times, abs(y_sim-y_measured), '-ok', label="|dq_real_measured - dq_simulation_est|")  # blue circles
        ax.set_xlabel("t")
        ax.set_ylabel(f"dq{joint_idx + 1}")
        ax.grid(True)
    ax.legend()
    # Adjust layout
    plt.tight_layout()
    plt.savefig( "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/{}_dq_measured_dq_simest_err.png".format(file_name), format="png",
                bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    # Plot for each joint (0 to 5)
    for joint_idx in range(6):
        ax = axes[joint_idx]
        # Downsample for visualization (every 100th point)
        # indices = np.arange(0, len(dq_measured), 100)

        y_commanded = dq_PI[closest_indices, joint_idx+1] + dq_SAC[closest_indices, joint_idx+1]
        ax.plot(closest_times, y_commanded, '-om', label="dq commanded - real", markersize=8)  # blue circles

        t_ = (dq_desired_measured[:, 0] - dq_desired_measured[0, 0]) * 1000
        # Target times: 0, 100, 200, ..., up to max(t)
        target_times_d = np.arange(0, t_[-1], 4.166666667)
        # Find indices in t closest to each target time
        closest_indices_d = np.array([np.abs(t_ - target).argmin() for target in target_times_d])[:2480]
        closest_times_d = t_[closest_indices_d]
        y_desired = dq_desired_measured[closest_indices_d, joint_idx + 1]
        ax.plot(closest_times_d, y_desired, '-og', label="dq desired - real", markersize=3)  # red circles
        ax.set_xlabel("t")
        ax.set_ylabel(f"dq{joint_idx + 1}")
        ax.grid(True)
    ax.legend()
    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        "/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/mismatch_learning/{}_dq_desired_commanded.png".format(file_name),
        format="png",
        bbox_inches='tight')
    plt.show()


    print("")
