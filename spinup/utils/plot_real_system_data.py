
import rosbag
import matplotlib.pyplot as plt
from datetime import datetime
import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

# # Path to your .bag file
# bag_file = "/home/mahdi/Downloads/bags/PIonly_noObserver_1_repeated.bag"
# # Open the bag file
# bag = rosbag.Bag(bag_file)
# # Specify the topic you want to read from
# topic_name = "/PRIMITIVE_velocity_controller/PRIMITIVE_messages/EEposition[1]"
# b = bagreader(bag_file)
# csvfiles = []
# for t in b.topics:
#     data = b.message_by_topic(t)
#     csvfiles.append(data)
# print(csvfiles[0])
# data = pd.read_csv(csvfiles[0])
# data.to_pickle("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/real_system_data/PIonly_noObserver_1_repeated.pkl")

data = pd.read_pickle("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/real_system_data/SAC_HW272_noObserver_1.pkl")
data_PIonly = pd.read_pickle("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/real_system_data/PIonly_noObserver_1_repeated.pkl")

EEposition_0=data.EEposition_0.to_numpy()
EEposition_1=data.EEposition_1.to_numpy()
EEposition_2=data.EEposition_2.to_numpy()
r_star_0=data.r_star_0.to_numpy()
r_star_1=data.r_star_1.to_numpy()
r_star_2=data.r_star_2.to_numpy()

EEposition_0_PIonly=data_PIonly.EEposition_0.to_numpy()
EEposition_1_PIonly=data_PIonly.EEposition_1.to_numpy()
EEposition_2_PIonly=data_PIonly.EEposition_2.to_numpy()
r_star_0_PIonly=data_PIonly.r_star_0.to_numpy()
r_star_1_PIonly=data_PIonly.r_star_1.to_numpy()
r_star_2_PIonly=data_PIonly.r_star_2.to_numpy()

t=data.Time.to_numpy()-data.Time.to_numpy()[0]
t_PIonly=data_PIonly.Time.to_numpy()-data_PIonly.Time.to_numpy()[0]

t0_tracking=t[np.argwhere(t>(4.91-3.91))[0]]
t0_tracking_PIonly=t_PIonly[np.argwhere(t_PIonly>(4.91-3.91))[0]]

# N=8000


fig3, axs3 = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(8, 12))
plt.rcParams['font.family'] = 'Serif'
axs3[0].plot(t-t0_tracking_PIonly, abs(EEposition_0-r_star_0)*1000,
             '-r', label='with SAC')
axs3[0].plot(t_PIonly-t0_tracking_PIonly, abs(EEposition_0_PIonly-r_star_0_PIonly)*1000,
             '-b', label='PI only')
# axs3[0].plot(np.arange(N) * 100, abs(plot_data_buffer[:, 30]) * 1000, 'r:',
#              label='error bound with SAC')
# axs3[0].plot(np.arange(N) * 100, abs(plot_data_buffer_no_SAC[:, 30]) * 1000, 'b:',
#              label='error bound PI only')
axs3[0].set_xlabel("t [s]")
axs3[0].set_ylabel("|x-xd| [mm]")
plt.legend()
# axs3[1].plot(np.arange(N) * 100,
#              abs(plot_data_buffer_no_SAC[:, 1] - plot_data_buffer_no_SAC[:, 4]) * 1000, 'b',
#              label='PI only')
axs3[1].plot(t-t0_tracking_PIonly, abs(EEposition_1-r_star_1)*1000,
             '-r', label='with SAC')
axs3[1].plot(t_PIonly-t0_tracking_PIonly, abs(EEposition_1_PIonly-r_star_1_PIonly)*1000,
             '-b', label='PI only')
# axs3[1].plot(np.arange(N) * 100, abs(plot_data_buffer[:, 31]) * 1000, 'r:',
#              label='error bound on with SAC')
# axs3[1].plot(np.arange(N) * 100, abs(plot_data_buffer_no_SAC[:, 31]) * 1000, 'b:',
#              label='error bound on PI only')
axs3[1].set_xlabel("t [s]")
axs3[1].set_ylabel("|y-yd| [mm]")
plt.legend()
# axs3[2].plot(np.arange(N) * 100,
#              abs(plot_data_buffer_no_SAC[:, 2] - plot_data_buffer_no_SAC[:, 5]) * 1000, 'b',
#              label='PI only')
axs3[2].plot(t-t0_tracking_PIonly, abs(EEposition_2-r_star_2)*1000,
             '-r', label='with SAC')
axs3[2].plot(t_PIonly-t0_tracking_PIonly, abs(EEposition_2_PIonly-r_star_2_PIonly)*1000,
             '-b', label='PI only')
# axs3[2].plot(np.arange(N) * 100, abs(plot_data_buffer[:, 32]) * 1000, 'r:',
#              label='error bound on with SAC')
# axs3[2].plot(np.arange(N) * 100, abs(plot_data_buffer_no_SAC[:, 32]) * 1000, 'b:',
#              label='error bound on PI only')
axs3[2].set_xlabel("t [s]")
axs3[2].set_ylabel("|z-zd| [mm]")
plt.legend()
# axs3[3].plot(np.arange(N) * 100,
#              np.linalg.norm(np.arange(N) * 100,
#                             (plot_data_buffer_no_SAC[:, 0:3] - plot_data_buffer_no_SAC[:, 3:6]), ord=2,
#                             axis=1) * 1000, 'b', label='PI only')
axs3[3].plot(t-t0_tracking, ((EEposition_0-r_star_0)**2+(EEposition_1-r_star_1)**2+(EEposition_2-r_star_2)**2)**0.5*1000,
             '-r', label='with SAC')
axs3[3].plot(t_PIonly-t0_tracking_PIonly, ((EEposition_0_PIonly-r_star_0_PIonly)**2+(EEposition_1_PIonly-r_star_1_PIonly)**2+(EEposition_2_PIonly-r_star_2_PIonly)**2)**0.5*1000,
             '-b', label='PI only')
# axs3[3].plot(np.arange(N) * 100,
#              np.linalg.norm(plot_data_buffer[:, 30:33], ord=2, axis=1) * 1000,
#              'r:', label='error bound on with SAC')
# axs3[3].plot(np.arange(N) * 100,
#              np.linalg.norm(plot_data_buffer_no_SAC[:, 30:33], ord=2, axis=1) * 1000,
#              'b:', label='error bound on PI only')

axs3[3].set_xlabel("t [s]")
axs3[3].set_ylabel("||r-rd||_2 [mm]")
# axs3[3].set_ylim([0, 10])
# axs3[3].set_yscale('log')
plt.legend()
plt.savefig("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/real_system_data/real_system_position_errors.pdf", format="pdf",
            bbox_inches='tight')
plt.show()

print("hi")
