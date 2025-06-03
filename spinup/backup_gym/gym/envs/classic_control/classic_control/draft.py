#!/usr/bin/env python

import rosbag
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from PRIMITIVEmessages.msg import PRIMITIVEMsg  # Replace with actual message type
import os

# --- Parameters ---
input_bag = "/home/mahdi/Downloads/SAC_1.bag"
output_bag = "/home/mahdi/Downloads/SAC_1_corrected.bag"
pose_topic = "/franka_state_controller/myfranka_ee_pose"
primitive_topic = "/PRIMITIVE_velocity_controller/PRIMITIVE_messages"

# --- Load Messages ---
pose_msgs = []
primitive_msgs = []

print("Reading bag file...")
with rosbag.Bag(input_bag, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[pose_topic, primitive_topic]):
        timestamp = t.to_sec()
        if topic == pose_topic:
            pose_msgs.append((timestamp, msg, t))
        elif topic == primitive_topic:
            primitive_msgs.append((timestamp, msg))

# --- Prepare time arrays ---
primitive_times = np.array([t for t, _ in primitive_msgs])
primitive_values = [msg for _, msg in primitive_msgs]


# --- Nearest neighbor matching ---
def find_nearest_msg(time_ref, timestamps, messages):
    idx = np.searchsorted(timestamps, time_ref)
    if idx == 0:
        return messages[0]
    elif idx >= len(timestamps):
        return messages[-1]
    else:
        before = timestamps[idx - 1]
        after = timestamps[idx]
        return messages[idx - 1] if abs(time_ref - before) < abs(time_ref - after) else messages[idx]


# --- Write aligned messages to new bag ---
print(f"Writing aligned bag: {output_bag}")
with rosbag.Bag(output_bag, 'w') as outbag:
    for pose_time, pose_msg, ros_time in pose_msgs:
        matched_primitive_msg = find_nearest_msg(pose_time, primitive_times, primitive_values)

        # Write the pose message
        outbag.write(pose_topic, pose_msg, ros_time)

        # Write the matched primitive message at the same timestamp
        outbag.write(primitive_topic, matched_primitive_msg, ros_time)

print("✅ Done: Aligned rosbag written.")
