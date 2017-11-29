# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:52:12 2017

@author: Sai Krishnan
"""

import load_data as ld
import p3_util as ut
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import bresenham2D as b2D
import cv2
import time
import mapping_functions_17 as map_func

def find_nearest_time(lidar_ts, joint_ts):
    joint_idx = np.zeros(np.shape((lidar_ts)))
    for i in range(0,len(np.transpose(lidar_ts))):
        joint_idx[:,i] = np.abs(joint_ts - lidar_ts[:,i]).argmin()
    return joint_idx

ti = time.time()
#%% Load data
lidar_data_index = 0

lidar_data = io.loadmat("data/train_lidar"+str(lidar_data_index)+".mat")
joint_data = io.loadmat("data/train_joint"+str(lidar_data_index)+".mat")
joint_ts = joint_data['ts']
temp = ld.get_lidar('data/train_lidar'+str(lidar_data_index))
head_angles = joint_data['head_angles']

#%% Sync time
lidar_ts = []
for i in range(len(temp)):
    ts = float(temp[i]['t'])
    lidar_ts.append(ts)

lidar_ts = np.matrix(lidar_ts)

joint_idx = find_nearest_time(lidar_ts, joint_ts).astype(int)

#%% Init MAP
MAP, prob_map = map_func.init_map()

#%% Init Particles
n = 1
part_pose = np.zeros([n, 3])
part_alpha = np.ones([n,1])*(1.0/n)
sigma = [0.001, 0.001, 0.001]
n_thresh = 5
#%% Prediction
angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
for i in range(1, np.max(np.shape(lidar_ts))):
#for i in range(1,6000):
    ranges = np.double(lidar_data['lidar'][0][i]['scan'][0][0]).T
    pose_t = np.double(lidar_data['lidar'][0][i]['pose'][0][0]).T
    pose_t_prev = np.double(lidar_data['lidar'][0][i-1]['pose'][0][0]).T
    part_pose = map_func.particle_predict(pose_t, pose_t_prev, part_pose, sigma, n)
    
#    max_part_pose, part_alpha = map_func.particle_update(pose, n, angles, MAP, head_angles, i, joint_idx, map_prev, n_thresh, part_alpha, lidar_data)

    plt.plot(part_pose[:,0], part_pose[:,1], 'ro', pose_t[0], pose_t[1], 'bo')
elapsed = time.time() - ti
print elapsed

plt.show()