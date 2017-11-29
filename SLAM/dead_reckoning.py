# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:39:35 2017

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
import mapping_functions as map_func

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

#%% Transform from lidar to body
# init MAP
MAP, prob_map = map_func.init_map()

#%% Loop
angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T

for i in range(1, np.max(np.shape(lidar_ts)), 5):
#for i in range(1,2):

    # Get ith scan and pose at that timestep
    scan, pose_t = map_func.get_scan(lidar_data, angles, i)
    
    # Get neck and head angles
    neck, head = map_func.get_angles(head_angles, joint_idx, i)    
    
    # Compute homogeneous transformations from lidar to body frame
    scan_body = map_func.lidar2body(neck, head, scan)
    
    # Compute homogeneous transformations from body to global frame
    scan_world = map_func.body2global(pose_t, scan_body)
    
    # Threshold z
    scan_world = map_func.threshold_z(scan_world)    
    
    # convert from meters to cells
    xis, yis = map_func.meters2cells(scan_world, MAP)

    prob_map[xis,yis] += (2 * np.log(9))

    start_x, start_y = map_func.meters2cells(pose_t, MAP)

    xis_2 = np.append(xis, start_x)
    yis_2 = np.append(yis, start_y)
    mask_2 = np.zeros(np.shape(prob_map))
    temp = np.array([yis_2, xis_2]).T.astype(np.int32)

    cv2.drawContours(image=mask_2, contours = [temp], contourIdx=0, color=np.log(1.0/9), thickness=-1)

    # add mask to log-odds map to accumulate log(1/9) in the free region
    prob_map += mask_2

elapsed = time.time() - ti
print elapsed

e = 1.0 - 1.0/(1.0 + np.exp(prob_map)) 

obst = e > 0.75
free = e < 0.35
unexp = (e > 0.35) & (e < 0.75)

e[obst] = 0
e[free] = 1
e[unexp] = 0.5
cv2.imshow('e', e)
cv2.waitKey(0)
cv2.destroyAllWindows()

