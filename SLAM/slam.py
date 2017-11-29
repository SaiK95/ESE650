# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:02:02 2017

@author: Sai Krishnan
"""

import load_data as ld
import numpy as np
from scipy import io
import cv2
import time
import mapping_functions as map_func
import matplotlib.pyplot as plt
import random

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

#%% Init Maps
MAP, prob_map = map_func.init_map()
nr, nc = np.shape(prob_map)
map_disp = np.zeros([nr, nc, 3])

#%% Loop
angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T

#%% Particle filter init
n = 100
part_pose = np.zeros([n, 3])
part_alpha = np.ones([n,1])*(1.0/n)
sigma = [0.002, 0.002, 0.002]
n_thresh = 5

part_traj = np.zeros([n,2])
robot_traj = []

#%% Loop
# First scan:

angles_0 = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
# Get ith scan and pose at that timestep
scan_0, pose_t_0 = map_func.get_scan(lidar_data, angles_0, 0)
# Get neck and head angles
neck_0, head_0 = map_func.get_angles(head_angles, joint_idx, 0)    
    
# Compute homogeneous transformations from lidar to body frame
scan_body_0 = map_func.lidar2body(neck_0, head_0, scan_0)
    
# Compute homogeneous transformations from body to global frame
scan_world_0 = map_func.body2global(pose_t_0, scan_body_0)
    
# Threshold z
scan_world_z_0 = map_func.threshold_z(scan_world_0)    
    
# convert from meters to cells
xis_0, yis_0 = map_func.meters2cells(scan_world_z_0, MAP)

prob_map[xis_0,yis_0] += (2 * np.log(9))

start_x_0, start_y_0 = map_func.meters2cells(pose_t_0, MAP)

xis_2_0 = np.append(xis_0, start_x_0)
yis_2_0 = np.append(yis_0, start_y_0)
mask_2_0 = np.zeros(np.shape(prob_map))
temp = np.array([yis_2_0, xis_2_0]).T.astype(np.int32)

cv2.drawContours(image = mask_2_0, contours = [temp], contourIdx=0, color=np.log(1.0/9), thickness=-1)

# add mask to log-odds map to accumulate log(1/9) in the free region
prob_map += mask_2_0

map_0 = 1.0 - 1.0/(1.0 + np.exp(prob_map))

obst = map_0 > 0.75
free = map_0 < 0.35
unexp = (map_0 > 0.35) & (map_0 < 0.75)

map_0[obst] = 0
map_0[free] = 1
map_0[unexp] = 0.5
cv2.imshow('map', map_0)
cv2.waitKey(1)
map_prev = map_func.get_map_prev(map_0)
cv2.imshow('map_prev', map_prev)
cv2.waitKey(1)

#%% Loop starts
robot_pose = np.transpose(pose_t_0)
max_part_traj = part_pose[0,:]
for i in range(1, np.max(np.shape(lidar_ts)), 5):    
#for i in range(1,2):    
    #%% Particle filter predict
    
    pose_t = np.double(lidar_data['lidar'][0][i]['pose'][0][0]).T
    pose_t_prev = np.double(lidar_data['lidar'][0][i-1]['pose'][0][0]).T
    part_pose = map_func.particle_predict(pose_t, pose_t_prev, part_pose, sigma, n)
#    max_part_pose = part_pose.T
    robot_pose = np.append(robot_pose, np.transpose(pose_t))

    #%% Particle Filter update
    ranges = np.double(lidar_data['lidar'][0][i]['scan'][0][0]).T
    max_part_pose, part_alpha, part_pose = map_func.particle_update(part_pose, n, angles, MAP, head_angles, i, joint_idx, map_prev, n_thresh, part_alpha, lidar_data)
    
    plt.plot(max_part_pose[0], max_part_pose[1], 'ro', pose_t[0], pose_t[1], 'bo')
    #plt.pause(0.001)
    
    max_part_traj = np.append(max_part_traj, np.transpose(max_part_pose))
    #%% Build map
    
    scan, pose_t = map_func.get_scan(lidar_data, angles, i)
    
    # Get neck and head angles
    neck, head = map_func.get_angles(head_angles, joint_idx, i)    
    
    # Compute homogeneous transformations from lidar to body frame
    scan_body = map_func.lidar2body(neck, head, scan)
    
    # Compute homogeneous transformations from body to global frame
    scan_world = map_func.body2global(max_part_pose, scan_body)
    
    # Threshold z
    scan_world_z = map_func.threshold_z(scan_world)    
    
    # convert from meters to cells
    xis, yis = map_func.meters2cells(scan_world_z, MAP)

    prob_map[xis,yis] += (2 * np.log(9))

    start_x = np.ceil((max_part_pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    start_y = np.ceil((max_part_pose[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    xis_2 = np.append(xis, start_x)
    yis_2 = np.append(yis, start_y)
    mask_2 = np.zeros(np.shape(prob_map))
    temp = np.array([yis_2, xis_2]).T.astype(np.int32)

    cv2.drawContours(image=mask_2, contours = [temp], contourIdx=0, color=np.log(1.0/9), thickness=-1)

    # add mask to log-odds map to accumulate log(1/9) in the free region
    prob_map += mask_2
    
    map_t = 1.0 - 1.0/(1.0 + np.exp(prob_map))

    obst = map_t > 0.75
    free = map_t < 0.35
    unexp = (map_t > 0.35) & (map_t < 0.75)

    map_t[obst] = 0
    map_t[free] = 1
    map_t[unexp] = 0.5
    
    map_prev = map_func.get_map_prev(map_t)
#%% Visualize
    
    map_disp[:,:,0] = map_t
    map_disp[:,:,1] = map_t
    map_disp[:,:,2] = map_t

    robot_pose = np.reshape(robot_pose, [np.max(np.shape(robot_pose))/3, 3])
    max_part_traj = np.reshape(max_part_traj, [np.max(np.shape(max_part_traj))/3, 3])
    
    robot_x = robot_pose[:,0]
    robot_y = robot_pose[:,1]
    robot_x_c = np.ceil((robot_x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    robot_y_c = np.ceil((robot_y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1   
    # Robot is blue 
    map_disp[robot_x_c, robot_y_c,:] = [255, 0, 0]
       
    part_x = max_part_traj[:,0]
    part_y = max_part_traj[:,1]
    part_x_c = np.ceil((part_x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    part_y_c = np.ceil((part_y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # Particle is red
    map_disp[part_x_c, part_y_c, :] = [0, 0, 255]
    map_display = cv2.resize(map_disp, (500, 500)) 
    map_display_prev = cv2.resize(map_prev, (500, 500))
    cv2.imshow('map', map_display)
    cv2.waitKey(1)    
    cv2.imshow('map_prev', map_display_prev)
    cv2.waitKey(1)
    #print i
elapsed = time.time() - ti
print elapsed
#cv2.imshow('map', map_display)
cv2.imwrite('Map_'+str(lidar_data_index)+'_slam.PNG', map_disp*255)
cv2.waitKey(1)
cv2.destroyAllWindows()
plt.show()
