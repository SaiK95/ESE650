# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:11:19 2017

@author: Sai Krishnan
"""

import numpy as np
import bresenham2D as b2D
import random
import scipy
import cv2

def init_map():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -40  #meters
    MAP['ymin']  = -40
    MAP['xmax']  =  40
    MAP['ymax']  =  40 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))    
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int16) #DATA TYPE: char or int16

    prob_map = np.zeros([MAP['sizex'], MAP['sizey']])
    
    return MAP, prob_map

def get_scan(lidar_data, angles, i):
    ranges = np.double(lidar_data['lidar'][0][i]['scan'][0][0]).T
    pose_t = np.double(lidar_data['lidar'][0][i]['pose'][0][0]).T
    
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # xy position in the sensor frame
    xs0 = (ranges*np.cos(angles)).T
    ys0 = (ranges*np.sin(angles)).T
    zs0 = np.zeros(np.shape(xs0))
    ones = np.ones(np.shape(xs0))
    scan = np.vstack((xs0, ys0, zs0, ones))
    
    return scan, pose_t

def get_angles(head_angles, joint_idx, i):
    
    [neck, head] = head_angles[:,joint_idx[:,i]]
    neck = float(neck)
    head = float(head)

    return neck, head
    
def lidar2body(neck, head, pos):
    
    rot_neck = [[np.cos(neck), -np.sin(neck), 0, 0], 
                [np.sin(neck), np.cos(neck), 0, 0], 
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
    rot_head = [[np.cos(head), 0, np.sin(head), 0], 
                 [0, 1, 0, 0], 
                 [-np.sin(head), 0, np.cos(head), 0],
                 [0, 0, 0, 1]]
    t_body2head = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0.33],
                   [0, 0, 0, 1]]
    rot_temp = np.dot(np.dot(t_body2head, rot_neck), rot_head)
    t_head2lidar = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.15],
                    [0, 0, 0, 1]]
    h = np.dot(rot_temp, t_head2lidar)
    pos_body = np.dot(h, pos)
    return pos_body
    
def body2global(pose_t, pos_body):
    theta = float(pose_t[2])
    rot_theta = [[np.cos(theta), -np.sin(theta), 0, 0], 
                [np.sin(theta), np.cos(theta), 0, 0], 
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
    t_lidar_btow = [[1, 0, 0, float(pose_t[0])],
                    [0, 1, 0, float(pose_t[1])],
                    [0, 0, 0, 0.93],
                    [0, 0, 0, 1]]
    h = np.dot(t_lidar_btow, rot_theta)
    pos_world = np.dot(h, pos_body)  
    return pos_world

def threshold_z(scan_world):

    z_thresh = 0.2
    mask = np.logical_not((scan_world[2,:] < z_thresh))
    scan_world_z = scan_world[:,mask]

    return scan_world_z
    
def meters2cells(scan, MAP):
    xis = np.ceil((scan[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((scan[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return xis, yis    
    
def find_obstacles(lidar_data, angles, MAP, head_angles, joint_idx, i):
    # Get ith scan and pose at that timestep
    scan, pose_t = get_scan(lidar_data, angles, i)
    
    # Get neck and head angles
    neck, head = get_angles(head_angles, joint_idx, i)    
    
    # Compute homogeneous transformations from lidar to body frame
    scan_body = lidar2body(neck, head, scan)
    
    # Compute homogeneous transformations from body to global frame
    scan_world = body2global(pose_t, scan_body)
    
    # Threshold z
    scan_world_z = threshold_z(scan_world)    
    
    # convert from meters to cells
    xis, yis = meters2cells(scan_world_z, MAP)
    
    return xis, yis
    
def build_map(MAP, pose_t, xis, yis, prob_map):

    prob_map[xis,yis] += (2 * np.log(9))

    start_x, start_y = meters2cells(pose_t, MAP)

    xis_2 = np.append(xis, start_x)
    yis_2 = np.append(yis, start_y)
    mask_2 = np.zeros(np.shape(prob_map))
    temp = np.array([yis_2, xis_2]).T.astype(np.int32)

    cv2.drawContours(image=mask_2, contours = [temp], contourIdx=0, color=np.log(1.0/9), thickness=-1)

    # add mask to log-odds map to accumulate log(1/9) in the free region
    prob_map += mask_2
    
    e = 1.0 - 1.0/(1.0 + np.exp(prob_map)) 
    obst = e > 0.65
    free = e < 0.35
    unexp = (e > 0.35) & (e < 0.65)
    #e[traj_x_cell, traj_y_cell] = 0.8
    e[obst] = 0
    e[free] = 1
    e[unexp] = 0.5

    return prob_map, e    

    
def resample(part_w, n, part_pose):
    # This was code written partly be Dr. Sebastian Thrun as part of an online course on udacity. I took the course and hence adapted the code from my code that I Wrote for the course.
    p_w = np.zeros([n,1])
    p_pose = np.zeros([n,3])
    index = int(random.random()*n)
    beta = 0.0
    mw = np.max(part_w)
    for i in range(n):
        beta += random.random()*2.0*mw
        while beta >= part_w[index]:
            beta -= part_w[index]
            index = (index + 1) % n
        p_w[i] = part_w[index]
        p_pose[i,:] = part_pose[index]
    return p_w, p_pose        
    
def particle_predict(pose_t, pose_t_prev, part_pose, sigma, n):
        # global frame
    temp = pose_t - pose_t_prev
    d_theta = temp[-1]
    d_pose = temp[0:2]
    # local frame
    yaw_prev = float(pose_t_prev[2])
    rot_theta = [[np.cos(yaw_prev), np.sin(yaw_prev)],
                [-np.sin(yaw_prev), np.cos(yaw_prev)]]
    d_pose_local = np.dot(rot_theta, d_pose)
    d_theta_local = d_theta
    # Apply to particles
    for i in range(n):
        yaw_est = part_pose[i][2]
        d_pose_global = np.dot([[np.cos(yaw_est), -np.sin(yaw_est)],
                                [np.sin(yaw_est), np.cos(yaw_est)]], d_pose_local)
        d_theta_global = d_theta_local
        noise = np.random.normal(0, sigma)
        part_pose[i][0] += d_pose_global[0] + noise[0]
        part_pose[i][1] += d_pose_global[1] + noise[1]
        part_pose[i][2] += d_theta_global + noise[2] 
    return part_pose
    
def particle_update(pose, n, angles, MAP, head_angles, i, joint_idx, map_prev, n_thresh, part_alpha, lidar_data):
    
    C = np.zeros([n, 1])
    scan, pose_t = get_scan(lidar_data, angles, i)
    neck, head = get_angles(head_angles, joint_idx, i)    
    
    # Compute homogeneous transformations from lidar to body frame
    scan_body = lidar2body(neck, head, scan)
    
    for k in range(n):        
        pose_k = pose[k][:]        
        
        # Compute homogeneous transformations from body to global frame
        scan_world = body2global(pose_k, scan_body)
    
        # Threshold z
        scan_world_z = threshold_z(scan_world)    
    
        # convert from meters to cells
        xis, yis = meters2cells(scan_world_z, MAP)
        C[k] = np.sum(map_prev[xis, yis]) 

    #%% Update weights - 2nd method, using w
    part_w = np.log(part_alpha)
    part_w_c = part_w + C
    part_w_t = part_w_c - scipy.misc.logsumexp(part_w_c)
    part_alpha = np.exp(part_w_t)
    
    #%% Update weights - 1st method, slightly incorrect
#    part_alpha = np.multiply(part_alpha, C)
#    part_alpha = part_alpha/np.sum(part_alpha)


    #%% Resample
    n_eff = 1.0/(np.sum(part_alpha)**2)
    max_corr = np.argmax(part_alpha)
    #print C[max_corr]
    max_part_pose = pose[max_corr]    
    print ""+str(C[max_corr])+", "+str(np.max(part_alpha))
    if n_eff < n_thresh:
        part_alpha, part_pose = resample(part_alpha, n, pose)    
    return max_part_pose, part_alpha, part_pose
    
def get_map_prev(map_0):
    obst = (map_0 == 0)
    map_prev = np.zeros(np.shape(map_0))
    map_prev[obst] = 1
    
    return map_prev