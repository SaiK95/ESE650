from scipy import io
import numpy as np
import os, sys
import glob

from Quaternion_functions import *
path = "imu"
dirs = os.listdir(path)

imu_file_index = 9
vicon_file_index = 9
imu = io.loadmat("imu\imuRaw"+str(imu_file_index)+".mat")
imu_vals = imu['vals']
imu_vals = np.transpose(imu_vals)
imu_ts = imu['ts']
imu_ts = np.transpose(imu_ts)

Vref = 3300

acc_x = -np.array(imu_vals[:,0])
acc_y = -np.array(imu_vals[:,1])
acc_z = np.array(imu_vals[:,2])
acc = [acc_x, acc_y, acc_z]

acc = np.array(acc)
acc = np.transpose(acc)
acc_sensitivity = 330.0
acc_scale_factor = Vref/1023.0/acc_sensitivity
acc_bias = acc[0] - (np.array([0, 0, 1])/acc_scale_factor)
acc_val = acc*acc_scale_factor
acc_val = acc_val - (acc_bias)*acc_scale_factor

gyro_x = np.array(imu_vals[:,4])
gyro_y = np.array(imu_vals[:,5])
gyro_z = np.array(imu_vals[:,3])

gyro = [gyro_x, gyro_y, gyro_z]
gyro = np.array(gyro)
gyro = np.transpose(gyro)
gyro_bias = gyro[0]
gyro_sensitivity = 3.33
gyro_scale_factor = Vref/1023/gyro_sensitivity
gyro_val = gyro*gyro_scale_factor
gyro_val = (np.array(gyro_val) - (gyro_bias*gyro_scale_factor))*(np.pi/180)

vicon_path = "vicon"
dirs = os.listdir(vicon_path)

vicon = io.loadmat("vicon/viconRot"+str(vicon_file_index)+".mat")

vicon_vals = vicon['rots']
vicon_ts = vicon['ts']

vicon_phi = np.zeros([np.shape(vicon_vals)[2], 1])
vicon_theta = np.zeros([np.shape(vicon_vals)[2], 1])
vicon_psi = np.zeros([np.shape(vicon_vals)[2], 1])
for i in range(np.shape(vicon_vals)[2]):
    R = vicon_vals[:,:,i]
    vicon_phi[i], vicon_theta[i], vicon_psi[i] = rot2euler(R)