'''
This code takes in the actual color images as input and returns an image that has predicted the approximate location of
the barrel or of all regions in the image that has a color intensity similar to the barrel.

Code created by Sai Krishnan Chandrasekar
Purpose: ESE 650, Project 1
'''

import cv2
import numpy as np
import os
import glob

pi = 22.0/7.0

# Load the file of red pixel values and reformat it

barrel_red_vals = np.load('barrel_values.npy')
barrel_red_vals_cov = np.transpose(barrel_red_vals)

#Estimate parameters for single gaussian for red color
mean_red = np.mean(barrel_red_vals, axis=0)
covariance_matrix_red = np.cov(barrel_red_vals_cov)


def single_gauss_prob(img, mu, cov_matrix):

    nr, nc, n_col = img.shape

    reshaped_img = np.reshape(img, (nr*nc, n_col))

    prefix = 1/(np.sqrt(((2*pi)**3)*(np.linalg.det(cov_matrix))))
    temp = reshaped_img - mu
    temp = np.matrix(temp)

    L = np.linalg.cholesky(np.linalg.inv(cov_matrix))
    L = np.matrix(L)

    output_probability_img = np.exp(-0.5*np.sum(np.square(temp*L),axis = 1))

    output_probability_img = np.reshape(output_probability_img, (nr, nc))

    output_img = np.zeros(image.shape)

    output_probability_img[output_probability_img > np.max(output_probability_img) / 17.5] = 255
    output_img[:, :, 2] = output_probability_img
    output_img = np.array(output_img)
    output_img = cv2.resize(output_img, (400, 300))

    return output_img

folder = "Test images"
folder2 = "Detected barrels"
os.mkdir(folder2)


for filename in os.listdir(folder):

    image = cv2.imread(os.path.join(folder,filename))
    cv2.imshow('Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    basename = os.path.basename(filename)
    print basename
    output_img = single_gauss_prob(image, mean_red, covariance_matrix_red)

    cv2.imshow('red_detected',output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(folder2, basename+'Barrel_detected'+'.png'), output_img)
