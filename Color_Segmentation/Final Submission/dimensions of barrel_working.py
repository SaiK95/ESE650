'''
This code takes in the predicted images (output of single_gauss.py) as input and returns the exact location of
 the barrel(s). It also returns the bottom Left, top right and centroid coordinates. It also returns the height, width,
 distance of barrel from the camera.

Code created by Sai Krishnan Chandrasekar
Purpose: ESE 650, Project 1
'''

import numpy as np
import cv2
import os
import glob

pi = 22.0/7.0

# Load the file of red pixel values and reformat it

barrel_red_vals = np.load('barrel_values.npy')
barrel_red_vals_cov = np.transpose(barrel_red_vals)

#Estimate parameters for single gaussian for red color
mean_red = np.mean(barrel_red_vals, axis=0)
covariance_matrix_red = np.cov(barrel_red_vals_cov)

def find_largest_contour(cont):
    """ Finds the largest contour in the image"""
    areas = [cv2.contourArea(c) for c in cont]
    max_index = np.argmax(areas)
    cnt = cont[max_index]

    return cnt


def get_contours(im):
    """ Gets all contours in the image"""
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return im2, contours, hierarchy


def estimate_depth(h,w):
    """Given height and width of the barrel, estimates the distance of the barrel to the camera"""

    actual_height = 0.57 #Given on Piazza
    actual_width = 0.4 #Given on Piazza
    f_est_height = 386.6873 #Estimated using similarity of triangles. F = (P*H)/D. P is estimated height in pixels, H is actual height, D is actual distance
    f_est_width = 372.9412 #Estimated using similarity of triangles. F = (P*W)/D. P is estimated width in pixels, W is actual width, D is actual distance

    d_height = (actual_height * f_est_height) / h #Estimating depth from estimated focal length using height
    d_width = (actual_width * f_est_width) / w #Estimating depth from estimated focal length using width
    d_avg = (d_height + d_width) / 2.0 #Averaging the two depths

    return d_avg


def estimate_barrel_parameters(x,y,h,w):
    """Estimates the bottom left X, bottom left Y, top right X, top right Y, centroid X, centroid Y of the barrel"""
    blX = x
    blY = y+h
    trX = x+w
    trY = y
    centreX = (x+x+w)/2
    centreY = (y+h+y)/2

    return blX, blY, trX, trY, centreX, centreY


def identify_barrel(im, orig_img):
    """Identifies the location of the barrel and puts a bounding box around it"""
    im2, contours, hierarchy = get_contours(im)

    areas = [cv2.contourArea(c) for c in contours]

    heights = []
    widths = []
    depths_average = []
    bottomLeftX = []
    bottomLeftY = []
    topRightY = []
    topRightX = []
    centroidX = []
    centroidY = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if contours.__len__() > 1:
            if np.max(areas)/area <= 4.4: # takes all areas that are upto 4.4 times as small as the largest area into consideration. Arrived at the value 4.4 by cross validation
                flag = 1
            else:
                flag = 0
        else:
            flag = 1
        print "h = ",h
        print "w = ",w
        h = float(h)
        w = float(w)

        if ((h/w) >= 1.15) and ((h/w) <= 2.6): #Takes all contours into consideration whose h/w ratio is within 1.15 and 2.6. Arrived at these figures by cross validation
            if flag == 1:

                h = int(h)
                w = int(w)
                d_avg = estimate_depth(h, w)
                bLeftX, bLeftY, tRightX, tRightY, centX, centY = estimate_barrel_parameters(x, y, h, w)

                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                bottomLeftX.append(bLeftX)
                bottomLeftY.append(bLeftY)
                topRightX.append(tRightX)
                topRightY.append(tRightY)
                centroidX.append(centX)
                centroidY.append(centY)

                heights.append(h)
                widths.append(w)
                depths_average.append(d_avg)


    return im, orig_img, heights, widths, depths_average, bottomLeftX, bottomLeftY, topRightX, topRightY, centroidX, centroidY


def preprocessing(im):
    """Blurs the image to remove salt and pepper noise and performs 'closing' operation to fill the holes"""
    im = cv2.medianBlur(im, 5)  # Median blur

    # Morphological Pre processing
    kernel = np.ones((10, 10), np.uint8)
    processed_im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

    return processed_im

folder = "Test images"
folder2 = "Detected barrels"
final_folder = "Final Output Images"
os.mkdir(final_folder)

#Final Parameters. To be saved and displayed
heights = ([])
widths = ([])
depths = ([])
bottomLeftX = []
bottomLeftY = []
topRightY = []
topRightX = []
centroidX = []
centroidY = []

for filename in os.listdir(folder2):

    basename = os.path.basename(filename)
    print basename

    # Reads the image
    img = cv2.imread(os.path.join(folder2,filename))
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    # Pre Processing
    processed_img = preprocessing(img)
    # Final Outputs
    barrel_det_img, barrel_in_orig_img, height, width, depth, bLeftX, bLeftY, tRightX, tRightY, centX, centY = identify_barrel(processed_img, img)

    cv2.imshow('Barrel', barrel_in_orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(final_folder, basename + 'Barrel_detected' + '.png'), barrel_in_orig_img)

    print "height = "+str(height)
    print "width = "+str(width)
    print "Depth = "+str(depth)
    print "Bottom Left X = ", bLeftX
    print "Bottom Left Y = ", bLeftY
    print "Top Right X = ", tRightX
    print "Top Right Y = ", tRightY
    print "Centroid X = ", centX
    print "Centroid Y = ", centY
    heights.append(height)
    widths.append(width)
    depths.append(depth)
    bottomLeftX.append(bLeftX)
    bottomLeftY.append(bLeftY)
    topRightX.append(tRightX)
    topRightY.append(tRightY)
    centroidX.append(centX)
    centroidY.append(centY)

dimensions = [heights, widths, depths, bottomLeftX, bottomLeftY, topRightX, topRightY, centroidX, centroidY]
dimensions = np.array(dimensions)
np.save('dimensions',dimensions)