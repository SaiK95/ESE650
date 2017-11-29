from roipoly import roipoly
import cv2
import pylab as pl
import numpy as np
import os, sys
import glob

path = "C:\Users\Sai Krishnan\Desktop\Penn\Courses\Spring 2017\ESE 650\Project 1 - Color Segmentation\Alternative Training set\Train"
dirs = os.listdir(path)
val = []

for filename in glob.glob('C:\Users\Sai Krishnan\Desktop\Penn\Courses\Spring 2017\ESE 650\Project 1 - Color Segmentation\Alternative Training set\Train/*.png'):

    img = cv2.imread(filename, cv2.COLOR_RGB2LAB)
    # Convert to RGB due to Pylab
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pl.imshow(img, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    pl.title("left click: line segment         right click: close region")

    # let user draw first ROI
    ROI1 = roipoly(roicolor='r')

    # Convert back to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Split into channels
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    # Get masks for each channel
    mask_r = ROI1.getMask(img)
    mask_g = ROI1.getMask(img)
    mask_b = ROI1.getMask(img)

    temp = img[mask_r]
    for i in range(temp.shape[0]):
        val.append(temp[i, :])

        #    val = np.append(val, img[mask_r])
    print img[mask_r]
    # print val

    # Get the image in each channel by multiplying mask with the image
    mask_r = mask_r * img_r
    mask_g = mask_g * img_g
    mask_b = mask_b * img_b

    # Get the mask by merging the channel
    mask = cv2.merge((mask_b, mask_g, mask_r))
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img.save(path2 + mask, "PNG")
    # cv2.imwrite(os.path.join(dirs2, filename), mask)
val = np.array(val)
print val.shape
nr, nc = val.shape
np.save('barrel_values', val)