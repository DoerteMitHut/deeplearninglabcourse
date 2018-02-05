# get shit
from load_gtsrb import load_gtsrb_images

# acquire data
datapath = "/home/fuckery/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(datapath,range(5),10)

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv

# change type from float32 to uint8 and convert to grayscale afterwards
greyscaleImages = [ cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in imgs]

# select files
image_files = [
    [datapath + "/00001/00027_00029.ppm"], # 30
    [datapath + "/00002/00010_00029.ppm"]  # 50
]

# the roi's limits in format [ymin, xmin, ymax, xmax]
roiCorners = [
    [10,8,85,80], # 30
    [10,10,104,104] # 50
]

for i, image_file in enumerate(image_files):
    #img_size = [32, 32]
    next_image = cv.imread(image_file, cv.IMREAD_COLOR)
    img = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)
    #img = cv.resize(img, tuple(img_size), interpolation = cv.INTER_CUBIC)
    plt.imshow(img[roiCorners[i][0]:roiCorners[i][2], roiCorners[i][1]:roiCorners[i][3]], 'gray')
    plt.show()
