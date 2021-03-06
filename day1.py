import os

# get shit
from load_gtsrb import load_gtsrb_images
sampleSize = 5

# acquire data
datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'GTSRB/Final_Training/Images/')
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(datapath,[1, 2],sampleSize)

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv

from util import *


# change type from float32 to uint8 and convert to grayscale afterwards
greyscaleImages = [ cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in imgs]

# select files
image_files = [
    datapath + "/00001/00027_00029.ppm", # 30
    datapath + "/00002/00010_00029.ppm"  # 50
]

# the roi's limits in format [ymin, xmin, ymax, xmax]
roiCorners = [
    [10,9,21,16], # 30
    [10,9,22,17] # 50
]

results = np.zeros((2,len(greyscaleImages)))
for i, image_file in enumerate(image_files):
    print("======================================")

    #read filterImage
    next_image = cv.imread(image_file, cv.IMREAD_COLOR)
    #greyscalify filterImage
    img = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)
    #resizing
    img_size = [32, 32]
    img = cv.resize(img, tuple(img_size), interpolation = cv.INTER_CUBIC)
    #extract ROI
    plt.imshow(img,'gray')
    plt.show()
    img = img[roiCorners[i][0]:roiCorners[i][2], roiCorners[i][1]:roiCorners[i][3]]

    #stc
    kernel = normalizeImage(img.astype(np.float32))
    #plot kernel
    plt.imshow(kernel, 'gray')
    plt.show()
    for j, testImg in enumerate(greyscaleImages):

        #convResult = cv.filter2D(normalizeImage(testImg),-1,kernel,borderType=cv.BORDER_CONSTANT)
        convResult = convolve(normalizeImage(testImg), kernel)

        results[i,j]=convResult.max()

        plt.subplot(2,1,1)
        plt.imshow(testImg,'gray')
        plt.subplot(2,1,2)
        plt.imshow(convResult,'gray')
        plt.show()

    print(results)

#plt.figure()
#plt.scatter(results[0,0:sampleSize-1],results[1,0:sampleSize-1],c="RED")
#plt.scatter(results[0,sampleSize:2*sampleSize-1],results[1,sampleSize:2*sampleSize-1],c="BLUE")
#plt.show()

# define separating line
w1 = 1
w2 = 1
b = 0

# count good and bad classifications by the separating line
good = 0
bad = 0
for i in range(0,2*sampleSize):
    if w1 * results[0,i] + w2 * results[1,i] > b :
        if i < sampleSize:
            good = good+1
        else:
            bad = bad + 1
    else:
        if i >= sampleSize:
            good = good+1
        else:
            bad = bad + 1

print((good,bad))

plt.figure()
plt.scatter(results[0,0:sampleSize-1],results[1,0:sampleSize-1],c="RED")
plt.scatter(results[0,sampleSize:2*sampleSize-1],results[1,sampleSize:2*sampleSize-1],c="BLUE")
plt.show()
