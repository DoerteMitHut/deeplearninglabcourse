from load_gtsrb import load_gtsrb_images

import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# configuration
chosenClasses = [0,13,20]
#chosenClasses = range(0,43)
maxSampleSize = 500
datapath = "/home/fuckery/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

maxNumMispredictedPreviews = 2

print("Loading data...")
[images, labels, class_descs, sign_ids] = load_gtsrb_images(datapath, chosenClasses, maxSampleSize)
# change type from float to uint8 and convert to grayscale afterwards
greyscaleImages = np.array([cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in images])

# this configuration worked out quite nicely for hog feature extraction
winSize = (32, 32) # use whole image, assume that the images all have the same size
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
numBins = 8

# how many cells do each block contain
blocksPerCellHorizontal =  np.floor(blockSize[0]/cellSize[0])
blocksPerCellVertical =  np.floor(blockSize[1]/cellSize[1])
blocksPerCell =  int(blocksPerCellVertical*blocksPerCellHorizontal)

numCellsHorizontal = np.floor((winSize[0]-blockSize[0])/blockStride[0])+1
numCellsVertical = np.floor((winSize[0]-blockSize[0])/blockStride[0])+1
numCells = int(numCellsVertical*numCellsHorizontal)

# calculate size of hog feature vector
print("Computing feature vector...")
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
vectors = np.zeros((len(greyscaleImages), numCells * blocksPerCell * numBins))
for i, img in enumerate(greyscaleImages):
    vectors[i] = np.array(hog.compute(img)).flatten()

# execute pricipal component analysis with 2 principal directions (lowering to a 2d subspace)
print("Executing principal component analysis...")
pca = PCA(n_components=2)
pca.fit(vectors)
pcs = pca.transform(vectors)

# present labeled data
for currentLabel in chosenClasses:
    currentImages = pcs[labels == currentLabel, :]
    plt.scatter(currentImages[:,0],currentImages[:,1], c=np.random.rand(3,), label=class_descs[currentLabel], alpha=0.3)
plt.legend()
plt.title('PCA')
plt.show()

# split set into test and training
print("Generating test and training sets...")
indices = np.arange(len(greyscaleImages))
np.random.shuffle(indices)
middle = int(len(greyscaleImages)/2)
trainingSetIndices = indices[0:middle]
testSetIndices = indices[middle:len(greyscaleImages)]

trainingSet = vectors[trainingSetIndices]
trainingSetLabels = labels[trainingSetIndices]
trainingSetImages = greyscaleImages[trainingSetIndices]

testSet = vectors[testSetIndices]
testSetLabels = labels[testSetIndices]
testSetImages = greyscaleImages[testSetIndices]

# train svm
print("Train svm...")
svc = SVC(decision_function_shape = 'ovo')
svc.fit(trainingSet, trainingSetLabels)
testSetPredictions = greyscaleImages[testSetIndices]

# test svm
predictedLabels = svc.predict(testSet)

# generate confusion matrix
cnfmat = confusion_matrix(testSetLabels, predictedLabels)
# select non diagonal elements (which are the wrong classified ones)
nonDiagonalMatrix = np.ones(cnfmat.shape)-np.diag(np.ones(cnfmat.shape[0]))
cnfMatErrs = cnfmat*nonDiagonalMatrix
# error rate is the number of wrong classified out of all classified entities
errorRate = cnfMatErrs.sum()/cnfmat.sum()

print("confusion matrix:")
print(cnfmat)
print("error rate: %f" % (errorRate))

# show an example of a misclassified image, if existing
misclassifiedSelector = testSetLabels != predictedLabels
if np.any(misclassifiedSelector):
    # gather misclassification results
    misclassifiedImages = testSetImages[misclassifiedSelector]
    misclassifiedLabels = testSetLabels[misclassifiedSelector]
    misclassifiedPredictions = predictedLabels[misclassifiedSelector]
    misclassifiedByIndex = np.arange(len(misclassifiedLabels))

    # clamp number of presented mispredictions
    maxNumMispredictedPreviews = np.clip(maxNumMispredictedPreviews, 1, len(misclassifiedLabels))

    # select random image by index
    np.random.shuffle(misclassifiedByIndex)
    for index in misclassifiedByIndex[0:maxNumMispredictedPreviews]:
        correctLabelName = class_descs[misclassifiedLabels[index]]
        currentLabelName = class_descs[misclassifiedPredictions[index]]

        plt.imshow(misclassifiedImages[index],'gray')
        plt.title('Falsely classified as "%s"\n should be "%s"' % (currentLabelName, correctLabelName))
        plt.show()
else:
    print('All images has been correctly classified')
