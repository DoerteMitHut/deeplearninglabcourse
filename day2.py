from load_gtsrb import load_gtsrb_images

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


chosenClasses = [0,13,20]
colors = ["RED", "BLUE", "GREEN"]
maxSampleSize = 900
datapath = "/home/fuckery/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(datapath, chosenClasses, maxSampleSize)
labels = np.array(labels)
# change type from float32 to uint8 and convert to grayscale afterwards
greyscaleImages = np.array([cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in imgs])

winSize = (32,32)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
nbins = 8


hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

vectors = np.zeros((len(greyscaleImages),2*2*7*7*8))
for i, img in enumerate(greyscaleImages):
    vectors[i] = np.array(hog.compute(img)).flatten()

pca = PCA(n_components=2)
pca.fit(vectors)
pcs = pca.transform(vectors)

for i,currentLabel in enumerate(chosenClasses):
    plt.scatter(pcs[labels == currentLabel,0],pcs[labels == currentLabel,1],c=colors[i])

plt.show()

indices = np.arange(len(greyscaleImages))
np.random.shuffle(indices)
middle = int(len(greyscaleImages)/2)
trainingIndices = indices[0:middle]
testIndices = indices[middle:len(greyscaleImages)]

svc = SVC(decision_function_shape = 'ovo')
#svc.fit(pcs[trainingIndices], labels[trainingIndices])
svc.fit(vectors[trainingIndices], labels[trainingIndices])

predictedLabels = svc.predict(vectors[testIndices])

cnfmat = confusion_matrix(labels[testIndices], predictedLabels)
print("confusion matrix:")
print(cnfmat)

print("error rate:")
print((cnfmat*np.array([[0,1,1],[1,0,1],[1,1,0]])).sum()/cnfmat.sum())

misclassifiedIndices = labels[testIndices] != predictedLabels
stuff = greyscaleImages[testIndices]
misclassifiedImages = stuff[misclassifiedIndices]

if len(misclassifiedImages):
    index = np.random.randint(len(misclassifiedImages))
    plt.imshow(misclassifiedImages[index],'gray')
    print('classified as')
    print(class_descs[predictedLabels[index]])
    print('should be')
    print(class_descs[labels[index]])
plt.show()
