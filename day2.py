from load_gtsrb import load_gtsrb_images

chosenClasses = [0,1,2]
sampleSize = 5
datapath = "/home/fuckery/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(datapath, chosenClasses, sampleSize)

# change type from float32 to uint8 and convert to grayscale afterwards
greyscaleImages = [cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in imgs]
