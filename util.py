import numpy as np


def normalizeImage(img):
    # convert to float point values
    toRet = img.astype(np.float32)
    # calculate the standard deviation
    sigma = np.sqrt(((toRet-toRet.mean())**2).mean())
    # and apply the actual normalization by deneaming and setting the standard deviation to 1
    toRet = (toRet-toRet.mean())/sigma

    return toRet


# convolution between an image and its kernel
def convolve(image, kernel):
    convolutedImage = np.zeros(image.shape)
    for yimage in range(0, image.shape[1]):
        for ximage in range(0, image.shape[0]):
            # convolution is simply the summation of the multiplication of each pixel the mask/kernel, when overlayed with the image on a pivot point. we hav chosen the center of the kernel as the pivot point.
            sum = 0
            for ykernel in range(0, kernel.shape[1]):
                for xkernel in range(0, kernel.shape[0]):
                    # use the center of the kernel as the center for the convolution and clamp the convolution coordinates, such that we recieve the value of the border pixel when the mask does not fit on the image
                    yconv = int(np.min( [image.shape[1]-1, np.max([0, yimage - kernel.shape[1]/2 + ykernel])]))
                    xconv = int(np.min( [image.shape[0]-1, np.max([0, ximage - kernel.shape[0]/2 + xkernel])]))
                    sum += image[xconv, yconv] * kernel[xkernel, ykernel]

            convolutedImage[ximage, yimage] = sum

    return convolutedImage