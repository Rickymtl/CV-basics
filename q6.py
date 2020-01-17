import numpy as np
import cv2
from matplotlib import pyplot as plt
import q4

# This is the code I used for q6, I've commented out the execution of functions I used for each part.

def addRandNoise(im, m):
    h, w = im.shape
    # generate noise in range[-0.05, 0.05]
    noise = np.random.rand(h, w) * m * 2 - m
    down_scale = im/255
    down_scale = down_scale + noise
    for x in range(w):
        for y in range(h):
            if down_scale[y, x] < 0:
                down_scale[y, x] = 0
            elif down_scale[y, x] > 1:
                down_scale[y, x] = 1
    return down_scale * 255

def addSaltAndPepperNoise(im, d):
    h, w = im.shape
    noise = np.random.rand(h, w)
    dst = im[:, :]
    for x in range(w):
        for y in range(h):
            if noise[y, x] < d/2:
                dst[y, x] = 0
            elif noise[y, x] < d:
                dst[y, x] = 255
    return dst


if __name__ == "__main__":
    # load image
    gray = cv2.imread('./gray.jpg')
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    # add random noise
    # dst = addRandNoise(gray, 0.05)
    # dst2 = addSaltAndPepperNoise(gray, 0.05)
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    # plt.imshow(dst, cmap='gray')
    # plt.title('S&P noise added')
    # plt.show()
    # use Gaussian filter to remove random noise
    # kernel = cv2.getGaussianKernel(5, 3)
    # denoise = filter.MyCorrelation(dst, kernel, 'same')
    # plt.imshow(denoise, cmap='gray')
    # plt.title("noise removed using gaussian")
    # plt.show()

    # use median blur to remove snp noise
    # denoise2 = cv2.medianBlur(dst2, 3)

    # color noise
    color = cv2.imread('./color.jpg')
    h, w, s = color.shape
    print(color[:,:,0].shape)
    dst = color[:,:,:]
    print(dst.shape)
    # add noise
    dst[:, :, 0] = addSaltAndPepperNoise(color[:, :, 0], 0.05)
    dst[:, :, 1] = addSaltAndPepperNoise(color[:, :, 1], 0.05)
    dst[:, :, 2] = addSaltAndPepperNoise(color[:, :, 2], 0.05)
    # rm noise
    dst[:, :, 0] = cv2.medianBlur(dst[:,:,0], 3)
    dst[:, :, 1] = cv2.medianBlur(dst[:,:,1], 3)
    dst[:, :, 2] = cv2.medianBlur(dst[:,:,2], 3)
    kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])/9
    dst[:, :, 0] = q4.MyCorrelation(dst[:, :, 0], kernel, 'same')
    dst[:, :, 1] = q4.MyCorrelation(dst[:, :, 1], kernel, 'same')
    dst[:, :, 2] = q4.MyCorrelation(dst[:, :, 2], kernel, 'same')
    # dst = cv2.medianBlur(dst, 3)
    # dst = cv2.medianBlur(dst, 3)
    # dst = cv2.medianBlur(dst, 3)

    dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    plt.imshow(dst2)
    plt.title('tried to remove artifact using moving average')
    plt.show()

