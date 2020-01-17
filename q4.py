from matplotlib import pyplot as plt
import cv2
import numpy as np
from math import floor as fl


def MyCorrelation(im, filter, mode):
    # filter height and width
    if mode not in ['full', 'same', 'valid']:
        return
    fh, fw = filter.shape
    # Create a copy of the image with a black border to perform correlation.
    ih, iw = im.shape
    ih = ih + 2 * (fh - 1)
    iw = iw + 2 * (fw - 1)
    im_cp = np.zeros((ih, iw))
    im_cp[fh-1:ih-fh + 1, fw-1:iw-fw + 1] = im[:, :]

    # create a blank image to store the result
    dst = np.zeros((ih, iw))
    # start and end point of correlation
    x_start = fl(fw/2)
    x_end = iw - fl(fw/2)
    y_start = fl(fh / 2)
    y_end = ih - fl(fh / 2)
    # iterate through each pixel proper distance away from border and perform correlation
    for x in range(x_start, x_end, 1):
        for y in range(y_start, y_end, 1):
            color = 0
            for x_f in range(fw):
                for y_f in range(fh):
                    x_delta = x_f - fl(fw/2)
                    y_delta = y_f - fl(fh/2)
                    color = color + filter[y_f, x_f] * im_cp[y + y_delta, x + x_delta]
            dst[y, x] = color

    # crop image according to the mode requirement
    if mode == 'full':
        return dst[y_start: y_end, x_start: x_end]
    elif mode == 'same':
        return dst[fh-1: ih - fh + 1, fw-1: iw - fw + 1]
    else:
        return dst[fh-1 + fl(fh/2): ih - fh - fl(fh/2) + 1, fw-1 + fl(fw/2): iw - fw - fl(fw/2) + 1]


def MyConvolution(im, ker, mode):
    ker = np.flip(ker, axis=0)
    ker = np.flip(ker, axis=1)
    return MyCorrelation(im, ker, mode)


def MyPortrait(im, mask):
    if im.shape != mask.shape:
        return
    ih, iw = im.shape
    kernel = cv2.getGaussianKernel(25, 12)
    dst = MyCorrelation(im, kernel, 'same')
    for x in range(iw):
        for y in range(ih):
            if mask[y, x] != 0:
                dst[y, x] = im[y, x]
    plt.imshow(dst, cmap='gray')
    plt.title('portrait')
    plt.show()
    return dst

if __name__ == "__main__":
    # input files:
    img = cv2.imread("./test.jpg")
    p_source = cv2.imread('./p_origin.jpg')
    p_mask = cv2.imread('./mask.jpg')

    # getting the gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    # p_gray = cv2.cvtColor(p_source, cv2.COLOR_BGR2GRAY)
    # mask_gray = cv2.cvtColor(p_mask, cv2.COLOR_BGR2GRAY)


    # Choose kernel (7x7 moving avg)
    # kernel = np.ones((15, 15), np.float32)/225
    #
    # # # show full
    # # dst_f = MyCorrelation(gray, kernel, "full")
    # # plt.imshow(dst_f, cmap='gray')
    # # plt.title("full")
    # # plt.show()
    # # # show same
    # # dst_s = MyCorrelation(gray, kernel, "same")
    # # plt.imshow(dst_s, cmap='gray')
    # # plt.title("same")
    # # plt.show()
    # # # show valid
    # # dst_v = MyCorrelation(gray, kernel, "valid")
    # # plt.imshow(dst_v, cmap='gray')
    # # plt.title("valid")
    # # plt.show()
    #
    #
    # # convolution:(Sobel filter)
    # kernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    #
    # dst_c = MyConvolution(gray, kernel2, 'same')
    # # plt.title('same')
    dst_c = cv2.Canny(img, 100, 200)
    plt.imshow(dst_c, cmap='gray')
    plt.show()
    # # dst2 = MyCorrelation(gray, kernel, mode)
    #
    # # portrait photo
    # MyPortrait(p_gray, mask_gray)
