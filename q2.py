import numpy as np
from matplotlib import pyplot as plt
import cv2
from math import floor as fl
import math


def harris(im):
    blur = cv2.GaussianBlur(im, (5, 5), 7)

    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    Ix2_blur = cv2.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv2.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv2.GaussianBlur(IxIy, (7, 7), 10)

    # implemented using det and trace
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur, IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    R = det - 0.05 * np.multiply(trace, trace)

    # implemented using eigen values
    # l0 = np.zeros(im.shape)
    # l1 = np.zeros(im.shape)
    # for i in range(im.shape[0]):
    #     for j in range(im.shape[1]):
    #         m = np.array([[Ix2_blur[i, j], IxIy_blur[i, j]], [IxIy_blur[i, j], Iy2_blur[i, j]]])
    #         l0[i, j], l1[i, j] = np.linalg.eig(m)[0]
    # R = np.multiply(l0, l1) - 0.05 * np.multiply(np.add(l0, l1), np.add(l0, l1))
    #
    # setting threshold
    limit = 0.1 * R.max()
    print(R.shape)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] < limit:
                R[i, j] = 0

    return R


def brown(im):
    # usual steps to calculate elements of M
    blur = cv2.GaussianBlur(im, (5, 5), 7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    Ix2_blur = cv2.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv2.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv2.GaussianBlur(IxIy, (7, 7), 10)

    # implemented using det and trace
    # det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur, IxIy_blur)
    # trace = Ix2_blur + Iy2_blur
    # result = np.divide(det, trace)

    # implemented using eigen values
    l0 = np.zeros(im.shape)
    l1 = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            m = np.array([[Ix2_blur[i, j], IxIy_blur[i, j]], [IxIy_blur[i, j], Iy2_blur[i, j]]])
            l0[i, j], l1[i, j] = np.linalg.eig(m)[0]
    result = np.divide(np.multiply(l0, l1), np.add(l0, l1))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if math.isnan(result[i, j]):
                result[i, j] = 0

    return result


def non_max_suppression(im, f_size):
    ih, iw = im.shape

    ih = ih + 2 * (f_size - 1)
    iw = iw + 2 * (f_size - 1)
    radius = fl(f_size / 2)
    im_cp = np.zeros((ih, iw))
    im_cp[f_size - 1:ih - f_size + 1, f_size - 1:iw - f_size + 1] = im[:, :]

    x_start = radius
    x_end = iw - radius
    y_start = radius
    y_end = ih - radius
    # try a 3x3 window for suppression first

    for x in range(x_start, x_end, 1):
        for y in range(y_start, y_end, 1):
            if im_cp[y, x] < im_cp[y - radius:y + radius, x - radius:x + radius].max():
                im_cp[y, x] = 0
            # elif im_cp[y, x] != 0:
            #     im_cp[y, x] = 1
    return im_cp[f_size - 1: ih - f_size + 1, f_size - 1: iw - f_size + 1]


def sift_by_LoG(im, s):
    # create a set of blank images to store key points at 10 different scales
    score_3d, scales = [], []
    h, w = im.shape
    sigma = 3
    for i in range(s + 2):
        blank = np.zeros((h + 2, w + 2))
        score_3d.append(blank)
        scales.append(sigma)
        sigma = sigma * 1.3

    # apply LoG at diff scale
    for i in range(s):
        blur = cv2.GaussianBlur(im, (0, 0), scales[i])
        LoG = cv2.Laplacian(blur, -1)

        # setting threshold
        # limit = max(0.8 * LoG.max(), 3.5)
        limit = 3.5
        print(limit)
        for y in range(LoG.shape[0]):
            for x in range(LoG.shape[1]):
                if LoG[y, x] < limit:
                    LoG[y, x] = 0
                # elif im[y,x] > 200:
                #     LoG[y,x] = 0
        LoG = non_max_suppression(LoG, 7)
        score_3d[i + 1][1:-1, 1:-1] = LoG

    # find local max in both location and scale and record the key points
    kp = []
    for i in range(s):
        radius = scales[i] / 2
        for y in range(h):
            for x in range(w):
                curr_pt = score_3d[i + 1][y + 1, x + 1]
                if curr_pt != 0:
                    prev_max = score_3d[i][y:y + 3, x:x + 3].max()
                    this_max = score_3d[i + 1][y:y + 3, x:x + 3].max()
                    next_max = score_3d[i + 2][y:y + 3, x:x + 3].max()
                    if curr_pt == max(prev_max, this_max, next_max):
                        kp.append([x, y, radius])

    return kp


if __name__ == "__main__":
    img = cv2.imread("./building.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    # dst = harris(gray)
    # plt.imshow(dst, cmap="gray")
    # plt.title("Setting threshold to R")
    # plt.show()

    # dst = non_max_suppression(dst, 7)
    # plt.imshow(dst, cmap="gray")
    # plt.title("Non-maxima suppression")
    # plt.show()

    # dst = brown(gray)
    # plt.imshow(dst, cmap="gray")
    # plt.title("Brown")
    # plt.show()

    # dst = harris(gray)
    # dst = non_max_suppression(dst, 7)
    # plt.imshow(dst, cmap="gray")
    # plt.title("harris using eigenvalues")
    # plt.show()

    # dst = brown(gray)
    # plt.imshow(dst, cmap="gray")
    # plt.title("Brown using eigenvalues")
    # plt.show()

    # h, w = gray.shape
    # M = cv2.getRotationMatrix2D((1000, 500), 60, 1)
    # dst = cv2.warpAffine(img, M, (w, h))
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # plt.imshow(dst, cmap="gray")
    # plt.title("rotated image")
    # plt.show()
    # dst = harris(dst)
    # dst = non_max_suppression(dst, 7)
    # plt.imshow(dst, cmap="gray")
    # plt.title("rotated corners")
    # plt.show()

    kp = sift_by_LoG(gray, 15)
    for item in kp:
        print(item)
        gray = cv2.circle(gray, (item[0], item[1]), int(item[2]), 0, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("sift using LoG")
    plt.show()
