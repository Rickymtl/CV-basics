import numpy as np
from matplotlib import pyplot as plt
import cv2


# mode 0 horizontal and 1 vertical
def stretch_4_times(im, mode):
    kernel = np.array([[0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0]])
    ih, iw = im.shape[0:2]

    if mode == 0:
        result_w = iw * 4 - 3
        inter = np.zeros([ih, result_w, 3])
        for i in range(0, iw):
            inter[:, i * 4, :] = im[:, i, :]
        result = cv2.filter2D(inter, -1, kernel)
        # result = MyCorrelation(result, kernel, "same")
    elif mode == 1:
        result_h = ih * 4 - 3
        inter = np.zeros([result_h, iw, 3])
        for i in range(0, ih, 1):
            inter[i * 4, :, :] = im[i, :, :]
        result = cv2.filter2D(inter, -1, kernel.T)
        # result = MyCorrelation(result, kernel.T, "same")

    return result


def triple(im):
    kernel_1d = np.array([0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, 0])
    kernel = np.outer(kernel_1d, kernel_1d)
    print(kernel.round(3))
    ih, iw = im.shape[0:2]
    result_h = ih * 3 - 2
    result_w = iw * 3 - 2
    result = np.zeros([result_h, result_w, 3])
    for i in range(ih):
        for j in range(iw):
            result[3 * i, 3 * j, :] = im[i, j, :]
    result = cv2.filter2D(result, -1, kernel)
    return result


if __name__ == "__main__":
    # q1 quadruple
    # img = cv2.imread("./bee.jpg") / 255
    # img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    # print(img.shape)
    # dst = stretch_4_times(img, 0)
    # plt.imshow(dst)
    # plt.title("quadruple horizontally")
    # plt.show()
    # print(dst.shape)
    # dst = stretch_4_times(dst, 1)
    # print(dst.shape)
    # plt.imshow(dst)
    # plt.title("quadruple on both direction")
    # plt.show()

    # q1 2d filter
    # kernel = np.array([[0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0]])
    # print(np.outer(kernel, kernel))

    # q2
    img = cv2.imread("./bee.jpg") / 255
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    dst = triple(img)
    plt.imshow(dst)
    plt.title("triple the size using a 2d filter")
    plt.show()
