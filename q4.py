import numpy as np
from matplotlib import pyplot as plt
import cv2
from math import floor as fl
import math


def l2_match(im1, im2, threshold):
    keypoints_sift1, descriptors1 = sift.detectAndCompute(im1, None)
    keypoints_sift2, descriptors2 = sift.detectAndCompute(im2, None)
    print(len(descriptors1))
    print(len(descriptors2))

    # normalize
    for i in range(len(descriptors1)):
        descriptors1[i] = np.divide(descriptors1[i], np.linalg.norm(descriptors1[i]))
    for i in range(len(descriptors2)):
        descriptors2[i] = np.divide(descriptors2[i], np.linalg.norm(descriptors2[i]))

    # clip value to 0.2
    descriptors1 = np.clip(descriptors1, 0, 0.2)
    descriptors2 = np.clip(descriptors2, 0, 0.2)

    # normalize again
    for i in range(len(descriptors1)):
        descriptors1[i] = np.divide(descriptors1[i], np.linalg.norm(descriptors1[i]))
    for i in range(len(descriptors2)):
        descriptors2[i] = np.divide(descriptors2[i], np.linalg.norm(descriptors2[i]))

    # use l-2 norm to find ratio
    # all_match = np.array([])
    all_match = {}
    all_ratios = []
    for i in range(len(descriptors1)):
        minimum = [0, 0, 100.]
        second_min = [0, 0, 100.]
        for j in range(len(descriptors2)):
            # L2
            distance = ((descriptors1[i] - descriptors2[j]) ** 2).sum() ** 0.5
            # L1
            # distance = (abs(descriptors1[i] - descriptors2[j])).sum()
            # L3
            # distance = (abs(descriptors1[i] - descriptors2[j]) ** 3).sum() ** (1. / 3)
            if distance < minimum[2]:
                second_min = minimum
                minimum = [i, j, distance]
            elif distance < second_min[2]:
                second_min = [i, j, distance]
        ratio = minimum[2] / second_min[2]
        # all_match = np.append(all_match, ratio)
        if ratio < threshold:
            all_match[ratio] = minimum
            all_ratios.append(ratio)

    # find the top n matching points
    all_ratios.sort()
    comb = np.concatenate((im1, im2), axis=1)
    for i in range(len(all_ratios)):
        line = all_match[all_ratios[i]][0:2]
        start = keypoints_sift1[line[0]].pt
        start = (round(start[0]), round(start[1]))
        end = keypoints_sift2[line[1]].pt
        end = (round(end[0] + 825), round(end[1]))
        cv2.line(comb, start, end, 255, 2)
    plt.imshow(comb, cmap="gray")
    plt.title("using grayscale")
    plt.show()

    return all_match


def sift_color(temp, search, threshold):
    sift = cv2.xfeatures2d.SIFT_create()
    comb = np.concatenate((temp, search), axis=1)
    t_g = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    s_g = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)

    kp_tg, des_tg = sift.detectAndCompute(t_g, None)
    kp_sg, des_sg = sift.detectAndCompute(s_g, None)

    temp_blue, temp_green, temp_red = temp[:, :, 0], temp[:, :, 1], temp[:, :, 2]
    search_blue, search_green, search_red = search[:, :, 0], search[:, :, 1], search[:, :, 2]

    kp_temp_blue, des_temp_blue = sift.detectAndCompute(temp_blue, None)
    kp_search_blue, des_search_blue = sift.detectAndCompute(search_blue, None)
    # kp_temp_green, des_temp_green = sift.detectAndCompute(temp_green, None)
    # kp_search_green, des_search_green = sift.detectAndCompute(search_green, None)
    kp_temp_red, des_temp_red = sift.detectAndCompute(temp_red, None)
    kp_search_red, des_search_red = sift.detectAndCompute(search_red, None)

    # normalize
    for i in range(len(des_temp_blue)):
        des_temp_blue[i] = np.divide(des_temp_blue[i], np.linalg.norm(des_temp_blue[i]))
    # for i in range(len(des_temp_green)):
    #     des_temp_green[i] = np.divide(des_temp_green[i], np.linalg.norm(des_temp_green[i]))
    for i in range(len(des_temp_red)):
        des_temp_red[i] = np.divide(des_temp_red[i], np.linalg.norm(des_temp_red[i]))
    for i in range(len(des_search_blue)):
        des_search_blue[i] = np.divide(des_search_blue[i], np.linalg.norm(des_search_blue[i]))
    # for i in range(len(des_search_green)):
    #     des_search_green[i] = np.divide(des_search_green[i], np.linalg.norm(des_search_green[i]))
    for i in range(len(des_search_red)):
        des_search_red[i] = np.divide(des_search_red[i], np.linalg.norm(des_search_red[i]))
    for i in range(len(des_tg)):
        des_tg[i] = np.divide(des_tg[i], np.linalg.norm(des_tg[i]))
    for i in range(len(des_sg)):
        des_sg[i] = np.divide(des_sg[i], np.linalg.norm(des_sg[i]))

    # clip value to 0.2
    des_temp_blue = np.clip(des_temp_blue, 0, 0.2)
    # des_temp_green = np.clip(des_temp_green, 0, 0.2)
    des_temp_red = np.clip(des_temp_red, 0, 0.2)
    des_search_blue = np.clip(des_search_blue, 0, 0.2)
    # des_search_green = np.clip( des_search_green, 0, 0.2)
    des_search_red = np.clip(des_search_red, 0, 0.2)
    des_tg = np.clip(des_tg, 0, 0.2)
    des_sg = np.clip(des_sg, 0, 0.2)

    # normalize again
    for i in range(len(des_temp_blue)):
        des_temp_blue[i] = np.divide(des_temp_blue[i], np.linalg.norm(des_temp_blue[i]))
    # for i in range(len(des_temp_green)):
    #     des_temp_green[i] = np.divide(des_temp_green[i], np.linalg.norm(des_temp_green[i]))
    for i in range(len(des_temp_red)):
        des_temp_red[i] = np.divide(des_temp_red[i], np.linalg.norm(des_temp_red[i]))
    for i in range(len(des_search_blue)):
        des_search_blue[i] = np.divide(des_search_blue[i], np.linalg.norm(des_search_blue[i]))
    # for i in range(len(des_search_green)):
    #     des_search_green[i] = np.divide(des_search_green[i], np.linalg.norm(des_search_green[i]))
    for i in range(len(des_search_red)):
        des_search_red[i] = np.divide(des_search_red[i], np.linalg.norm(des_search_red[i]))
    for i in range(len(des_tg)):
        des_tg[i] = np.divide(des_tg[i], np.linalg.norm(des_tg[i]))
    for i in range(len(des_sg)):
        des_sg[i] = np.divide(des_sg[i], np.linalg.norm(des_sg[i]))

    all_match = {}
    all_ratios = []
    for i in range(len(des_tg)):
        minimum = [0, 0, 100.]
        second_min = [0, 0, 100.]
        for j in range(len(des_sg)):
            # L2
            distance = ((des_tg[i] - des_sg[j]) ** 2).sum() ** 0.5
            # L1
            # distance = (abs(descriptors1[i] - descriptors2[j])).sum()
            # L3
            # distance = (abs(descriptors1[i] - descriptors2[j]) ** 3).sum() ** (1. / 3)
            if distance < minimum[2]:
                second_min = minimum
                minimum = [i, j, distance]
            elif distance < second_min[2]:
                second_min = [i, j, distance]
        ratio = minimum[2] / second_min[2]
        # all_match = np.append(all_match, ratio)
        if ratio < threshold:
            all_match[ratio] = minimum
            all_ratios.append(ratio)

    all_match_blue = {}
    all_ratios_blue = []
    for i in range(len(des_temp_blue)):
        minimum = [0, 0, 100.]
        second_min = [0, 0, 100.]
        for j in range(len(des_search_blue)):
            distance = ((des_temp_blue[i] - des_search_blue[j]) ** 2).sum() ** 0.5
            if distance < minimum[2]:
                second_min = minimum
                minimum = [i, j, distance]
            elif distance < second_min[2]:
                second_min = [i, j, distance]
        ratio = minimum[2] / second_min[2]
        if ratio < threshold:
            all_match_blue[ratio] = minimum
            all_ratios_blue.append(ratio)
            print(minimum)

    # all_match_green = {}
    # all_ratios_green = []
    # for i in range(len(des_temp_green)):
    #     minimum = [0, 0, 100.]
    #     second_min = [0, 0, 100.]
    #     for j in range(len(des_search_green)):
    #         distance = ((des_temp_green[i] - des_search_green[j]) ** 2).sum() ** 0.5
    #         if distance < minimum[2]:
    #             second_min = minimum
    #             minimum = [i, j, distance]
    #         elif distance < second_min[2]:
    #             second_min = [i, j, distance]
    #     ratio = minimum[2] / second_min[2]
    #     if ratio < threshold:
    #         all_match_green[ratio] = minimum
    #         all_ratios_green.append(ratio)
    #         print(minimum)

    all_match_red = {}
    all_ratios_red = []
    for i in range(len(des_temp_red)):
        minimum = [0, 0, 100.]
        second_min = [0, 0, 100.]
        for j in range(len(des_search_red)):
            distance = ((des_temp_red[i] - des_search_red[j]) ** 2).sum() ** 0.5
            if distance < minimum[2]:
                second_min = minimum
                minimum = [i, j, distance]
            elif distance < second_min[2]:
                second_min = [i, j, distance]
        ratio = minimum[2] / second_min[2]
        if ratio < threshold:
            all_match_red[ratio] = minimum
            all_ratios_red.append(ratio)
            print(minimum)

    all_ratios_blue.sort()
    # all_ratios_green.sort()
    all_ratios_red.sort()
    all_ratios.sort()

    for i in range(len(all_ratios)):
        line = all_match[all_ratios[i]][0:2]
        start = kp_tg[line[0]].pt
        start = (round(start[0]), round(start[1]))
        end = kp_sg[line[1]].pt
        end = (round(end[0] + 825), round(end[1]))
        cv2.line(comb, start, end, (255, 255, 255), 2)

    for i in range(len(all_ratios_blue)):
        line = all_match_blue[all_ratios_blue[i]][0:2]
        start = kp_temp_blue[line[0]].pt
        start = (round(start[0]), round(start[1]))
        end = kp_search_blue[line[1]].pt
        end = (round(end[0] + 825), round(end[1]))
        cv2.line(comb, start, end, (255, 255, 255), 2)

    # for i in range(len(all_ratios_green)):
    #     line = all_match_green[all_ratios_green[i]][0:2]
    #     start = kp_temp_green[line[0]].pt
    #     start = (round(start[0]), round(start[1]))
    #     end = kp_search_green[line[1]].pt
    #     end = (round(end[0] + 825), round(end[1]))
    #     cv2.line(comb, start, end, (255,255,255), 2)

    for i in range(len(all_ratios_red)):
        line = all_match_red[all_ratios_red[i]][0:2]
        start = kp_temp_red[line[0]].pt
        start = (round(start[0]), round(start[1]))
        end = kp_search_red[line[1]].pt
        end = (round(end[0] + 825), round(end[1]))
        cv2.line(comb, start, end, (255, 255, 255), 2)
    comb = cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
    plt.imshow(comb)
    plt.title("sift matching on each colour channel")
    plt.show()
    return


if __name__ == "__main__":
    sift = cv2.xfeatures2d.SIFT_create()
    s1 = cv2.imread("./sample1.jpg")
    s2 = cv2.imread("./sample2.jpg")
    gray1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
    print(gray2.shape)

    # part a
    #
    # keypoints_sift1, descriptors1 = sift.detectAndCompute(gray1, None)
    # keypoints_sift2, descriptors2 = sift.detectAndCompute(gray2, None)
    # dst1 = cv2.drawKeypoints(gray1, keypoints_sift1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # dst2 = cv2.drawKeypoints(gray2, keypoints_sift2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.subplots(1,1),
    # plt.imshow(dst1)
    # plt.title("sample1 sift")
    # plt.show()
    # plt.subplots(1,2),
    # plt.imshow(dst2)
    # plt.title("sample2 sift")
    # plt.show()

    # part b plot
    # match_ratios = l2_match(gray1, gray2, 1)
    # x,y = [],[]
    # for i in range(1,20):
    #     threshold = 0.05*i
    #     x.append(threshold)
    #     y.append(len(match_ratios[match_ratios<threshold]))
    # plt.plot(x,y)
    # plt.xlabel("ratio of distances(closest/next closest)")
    # plt.ylabel("number of matches")
    # plt.show()

    # part b match
    # match_ratios = l2_match(gray1, gray2, 0.4)

    # d add noise
    # print(gray2.dtype)
    # gray1 = gray1 / 255
    # gray2 = gray2 / 255
    # h1, w1 = gray1.shape
    # h2, w2 = gray2.shape
    # g_noise1 = np.random.normal(0, 0.08, (h1, w1))
    # g_noise2 = np.random.normal(0, 0.08, (h2, w2))
    # noisy1 = gray1 + g_noise1
    # noisy2 = gray2 + g_noise2
    # noisy1 = (np.clip(noisy1, 0, 1) * 255).round().astype(np.uint8)
    # noisy2 = (np.clip(noisy2, 0, 1) * 255).round().astype(np.uint8)

    # plot sift key points
    # keypoints_sift1, descriptors1 = sift.detectAndCompute(noisy1, None)
    # keypoints_sift2, descriptors2 = sift.detectAndCompute(noisy2, None)
    # print(len(keypoints_sift1))
    # print(len(keypoints_sift2))
    # dst1 = cv2.drawKeypoints(noisy1, keypoints_sift1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # dst2 = cv2.drawKeypoints(noisy2, keypoints_sift2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # fig, plot = plt.subplots(1, 2)
    # plot[0].imshow(dst1)
    # # plot[0].title("noisy sample1 sift")
    # # plt.show()
    # plot[1].imshow(dst2)
    # # plot[1].title("noisy sample2 sift")
    # plt.show()

    # plot matches vs ratio
    # match_ratios = l2_match(noisy1, noisy2, 1)
    # x, y = [], []
    # for i in range(1, 20):
    #     threshold = 0.05 * i
    #     x.append(threshold)
    #     y.append(len(match_ratios[match_ratios < threshold]))
    # plt.plot(x, y)
    # plt.xlabel("ratio of distances(closest/next closest)")
    # plt.ylabel("number of matches")
    # plt.show()
    # l2_match(noisy1, noisy2, 0.4)

    # 4 e
    temp = cv2.imread("./colourTemplate.png")
    search = cv2.imread("./colourSearch.png")
    temp = cv2.resize(temp, (825, 825))
    # t_g = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    # s_g = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    # l2_match(t_g, s_g, 0.6)

    sift_color(temp, search, 0.6)
