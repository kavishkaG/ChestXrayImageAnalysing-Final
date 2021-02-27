import numpy as np
import cv2

from Segmentation.drew_boundary_edge import drew_boundary_edge


def grayscaleToBinaryConvertion(image):
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image[y, x] == 1 else 0

    # return the grayscale image
    return image


def deleteElement(array, index):
    print('del index', index)
    delete_area = np.delete(array, index)
    return delete_area


def filter_by_size(image):
    mask = np.zeros(image.shape, np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    dilation_image = cv2.dilate(closing, kernel, iterations=1)

    # cv2.imshow('dilation_image', dilation_image)

    contour, hier = cv2.findContours(dilation_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contour, key=cv2.contourArea, reverse=True)
    largest_area_1 = cv2.contourArea(cnt_sorted[0])
    largest_area_2 = cv2.contourArea(cnt_sorted[1])

    sensitivity_1 = 0.14
    sensitivity_2 = 2

    area_ratio_1 = (largest_area_1 - largest_area_2) / largest_area_1
    print(area_ratio_1)
    considered_cnt_lst = [cnt_sorted[0], cnt_sorted[1]]

    if area_ratio_1 > sensitivity_1:

        for i in range(2, len(cnt_sorted)):
            if largest_area_2 < (cv2.contourArea(cnt_sorted[i]) * sensitivity_2):
                considered_cnt_lst.append(cnt_sorted[i])
            else:
                break

    for cnt in considered_cnt_lst:
        cv2.drawContours(mask, [cnt], -1, (255, 0, 0), -1)

    return mask


def segment(resized_img, noise_removal_gray_img):

    thresh_img = cv2.adaptiveThreshold(noise_removal_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       699, 6)

    # cv2.imshow('adaptive_threshold_image', thresh_img)

    kernel = np.ones((3, 3), np.uint8)
    erosion_img = cv2.erode(thresh_img, kernel, iterations=1)

    # cv2.imshow('erosion_image', erosion_img)

    constant = drew_boundary_edge(thresh_img)

    # divided to clusters
    markers = cv2.connectedComponentsWithStats(constant, 8, cv2.CV_32S)
    print('markers', markers)

    marker_area_2 = markers[2]
    print('marker_area_2', markers[1])
    lung_mask_2 = markers[1] == -1

    for index, i in enumerate(marker_area_2):
        # print('index', index)
        if i[0] == 0 and i[1] == 0:
            marker_delete_area = deleteElement(marker_area_2, index)
            print('maker_delete_area', marker_delete_area)
        else:
            lung_mask_2 = lung_mask_2 + (markers[1] == index)

        marker_area = i[4]

    print('marker_area', marker_area)
    thresh2 = erosion_img
    thresh2[lung_mask_2 == False] = 0

    filterSizeImage = filter_by_size(thresh2)

    # cv2.imshow('area_filtered_image', filterSizeImage)

    kernel = np.ones((10, 10), np.uint8)

    dilation = cv2.dilate(filterSizeImage, kernel, iterations=1)

    lung_out = resized_img.copy()
    lung_out[dilation == False] = 0
    # cv2.imshow('lung_out', lung_out)

    return thresh2, lung_out
