import numpy as np
import cv2


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


def segment(gray1, constant, thresh1):
    # divided to clusters
    markers = cv2.connectedComponentsWithStats(constant, 8, cv2.CV_32S)
    print('markers', markers)

    marker_area_2 = markers[2]
    print('marker_area_2', marker_area_2)
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
    thresh2 = thresh1
    thresh2[lung_mask_2 == False] = 0

    lung_out = gray1.copy()
    lung_out[thresh2 == False] = 0
    cv2.imshow('lung_out', lung_out)

    equalize_lung_out = cv2.equalizeHist(lung_out)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_lung_out = clahe.apply(equalize_lung_out)

    print('clahe_lung_out-max', list(map(max, clahe_lung_out)))

    return thresh2, lung_out, clahe_lung_out
