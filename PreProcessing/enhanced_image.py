import numpy as np
import cv2


def enhanced_image_For(gray_img):
    gray_img_arr = np.asarray(gray_img)

    img_arr = gray_img_arr.copy()

    # print(img_arr[400])

    maxList = list(map(max, img_arr))
    # print('maxList', maxList)
    maxGrayLevel = min(map(max, img_arr))

    if maxGrayLevel > 200:
        maxNumber = maxGrayLevel
    else:
        maxNumber = 200

    for y in range(0, (gray_img.shape[0] - 1)):
        for x in range(0, (gray_img.shape[1] - 1)):

            if img_arr[y][x] > (maxNumber - 20):
                img_arr[y][x] = 255

    # cv2.imshow('img_arr - first', img_arr)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    first_clahe_gray_img = clahe.apply(img_arr)

    for y in range(0, (gray_img.shape[0] - 1)):
        for x in range(0, (gray_img.shape[1] - 1)):

            if first_clahe_gray_img[y][x] > (maxNumber - 30):
                first_clahe_gray_img[y][x] = 255

    return first_clahe_gray_img
