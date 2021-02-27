import cv2


def convert_to_gray(img):

    if len(img.shape) > 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif len(img.shape) == 2:
        gray_img = img

    else:
        gray_img = img

    # cv2.imshow('gray_img', gray_img)

    return gray_img
