import cv2


def convert_to_gray(img):
    num_channels = 0

    if len(img.shape) > 2:
        num_channels = img.shape[2]
    elif len(img.shape) == 2:
        num_channels = 1
    else:
        print('Resized img leng : ', len(img.shape))

    gray_img = img

    # convert color image to gray
    if num_channels != 1:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray_img', gray_img)

    return gray_img
