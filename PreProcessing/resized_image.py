import cv2


def resized_image(img, screen_height, screen_width):
    img_height = img.shape[0]
    img_width = img.shape[1]

    # print('img_height-->', img.shape[0], ' img_width-->', img.shape[1])

    # select scale percentage according to screen resolution without changing image ratio
    scale_percent = 0
    if (img_height < screen_height) & (img_width < screen_width):
        scale_percent = 100  # percent of original size
    elif (img_height * 90 / 100 < screen_height) & (img_width * 90 / 100 < screen_width):
        scale_percent = 90  # percent of original size
    elif (img_height * 80 / 100 < screen_height) & (img_width * 80 / 100 < screen_width):
        scale_percent = 80
    elif (img_height * 70 / 100 < screen_height) & (img_width * 70 / 100 < screen_width):
        scale_percent = 70
    elif (img_height * 60 / 100 < screen_height) & (img_width * 60 / 100 < screen_width):
        scale_percent = 60
    elif (img_height * 50 / 100 < screen_height) & (img_width * 50 / 100 < screen_width):
        scale_percent = 50
    elif (img_height * 40 / 100 < screen_height) & (img_width * 40 / 100 < screen_width):
        scale_percent = 40
    elif (img_height * 30 / 100 < screen_height) & (img_width * 30 / 100 < screen_width):
        scale_percent = 30
    elif (img_height * 20 / 100 <= screen_height) & (img_width * 20 / 100 <= screen_width):
        scale_percent = 20
    else:
        scale_percent = 10

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # print(width)
    # print(height)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized
