import cv2


def drew_boundary_edge(thresh_img):
    x = thresh_img.shape[1]
    y = thresh_img.shape[0]

    # drawing boundary edge
    constant = cv2.line(thresh_img, (0, 0), (x, 0), (255, 255, 255), 5)
    constant = cv2.line(thresh_img, (0, 0), (0, y), (255, 255, 255), 5)
    constant = cv2.line(thresh_img, (x, 0), (x, y), (255, 255, 255), 5)
    constant = cv2.line(thresh_img, (0, y), (x, y), (255, 255, 255), 5)

    return constant
