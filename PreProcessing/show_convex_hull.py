import cv2
import numpy as np
import math


def show_convex_hull(segmentImage, image):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(segmentImage, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours = find_contours(segmentImage)
    # find and store hull points
    hull = []
    for i in contours:
        hull.append(cv2.convexHull(i, False))
    # create a mask from hull points
    # hull = mask_from_contours(img, hull)

    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    cv2.imshow("hullmask", mask)

    img = segmentImage.copy()
    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lung_areas = []

    contour, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(contour) < 2:
        return lung_areas

    else:
        cnt_sorted = sorted(contour, key=cv2.contourArea, reverse=True)

        largest_area_1 = cnt_sorted[0]
        largest_area_2 = cnt_sorted[1]

        M = cv2.moments(largest_area_1)
        cx1 = int(M['m10'] / M['m00'])
        cy1 = int(M['m01'] / M['m00'])

        M = cv2.moments(largest_area_2)
        cx2 = int(M['m10'] / M['m00'])
        cy2 = int(M['m01'] / M['m00'])

        center_right = [cx1, cy1]
        center_left = [cx2, cy2]

        if cx2 > cx1:
            left_lung = largest_area_2
            right_lung = largest_area_1

        else:
            right_lung = largest_area_2
            left_lung = largest_area_1
            center_left = [cx1, cy1]
            center_right = [cx2, cy2]

    right_lung_hull = []
    for cnt in right_lung:
        hull_right = cv2.convexHull(cnt)
        right_lung_hull.append(hull_right[0][0])
        # cv2.drawContours(image, [cnt], -1, (255, 0, 0), 1)

    hull_right = cv2.convexHull(right_lung)
    cv2.drawContours(image, [hull_right], -1, (0, 255, 0), 1)
    #
    left_lung_hull = []
    for cnt in left_lung:
        hull_left = cv2.convexHull(cnt)
        left_lung_hull.append(hull_left[0][0])
        # cv2.drawContours(image, [cnt], -1, (255, 0, 0), 1)

    hull_left = cv2.convexHull(left_lung)
    cv2.drawContours(image, [hull_left], -1, (0, 255, 0), 1)

    # hull_left = cv2.convexHull(left_lung)
    # hull_right = cv2.convexHull(right_lung)

    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.drawContours(mask, hull_left, -1, (255, 255, 255), -1)
    #     # return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #
    # hull = mask_from_contours(image, hull_left)

    # cv2.drawContours(original, [hull_left], -1, (0, 255, 255), 1)
    # cv2.drawContours(original, [hull_right], -1, (0, 255, 255), 1)

    cv2.imshow('hull', mask)

    cv2.waitKey(0)

    return hull_left
