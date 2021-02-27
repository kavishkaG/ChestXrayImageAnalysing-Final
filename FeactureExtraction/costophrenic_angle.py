import cv2
import numpy as np
import math
from math import acos, degrees


def slope(x1, y1, x2, y2):  # Line slope given two points:
    return (y2 - y1) / (x2 - x1)


def angle(s1, s2):
    return math.degrees(math.atan((s1 - s2) / (1 + (s2 * s1))))


def find_angle(lungs, image):
    print('aaaaaaaaaaaaaaaaaa')
    original = image.copy()
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    left_lung = lungs[0]
    right_lung = lungs[1]

    cx_left = lungs[2][0]
    cy_left = lungs[2][1]

    cx_right = lungs[3][0]
    cy_right = lungs[3][1]

    # print('cx_left', cx_left)
    # print('cy_left', cy_left)
    # print('cx_right', cx_right)
    # print('cy_right', cy_right)

    val_left = []
    val_right = []

    for val_x in left_lung:

        x_cord = val_x[0][0]
        y_cord = val_x[0][1]

        if x_cord > cx_left and y_cord > cy_left:
            val_left.append(val_x)
            # cv2.drawContours(original, [val_x], -1, (255, 0, 0), 2)

    points_left = np.vstack(val_left)

    M1 = cv2.moments(points_left)
    ccx_left = int(M1['m10'] / M1['m00'])
    ccy_left = int(M1['m01'] / M1['m00'])

    left_li = []
    for val in left_lung:

        x_cord = val[0][0]
        y_cord = val[0][1]

        if x_cord > ccx_left and y_cord > ccy_left:
            left_li.append(val)
            cv2.drawContours(original, [val], -1, (0, 0, 255), 3)

    pointl_left = np.vstack(left_li)

    hull_pointL = cv2.convexHull(pointl_left)
    # print('hull_point left', hull_pointL)

    x_min = 5000
    y_min = 5000
    x_tuple = []
    y_tuple = []
    for corner in hull_pointL:

        x_val = corner[0][0]
        y_val = corner[0][1]

        if x_val < x_min:
            x_min = x_val
            x_tuple = corner[0]
        if y_val < y_min:
            y_min = y_val
            y_tuple = corner[0]

    corner1 = x_tuple
    corner2 = y_tuple
    c_index = len(hull_pointL - 1) // 2
    corner3 = hull_pointL[c_index][0]

    # def slope(x1, y1, x2, y2):  # Line slope given two points:
    #     return (y2 - y1) / (x2 - x1)
    #
    # def angle(s1, s2):
    #     # return math.degrees(math.atan((s2 - s1) / (1 + (s2 * s1))))
    #     return math.degrees(math.atan((s1-s2) / (1 + (s2 * s1))))

    # line1 = (tuple(corner1), tuple(corner3))
    # line2 = (tuple(corner3), tuple(corner2))

    slope1 = slope(corner1[0], corner1[1], corner3[0], corner3[1])
    slope2 = slope(corner3[0], corner3[1], corner2[0], corner2[1])
    # print('slope1', slope1)
    # print('slope2', slope2)
    ang = angle(slope1, slope2)
    print('angle of left lung', ang)
    # hull_pointL.sort()
    # print('hull_point L sorted ', hull_pointL)
    # original = cv2.circle(original, tuple(corner1), 0, (255, 0, 0), 5)
    # original = cv2.circle(original, tuple(corner2), 0, (255, 0, 0), 5)
    # original = cv2.circle(original, tuple(corner3), 0, (255, 0, 0), 5)

    # cv2.line(original, tuple(corner1), tuple(corner2), (255, 0, 0), 2)
    # cv2.line(original, tuple(corner1), tuple(corner3), (255, 0, 0), 2)
    # cv2.line(original, tuple(corner3), tuple(corner2), (255, 0, 0), 2)

    cv2.drawContours(original, [hull_pointL], -1, (0, 255, 0), 3)

    for val_x in right_lung:

        x_cord = val_x[0][0]
        y_cord = val_x[0][1]

        if x_cord < cx_right and y_cord > cy_right:
            # print('val_x right', val_x)
            val_right.append(val_x)
            # cv2.drawContours(original, [val_x], -1, (255, 0, 0), 2)

    # print('val_right', val_right)
    points_right = np.vstack(val_right)

    M1 = cv2.moments(points_right)
    ccx_right = int(M1['m10'] / M1['m00'])
    ccy_right = int(M1['m01'] / M1['m00'])

    right_li = []
    for val in right_lung:

        x_cord = val[0][0]
        y_cord = val[0][1]

        if x_cord < ccx_right and y_cord > ccy_right:
            right_li.append(val)
            cv2.drawContours(original, [val], -1, (0, 0, 255), 3)

    pointl_right = np.vstack(right_li)

    hull_pointR = cv2.convexHull(pointl_right)
    # print('hull_point right  ', hull_pointR)
    x_max = 0
    y_min = 5000
    x_tuple = []
    y_tuple = []
    for corner in hull_pointR:

        x_val = corner[0][0]
        y_val = corner[0][1]

        if x_val > x_max:
            x_max = x_val
            x_tuple = corner[0]
        if y_val < y_min:
            y_min = y_val
            y_tuple = corner[0]

    corner1 = x_tuple
    corner2 = y_tuple
    c_index = len(hull_pointR - 1) // 2
    corner3 = hull_pointR[c_index][0]

    # print('corner1 : ', corner1, ', corner2 : ', corner2, ', corner3: ', corner3)
    # original = cv2.circle(original, tuple(corner1), 0, (255,0,0), 5)
    # original = cv2.circle(original, tuple(corner2), 0, (255,0,0), 5)
    # original = cv2.circle(original, tuple(corner3), 0, (255,0,0), 5)

    # cv2.line(original, tuple(corner1), tuple(corner2), (255, 0, 0), 2)
    # cv2.line(original, tuple(corner1), tuple(corner3), (255, 0, 0), 2)
    # cv2.line(original, tuple(corner3), tuple(corner2), (255, 0, 0), 2)

    # return math.degrees(math.atan((s1-s2) / (1 + (s2 * s1))))

    # line1 = (tuple(corner1), tuple(corner3))
    # line2 = (tuple(corner3), tuple(corner2))

    slope1 = slope(corner1[0], corner1[1], corner3[0], corner3[1])
    slope2 = slope(corner2[0], corner2[1], corner3[0], corner3[1])

    # length1 = cv2.norm(corner1 - corner3)
    # length2 = cv2.norm(corner2-corner3)
    # length3 = cv2.norm(corner1- corner2)

    # ang = degrees(acos((length1 * length1 + length2 * length2 - length3 * length3) / (2.0 * length1 * length2)))
    # print('slope1', slope1)
    # print('slope2', slope2)
    ang = angle(slope1, slope2)
    print('angle of right lung', ang)
    cv2.drawContours(original, [hull_pointR], -1, (0, 255, 0), 3)

    cv2.imshow('costophrenic anlge', original)
    # cv2.imwrite('corner.png',original)
    cv2.waitKey(0)
