import cv2
import numpy as np
import math


def find_ROI(image):
    img = image.copy()
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

        center_rigth = [cx1, cy1]
        center_left = [cx2, cy2]

        if cx2 > cx1:
            left_lung = largest_area_2
            right_lung = largest_area_1

        else:
            right_lung = largest_area_2
            left_lung = largest_area_1
            center_left = [cx1, cy1]
            center_rigth = [cx2, cy2]

        right_lung_hull = []
        for cnt in right_lung:
            hull_right = cv2.convexHull(cnt)
            right_lung_hull.append(hull_right[0][0])
            cv2.drawContours(original, [cnt], -1, (255, 0, 0), 1)

        for cnt in right_lung:
            for cnt_right in hull_right[0]:
                if cnt[0].any() == cnt_right[0].any():
                    cv2.drawContours(original, [cnt], -1, (0, 0, 255), 1)

        hull_right = cv2.convexHull(right_lung)
        cv2.drawContours(original, [hull_right], -1, (0, 255, 0), 1)

        x = 0
        maxRt = 0
        while x < len(hull_right) - 1:
            x1 = hull_right[x][0][0]
            y1 = hull_right[x][0][1]
            x2 = hull_right[x + 1][0][0]
            y2 = hull_right[x + 1][0][1]
            screenHeight = original.shape[0]
            screenHalfHeight = int(screenHeight / 2)

            if ((x1 != x2) & (y1 != y2)) & (y1 > screenHalfHeight) & (y2 > screenHalfHeight):
                mRt = (x2 - x1) / (y2 - y1)
                mInt = int(mRt)
                if mInt <= 0:
                    if maxRt < abs(x1 - x2):
                        centerXRt = abs(math.ceil((x2 + x1) / 2))
                        centerYRt = abs(math.ceil((y2 + y1) / 2))
                        maxXRt1 = x1
                        maxXRt2 = x2
                        maxYRt1 = y1
                        maxYRt2 = y2
                        maxRt = abs(x1 - x2)
                        cRt = centerYRt + (mRt * centerXRt)
                        mNewRt = mRt

            x += 1

        if maxRt != 0:
            newXRt = 0
            newYRt = math.ceil(cRt)
            cv2.line(original, (centerXRt, centerYRt), (newXRt, newYRt), (255, 0, 255), thickness=1, lineType=8)

        cv2.line(original, (maxXRt1, maxYRt1), (maxXRt2, maxYRt2), (255, 0, 0), thickness=1, lineType=8)

        minRt = original.shape[0]
        minPointRtX = 0
        minPointRtY = 0
        perpendicularLengthRt = 0
        for cnt in right_lung:
            xRtCut = centerXRt
            while xRtCut > 0:
                yRtCut = (xRtCut*((-1)*mNewRt)) + cRt
                if (cnt[0][0] == math.ceil(xRtCut)) & (cnt[0][1] == math.ceil(yRtCut)):
                    perLengthRt = math.sqrt(math.pow(abs(centerXRt - cnt[0][0]), 2) + math.pow(abs(centerYRt - cnt[0][1]), 2))
                    if minRt > perLengthRt:
                        minPointRtX = cnt[0][0]
                        minPointRtY = cnt[0][1]
                        perpendicularLengthRt = perLengthRt
                        minRt = perLengthRt
                xRtCut -= 1

        for cnt in right_lung:
            yRtCut = centerYRt
            while yRtCut > int(screenHeight/2):
                xRtCut = (yRtCut - cRt)/((-1)*mNewRt)
                if (cnt[0][0] == math.ceil(xRtCut)) & (cnt[0][1] == math.ceil(yRtCut)):
                    perLengthRt = math.sqrt(math.pow(abs(centerXRt - cnt[0][0]), 2) + math.pow(abs(centerYRt - cnt[0][1]), 2))
                    if minRt > perLengthRt:
                        minPointRtX = cnt[0][0]
                        minPointRtY = cnt[0][1]
                        perpendicularLengthRt = perLengthRt
                        minRt = perLengthRt
                yRtCut -= 1

        if (minPointRtX != 0) & (minPointRtY != 0):
            cv2.circle(original, (minPointRtX, minPointRtY), radius=20, color=(0, 0, 255), thickness=3)

        for cnt in left_lung:
            hull_right = cv2.convexHull(cnt)
            cv2.drawContours(original, [cnt], -1, (0, 0, 255), 1)

        hull_left = cv2.convexHull(left_lung)
        draw_cont = cv2.drawContours(original, [hull_left], -1, (0, 255, 0), 1)

        x = 0
        maxLt = 0
        while x < len(hull_left) - 1:
            x1 = hull_left[x][0][0]
            y1 = hull_left[x][0][1]
            x2 = hull_left[x + 1][0][0]
            y2 = hull_left[x + 1][0][1]
            screenHeight = original.shape[0]
            screenHalfHeight = int(screenHeight / 2)

            if ((x1 != x2) & (y1 != y2)) & (y1 > screenHalfHeight) & (y2 > screenHalfHeight):
                mLt = (x2 - x1) / (y2 - y1)
                mInt = math.ceil(mLt)

                if mInt > 0:
                    if maxLt < abs(x1 - x2):
                        centerXLt = abs(math.ceil((x2 + x1) / 2))
                        centerYLt = abs(math.ceil((y2 + y1) / 2))
                        maxXLt1 = x1
                        maxXLt2 = x2
                        maxYLt1 = y1
                        maxYLt2 = y2
                        maxLt = abs(x1 - x2)
                        cLt = centerYLt + (mLt * centerXLt)
                        cutLtX = cLt / mLt
                        mNewLt = mLt

            x += 1

        if maxLt != 0:
            newX = math.ceil(cutLtX)
            newY = 0
            cv2.line(original, (centerXLt, centerYLt), (newX, newY), (255, 0, 255), thickness=1, lineType=8)

        cv2.line(original, (maxXLt1, maxYLt1), (maxXLt2, maxYLt2), (255, 0, 0), thickness=1, lineType=8)

        minLt = original.shape[0]
        minPointLtX = 0
        minPointLtY = 0
        perpendicularLengthLt = 0
        for cnt in left_lung:
            xLtCut = centerXLt
            while xLtCut < int(original.shape[1]):
                yLtCut = (xLtCut*((-1)*mNewLt)) + cLt
                if (cnt[0][0] == math.ceil(xLtCut)) & (cnt[0][1] == math.ceil(yLtCut)):
                    perLengthLt = math.sqrt(math.pow(abs(centerXLt - cnt[0][0]), 2) + math.pow(abs(centerYLt - cnt[0][1]), 2))
                    if minLt > perLengthLt:
                        minPointLtX = cnt[0][0]
                        minPointLtY = cnt[0][1]
                        perpendicularLengthLt = perLengthLt
                        minLt = perLengthLt
                xLtCut += 1

        for cnt in left_lung:
            yLtCut = centerYLt
            while yLtCut > int(original.shape[0]/2):
                xLtCut = (yLtCut - cLt)/((-1)*mNewLt)
                if (cnt[0][0] == math.ceil(xLtCut)) & (cnt[0][1] == math.ceil(yLtCut)):
                    perLengthLt = math.sqrt(math.pow(abs(centerXLt - cnt[0][0]), 2) + math.pow(abs(centerYLt - cnt[0][1]), 2))
                    if minLt > perLengthLt:
                        minPointLtX = cnt[0][0]
                        minPointLtY = cnt[0][1]
                        perpendicularLengthLt = perLengthLt
                        minLt = perLengthLt
                yLtCut -= 1

        if (minPointLtX != 0) & (minPointLtY != 0):
            cv2.circle(original, (minPointLtX, minPointLtY), radius=20, color=(0, 0, 255), thickness=3)

        print('perpendicular length of right lung --> ', perpendicularLengthRt)
        print('perpendicular length of left lung --> ', perpendicularLengthLt)

        x, y, w, h = cv2.boundingRect(left_lung)

        # cv2.imshow('Gaussian Blur', y)
        x, y, w, h = cv2.boundingRect(right_lung)

        gray_img_arr = np.asarray(original)

        cv2.imshow('ROI', original)

        cv2.waitKey(0)
        lungs = [left_lung, right_lung, center_left, center_rigth]
        return lungs
