import numpy as np


def accuracyCalculator(segmentImage, accurateImage):
    segmentImageArr = np.asarray(segmentImage)
    accurateImageArr = np.asarray(accurateImage)

    accurateCount = 0
    nonAccurateCount = 0
    for y in range(0, (segmentImageArr.shape[0] - 1)):
        for x in range(0, (segmentImageArr.shape[1] - 1)):

            if segmentImageArr[y][x] == accurateImageArr[y][x]:
                accurateCount = accurateCount + 1
            else:
                nonAccurateCount = nonAccurateCount + 1

    # print('accurateCount', accurateCount)
    # print('nonAccurateCount', nonAccurateCount)

    accuracy = accurateCount / (accurateCount + nonAccurateCount)

    return accuracy, accurateCount, nonAccurateCount
