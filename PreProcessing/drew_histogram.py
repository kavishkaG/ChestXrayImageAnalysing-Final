import cv2
from matplotlib import pyplot as plt


def drew_histogram(image, equ_image):
    histNormal = cv2.calcHist(image, [0], None, [256], [0, 256])
    histEqu = cv2.calcHist(equ_image, [0], None, [256], [0, 256])

    plt.subplot(121)
    plt.title("histNormal")
    plt.xlabel('bins')
    plt.ylabel("No of pixels")
    plt.plot(histNormal)
    plt.subplot(122)
    plt.title("histEqu")
    plt.xlabel('bins')
    plt.ylabel("No of pixels")
    plt.plot(histEqu)
    plt.show()
