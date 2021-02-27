import cv2
from skimage.feature import greycomatrix, greycoprops


def textureFeatures(left_lung, right_lung, resized_img):
    x, y, w, h = cv2.boundingRect(left_lung)
    leftLung = resized_img[y:y + h, x:x + w]
    cv2.imshow('leftLung', leftLung)

    g = greycomatrix(leftLung, [1], [0], levels=256, symmetric=False, normed=True)
    print('g', g[0][0])

    contrast_L = greycoprops(g, 'contrast')[0][0]
    energy_L = greycoprops(g, 'energy')[0][0]
    homogeneity_L = greycoprops(g, 'homogeneity')[0][0]
    correlation_L = greycoprops(g, 'correlation')[0][0]
    asm_L = greycoprops(g, 'ASM')[0][0]
    dissimilarity_L = greycoprops(g, 'dissimilarity')[0][0]

    x, y, w, h = cv2.boundingRect(right_lung)
    rightLung = resized_img[y:y + h, x:x + w]
    cv2.imshow('rightLung', rightLung)

    g = greycomatrix(rightLung, [1], [0], levels=256, symmetric=False, normed=True)

    contrast_R = greycoprops(g, 'contrast')[0][0]
    energy_R = greycoprops(g, 'energy')[0][0]
    homogeneity_R = greycoprops(g, 'homogeneity')[0][0]
    correlation_R = greycoprops(g, 'correlation')[0][0]
    asm_R = greycoprops(g, 'ASM')[0][0]
    dissimilarity_R = greycoprops(g, 'dissimilarity')[0][0]

    leftLungFeatures = [contrast_L, energy_L, homogeneity_L, correlation_L, asm_L, dissimilarity_L]
    rightLungFeatures = [contrast_R, energy_R, homogeneity_R, correlation_R, asm_R, dissimilarity_R]

    return leftLungFeatures, rightLungFeatures
