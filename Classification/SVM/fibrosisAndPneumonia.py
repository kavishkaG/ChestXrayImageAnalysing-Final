from skimage.feature import hog
from PreProcessing.convert_to_gray import convert_to_gray
from skimage.feature import greycomatrix, greycoprops
import pickle
import cv2
import numpy as np


def fibrosisOrPneumoniaSVM(image):

    gray_img = convert_to_gray(image)
    dim = (64, 128)
    closing_new = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)

    filename = 'finalized_model_for_both.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    hist, hog_image_right = hog(closing_new, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                visualize=True, transform_sqrt=True, block_norm='L2-Hys', multichannel=False)

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    distance = [1]

    g = greycomatrix(gray_img, distances=distance, angles=angles, levels=256, symmetric=False, normed=True)

    contrast_R = greycoprops(g, 'contrast')
    energy_R = greycoprops(g, 'energy')
    homogeneity_R = greycoprops(g, 'homogeneity')
    correlation_R = greycoprops(g, 'correlation')
    asm_R = greycoprops(g, 'ASM')
    dissimilarity_R = greycoprops(g, 'dissimilarity')

    newarray = np.append(hist, [contrast_R[0][0], contrast_R[0][1], contrast_R[0][2], contrast_R[0][3], energy_R[0][0],
                                energy_R[0][1], energy_R[0][2], energy_R[0][3], homogeneity_R[0][0],
                                homogeneity_R[0][1], homogeneity_R[0][2], homogeneity_R[0][3], correlation_R[0][0],
                                correlation_R[0][1], correlation_R[0][2], correlation_R[0][3], asm_R[0][0], asm_R[0][1],
                                asm_R[0][2], asm_R[0][3], dissimilarity_R[0][0], dissimilarity_R[0][1],
                                dissimilarity_R[0][2], dissimilarity_R[0][3]])

    sample = np.float32(newarray)

    resp = loaded_model.predict(sample.reshape(1, -1))
    # 0 ---> Pne
    # 1 ---> fib
    # 2 ---> abnormal/other

    print(resp)

    return resp
