import cv2 as cv
import numpy as np
import os
import glob
from skimage.feature import hog
from PreProcessing.convert_to_gray import convert_to_gray
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import pickle
import pandas as pd


samples = []
glcm_samples = []
sample_new = []
labels = []

fibrosisPath = 'E:\\01 - Final Year Research\\DataSets\\SVM Train Dataset\\Fibrosis'
pneumoniaPath = 'E:\\01 - Final Year Research\\DataSets\\SVM Train Dataset\\Pneumonia'
other_path = 'E:\\01 - Final Year Research\\DataSets\\SVM Train Dataset\\Effusion'

for filename in glob.glob(os.path.join(other_path, '*.png')):
    img = cv.imread(filename)
    gray_img = convert_to_gray(img)
    dim = (64, 128)
    closing_new = cv.resize(gray_img, dim, interpolation=cv.INTER_AREA)
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

    samples.append(newarray)
    labels.append(2)


for filename in glob.glob(os.path.join(fibrosisPath, '*.png')):
    img = cv.imread(filename)
    gray_img = convert_to_gray(img)
    dim = (64, 128)
    closing_new = cv.resize(gray_img, dim, interpolation=cv.INTER_AREA)
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

    print('hist before', len(hist))

    newarray = np.append(hist, [contrast_R[0][0], contrast_R[0][1], contrast_R[0][2], contrast_R[0][3], energy_R[0][0], energy_R[0][1], energy_R[0][2], energy_R[0][3], homogeneity_R[0][0], homogeneity_R[0][1], homogeneity_R[0][2], homogeneity_R[0][3], correlation_R[0][0], correlation_R[0][1], correlation_R[0][2], correlation_R[0][3], asm_R[0][0], asm_R[0][1], asm_R[0][2], asm_R[0][3], dissimilarity_R[0][0], dissimilarity_R[0][1], dissimilarity_R[0][2], dissimilarity_R[0][3]])

    print('hist after', len(newarray))
    samples.append(newarray)
    labels.append(1)


for filename in glob.glob(os.path.join(pneumoniaPath, '*.png')):
    img = cv.imread(filename)
    gray_img = convert_to_gray(img)
    dim = (64, 128)
    closing_new = cv.resize(gray_img, dim, interpolation=cv.INTER_AREA)
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

    samples.append(newarray)
    labels.append(0)


print('samples', newarray)

samples = np.float32(samples)
labels = np.asarray(labels)

print('length feature', len(samples))
print('length labels', len(labels))

rand = np.random.RandomState(421)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]
# samples = np.random.shuffle(samples)
# labels = np.random.shuffle(labels)

X_train, X_test, Y_train, Y_test = train_test_split(samples, labels, test_size=0.1, random_state=4)

print(X_train.shape)

print(Y_train.shape)

print(X_test[0:1][:].shape)

print(Y_test.shape)

classifier = svm.SVC(kernel='linear', gamma='auto', C=2).fit(samples, labels)

y_pred = classifier.predict(X_test)

print("Accuracy: "+str(accuracy_score(Y_test, y_pred)))
print('\n')
print(classification_report(Y_test, y_pred))

result = pd.DataFrame({'original': Y_test, 'predicted': y_pred})

print('result', result)
# save the model to disk
filename = '../../finalized_model_for_both.sav'
pickle.dump(classifier, open(filename, 'wb'))