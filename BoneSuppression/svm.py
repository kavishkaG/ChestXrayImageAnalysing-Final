import cv2 as cv
import numpy as np
import os
import glob
from skimage.feature import hog
from PreProcessing.convert_to_gray import convert_to_gray
import csv

from ROI.findAngle import find_angle
from ROI.find_roi import find_ROI

#variables
samples = [] #for angles
labels = []  #for classes

# specify your paths to images

positive_path_1 = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Effusion'
negative_path = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\TB'

#Get positive samples effusion
for filename in glob.glob(os.path.join(positive_path_1, '*.png')):
    img = cv.imread(filename)
    gray_img = convert_to_gray(img)
    gray_img = cv.resize(gray_img, (512, 512))
    lungs = find_ROI(gray_img)
    angles = find_angle(lungs, gray_img)

    # dim = (64, 128)
    # closing_new = cv.resize(gray_img, dim, interpolation=cv.INTER_AREA)
    # hist, hog_image_right = hog(closing_new, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
    #                             visualize=True, transform_sqrt=True, block_norm='L2-Hys', multichannel=False)
    samples.append(angles)
    labels.append(1)


#Get negative samples
for filename in glob.glob(os.path.join(negative_path, '*.png')):
    img = cv.imread(filename)
    gray_img = convert_to_gray(img)
    gray_img = cv.resize(gray_img, (512, 512))
    lungs = find_ROI(gray_img)
    angles = find_angle(lungs, gray_img)
    # dim = (64, 128)
    # closing_new = cv.resize(gray_img, dim, interpolation=cv.INTER_AREA)
    # hist, hog_image_right = hog(closing_new, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
    #                             visualize=True, transform_sqrt=True, block_norm='L2-Hys', multichannel=False)

    samples.append(angles)
    labels.append(0)

samples = np.float32(samples)
labels = np.array(labels)

rand = np.random.RandomState(421)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]

svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setGamma(5.383)
svm.setC(2.67)

svm.train(samples, cv.ml.ROW_SAMPLE, labels)
svm.save('svm_effusion.dat')

# imageDir = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\img\\00012488_004.png'
# imageDir = "D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Effusion\\00000061_006.png"
# imageDir = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Effusion\\00004501_004.png'
imageDir = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Effusion\\00004525_007.png'
# imageDir = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Effusion\\00000099_009.png'
# imageDir = "D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Normal\\CHNCXR_0003_0.png"
#specify path to unclassified images
image = cv.imread(imageDir, cv.IMREAD_UNCHANGED) #read sample.jpg
gray_img = convert_to_gray(image)
gray_img = cv.resize(gray_img, (512,512))
# dim = (64, 128)
# closing_new = cv.resize(gray_img, dim, interpolation=cv.INTER_AREA)
cv.imshow('image', gray_img)

# #Wait for user close image window
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.waitKey(1)

svm = cv.ml.SVM_load('svm_effusion.dat')

lungs = find_ROI(gray_img)
angles = find_angle(lungs, gray_img)
# hist = hog(image)
# hist, hog_image_right = hog(closing_new, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
#                             visualize=True, transform_sqrt=True, block_norm='L2-Hys', multichannel=False)

sample = np.float32(angles)

resp = svm.predict(sample.reshape(1, -1))
print(resp[1].ravel()[0])

