import cv2
import tkinter as tk
import glob
import csv
import os

from EvaluateClassification.findROIAngle import findROIAngle
from PreProcessing.convert_to_gray import convert_to_gray
from PreProcessing.resized_image import resized_image

header = ['Image Name', 'Abnormal/Normal', 'Perpendicular Length Rt', 'Perpendicular Length Lt']

segmentedImagesPath = glob.glob("E:\\EvaluateSegmentation\\Fibrosis\\segmented\\*.png")

for image in segmentedImagesPath:

    evaluateImg = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    imageName = image.split(sep="E:\\EvaluateSegmentation\\Fibrosis\\segmented\\", maxsplit=1)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    resized_evaluateImg = resized_image(evaluateImg, screen_height, screen_width)

    gray_evaluateImg = convert_to_gray(resized_evaluateImg)

    th, otsuBinarizedImage = cv2.threshold(gray_evaluateImg, 1, 255, cv2.THRESH_OTSU)

    lungsInAccurateImage, lungsInAccurateImage_resized_img_with_hull, accurateImagePerpendicularLengthRt, accurateImagePerpendicularLengthLt = findROIAngle(
        otsuBinarizedImage, resized_evaluateImg)

    data = [imageName[1], 'Abnormal', accurateImagePerpendicularLengthRt, accurateImagePerpendicularLengthLt]

    file_exists = os.path.isfile('E:\\EvaluateSegmentation\\EvaluateRoiLength.csv')
    with open('E:\\EvaluateSegmentation\\EvaluateRoiLength.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writerHeader = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=header)

        if not file_exists:
            writerHeader.writeheader()

        writer.writerow(data)

