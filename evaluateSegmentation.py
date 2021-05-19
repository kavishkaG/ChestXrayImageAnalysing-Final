import cv2
import tkinter as tk
import glob
import csv
import os

from EvaluateSegmentation.lungSegmentation import lungSegment
from PreProcessing.convert_to_gray import convert_to_gray
from PreProcessing.enhanced_image import enhanced_image_For
from PreProcessing.resized_image import resized_image

from EvaluateSegmentation.segmentImageAccuracy import accuracyCalculator

header = ['Image Name', 'Accurate Count', 'Non Accurate Count', 'Accuracy']

segmentedImagesPath = glob.glob("E:\\EvaluateSegmentation\\Fibrosis\\segmented\\*.png")
withoutSegmentedImagesPath = "E:\\EvaluateSegmentation\\Fibrosis\\without\\"


for image in segmentedImagesPath:
    imageName = image.split(sep = "E:\\EvaluateSegmentation\\Fibrosis\\segmented\\", maxsplit=1)
    img = cv2.imread(withoutSegmentedImagesPath + imageName[1], cv2.IMREAD_UNCHANGED)
    segmentedImg = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    resized_img = resized_image(img, screen_height, screen_width)
    resized_evaluateImg = resized_image(segmentedImg, screen_height, screen_width)

    gray_img = convert_to_gray(resized_img)
    gray_evaluateImg = convert_to_gray(resized_evaluateImg)

    enhanced_image = enhanced_image_For(gray_img)

    noise_removal_gray_img = cv2.medianBlur(enhanced_image, 5)

    closing = lungSegment(gray_img, noise_removal_gray_img)

    th, otsuBinarizedImage = cv2.threshold(gray_evaluateImg, 1, 255, cv2.THRESH_OTSU)

    accuracy, accurateCount, nonAccurateCount = accuracyCalculator(closing[2], otsuBinarizedImage)

    data = [imageName[1], accurateCount, nonAccurateCount, accuracy]

    file_exists = os.path.isfile('E:\\EvaluateSegmentation\\EvaluateSegmentation.csv')
    with open('E:\\EvaluateSegmentation\\EvaluateSegmentation.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writerHeader = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=header)

        if not file_exists:
            writerHeader.writeheader()

        writer.writerow(data)

    # print(accuracy)

sumAccuracy = 0
rowCount = 0
with open('E:\\EvaluateSegmentation\\EvaluateSegmentation.csv', 'r')as csvFile:
    csvReader = csv.reader(csvFile)

    for row in csvReader:
        if row[0] != 'Image Name':
            sumAccuracy += float(row[3])
            rowCount += 1

print(sumAccuracy)
print(rowCount)
print((sumAccuracy/rowCount)*100)
