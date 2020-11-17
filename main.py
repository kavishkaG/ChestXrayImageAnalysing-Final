import numpy as np
import cv2
import tkinter as tk

from PreProcessing.convert_to_gray import convert_to_gray
from PreProcessing.drew_histogram import drew_histogram
from PreProcessing.enhanced_image import enhanced_image
from PreProcessing.resized_image import resized_image
from Segmentation.drew_boundary_edge import drew_boundary_edge
from Segmentation.segment_lungs import segment

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
print('screen resolution : ', screen_width, ',', screen_height)

# img = cv2.imread('C:\\Users\\Dell\\Desktop\\Effusion\\00000011_000.png', cv2.IMREAD_UNCHANGED)
# img = cv2.imread('C:\\Users\\Dell\\Desktop\\No Findings\\abcd.jpeg', cv2.IMREAD_UNCHANGED)
# img = cv2.imread('C:\\Users\\Dell\\Desktop\\No Findings\\00004531_002.png', cv2.IMREAD_UNCHANGED)
# img = cv2.imread('C:\\Users\\Dell\\Desktop\\TB\\CHNCXR_0335_1.png', cv2.IMREAD_UNCHANGED)
# img = cv2.imread('C:\\Users\\Dell\\Desktop\\Pneumonia\\00014234_000.png', cv2.IMREAD_UNCHANGED)
img = cv2.imread('C:\\Users\\Dell\\Desktop\\Fibrosis\\00000067_000.png', cv2.IMREAD_UNCHANGED)

resized_img = resized_image(img, screen_height, screen_width)

gray_img = convert_to_gray(resized_img)

enhanced_image = enhanced_image(gray_img)

drew_histogram(gray_img, enhanced_image)

noise_removal_gray_img = cv2.medianBlur(enhanced_image, 5)

thresh_img = cv2.adaptiveThreshold(noise_removal_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   699, 6)

kernel = np.ones((3, 3), np.uint8)
erosion_img = cv2.erode(thresh_img, kernel, iterations=1)

constant = drew_boundary_edge(thresh_img)

closing = segment(gray_img, constant, thresh_img)

cv2.waitKey(0)
