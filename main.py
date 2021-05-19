import numpy as np
import os
import cv2
import tkinter as tk
from werkzeug.utils import secure_filename

from Classification.SVM.fibrosisAndPneumonia import fibrosisOrPneumoniaSVM
from PreProcessing.convert_to_gray import convert_to_gray
from PreProcessing.enhanced_image import enhanced_image_For
from PreProcessing.resized_image import resized_image
from ROI.findAngle import find_angle
from Segmentation.segment_lungs import segment
from ROI.find_roi import find_ROI
from BoneSuppression.bone_suppresion import predict

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)

        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        # resized_img = resized_image(img, screen_height, screen_width)
        resized_img = cv2.resize(img, (512, 512))

        cv2.imshow('resized_img', resized_img)

        gray_img = convert_to_gray(resized_img)

        output = predict(gray_img)

        resizedBoneImg = cv2.resize(output, (512, 512))
        cv2.imshow('bone suppressed', resizedBoneImg)

        enhanced_image = enhanced_image_For(resizedBoneImg)

        cv2.imshow('enhanced_image', enhanced_image)

        noise_removal_gray_img = cv2.medianBlur(enhanced_image, 5)

        cv2.imshow('noise_removal_gray_img', noise_removal_gray_img)

        closing = segment(gray_img, noise_removal_gray_img)

        cv2.imshow('closing', closing[1])

        resized_img_with_hull, lungs, perpendicularLengthRt, perpendicularLengthLt = find_ROI(closing[2], resized_img)

        # lungsInAccurateImage, lungsInAccurateImage_resized_img_with_hull, accurateImagePerpendicularLengthRt, accurateImagePerpendicularLengthLt = find_ROI_angle(otsuBinarizedImage, resized_evaluateImg)

        if perpendicularLengthRt == 0 or perpendicularLengthLt == 0:
            costophrenicAnlge = find_angle(lungs, closing[1])
            if costophrenicAnlge != 30:
                return 'Effusion'
            else:
                response = fibrosisOrPneumoniaSVM(closing[1])
                if response == 0:
                    return 'Pneumonia'
                elif response == 1:
                    return 'fibrosis'
                else:
                    return 'Abnormal'

        else:
            if 10 < perpendicularLengthRt < 20 and 10 < perpendicularLengthLt < 20:
                return 'Normal'
            else:
                costophrenicAnlge = find_angle(lungs, closing[1])
                if costophrenicAnlge == 30:
                    return 'Effusion'
                else:
                    response = fibrosisOrPneumoniaSVM(closing[1])
                    if response == 0:
                        return 'Pneumonia'
                    elif response == 1:
                        return 'fibrosis'
                    else:
                        return 'Abnormal'

        cv2.waitKey(0)

    return None


if __name__ == "__main__":
    app.run(debug=True)

