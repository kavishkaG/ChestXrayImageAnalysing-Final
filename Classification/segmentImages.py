import cv2
import glob

from PreProcessing.convert_to_gray import convert_to_gray
from PreProcessing.enhanced_image import enhanced_image_For
from Segmentation.segment_lungs import segment
from BoneSuppression.bone_suppresion import predict


imagesPath = glob.glob("E:\\01 - Final Year Research\\DataSets\\Normal\\Effusion\\*.png")

for image in imagesPath:

    print(image)

    imageName = image.split(sep="E:\\01 - Final Year Research\\DataSets\\Normal\\Effusion\\", maxsplit=1)

    print(imageName)

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    resized_img = cv2.resize(img, (512, 512))

    gray_img = convert_to_gray(resized_img)

    output = predict(gray_img)

    resizedBoneImg = cv2.resize(output, (512, 512))

    enhanced_image = enhanced_image_For(resizedBoneImg)

    noise_removal_gray_img = cv2.medianBlur(enhanced_image, 5)

    closing = segment(gray_img, noise_removal_gray_img)

    path = 'E:\\01 - Final Year Research\\DataSets\\SVM Train Dataset\\Effusion'

    cv2.imwrite(path + '\\' + imageName[1], closing[1])
