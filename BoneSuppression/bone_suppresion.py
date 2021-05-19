from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

# path = 'D:\\L4\\Level 4 group project\\data sets\\Segmented lung images\\Effusion\\pen tool\\black\\00000013_021.png'


def custom_loss(Y_clear, Y):
    alpha = 0.84
    # MSE
    mse = tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(Y_clear, Y), 1))
    # MS SSIM
    ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(Y_clear, Y, 1))
    # Mixed cost
    cost = alpha*ssim + (1 - alpha)*mse
    return cost


def MSE(Y_clear, Y):
    mse = tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(Y_clear, Y), 1))
    return mse


def SSIM(Y_clear, Y):
    ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(Y_clear, Y, 1))
    return ssim


def predict(path):
    filename = 'Bone_suppresion_V3.hdf5'
    # cv2.imshow('inside bone sup', path)
    # cv2.waitKey(0)
    image = []
    model = load_model(filename, custom_objects={'custom_loss': custom_loss, 'MSE': MSE, 'SSIM': SSIM})

    # image_single = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # image_single = cv2.resize(image_single, (1024, 1024))
    image_single = cv2.resize(path, (1024, 1024))
    image_single = image_single / 255
    image.append(image_single)
    image = np.reshape(image, (len(image), 1024, 1024, 1))

    decoded_img = model.predict(image)
    decoded_img = decoded_img * 255

    for i, img in enumerate(decoded_img):
        out = img.reshape(1024, 1024)

    out = np.array(out, dtype='uint8')

    return out
# output = predict(path)
# cv2.imshow('bone suppressed', output)
# cv2.waitKey(0)
# cv2.imwrite('bone suppressed effusion.png', output)
#
# print('output', output)