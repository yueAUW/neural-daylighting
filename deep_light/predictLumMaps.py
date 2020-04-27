#
#
#  Copyright (c) 2020.  Yue Liu
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  If you find this code useful please cite:
#  Predicting Annual Equirectangular Panoramic Luminance Maps Using Deep Neural Networks,
#  Yue Liu, Alex Colburn and and Mehlika Inanici.  16th IBPSA International Conference and Exhibition, Building Simulation 2019.
#
#
#



from __future__ import absolute_import, division, print_function

import os
import threading
import time

import math
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import model_from_json

from deep_light import genData, plot
from deep_light.genData import get_data_path

from deep_light.pano2fish_lum import save_im_falsecolor, plotDGPAnalysis_whole, plot_DGPvsSC_Analysis, plotLog10MSEAnalysis, \
    plotLog10RERAnalysis, plotLog10MSEAnalysis01, plotLog10RERAnalysis01

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tf.compat.v1.disable_eager_execution()

matplotlib.use('Agg')

#

# from RAMMG import plotRAMMGAnalysis, getAllRAMMGs

VAL_PERCENTAGE = 0.2
SHUFFLE = True
COLORMAP = "jet"

NUM_NODES = 600
# TRAIN VARIABLES
MIN_LR = 0.0000000000001
NUM_EPOCHS = 30  # 30
INITIAL_BATCH_SIZE = 6  # 6
BATCH_REDUCTIONS = 4  # 4
MINI_BATCHSIZE = 1  # 1
TARGET_LOSS = 1e-5
LOSS_VAR = 10

IMAGE_SIZE = (230, 115)
# display compensation
EPSILON = 0  # 0.2


def hdr_to_image(lu, gamma):
    g = 1.0 / gamma
    lu = np.power(lu, g)

    return lu


def image_to_hdr(lu, gamma):
    g = 1.0 / gamma
    lu = np.power(lu, 1.0 / g)

    return lu


# restore the normalized value to the original luminance value


def restore_Normalization(output_dim, im, type, gamma):
    # first pair of min and max when conduct 1st normalization
    file = open("data/processed_data/" + type + 'restoration_min_max.txt', 'r')
    data = file.read().split(',')

    min_1 = float(data[0])
    max_1 = float(data[1])

    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    # restore: 1) reverse restore the normalization by multiple the max, 2)
    # reverse gamma correction, 3) reverse normalization restore

    im[:, :, -output_dim:] = image_to_hdr(im[:, :, -output_dim:], gamma)

    im[:, :, -output_dim:] = im[:, :, -output_dim:] * (max_1 - min_1) + min_1

    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
    return im


def reverse_angle_correction(py, p_lum):
    return p_lum / \
           ((math.cos(math.pi * (py - IMAGE_SIZE[1] / 2) / IMAGE_SIZE[1]) + EPSILON) / (1 + EPSILON))
    # return p_lum / \
    # ((math.cos(math.pi * (py - IMAGE_SIZE[1] / 2) / IMAGE_SIZE[1]) + EPSILON)/(1+ EPSILON))


def reverse_angle_correction_im(im):
    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        x, y = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        im[x, y, :] = reverse_angle_correction(y, im[x, y, :])
    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
    return im


def post_process_angle(n_im, p_im):
    n_im = reverse_angle_correction_im(n_im)
    p_im = reverse_angle_correction_im(p_im)
    diff = abs(p_im - n_im)
    return diff, n_im, p_im


def post_process_log(n_im, p_im, log_base):
    n_im = reverse_log_processing(n_im, log_base)
    p_im = reverse_log_processing(p_im, log_base)
    diff = abs(p_im - n_im)
    return diff, n_im, p_im


def reverse_log_processing(im, log_base):
    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        x, y = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        im[x, y, :] = np.power(log_base, im[x, y, :])
    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
    return im


######################## Result Analysis Functions ####################

# function to display the comparison between ground truth, predicted and
# difference images on 1 image


def saveComparisonImage(
        error,
        i,
        n_im,
        p_im,
        diff,
        type,
        mse,
        dir,
        dif,
        al,
        az,
        sc_p,
        sc_t,
        sc_diff,
        solid_angle_mse,
        solid_angle_rer,
        gradient,
        vmax):
    # setup the figure
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(
        type +
        " Image Comparison: %.d, Altitude %d, Azimuth %d, Direct Irradiance %.d, Diffuse Irradiance %d," %
        (i,
         al,
         az,
         dir,
         dif))
    plt.figtext(.5, .93,
                " Average Error Rate: %.4f, MSE: %.5e, Solid Angle Weighted MSE: %.5e, Solid Angle Weighted RER: %.4f"
                % (error, mse, solid_angle_mse, solid_angle_rer), fontsize=10, ha='center')
    plt.figtext(.5, .91,
                " Spatial Contrast(SC) Error: %.5e, Predicted SC: %.5e, Truth SC: %.5e, Gradient(sobel) MAE: %.5e"
                % (sc_diff, np.mean(sc_p), np.mean(sc_t), gradient), fontsize=10, ha='center')
    plt.axis("off")

    n_im = np.clip(n_im, 1, vmax)
    p_im = np.clip(p_im, 1, vmax)
    diff = np.clip(diff, 1, vmax)

    # possible color map options
    # jet? gnuplot2? nipy_spectral, CMRmap
    colormap = COLORMAP
    fig.add_subplot(3, 1, 1)
    implot1 = plt.imshow(n_im, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Truth")
    plt.axis("off")

    # show the predicted image
    fig.add_subplot(3, 1, 2)
    implot1 = plt.imshow(p_im, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Prediction")
    plt.axis("off")

    # show the difference image
    ax = fig.add_subplot(3, 1, 3)
    implot1 = plt.imshow(diff, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Absolute Difference")
    plt.axis("off")

    # add color index bar
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])

    if (vmax == 3000):
        formatter = LogFormatter(3, labelOnlyBase=False)
        cb = plt.colorbar(
            ticks=[
                1,
                4.1,
                12.2,
                36.5,
                109,
                325.8,
                973.4,
                3000],
            format=formatter,
            cax=cbaxes)

    else:
        formatter = LogFormatter(3.7, labelOnlyBase=False)
        cb = plt.colorbar(
            ticks=[
                1,
                3.7,
                14,
                55,
                204,
                755,
                2794,
                10000],
            format=formatter,
            cax=cbaxes)

    cb.set_label('Luminance (cd/m2)', rotation=90)

    # save the image
    plt.savefig(
        os.path.join(
            "results/result_combo/" +
            str(al) +
            "_" +
            str(az) +
            "_" +
            str(dir) +
            "_" +
            str(dif) +
            "_" +
            type +
            "_" +
            str(vmax) +
            ".png"))
    plt.close()


def save_sc_ims(
        sc_p,
        sc_t,
        al,
        az,
        dir,
        dif,
        solid_angle_rer,
        solid_angle_mse,
        vmax):
    sc_one_p = np.mean(sc_p)
    sc_one_t = np.mean(sc_t)
    sc_error = np.mean(abs(sc_p - sc_t))
    sc_mse = np.mean(np.square(sc_p - sc_t))
    # setup the figure
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(
        "Image Comparison: Altitude %d, Azimuth %d, Direct Irradiance %.d, Diffuse Irradiance %d," %
        (al, az, dir, dif))
    plt.figtext(.5, .93, " Solid Angle Weighted MSE: %.5e, Solid Angle Weighted RER: %.4f, "
                         "Predicted SC : %.4f, Truth SC: %.4f, SC MAE: %.4f, SC_MSE: %.4e" %
                (solid_angle_mse, solid_angle_rer, sc_one_p, sc_one_t, sc_error, sc_mse), fontsize=6, ha='center')

    plt.axis("off")

    n_im = np.clip(sc_t, 1, vmax)
    p_im = np.clip(sc_p, 1, vmax)
    diff = np.clip(abs(sc_t - sc_p), 1, vmax)

    # possible color map options
    # jet? gnuplot2? nipy_spectral, CMRmap
    colormap = COLORMAP
    fig.add_subplot(3, 1, 1)
    implot1 = plt.imshow(n_im, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Truth")
    plt.axis("off")

    # show the predicted image
    fig.add_subplot(3, 1, 2)
    implot1 = plt.imshow(p_im, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Prediction")
    plt.axis("off")

    # show the difference image
    ax = fig.add_subplot(3, 1, 3)
    implot1 = plt.imshow(diff, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Absolute Difference")
    plt.axis("off")

    # add color index bar
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])

    if (vmax == 3000):
        formatter = LogFormatter(3, labelOnlyBase=False)
        cb = plt.colorbar(
            ticks=[
                1,
                4.1,
                12.2,
                36.5,
                109,
                325.8,
                973.4,
                3000],
            format=formatter,
            cax=cbaxes)

    else:
        formatter = LogFormatter(3.7, labelOnlyBase=False)
        cb = plt.colorbar(
            ticks=[
                1,
                3.7,
                14,
                55,
                204,
                755,
                2794,
                10000],
            format=formatter,
            cax=cbaxes)

    cb.set_label('Luminance (cd/m2)', rotation=90)

    if (os.path.exists("results/result_combo_SC/") == False):
        os.mkdir("results/result_combo_SC/")
    # save the image
    plt.savefig(
        os.path.join(
            "results/result_combo_SC/" +
            str(al) +
            "_" +
            str(az) +
            "_" +
            str(dir) +
            "_" +
            str(dif) +
            "_" +
            str(vmax) +
            ".png"))
    plt.close()


# save gradient images


def save_gradient_ims(
        sobel_p,
        sobel_n,
        solid_angle_mse,
        solid_angle_rer,
        al,
        az,
        dir,
        dif,
        vmax):
    sobel_mse = np.mean(np.square(sobel_p - sobel_n))
    sobel_mae = np.mean(abs(sobel_p - sobel_n))
    # setup the figure
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(
        "Image Comparison: Altitude %d, Azimuth %d, Direct Irradiance %.d, Diffuse Irradiance %d," %
        (al, az, dir, dif))
    plt.figtext(.5, .93, " Solid Angle Weighted MSE: %.5e, Solid Angle Weighted RER: %.4f, "
                         "Sobel MAE: %.4f, Sobel MSE: %.4e" %
                (solid_angle_mse, solid_angle_rer, sobel_mae, sobel_mse), fontsize=10, ha='center')

    plt.axis("off")

    n_im = np.clip(sobel_n, 1, vmax)
    p_im = np.clip(sobel_p, 1, vmax)
    diff = np.clip(abs(sobel_n - sobel_p), 1, vmax)

    # possible color map options
    # jet? gnuplot2? nipy_spectral, CMRmap
    colormap = COLORMAP
    fig.add_subplot(3, 1, 1)
    implot1 = plt.imshow(n_im, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Truth")
    plt.axis("off")

    # show the predicted image
    fig.add_subplot(3, 1, 2)
    implot1 = plt.imshow(p_im, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Prediction")
    plt.axis("off")

    # show the difference image
    ax = fig.add_subplot(3, 1, 3)
    implot1 = plt.imshow(diff, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
    plt.title("Absolute Difference")
    plt.axis("off")

    # add color index bar
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])

    if (vmax == 3000):
        formatter = LogFormatter(3, labelOnlyBase=False)
        cb = plt.colorbar(
            ticks=[
                1,
                4.1,
                12.2,
                36.5,
                109,
                325.8,
                973.4,
                3000],
            format=formatter,
            cax=cbaxes)

    else:
        formatter = LogFormatter(3.7, labelOnlyBase=False)
        cb = plt.colorbar(
            ticks=[
                1,
                3.7,
                14,
                55,
                204,
                755,
                2794,
                10000],
            format=formatter,
            cax=cbaxes)

    cb.set_label('Luminance (cd/m2)', rotation=90)

    if (os.path.exists("results/result_combo_Sobel/") == False):
        os.mkdir("results/result_combo_Sobel/")
    # save the image
    plt.savefig(
        os.path.join(
            "results/result_combo_Sobel/" +
            str(al) +
            "_" +
            str(az) +
            "_" +
            str(dir) +
            "_" +
            str(dif) +
            "_" +
            str(vmax) +
            ".png"))
    plt.close()


# function to display the comparison between ground truth, predicted and
# difference images seperately


def saveComparisonImageSeperate(i, n_im, p_im, diff, type, dir, dif, al, az):
    p_im = p_im.reshape(IMAGE_SIZE[0] * IMAGE_SIZE[1])
    n_im = n_im.reshape(IMAGE_SIZE[0] * IMAGE_SIZE[1])

    p_im = p_im / 179
    n_im = n_im / 179

    p_im_1 = np.array([p_im, p_im, p_im])
    n_im_1 = np.array([n_im, n_im, n_im])

    p_im_2 = p_im_1.T
    n_im_2 = n_im_1.T

    if (os.path.exists("result_seperate/")) == False:
        os.makedirs("result_seperate/")
    np.savetxt(
        os.path.join(
            "results/result_seperate/" +
            str(al) +
            "_" +
            str(az) +
            "_" +
            str(dir) +
            "_" +
            str(dif) +
            "_" +
            "prediction_" +
            ".txt"),
        p_im_2)
    np.savetxt(
        os.path.join(
            "results/result_seperate/" +
            str(al) +
            "_" +
            str(az) +
            "_" +
            str(dir) +
            "_" +
            str(dif) +
            "_" +
            "prediction_" +
            ".txt"),
        n_im_2)


"""
def plotSampleDistribution(train,val,test):
    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX =getMaxMinPrams("train_")
    train_dir =restorNorm(train[:, 0, 4], DIR_MAX, DIR_MIN)
    train_dif =restorNorm(train[:, 0, 5], DIF_MAX, DIF_MIN)
    train_al =restorNorm(train[:, 0, 2],AL_MAX, AL_MIN)
    train_az =restorNorm(train[:, 0, 3],AZ_MAX, AZ_MIN)

    val_dir =restorNorm(val[:, 0, 4], DIR_MAX, DIR_MIN)
    val_dif =restorNorm(val[:, 0, 5], DIF_MAX, DIF_MIN)
    val_al =restorNorm(val[:, 0, 2],AL_MAX, AL_MIN)
    val_az =restorNorm(val[:, 0, 3],AZ_MAX, AZ_MIN)

    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX =getMaxMinPrams("test_")
    test_dir =restorNorm(test[:, 0, 4], DIR_MAX, DIR_MIN)
    test_dif =restorNorm(test[:, 0, 5], DIF_MAX, DIF_MIN)
    test_al =restorNorm(test[:, 0, 2],AL_MAX, AL_MIN)
    test_az =restorNorm(test[:, 0, 3],AZ_MAX, AZ_MIN)

    fig = plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(train_dir, train_dif, s=10, color='r', label ="train")
    plt.scatter(val_dir, val_dif, s=10, color='g', label ="validation")
    plt.scatter(test_dir, test_dif, s=1, color='b', label ="test")
    plt.title("Sky Direct and Diffuse Irradiances Distribution", size = 16)
    plt.xlabel('Direct')
    plt.ylabel('Diffuse')
    plt.legend(fancybox=True)
    if (os.path.exists("results/result_combo/")) == False:
        os.makedirs("results/result_combo/")
    plt.savefig('results/result_combo/sky.png')
    plt.close()

    fig = plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(train_al, train_az, s=10, color='r', label ="train")
    plt.scatter(val_al, val_az, s=10, color='g', label ="validation")
    plt.scatter(test_al, test_az, s=1, color='b', label ="test")
    plt.title("Sun Altitude and Azimuth Distribution", size = 16)
    plt.xlabel('Altitude')
    plt.ylabel('Azimuth')
    plt.legend(fancybox=True)
    plt.savefig('results/result_combo/sun.png')
    plt.close()

"""


def plotSampleDistribution(train, val, test):
    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX = getMaxMinPrams(
        "train_")
    train_dir = restorNorm(train[:, 0, 4], DIR_MAX, DIR_MIN)
    train_dif = restorNorm(train[:, 0, 5], DIF_MAX, DIF_MIN)
    train_al = restorNorm(train[:, 0, 2], AL_MAX, AL_MIN)
    train_az = restorNorm(train[:, 0, 3], AZ_MAX, AZ_MIN)

    val_dir = restorNorm(val[:, 0, 4], DIR_MAX, DIR_MIN)
    val_dif = restorNorm(val[:, 0, 5], DIF_MAX, DIF_MIN)
    val_al = restorNorm(val[:, 0, 2], AL_MAX, AL_MIN)
    val_az = restorNorm(val[:, 0, 3], AZ_MAX, AZ_MIN)

    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX = getMaxMinPrams(
        "test_")
    test_dir = restorNorm(test[:, 0, 4], DIR_MAX, DIR_MIN)
    test_dif = restorNorm(test[:, 0, 5], DIF_MAX, DIF_MIN)
    test_al = restorNorm(test[:, 0, 2], AL_MAX, AL_MIN)
    test_az = restorNorm(test[:, 0, 3], AZ_MAX, AZ_MIN)

    fig = plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(train_dir, train_dif, s=10, color='black')
    plt.scatter(val_dir, val_dif, s=10, color='black')
    plt.title("Sky Direct and Diffuse Irradiances Distribution", size=16)
    plt.xlabel('Direct')
    plt.ylabel('Diffuse')
    plt.legend(fancybox=True)
    if (os.path.exists("results/result_combo/")) == False:
        os.makedirs("results/result_combo/")
    plt.savefig('results/result_combo/sky.png')
    plt.close()

    fig = plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(train_al, train_az, s=10, color='black')
    plt.scatter(val_al, val_az, s=10, color='black')
    plt.title("Sun Altitude and Azimuth Distribution", size=16)
    plt.xlabel('Altitude')
    plt.ylabel('Azimuth')
    plt.legend(fancybox=True)
    plt.savefig('results/result_combo/sun.png')
    plt.close()


#################### call back functions ####################
class printbatch(callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        if batch % 10 == 0:
            print("Batch " + str(batch) + " ends")
            # print(logs)

    def on_epoch_begin(self, epoch, logs={}):
        print(logs)

    def on_epoch_end(self, epoch, logs={}):
        print(logs)


class AttentionLoss(callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    # customize your behavior

    def on_epoch_end(self, epoch, logs={}):
        print("alpha:" + str(K.get_value(self.alpha)))
        print("beta:" + str(K.get_value(self.beta)))
        if epoch > 2:  # 30
            print("yes, epoch")
            K.set_value(self.alpha, K.get_value(self.alpha) - 0.05)
            K.set_value(self.beta, K.get_value(self.beta) + 0.05)  # 0.05


#################### Now make the data generator threadsafe ##############


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def trainGenerator_Dense(input_dim, x_full, x_sun, y, batch_size):
    batch_size_pixels = batch_size * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    num_examples = x_full.shape[0] * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    steps_per_epoch = int(num_examples / batch_size_pixels)

    if num_examples < batch_size_pixels:
        batch_size_pixels = num_examples

    index_array = np.arange(num_examples)
    x_full = x_full.reshape(num_examples, input_dim - 1)
    x_sun = x_sun.reshape(num_examples, 1)
    y = y.reshape(num_examples, 1)

    while True:
        np.random.shuffle(index_array)
        start = 0
        for i in range(steps_per_epoch):
            if start + batch_size_pixels > num_examples:
                start = 0
                np.random.shuffle(index_array)
            end = start + batch_size_pixels
            idx = index_array[start: end]
            x_room_b = x_full[idx, :]
            x_sun_b = x_sun[idx, :]
            y_b = y[idx, :]
            start += batch_size_pixels
            yield [x_sun_b, x_room_b], y_b


def testGenerator_Dense(input_dim, x_full, x_sun, batch_size):
    num_examples = x_full.shape[0] * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    batch_size_pixels = batch_size * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    steps_per_epoch = int(num_examples / batch_size_pixels)

    while True:
        for i in range(steps_per_epoch):
            x_room_b = x_full[i, :, :, :].reshape(
                batch_size_pixels, input_dim - 1)
            x_sun_b = x_sun[i, :, :, :].reshape(batch_size_pixels, 1)
            yield [x_sun_b, x_room_b]


@threadsafe_generator
def trainGenerator_CNN(x_full, x_sun, y, batch_size):
    num_examples = x_full.shape[0]
    steps_per_epoch = int(num_examples / batch_size)

    if num_examples < batch_size:
        batch_size = num_examples

    index_array = np.arange(num_examples)

    while True:
        np.random.shuffle(index_array)
        start = 0
        for i in range(steps_per_epoch):
            if start + batch_size > num_examples:
                start = 0
                np.random.shuffle(index_array)
            end = start + batch_size
            idx = index_array[start:end]
            x_full_b = x_full[idx, :]
            x_sun_b = x_sun[idx, :]
            y_b = y[idx, :]

            start += batch_size
            yield [x_sun_b, x_full_b], y_b


def testGenerator_CNN(x_full, x_sun, batch_size):
    num_examples = x_full.shape[0]
    steps_per_epoch = int(num_examples / batch_size)

    if num_examples < batch_size:
        batch_size = num_examples

    index_array = np.arange(num_examples)

    while True:
        start = 0
        for i in range(steps_per_epoch):
            end = start + batch_size
            idx = index_array[start:end]
            x_full_b = x_full[idx, :]
            x_sun_b = x_sun[idx, :]
            start += batch_size
            yield [x_sun_b, x_full_b]


# dipslay the image and check the data
# reshape the data to the format of (width, depth and rgb)
def genSampleImage(output_dim, data, index, type):
    # dipslay the image and check the data(could be commented)
    # reshape the data to the format of (width, depth and rgb)
    NUM_IM = 0
    im = data[NUM_IM, :, index]
    im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], output_dim)
    if output_dim == 1:
        im = im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])

    # save the image sample to check before proceed
    print(im.shape)
    sample_image = plt.imshow(im)
    plt.savefig(os.path.join("test_pics/sample_image_" + type + ".png"))
    plt.close()


########## Data generator is now threadsafe and should work with multiple


def loadData(input_dim, output_dim):
    # load train and validation data
    train, val, test = genData.gen_training_data(VAL_PERCENTAGE, SHUFFLE)
    print("Total number of images in training set: ", train.shape)
    print("Total number of images in validation set: ", val.shape)
    print("Total number of images in test set: ", test.shape)
    plotSampleDistribution(train, val, test)
    plotDataSampleDistribution(train, val, test)

    genSampleImage(output_dim, train, -1, "full")
    genSampleImage(output_dim, train, -2, "sun")

    # construct input and output of train and validation data
    train_unwrapped = train.reshape(
        train.shape[0],
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        input_dim + output_dim)
    x_total = train_unwrapped[..., :input_dim - 1]
    x_sun_total = train_unwrapped[..., -2]
    x_sun_total = x_sun_total.reshape(
        train.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    y_total = train_unwrapped[..., -1]
    y_total = y_total.reshape(train.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    val_unwrapped = val.reshape(
        val.shape[0],
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        input_dim + output_dim)
    x_val = val_unwrapped[..., :input_dim - 1]
    x_sun_val = val_unwrapped[..., -2]
    x_sun_val = x_sun_val.reshape(
        val.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    y_val = val_unwrapped[..., -1]
    y_val = y_val.reshape(val.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    test_unwrapped = test.reshape(
        test.shape[0],
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        input_dim + output_dim)
    x_test = test_unwrapped[..., :input_dim - 1]
    x_sun_test = test_unwrapped[..., -2]
    x_sun_test = x_sun_test.reshape(
        test.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    y_test = test_unwrapped[..., -1]
    y_test = y_test.reshape(test.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    print("done")
    print(x_total.shape)
    print(y_total.shape)
    print(x_sun_total.shape)
    return x_total, x_sun_total, y_total, \
           x_val, x_sun_val, y_val, \
           x_test, x_sun_test, y_test


########### evaluation functions ####################
# create a solid angle weight for loss calculation purposes


def get_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def get_log10_mse(y_true, y_pred):
    return np.mean(np.square(np.log10(y_true) - np.log10(y_pred)))


def get_log10_relative_error(y_true, y_pred):
    sum_err_sqr = np.sum(np.square(np.log10(y_true) - np.log10(y_pred)))
    sum_val_sqr = np.sum(np.square(np.log10(y_true)))
    return (np.sqrt(sum_err_sqr / sum_val_sqr))


def get_rmse(y_true, y_pred):
    return (np.sqrt(np.mean(np.square(y_true - y_pred)))) / np.mean(y_true)


def get_relative_error(y_true, y_pred):
    sum_err_sqr = np.sum(np.square(y_true - y_pred))
    sum_val_sqr = np.sum(np.square(y_true))
    return (np.sqrt(sum_err_sqr / sum_val_sqr))


def get_relative_error_pixel(y_true, y_pred):
    err_sqr = np.square(y_true - y_pred)
    val_sqr = max(np.square(y_true), 0.0000000001)
    return np.sqrt(np.sum(err_sqr / val_sqr))


def im_angle():
    image_frame = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype='float32')
    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        index1, index2 = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        image_frame[index1][index2] = angle_correction(index2)
    return image_frame.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def angle_correction(py):
    return math.cos(
        math.pi * (py - IMAGE_SIZE[1] / 2) / IMAGE_SIZE[1]) + EPSILON


def get_solid_angle_mse(y_true, y_pred):
    return np.mean(np.square(y_pred * im_angle() - y_true * im_angle()))


def get_solid_angle_rer(y_true, y_pred):
    return get_relative_error(y_true * im_angle(), y_pred * im_angle())


"""
#get spatial contrast one number value
def spatial_contrast_one_im(im):
    sum = 0;
    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        x, y = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        #print(str(x) + ", " + str(y))
        sc = (abs(im[x, y] - im[max(x - 1, 0), y]) + abs(im[x, y] - im[min(x + 1, IMAGE_SIZE[0]-1), y])
              + abs(im[x, y] - im[x, max(0, y - 1)]) + abs(im[x, y] - im[x, min(y + 1, IMAGE_SIZE[1]-1)]))/4
        sum = sum + sc;
    return sum / (IMAGE_SIZE[0]*IMAGE_SIZE[1])

def spatial_contrast_eval(ims):
    # spatial contrast, too slow to add to the loss
    sum = 0;
    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        x, y = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        # print(str(x) + ", " + str(y))
        sc = (abs(ims[:, x, y] - ims[:, max(x - 1, 0), y]) + abs(
            ims[:, x, y] - ims[:, min(x + 1, IMAGE_SIZE[0] - 1), y])
                + abs(ims[:, x, y] - ims[:, x, max(0, y - 1)]) + abs(
                    ims[:, x, y] - ims[:, x, min(y + 1, IMAGE_SIZE[1] - 1)])) / 4
        sum = sum + sc;
    return sum / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
"""


# pixel wise spatial contrast evaluation, return a spatial contrast im.


def spatial_contrast_one_im(im):
    pad_ims = np.pad(im, [1, 1], mode="constant")
    sc_im = (abs(im - pad_ims[0:-2, 1:-1]) +
             abs(im - pad_ims[1:-1, 0:-2]) +
             abs(im - pad_ims[2:, 1:-1]) +
             abs(im - pad_ims[1:-1, 2:])) / 4
    return sc_im


# pixel wise spatial contrast evaluation, return spatial contrast ims.


def spatial_contrast_eval(ims):
    pad = ((0, 0), (1, 1), (1, 1), (0, 0))
    pad_ims = np.pad(ims, pad_width=pad, mode="constant", constant_values=0)
    sc_im = (abs(ims - pad_ims[:, 0:-2, 1:-1, :]) +
             abs(ims - pad_ims[:, 1:-1, 0:-2, :]) +
             abs(ims - pad_ims[:, 2:, 1:-1, :]) +
             abs(ims - pad_ims[:, 1:-1, 2:, :])) / 4
    return sc_im


def get_spatial_contrast_MAE(y_true, y_pred):
    return np.mean(abs(spatial_contrast_eval(
        y_true) - spatial_contrast_eval(y_pred)))


def get_spatial_contrast_RER(y_true, y_pred):
    return get_relative_error(
        spatial_contrast_eval(y_pred),
        spatial_contrast_eval(y_true))


def get_gradient_ims(img):
    from scipy import ndimage
    # Get x-gradient
    sx = ndimage.sobel(img, axis=1, mode='constant')
    # Get y-gradient
    sy = ndimage.sobel(img, axis=2, mode='constant')
    # Get square root of sum of squares
    sobel = np.hypot(sx, sy)
    return sobel


def get_gradient_one_im(im):
    from scipy import ndimage
    # Get x-gradient
    sx = ndimage.sobel(im, axis=0, mode='constant')
    # Get y-gradient
    sy = ndimage.sobel(im, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel = np.hypot(sx, sy)
    return sobel


def get_gradient_MAE(y_true, y_pred):
    return np.mean(abs(get_gradient_ims(y_true) - get_gradient_ims(y_pred)))


################ loss functions ############
# define custom loss function, mse of rgb and luminance
# rer


def relative_err(y_true, y_pred):
    sum_err_sqr = K.sum(K.square(y_true - y_pred))
    sum_val_sqr = K.clip(K.sum(K.square(y_true)), K.epsilon(), None)
    return (K.sqrt(sum_err_sqr / sum_val_sqr))


def relative_err_pixel(y_true, y_pred):
    err_sqr = K.square(y_true - y_pred)
    val_sqr = K.clip(K.square(y_true), K.epsilon(), None)
    return K.sqrt(K.sum(err_sqr / val_sqr))


# mse


def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# mse log


def mse_log(y_true, y_pred):
    return losses.mean_squared_logarithmic_error(y_true, y_pred)


# combination of mse and rer


def mse_rer_loss(y_true, y_pred):
    mse_lum = K.mean(K.square(y_pred - y_true), axis=-1)
    return mse_lum + relative_err(y_true, y_pred) * LOSS_VAR


# combination of mse and rer


def mse_rer_loss_old(y_true, y_pred):
    mse_lum = K.mean(K.square(y_pred - y_true), axis=-1)
    return mse_lum + relative_err(y_true, y_pred) * \
           LOSS_VAR + 10 * mse_log(y_true, y_pred)


def reverse_math_version(x, gamma):
    r_g = x ** gamma
    r_n = r_g * math.log(10 ** 7.8, math.e)
    r_l = (math.e ** r_n) - 1
    n = r_l / (10 ** 7.8)
    g = n ** (1 / gamma)
    return g


def reverse(x):
    r_g = K.pow(x, 4)
    r_n = r_g * math.log(10 ** 7.8, math.e)
    r_l = K.exp(r_n) - 1
    n = r_l / (10 ** 7.8)
    g = K.sqrt(K.sqrt(n))
    return g


# combination of mse and rer


def mse_rer_loss_handle_log(y_true, y_pred):
    y_true1 = reverse(y_true)
    y_pred1 = reverse(y_pred)
    return mse_rer_loss_old(y_true1, y_pred1)


# solid angle weighted mse


def solid_angle_mse(y_true, y_pred):
    return K.mean(K.square(y_pred * im_angle() - y_true * im_angle()), axis=-1)


# solid angle weighted rer


def solid_anlge_mse_rer_loss(y_true, y_pred):
    return mse_rer_loss(y_true * im_angle(), y_pred * im_angle())


# gradient sobel filter


def expandedSobel(inputTensor):
    sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                              [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                              [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])
    # this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(
        inputTensor[0, 0, 0, :]), (1, 1, -1, 1))
    # if you're using 'channels_first', use inputTensor[0,:,0,0] above
    # inputChannels = K.reshape(K.ones_like(inputTensor[0,:,0,0]),(1,1,-1,1))

    return sobelFilter * inputChannels


# gradient sobel loss


def sobelLoss(yTrue, yPred):
    # get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)

    # calculate the sobel filters for yTrue and yPred
    # this generates twice the number of input channels
    # a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue, filt)
    sobelPred = K.depthwise_conv2d(yPred, filt)

    # now you just apply the mse:
    return K.mean(abs(sobelTrue - sobelPred))


# combination of solid angle weighted mse, rer and gradient loss


def solid_angle_mse_rer_and_gradient_loss(y_true, y_pred):
    return 0.1 * mse_loss(y_true * im_angle(),
                          y_pred * im_angle()) + 0.8 * relative_err(y_true * im_angle(),
                                                                    y_pred * im_angle()) + 10 * sobelLoss(
        y_true * im_angle(),
        y_pred * im_angle())


# spatial contrast


def spatial_contrast(ims):
    # ims_new = tf.reshape(ims, [ims.shape[0], ims.shape[1], ims.shape[2]])
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    pad_ims = tf.pad(ims, paddings, "REFLECT")
    sc_im1 = K.abs(ims - pad_ims[:, :-2, 1:-1, :])
    sc_im2 = K.abs(ims - pad_ims[:, 1:-1, :-2, :])
    sc_im3 = K.abs(ims - pad_ims[:, 2:, 1:-1, :])
    sc_im4 = K.abs(ims - pad_ims[:, 1:-1, 2:, :])
    return (sc_im1 + sc_im2 + sc_im3 + sc_im4) / 4


def spatial_contrast_rer_loss(y_true, y_pred):
    sc_im_true = spatial_contrast(y_true)
    sc_im_pred = spatial_contrast(y_pred)

    sum_err_sqr = K.sum(K.square(sc_im_true - sc_im_pred))
    sum_val_sqr = K.clip(K.sum(K.square(sc_im_true)), K.epsilon(), None)
    return (K.sqrt(sum_err_sqr / sum_val_sqr))


def spatial_contrast_mae_loss(y_true, y_pred):
    sc_im_true = spatial_contrast(y_true)
    sc_im_pred = spatial_contrast(y_pred)
    return K.mean(abs(sc_im_true - sc_im_pred))


def solid_angle_mse_rer_and_spatial_contrast_loss(y_true, y_pred):
    return 0.8 * solid_anlge_mse_rer_loss(y_true,
                                          y_pred) + 1 * spatial_contrast_mae_loss(y_true,
                                                                                  y_pred)


############end of loss functions#############


def two_inputs_CNN_model(input_dim, output_dim):
    # first input model:sun patch images
    visible1 = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1,))
    conv11 = Conv2D(200, kernel_size=1, activation='relu')(visible1)

    # second input model:full light images
    visible2 = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], input_dim - 1,))
    conv21 = Conv2D(600, kernel_size=1, activation='relu')(visible2)
    conv22 = Conv2D(600, kernel_size=1, activation='relu')(conv21)
    conv23 = Conv2D(600, kernel_size=1, activation='relu')(conv22)
    conv24 = Conv2D(600, kernel_size=1, activation='relu')(conv23)

    # merge input models
    merge = concatenate([conv11, conv24])
    # interpretation model
    conv1 = Conv2D(600, kernel_size=1, activation='relu')(merge)
    output = Conv2D(output_dim, kernel_size=1, activation='relu')(conv1)

    model = Model(inputs=[visible1, visible2], outputs=output)

    # print(model.summary())

    return model


def two_inputs_dense_model(input_dim, output_dim):
    # first input model:sun patch images
    visible1 = Input(shape=(1,))
    dense11 = Dense(600, activation='relu')(visible1)

    # second input model:full light images
    visible2 = Input(shape=(input_dim - 1,))
    dense21 = Dense(600, activation='relu')(visible2)
    dense22 = Dense(600, activation='relu')(dense21)
    dense23 = Dense(600, activation='relu')(dense22)
    dense24 = Dense(600, activation='relu')(dense23)

    # merge input models
    merge = concatenate([dense11, dense24])
    # interpretation model
    dense1 = Dense(600, activation='relu')(merge)
    output = Dense(output_dim, activation='relu')(dense1)

    model = Model(inputs=[visible1, visible2], outputs=output)

    # print(model.summary())

    return model


def three_inputs_dense_model(input_dim):
    # first input model:sun patch images
    visible1 = Input(shape=(1,))
    dense11 = Dense(NUM_NODES, activation='relu')(visible1)

    # second input model:full light images
    visible2 = Input(shape=(input_dim - 2,))
    dense21 = Dense(NUM_NODES, activation='relu')(visible2)
    dense22 = Dense(NUM_NODES, activation='relu')(dense21)
    dense23 = Dense(NUM_NODES, activation='relu')(dense22)
    dense24 = Dense(NUM_NODES, activation='relu')(dense23)

    visible3 = Input(shape=(1,))
    dense31 = Dense(NUM_NODES, activation='relu')(visible3)

    # merge input models
    merge = concatenate([dense11, dense24, dense31])
    # interpretation model
    dense1 = Dense(NUM_NODES, activation='relu')(merge)
    output = Dense(1, activation='relu')(dense1)

    model = Model(inputs=[visible1, visible2, visible3], outputs=output)

    # print(model.summary())

    return model


def retrain_model():
    # load pre-trained model
    # load json and create model
    if (os.path.exists("results/model/") == False):
        os.mkdir("results/model/")

    json_file = open('results/model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("results/model/model.h5")
    print("Loaded model from disk")
    return loaded_model


def plotLoss(history):
    # save train detail image
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('validation loss')
    plt.ylabel('validation error')
    plt.xlabel('epoch')
    plt.legend(['val_loss', "loss"], loc='upper left')
    plt.xlim([0, NUM_EPOCHS])
    plt.ylim([0, history.history['loss'][0]])
    plt.savefig(os.path.join("results/validation_error.png"))
    plt.close()


def np2txt(array, al, az, dir, dif):
    array = array / 179
    print(array.shape)
    np.savetxt(
        "results/final_predictions_txt/" +
        str(al) +
        "_" +
        str(az) +
        "_" +
        str(dir) +
        "_" +
        str(dif) +
        '_three_columns.txt',
        np.c_[
            array,
            array,
            array])


def processData(output_dim, p_im_val, y_val, num, i):
    p_im = p_im_val.reshape(num, IMAGE_SIZE[0] * IMAGE_SIZE[1], output_dim)
    p_im = p_im[i, :, :]
    p_im = p_im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
    print(p_im.shape)

    n_im = y_val.reshape(num, IMAGE_SIZE[0] * IMAGE_SIZE[1], output_dim)
    n_im = n_im[i, :, :]
    n_im = n_im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
    print(n_im.shape)

    # clip the final image brightness to 0 -1
    n_im = np.clip(n_im, 0, 1)
    p_im = np.clip(p_im, 0, 1)

    diff = abs(p_im - n_im)

    return p_im, n_im, diff


# post process the image for better display


def postProcessIms(output_dim, n_im, p_im, type, gamma):
    p_im = restore_Normalization(output_dim, p_im, type, gamma)
    n_im = restore_Normalization(output_dim, n_im, type, gamma)

    diff = abs(p_im - n_im)

    return diff, n_im, p_im


def getMaxMinPrams(type):
    file = open(
        "data/processed_data/" +
        type +
        'restoration_min_max_parameters.txt',
        'r')
    data = file.read().split(',')
    return float(
        data[0]), float(
        data[1]), float(
        data[2]), float(
        data[3]), float(
        data[4]), float(
        data[5]), float(
        data[6]), float(
        data[7])


def plotDataSampleDistribution(train, val, test):
    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX = getMaxMinPrams(
        "train_")
    train_dir = restorNorm(train[:, 0, 4], DIR_MAX, DIR_MIN)
    train_dif = restorNorm(train[:, 0, 5], DIF_MAX, DIF_MIN)
    train_al = restorNorm(train[:, 0, 2], AL_MAX, AL_MIN)
    train_az = restorNorm(train[:, 0, 3], AZ_MAX, AZ_MIN)

    val_dir = restorNorm(val[:, 0, 4], DIR_MAX, DIR_MIN)
    val_dif = restorNorm(val[:, 0, 5], DIF_MAX, DIF_MIN)
    val_al = restorNorm(val[:, 0, 2], AL_MAX, AL_MIN)
    val_az = restorNorm(val[:, 0, 3], AZ_MAX, AZ_MIN)

    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX = getMaxMinPrams(
        "test_")
    test_dir = restorNorm(test[:, 0, 4], DIR_MAX, DIR_MIN)
    test_dif = restorNorm(test[:, 0, 5], DIF_MAX, DIF_MIN)
    test_al = restorNorm(test[:, 0, 2], AL_MAX, AL_MIN)
    test_az = restorNorm(test[:, 0, 3], AZ_MAX, AZ_MIN)

    num = int(test_dir.shape[0])
    for i in range(num):
        al = int(test_al[i])
        az = int(test_az[i])
        dir = int(test_dir[i])
        dif = int(test_dif[i])

        file_name = str(al) + "_" + str(az) + "_" + str(dir) + "_" + str(dif)

        fig = plt.figure(figsize=(10, 10), dpi=150)

        plt.scatter(train_dir, train_dif, s=10, color='g', label="train")
        plt.scatter(val_dir, val_dif, s=10, color='black', label="validation")
        plt.scatter(test_dir, test_dif, s=1, color='b', label="test")
        plt.scatter(dir, dif, s=50, color='r', label="predicted_sample")
        plt.title("Sky Direct and Diffuse Irradiances Distribution", size=16)
        plt.xlabel('Direct')
        plt.ylabel('Diffuse')
        plt.legend(fancybox=True)
        plt.savefig('results/result_combo/' + file_name + "_sky.png")
        plt.close()

        fig = plt.figure(figsize=(10, 10), dpi=150)

        plt.scatter(train_al, train_az, s=10, color='g', label="train")
        plt.scatter(val_al, val_az, s=10, color='black', label="validation")
        plt.scatter(test_al, test_az, s=1, color='b', label="test")
        plt.scatter(al, az, s=50, color='r', label="predicted_sample")
        plt.title("Sun Altitude and Azimuth Distribution", size=16)
        plt.xlabel('Altitude')
        plt.ylabel('Azimuth')
        plt.legend(fancybox=True)
        plt.savefig('results/result_combo/' + file_name + "_sun.png")
        plt.close()

    print("done plot")


def restorNorm(num, max, min):
    return num * (max - min) + min


def plotAnalysis(
        output_dim,
        log_process,
        y_predict,
        y,
        x,
        type,
        gamma,
        log_base):
    num = int(x.shape[0])
    print(x.shape)
    print(y_predict.shape)
    print(y.shape)
    print(x[..., 2].shape)
    x_al = x[..., 2].reshape(num, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    x_az = x[..., 3].reshape(num, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    x_dir = x[..., 4].reshape(num, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    x_dif = x[..., 5].reshape(num, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    dgps_p = []
    dgps_t = []
    spatial_contrasts_t = []
    dgps_t_mean = []
    dgps_t_max = []
    log_mse = []
    log_rer = []

    for i in range(num):
        p_im, n_im, diff = processData(output_dim, y_predict, y, num, i)

        AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX = getMaxMinPrams(
            type)
        al = int(restorNorm((x_al[i][0]), AL_MAX, AL_MIN))
        az = int(restorNorm(x_az[i][0], AZ_MAX, AZ_MIN))
        dir = int(restorNorm(x_dir[i][0], DIR_MAX, DIR_MIN))
        dif = int(restorNorm(x_dif[i][0], DIF_MAX, DIF_MIN))
        print(
            "direct irradiance and diffuse irradiance is: " +
            str(dir) +
            ", " +
            str(dif))
        name = str(al) + "_" + str(az) + "_" + str(dir) + "_" + str(dif)

        # calculate the loss and error according to paper definition
        loss = np.sum(np.square(p_im - n_im))
        truth = np.sum(np.square(n_im))
        error = np.divide(loss, truth)
        print("Total loss is: ", loss)
        print("Error rate is: ", error)
        mse = ((p_im - n_im) ** 2).mean(axis=None)
        print("Mse is: ", mse)

        # calculate mse with solid angle weight
        angle_filter = im_angle().reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
        sa_weighted_p_im = p_im * angle_filter
        sa_weighted_n_im = n_im * angle_filter
        solid_anlge_weighted_mse = (
                (sa_weighted_n_im - sa_weighted_p_im) ** 2).mean()

        # calculate solid angle weighted rer
        loss = np.sum(np.square(sa_weighted_p_im - sa_weighted_n_im))
        truth = np.sum(np.square(sa_weighted_n_im))
        solid_angle_weighted_error = np.divide(loss, truth)

        # calculate nomralized spatial contrast
        norm_sc_t = np.mean(spatial_contrast_one_im(n_im))
        # spatial_contrasts_t.append(norm_sc_t * 100)

        # diff, n_im, p_im = post_process_angle(n_im, p_im)
        diff, n_im, p_im = postProcessIms(output_dim, n_im, p_im, type, gamma)
        if log_process:
            diff, n_im, p_im = post_process_log(n_im, p_im, log_base)

        log10_mse = get_log10_mse(n_im, p_im)
        log10_rer = get_log10_relative_error(n_im, p_im)

        # calculate spatial contrast mae
        sc_p = spatial_contrast_one_im(p_im)
        sc_t = spatial_contrast_one_im(n_im)
        sc_diff = np.mean(abs(sc_p - sc_t))
        # save spatial contrast images
        save_sc_ims(
            sc_p,
            sc_t,
            al,
            az,
            dir,
            dif,
            solid_angle_weighted_error,
            solid_anlge_weighted_mse,
            3000)
        save_sc_ims(
            sc_p,
            sc_t,
            al,
            az,
            dir,
            dif,
            solid_angle_weighted_error,
            solid_anlge_weighted_mse,
            10000)
        spatial_contrasts_t.append(np.mean(sc_t))

        # calculate gradient loss
        sobel_n = get_gradient_one_im(n_im)
        sobel_p = get_gradient_one_im(p_im)
        gradient = np.mean(abs(sobel_n - sobel_p))
        # save gradient images
        save_gradient_ims(
            sobel_p,
            sobel_n,
            solid_anlge_weighted_mse,
            solid_angle_weighted_error,
            al,
            az,
            dir,
            dif,
            3000)
        save_gradient_ims(
            sobel_p,
            sobel_n,
            solid_anlge_weighted_mse,
            solid_angle_weighted_error,
            al,
            az,
            dir,
            dif,
            10000)

        saveComparisonImage(
            error,
            i,
            n_im,
            p_im,
            diff,
            type,
            mse,
            dir,
            dif,
            al,
            az,
            sc_p,
            sc_t,
            sc_diff,
            solid_anlge_weighted_mse,
            solid_angle_weighted_error,
            gradient,
            3000)
        saveComparisonImage(
            error,
            i,
            n_im,
            p_im,
            diff,
            type,
            mse,
            dir,
            dif,
            al,
            az,
            sc_p,
            sc_t,
            sc_diff,
            solid_anlge_weighted_mse,
            solid_angle_weighted_error,
            gradient,
            10000)
        # saveComparisonImageSeperate(i, n_im, p_im, diff, type, dir, dif, al, az)
        if (os.path.exists("results/results_3000/") == False):
            os.mkdir("results/results_3000/")
        if (os.path.exists("results/results_10000/") == False):
            os.mkdir("results/results_10000/")
        if (os.path.exists("results/DGPs/") == False):
            os.mkdir("results/DGPs/")
        p, t, t_mean, t_max = save_im_falsecolor(
            p_im, n_im, name, 3000, "results/results_3000/", IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[1])
        dgps_p.extend(p)
        dgps_t.extend(t)
        dgps_t_mean.append(t_mean)
        dgps_t_max.append(t_max)
        log_mse.append(log10_mse)
        log_rer.append(log10_rer)
        save_im_falsecolor(
            p_im,
            n_im,
            name,
            10000,
            "results/results_10000/",
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            IMAGE_SIZE[1])

    np.save("results/dgps_t.npy", dgps_t)
    np.save("results/dgps_p.npy", dgps_p)
    np.save("results/log10mse.npy", log_mse)
    np.save("results/log10rer.npy", log_rer)
    plotDGPAnalysis_whole(dgps_t, dgps_p)
    plot_DGPvsSC_Analysis(dgps_t_mean, spatial_contrasts_t, "mean")
    plot_DGPvsSC_Analysis(dgps_t_max, spatial_contrasts_t, "max")
    plotLog10MSEAnalysis01(log_mse)
    plotLog10RERAnalysis01(log_rer)
    plotLog10MSEAnalysis(log_mse)
    plotLog10RERAnalysis(log_rer)


def printStatistics(truth, prediction, file_name):
    # save statistic informaiton to txt file
    with open("results/" + file_name, "w") as myfile:
        myfile.write(
            "Test set statistic is: MSE %.5e, RMSE %.2f, LOG10_MSE %.5e, LOG10_RER %.5e"
            "Relative Average Error Rate %.3f, "
            "test solid angle MSE %.5e, test_Solid_Angle_RER %.3f, "
            "test sc %.5e, test_gradient %.5e," %
            (get_mse(
                truth, prediction), get_rmse(
                truth, prediction), get_log10_mse(
                truth, prediction), get_log10_relative_error(
                truth, prediction), get_relative_error(
                truth, prediction), get_solid_angle_mse(
                truth, prediction), get_solid_angle_rer(
                truth, prediction), get_spatial_contrast_MAE(
                truth, prediction), get_gradient_MAE(
                truth, prediction)))
        myfile.close()


def postProcessData(output_dim, log_reverse, n_ims, p_ims, gamma, log_base):
    num_ims = n_ims.shape[0]
    im_frame_n = np.zeros(
        (num_ims,
         IMAGE_SIZE[0],
         IMAGE_SIZE[1],
         1),
        dtype='float32')
    im_frame_p = np.zeros(
        (num_ims,
         IMAGE_SIZE[0],
         IMAGE_SIZE[1],
         1),
        dtype='float32')
    for i in range(num_ims):
        n_im = n_ims[i, :, :]
        p_im = p_ims[i, :, :]
        diff, n_im, p_im = postProcessIms(
            output_dim, n_im, p_im, "test_", gamma)
        if log_reverse:
            diff, n_im, p_im = post_process_log(n_im, p_im, log_base)
        im_frame_n[i, :, :] = n_im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
        im_frame_p[i, :, :] = p_im.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    return im_frame_n, im_frame_p


def train(
        retrain,
        loss_function,
        gamma,
        log_base,
        log_process,
        input_dim,
        output_dim,
        train_model,
        learning_rate):
    # start record time
    start_time = time.time()

    tf.keras.backend.clear_session()
    val_losses = []
    train_losses = []
    # load data
    x_total, x_sun_total, y_total, \
    x_val, x_sun_val, y_val, \
    x_test, x_sun_test, y_test = loadData(input_dim, output_dim)

    # load model
    if train_model == "CNN":
        model = two_inputs_CNN_model(input_dim, output_dim)
    else:
        model = two_inputs_dense_model(input_dim, output_dim)

    if retrain:
        model = retrain_model()

    # compile and train the model
    # save the model after each epoch if loss is decreasing
    # reduce the learning rate by 0.5, if loss is not decreasing after each
    # epoch until reach the minimum learning rate

    alpha = K.variable(1.0)  # 1
    beta = K.variable(0.0)  # 0

    def loss_update(alpha, beta):
        def loss(y_true, y_pred):
            a = K.get_value(alpha)
            b = K.get_value(beta)
            print("b:" + str(beta))
            combo_loss = a * \
                         mse_rer_loss_old(y_true, y_pred) + b * mse_rer_loss_handle_log(y_true, y_pred)
            return combo_loss

        return loss

    model.compile(
        loss=loss_function,
        optimizer=keras.optimizers.adam(
            lr=learning_rate),
        metrics=[
            metrics.mse,
            metrics.mae,
            relative_err,
            mse_log])

    batch_size = INITIAL_BATCH_SIZE
    min_batch = batch_size - 1
    if BATCH_REDUCTIONS > 0:
        batch_reductions = 2 ** (int(BATCH_REDUCTIONS))
        min_batch = np.max([MINI_BATCHSIZE, int(
            INITIAL_BATCH_SIZE / BATCH_REDUCTIONS)])
    loss = 1e10
    history = None
    if (os.path.exists("results/model/") == False):
        os.mkdir("results/model/")

    # first train until loss not decreasing, then reduce the learning rate
    # when learning rate is reduced to min_learning rate, reduce batch size by half
    # stop while loss is less than 1e-5, or batch size reach min_batch size
    while batch_size > min_batch and loss > TARGET_LOSS:
        pb = printbatch()
        if train_model == "CNN":
            train_generator = trainGenerator_CNN(
                x_total, x_sun_total, y_total, batch_size)
            val_generator = trainGenerator_CNN(
                x_val, x_sun_val, y_val, batch_size)
            val_test_generator = testGenerator_CNN(x_val, x_sun_val, 1)
            test_generator = testGenerator_CNN(x_test, x_sun_test, 1)
        else:
            train_generator = trainGenerator_Dense(
                input_dim, x_total, x_sun_total, y_total, batch_size)
            val_generator = trainGenerator_Dense(
                input_dim, x_val, x_sun_val, y_val, batch_size)
            val_test_generator = testGenerator_Dense(
                input_dim, x_val, x_sun_val, 1)
            test_generator = testGenerator_Dense(
                input_dim, x_test, x_sun_test, 1)

        print("Built the generator")
        print(x_total.shape)
        history = model.fit_generator(train_generator, steps_per_epoch=int(x_total.shape[0] / batch_size),
                                      epochs=NUM_EPOCHS, verbose=2,
                                      validation_data=val_generator, validation_steps=int(
                x_val.shape[0]),
                                      # callbacks=[AttentionLoss(alpha, beta),
                                      # pb,
                                      callbacks=[pb,
                                                 ReduceLROnPlateau(
                                                     monitor='loss', factor=0.5, patience=2, min_lr=MIN_LR, verbose=1),
                                                 ModelCheckpoint(filepath='results/model/weights.hdf5', verbose=1,
                                                                 save_best_only=True)])

        batch_size = int(batch_size / 2.0)
        # batch_size = batch_size-1
        print("batch size reduce to: " + str(batch_size))

        val_losses = np.append(history.history['val_loss'], val_losses)
        train_losses = np.append(history.history['loss'], train_losses)

        loss = history.history['val_loss'][-1]
        print("loss is: " + str(loss))
        print("target loss is: " + str(TARGET_LOSS))
        print("min batch size is: " + str(min_batch))

    plt.plot(
        np.arange(
            len(val_losses)),
        val_losses,
        c='b',
        label="validation error")
    plt.plot(
        np.arange(
            len(train_losses)),
        train_losses,
        c='r',
        label="training error")
    plt.legend(fancybox=True)
    plt.ylabel('validation error')
    plt.xlabel('epoch')
    plt.xlim(0, 100)
    plt.ylim(0, 0.10)
    plt.savefig("results/error.png")
    # save the model
    # serialize model to JSON

    model_json = model.to_json()
    with open("results/model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("results/model/model.h5", overwrite=True)
    print("Saved trained model to disk")

    # save train detail image
    plotLoss(history)

    # load predicted images using the trained model and validation/train data
    p_im_val = model.predict_generator(
        val_test_generator, steps=y_val.shape[0])
    print(p_im_val.shape)
    p_im_val = p_im_val.reshape(
        y_val.shape[0],
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        1)

    p_im_test = model.predict_generator(test_generator, steps=y_test.shape[0])
    print(p_im_test.shape)
    p_im_test = p_im_test.reshape(
        y_test.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    print(p_im_test.max())

    # print(p_im_test_absolute.max())
    np.save("results/test_prediction.npy", p_im_test)
    np.save("results/test_truth.npy", y_test)
    np.save("results/test_parameters.npy", x_test)
    np.save("results/test_sun_p.npy", x_sun_test)

    print("finished predicting images using trained model")

    # save statistic informaiton to txt file
    printStatistics(y_test, p_im_test, "statistics.txt")

    print("done!")
    # print time information
    duration = int(time.time() - start_time)
    minutes, seconds = duration // 60, duration % 60
    print(
        "Time spend on load all captured data: " +
        str(minutes) +
        ':' +
        str(seconds))


def model_analysis(gamma,
                   log_base,
                   log_process,
                   output_dim):
    start_time = time.time()
    print("start saving predicted images process")

    p_im_test = np.load("results/test_prediction.npy")
    y_test = np.load("results/test_truth.npy")
    x_test = np.load("results/test_parameters.npy")
    x_sun_test = np.load("results/test_sun_p.npy")

    plotAnalysis(
        output_dim,
        log_process,
        p_im_test,
        y_test,
        x_test,
        "test_",
        gamma,
        log_base)

    y_test_absolute, p_im_test_absolute = postProcessData(
        output_dim, log_process, y_test, p_im_test, gamma, log_base)
    print(p_im_test.max())
    printStatistics(
        y_test_absolute,
        p_im_test_absolute,
        "absolute_statistics.txt")

    print("Analysis done!")
    # print time information
    duration = int(time.time() - start_time)
    minutes, seconds = duration // 60, duration % 60
    print(
        "Analysis time spent: " +
        str(minutes) +
        ':' +
        str(seconds))


def generate_train_val_data(data_root='./ALL_DATA_FP32',
                RETRAIN=False,
                LOSS_FUNCTION_TYPE=relative_err,
                LOG_BOOL=True,
                SKYMAP_BOOL=False,
                MODEL_TYPE="Dense",
                GAMMA_VALUE=1.5,  # 2.2
                LOG_BASE_VALUE=10,
                NUM_CLUSTERS=250,
                LATITUDE=47,
                LONGITUDE=122,
                SM=120):
    
    AB4_DIR = data_root + get_data_path('AB4')
    AB0_DIR = data_root + get_data_path('AB0')
    SKY_DIR = data_root + get_data_path('SKY')

    genData.preprocess(
        log=LOG_BOOL,
        gamma=GAMMA_VALUE,
        skymap=SKYMAP_BOOL,
        log_base=LOG_BASE_VALUE,
        num_clusters=NUM_CLUSTERS,
        gen_trainSet=True,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        sm=SM,
        ab4_dir=AB4_DIR,
        ab0_dir=AB0_DIR,
        sky_dir=SKY_DIR)
    

def train_model(data_root='./ALL_DATA_FP32',
                RETRAIN=False,
                LOSS_FUNCTION_TYPE=relative_err,
                LOG_BOOL=True,
                SKYMAP_BOOL=False,
                MODEL_TYPE="Dense",
                GAMMA_VALUE=1.5,  # 2.2
                LOG_BASE_VALUE=10,
                NUM_CLUSTERS=250,
                LATITUDE=47,
                LONGITUDE=122,
                SM=120):
    
    AB4_DIR = data_root + get_data_path('AB4')
    AB0_DIR = data_root + get_data_path('AB0')
    SKY_DIR = data_root + get_data_path('SKY')

    genData.preprocess(
        log=LOG_BOOL,
        gamma=GAMMA_VALUE,
        skymap=SKYMAP_BOOL,
        log_base=LOG_BASE_VALUE,
        num_clusters=NUM_CLUSTERS,
        gen_trainSet=False,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        sm=SM,
        ab4_dir=AB4_DIR,
        ab0_dir=AB0_DIR,
        sky_dir=SKY_DIR)
    
    INPUT_DIM = 8
    if SKYMAP_BOOL:
        INPUT_DIM = 9
    LR = 0.0001
    if MODEL_TYPE == "CNN":
        LR = 0.001
    train(
        retrain=RETRAIN,
        loss_function=LOSS_FUNCTION_TYPE,
        gamma=GAMMA_VALUE,
        log_base=LOG_BASE_VALUE,
        log_process=LOG_BOOL,
        input_dim=INPUT_DIM,
        output_dim=1,
        train_model=MODEL_TYPE,
        learning_rate=LR)


def analyze_model(data_root='./ALL_DATA_FP32',
                  RETRAIN=False,
                  LOSS_FUNCTION_TYPE=relative_err,
                  LOG_BOOL=True,
                  SKYMAP_BOOL=False,
                  MODEL_TYPE="Dense",
                  GAMMA_VALUE=1.5,  # 2.2
                  LOG_BASE_VALUE=10,
                  NUM_CLUSTERS=250,
                  LATITUDE=47,
                  LONGITUDE=122,
                  SM=120):
    
    AB4_DIR = data_root + get_data_path('AB4')
    AB0_DIR = data_root + get_data_path('AB0')
    SKY_DIR = data_root + get_data_path('SKY')

    model_analysis(gamma = GAMMA_VALUE,
                   log_base = LOG_BASE_VALUE,
                   log_process=LOG_BOOL,
                   output_dim=1)
    version = ""
    result_path = "results/"

    t = np.load(result_path + "test_truth.npy")
    p = np.load(result_path + "test_prediction.npy")
    process_data_path = "data/processed_data" + version + "/"

    # im post process, reverse log, normalization, gamma
    diff, t, p = plot.postProcessIms(
        t, p, "test_", t.shape[0], 1.5, process_data_path)
    diff, t, p = plot.post_process_log(t, p, True, 10, t.shape[0])

    ### just importing these modules does something... 
    ### need to sanitize this code
    from deep_light.RAMMG import plotRAMMGAnalysis, getAllRAMMGs
    
    # im apply angle filter
    r_ts, r_ps = getAllRAMMGs(
        t, p, level_start=0, level_end=5, angle_bool=False, save_path=result_path)
    plotRAMMGAnalysis(r_ps, r_ts, save_path=result_path)

    plot.plotAnalysis(
        log_process=LOG_BOOL,
        log_base=LOG_BASE_VALUE,
        gamma=GAMMA_VALUE)

#
# train_model(data_root='W:')
