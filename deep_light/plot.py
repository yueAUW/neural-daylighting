

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

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
import math
from mpl_toolkits.mplot3d import Axes3D
IMAGE_SIZE = (230, 115)
EPSILON = 0

def hdr_to_image(RGBColor, gamma):
    g = 1.0 / gamma
    RGB = np.power(RGBColor, g)

    return RGB

def image_to_hdr(lu, gamma):
    lu = np.power(lu, gamma)

    return lu

def getMaxMinPrams(type, path):
    file = open(path + type +'restoration_min_max_parameters.txt', 'r')
    data = file.read().split(',')
    return float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])

def restorNorm(num, max, min):
    return num*(max-min)+min

# post process the image for better display
def postProcessIms(n_im, p_im, type, num_images, gamma, process_path):

    p_im = restore_Normalization(p_im, type, num_images, gamma, process_path)
    n_im = restore_Normalization(n_im, type, num_images, gamma, process_path)

    diff = abs(p_im - n_im)

    return diff, n_im, p_im

# restore the normalized value to the original luminance value
def restore_Normalization(ims, type, num_images, gamma, path):
    #first pair of min and max when conduct 1st normalization
    file = open(path + type + 'restoration_min_max.txt', 'r')
    data = file.read().split(',')

    min_1 = float(data[0])
    max_1 = float(data[1])

    ims = ims.reshape(num_images, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    # restore: 1) reverse restore the normalization by multiple the max, 2) reverse gamma correction, 3) reverse normalization restore

    ims = image_to_hdr(ims, gamma)

    ims = ims * (max_1 - min_1) + min_1


    ims = ims.reshape(num_images, IMAGE_SIZE[0], IMAGE_SIZE[1])
    return ims

def get_relative_error(y_true, y_pred):
    sum_err_sqr = np.sum(np.square(y_true - y_pred))
    sum_val_sqr = np.sum(np.square(y_true))
    return (np.sqrt(sum_err_sqr / sum_val_sqr))

def im_angle():
    image_frame = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype='float32')
    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        index1, index2 = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        image_frame[index1][index2] = angle_correction(index2)
    return image_frame.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

def angle_correction(py):
    return math.cos(math.pi * (py - IMAGE_SIZE[1] / 2) / IMAGE_SIZE[1]) + EPSILON

def post_process_log(n_ims, p_ims, log_process, log_base, num_ims):
    if log_process:
        n_ims = reverse_log_processing(n_ims, log_base, num_ims)
        p_ims = reverse_log_processing(p_ims, log_base, num_ims)
    diff = abs(p_ims - n_ims)
    return diff, n_ims, p_ims

def reverse_log_processing(ims, log_base, num_ims):
    ims = ims.reshape (num_ims, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    ims = np.power(log_base, ims)
    ims = ims.reshape(num_ims, IMAGE_SIZE[0], IMAGE_SIZE[1])
    return ims


def plotAnalysis(log_process, log_base, gamma):
    dir_path = "results/"
    process_data_path = "data/processed_data/"
    diagram_path = "results/"
    if (os.path.exists(diagram_path) == False):
        os.mkdir(diagram_path)

    p = np.load(dir_path + "test_prediction.npy")
    t = np.load(dir_path + "test_truth.npy")
    parameters = np.load(dir_path + "test_parameters.npy")
    sun = np.load(dir_path + "test_sun_p.npy")
    dgps_p = np.load(dir_path + "dgps_p.npy")
    dgps_t = np.load(dir_path + "dgps_t.npy")

    num_images = p.shape[0]

    dgps_acc_p = np.zeros(num_images)
    dgps_acc_t = np.zeros(num_images)
    dgps_acc_diff = np.zeros(num_images)
    dgps_mean_diff = np.zeros(num_images)
    dgps_max_diff = np.zeros(num_images)

    for i in range(num_images):
        for j in range(10):
            index = i*10 + j
            dgps_acc_p[i] = dgps_acc_p[i] + dgps_p[index]
            dgps_acc_t[i] = dgps_acc_t[i] + dgps_t[index]
            dgps_acc_diff[i] = dgps_acc_diff[i] + abs(dgps_p[index] - dgps_t[index])
            dgps_max_diff[i] = max(dgps_max_diff[i], abs(dgps_p[index] - dgps_t[index]))
        dgps_mean_diff[i] = dgps_acc_diff[i]/10
        dgps_acc_p[i] = dgps_acc_p[i]/10
        dgps_acc_t[i] = dgps_acc_t[i]/10



    diff_origin = abs(p-t)
    diff_process, t, p = postProcessIms(t, p, "test_", num_images, gamma, process_data_path)

    al = parameters[:, :, :, 2]
    al=al[:, 0, 0]
    az = parameters[:, :, :, 3]
    az=az[:, 0, 0]
    dir = parameters[:, :, :, 4]
    dir=dir[:, 0, 0]
    dif = parameters[:, :, :, 5]
    dif=dif[:, 0, 0]


    AL_MIN, AL_MAX, AZ_MIN, AZ_MAX, DIR_MIN, DIR_MAX, DIF_MIN, DIF_MAX = getMaxMinPrams("test_", process_data_path)
    al = restorNorm(al, AL_MAX, AL_MIN)
    az = restorNorm(az, AZ_MAX, AZ_MIN)
    dir = restorNorm(dir, DIR_MAX, DIR_MIN)
    dif = restorNorm(dif, DIF_MAX, DIF_MIN)

    angle = im_angle().reshape(230, 115)
    mse_angle =[]
    rer_angle =[]
    for i in range(num_images):
        mse_angle_im = np.mean((t[i, :]*angle - p[i, :]*angle)**2)
        rer_angle_im = get_relative_error(t[i, :]*angle, p[i, :]*angle)
        mse_angle.append(mse_angle_im)
        rer_angle.append(rer_angle_im)

    #3)solid angle mse scatter plot;
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter3D(al, az, mse_angle, s=1, c='red')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(mse_angle),max(mse_angle))
    plt.xlabel("Sun Altitude")
    plt.ylabel("Sun Azimuth")
    plt.title("Solid Angle MSE 3D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "mse_scatter.png")
    plt.close()

    #4)solid angle rer scatter plot;
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter3D(al, az, rer_angle, s=1, c='red', vmin = min(rer_angle), vmax = max(rer_angle), marker ='o')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(rer_angle),max(rer_angle))
    plt.xlabel("Sun Altitude")
    plt.ylabel("Sun Azimuth")
    plt.title("Solid Angle RER 3D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "rer_scatter.png")
    plt.close()

    diff_log, t, p = post_process_log(t, p, log_process, log_base, num_images)
    diff_process = diff_process.reshape(num_images, 230, 115, 1)
    diff_log = diff_log.reshape(num_images, 230, 115, 1)

    print("individual mse values:")
    for i in range(t.shape[0]):
        print(np.mean(np.square(t[i, :]-p[i, :])))
    print("finish individual mse values:")

    print("individual rer values:")
    for i in range(t.shape[0]):
        print(get_relative_error(t[i, :], p[i, :]))
    print("finish individual rer values:")

    #delete pixel when diff > e+06
    for i in range(diff_log.shape[0]):
        for j in range(diff_log.shape[1]):
            for x  in range(diff_log.shape[2]):
                if diff_log[i, j, x] > 10**6:
                    diff_log[i, j, x] = 0
                    t[i, j, x] = 0
                    p[i, j, x] = 0
    print("new mse is: ")
    print(np.mean(np.square(t - p)))

    #1)accumulated mse:
    origin_sum = np.sum(diff_origin, axis = 0)/num_images
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(origin_sum.reshape(230, 115), cmap = "jet", norm=LogNorm(vmin=origin_sum.min(), vmax = origin_sum.max()))
    # add color index bar
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    formatter = LogFormatter(2, labelOnlyBase=False)
    #cb = plt.colorbar(ticks=[1, 4.1, 12.2, 36.5, 109, 325.8, 973.4, 3000], format=formatter, cax=cbaxes)
    cb = plt.colorbar(format=formatter, cax=cbaxes)
    cb.set_label('Luminance (cd/m2)', rotation=90)
    # save the image
    plt.savefig(os.path.join(diagram_path + "average_accumulated_absolute_error_origin.png"))
    plt.close()


    #1)accumulated mse:
    process_sum = np.sum(diff_process, axis = 0)/num_images
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(process_sum.reshape(230, 115), cmap = "jet", norm=LogNorm(vmin=process_sum.min(), vmax = process_sum.max()))
    # add color index bar
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    formatter = LogFormatter(2, labelOnlyBase=False)
    #cb = plt.colorbar(ticks=[1, 4.1, 12.2, 36.5, 109, 325.8, 973.4, 3000], format=formatter, cax=cbaxes)
    cb = plt.colorbar(format=formatter, cax=cbaxes)
    cb.set_label('Luminance (cd/m2)', rotation=90)
    # save the image
    plt.savefig(os.path.join(diagram_path + "average_accumulated_absolute_error_process.png"))
    plt.close()


    #1)accumulated mse:
    log_sum = np.sum(diff_log, axis = 0)/num_images
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(log_sum.reshape(230, 115), cmap = "jet", norm=LogNorm(vmin=log_sum.min(), vmax = log_sum.max()))
    # add color index bar
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    formatter = LogFormatter(10, labelOnlyBase=False)
    #cb = plt.colorbar(ticks=[1, 4.1, 12.2, 36.5, 109, 325.8, 973.4, 3000], format=formatter, cax=cbaxes)
    cb = plt.colorbar(format=formatter, cax=cbaxes)
    cb.set_label('Luminance (cd/m2)', rotation=90)
    # save the image
    plt.savefig(os.path.join(diagram_path + "average_accumulated_absolute_luminance_error_log.png"))
    plt.close()


    mse_angle =[]
    rer_angle =[]
    for i in range(num_images):
        mse_angle_im = np.mean((t[i, :]*angle - p[i, :]*angle)**2)
        rer_angle_im = get_relative_error(t[i, :]*angle, p[i, :]*angle)
        mse_angle.append(mse_angle_im)
        rer_angle.append(rer_angle_im)

    #3)solid angle mse scatter plot;
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter3D(al, az, mse_angle, s=1, c='red')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(mse_angle),max(mse_angle))
    plt.xlabel("Sun Altitude")
    plt.ylabel("Sun Azimuth")
    plt.title("Solid Angle MSE 3D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "mse_scatter_after.png")
    plt.close()

    #4)solid angle rer scatter plot;
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter3D(al, az, rer_angle, s=1, c='red', vmin = min(rer_angle), vmax = max(rer_angle), marker ='o')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(rer_angle),max(rer_angle))
    plt.xlabel("Sun Altitude")
    plt.ylabel("Sun Azimuth")
    plt.title("Solid Angle RER 3D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "rer_scatter_after.png")
    plt.close()

    #dgp scatter 3d plot
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter3D(al, az, dgps_mean_diff, s=3, c='red', vmin = min(dgps_mean_diff), vmax = max(dgps_mean_diff), marker ='o')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(dgps_mean_diff), max(dgps_mean_diff))
    plt.xlabel("Sun Altitude")
    plt.ylabel("Sun Azimuth")
    plt.title("Mean DGP Diff 3D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "DGP_scatter_after.png")
    plt.close()


    #dgp scatter 3d plot
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter3D(al, az, dgps_acc_diff, s=3, c='red', vmin = min(dgps_acc_diff), vmax = max(dgps_acc_diff), marker ='o')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(dgps_acc_diff), max(dgps_acc_diff))
    plt.xlabel("Sun Altitude")
    plt.ylabel("Sun Azimuth")
    plt.title("Accumulated DGP Diff 3D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "acc_DGP_scatter_after.png")
    plt.close()

    #5)dgp scatter plot per hour
    fig = plt.figure()
    plt.scatter(np.arange(num_images), dgps_mean_diff, s=3, c='red', vmin = min(dgps_mean_diff), vmax = max(dgps_mean_diff), marker ='o')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(dgps_mean_diff), max(dgps_mean_diff))
    plt.xlabel("Point of Time")
    plt.ylabel("Mean DGP Diff")
    plt.title("Mean DGP Diff 2D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "DGP_2dscatter_after.png")
    plt.close()

    #5)dgp scatter plot per hour
    fig = plt.figure()
    plt.scatter(np.arange(num_images), dgps_max_diff, s=3, c='red', vmin = min(dgps_max_diff), vmax = max(dgps_max_diff), marker ='o')
    # Plot settings:
    ax.set_xlim3d(min(al),max(al))
    ax.set_ylim3d(min(az),max(az))
    ax.set_zlim3d(min(dgps_max_diff), max(dgps_max_diff))
    plt.xlabel("Point of Time")
    plt.ylabel("Max DGP Diff")
    plt.title("Max DGP Diff 2D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "max_DGP_2dscatter_after.png")
    plt.close()


    #5)dgp scatter plot per al
    fig = plt.figure()
    plt.scatter(al, dgps_max_diff, s=3, c='red', vmin = min(dgps_max_diff), vmax = max(dgps_max_diff), marker ='o')
    plt.xlabel("Altitude")
    plt.ylabel("Max DGP Diff")
    plt.title("Max DGP Diff 2D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "max_DGP_2dscatter_al_after.png")
    plt.close()


    #5)dgp scatter plot per az
    fig = plt.figure()
    plt.scatter(az, dgps_max_diff, s=3, c='red', vmin = min(dgps_max_diff), vmax = max(dgps_max_diff), marker ='o')
    # Plot settings:
    plt.xlabel("Azimuth")
    plt.ylabel("Max DGP Diff")
    plt.title("Max DGP Diff 2D Scatter Plot")
    #ax.set_zlabel("Solid Angle Weighted MSE")
    plt.savefig(diagram_path + "max_DGP_2dscatter_az_after.png")
    plt.close()

"""
#import data
IMAGE_SIZE =(230, 115)
EPSILON = 0

LOG = True
LOG_BASE = 10
GAMMA = 1.5


plotAnalysis(log_process = LOG, log_base = LOG_BASE, gamma = GAMMA)
"""