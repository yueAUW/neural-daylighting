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
import sys
import time

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from deep_light import time_to_sun_angles
from deep_light.kmeans_threeInputs import kmeans

NUM_CLUSTERS = 250

# Fixed Variables
IMAGE_SIZE = (230, 115)
NUM_NODES = 400  # number of nodes for each layer
LATITUDE = 47
LONGITUDE = 122
SM = 120

NPY_AB4 = "data/original_data/kmeans_" + str(NUM_CLUSTERS) + "_npy_ab4/"
NPY_AB0 = "data/original_data/kmeans_" + str(NUM_CLUSTERS) + "_npy_ab0/"
NPY_SKY = "data/original_data/kmeans_" + str(NUM_CLUSTERS) + "_npy_sky/"
TRAIN_NPY_DIR = "data/processed_data/train_" + str(NUM_CLUSTERS) + "_npy/"
TRAIN_SAVE_P = "data/processed_data/train_" + str(NUM_CLUSTERS) + "_combine.npy"
TRAIN_PROCESS_P = "data/final_data/processed_train_" + str(NUM_CLUSTERS) + "_combine.npy"

TEST_NPY_AB4 = 'data/original_test_all/test_ab4/'
TEST_NPY_AB0 = 'data/original_test_all/test_ab0/'
TEST_NPY_SKY = 'data/original_test_all/test_sky_ab4/'
TEST_NPY_DIR = 'data/processed_data/test_500_npy/'
TEST_SAVE_P = 'data/processed_data/test_500_combine.npy'
TEST_PROCESS_P = 'data/final_data/processed_test_500_combine.npy'


GLOBAL_PATH_NAMES = {
    'AB4' : "/AB4_ANNUAL_PANORAMA_npy/",
    'AB0' : "/AB0_ANNUAL_PANORAMA_npy/",
    'SKY' : "/AB4_SKY_ANNUAL_2018-11-30_npy/",
    }

def get_data_path(key):
    if key in GLOBAL_PATH_NAMES.keys():
        return GLOBAL_PATH_NAMES[key]

def set_data_path(key, path):
    GLOBAL_PATH_NAMES[key] = path
    

def selectSamples(ims, num_clusters):
    # read all the samples,
    # kmeans select 5 ones over 4 dimensionals
    data = ims[:, 0, :]
    samples = KMeans(n_clusters=num_clusters, init='random', n_init=20).fit(data[:, 2:6])
    n_ims = np.zeros([num_clusters, ims.shape[1]])
    n_nbrs = np.zeros([num_clusters, ims.shape[1], 4])

    i = 0
    for value in samples.cluster_centers_:
        nearest_im = findNearest(data[:, 2:6], value, ims)
        n_ims[i] = nearest_im[:, -1]
        n_nbrs[i] = nearest_im[:, 2:6]
        i = i + 1

    # plot the kmeans centers and nearest neighbors
    plot_distribution(samples, n_nbrs[:, 0, :], data[:, 2:6], "./5samples", num_clusters)
    ims[:, :, 7:12] = np.repeat(n_ims.T[np.newaxis, :, :], data.shape[0], axis=0)
    return ims


# find the nearest sample data to each cluster center
def findNearest(arr, value, data):
    d = ((arr - value) ** 2).sum(axis=1)
    ndx = d.argsort()
    nearest_idx = ndx[0]
    nearest_neiborgh = data[nearest_idx]
    smallest_distance = d[nearest_idx]
    return nearest_neiborgh


# plot the distribution diagrams
def plot_distribution(y_pred, n_nbrs, data, path, num_clusters):
    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.scatter(y_pred.cluster_centers_[:, 0], y_pred.cluster_centers_[:, 1], s=5, color='b', label="kmeans_centroids")
    plt.scatter(n_nbrs[:, 0], n_nbrs[:, 1], s=10, color='r', label="train")
    plt.scatter(data[:, 0], data[:, 1], s=1, color='g', label="test")
    plt.title("Sky Direct and Diffuse Irradiances Distribution", size=16)
    plt.xlabel('Direct')
    plt.ylabel('Diffuse')
    plt.legend(fancybox=True)
    plt.savefig(path + 'sky_pre_kmeans_' + str(num_clusters) + '.png')
    plt.close()

    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.scatter(y_pred.cluster_centers_[:, 2], y_pred.cluster_centers_[:, 3], s=5, color='b', label="kmeans_centroids")
    plt.scatter(n_nbrs[:, 2], n_nbrs[:, 3], s=10, color='r', label="train")
    plt.scatter(data[:, 2], data[:, 3], s=1, color='g', label="test")
    plt.title("Sun Altitude and Azimuth Distribution", size=16)
    plt.xlabel('Altitude')
    plt.ylabel('Azimuth')
    plt.legend(fancybox=True)
    plt.savefig(path + 'sun_pre_kmeans_' + str(num_clusters) + '.png')
    plt.close()


def angle_correction(py, p_lum):
    return p_lum * ((math.cos(math.pi * (py - IMAGE_SIZE[1] / 2) / IMAGE_SIZE[1]) + EPSILON) / (1 + EPSILON))


# convert radiance generated pic files to txt files
def pic2txt(bash_name):
    rad_command = "bash " + bash_name
    os.system(rad_command)


# convert txt files to numpy array
def txt2np(file_name):
    array = np.loadtxt(file_name)
    print(array.shape)
    array = np.reshape(array, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

    # Display.
    # plt.imshow(array, cmap = "gray")
    # plt.show()
    print("txt to numpy shape is: ", array.shape)
    return array


# read file_name and return month, date, and standard time seprately
def get_month_date(file_name):
    month = int(file_name.split('_')[1])
    date = int(file_name.split('_')[2])
    standard_time = float(file_name.split('_')[3])
    print("month, date and standard time is: ", month, date, standard_time)
    return month, date, standard_time


# read file_name and return direct irradiance and diffuse irradiance
def get_dir_dif(file_name):
    dir = int(file_name.split('_')[4].split('.')[0])
    dif = int(file_name.split('_')[5].split('.')[0])
    print("direct and diffuse irradiances are: ", dir, dif)
    return dir, dif


# read one image in txt format of file_name,
# return the numpy array with organized information of 5 labels, and 1 output value
def read_one_image(log_process, log_base, sky_map, input_dim, output_dim, dir_ab4, dir_ab0, dir_sky, dir_npy,
                   file_name):
    # load an image (numpy arrays) from a txt file
    im = np.load(dir_ab4 + file_name).astype(np.float64)
    im_resized = cv2.resize(im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # im_resized = im
    sun_im = np.load(dir_ab0 + file_name[:-4] + "_ab0.npy").astype(np.float64)
    sun_im_resized = cv2.resize(sun_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # sun_im_resized = sun_im
    sky_im = np.load(dir_sky + file_name).astype(np.float64)
    sky_im_resized = cv2.resize(sky_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # sky_im_resized = sky_im

    # create a new image frame and store all image information in this frame
    image_frame = np.zeros((IMAGE_SIZE[1] * IMAGE_SIZE[0], input_dim + output_dim), dtype='float32')

    file = file_name.split('/')[0]
    # get month, date and time from file name
    month, date, standard_time = get_month_date(file)
    dir, dif = get_dir_dif((file))

    # read image pixel by pixel, reorganize the data
    # return image with shape(PIXELS, 6(px, py, altitude, azimuth, ave_luminance(empty for now), real_luminance))
    for i in range(IMAGE_SIZE[1] * IMAGE_SIZE[0]):
        index1, index2 = np.unravel_index(i, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

        full_im_lum = im_resized[index1, index2]
        sun_patch_lum = sun_im_resized[index1, index2]
        sky_lum = sky_im_resized[index1, index2]

        if log_process == True:
            if full_im_lum < 1:
                image_frame[i, -1] = sys.float_info.min
            else:
                image_frame[i, -1] = math.log(full_im_lum, log_base)  # save sun patch image

            if sun_patch_lum < 1:
                image_frame[i, -2] = sys.float_info.min
            else:
                image_frame[i, -2] = math.log(sun_patch_lum, log_base)  # save sun patch image

            if sky_map:
                if sky_lum < 1:
                    image_frame[i, -3] = sys.float_info.min
                else:
                    image_frame[i, -3] = math.log(sky_lum, log_base)  # save sky luminance panorama image

        else:
            image_frame[i, -1] = full_im_lum
            image_frame[i, -2] = sun_patch_lum
            if sky_map:
                image_frame[i, -3] = sky_lum

        image_frame[i, 0:2] = index1, index2  # save px, py
        image_frame[i, 2:4] = time_to_sun_angles.timeToAltitudeAzimuth(date, month, standard_time, LATITUDE, LONGITUDE,
                                                                       SM)  # save altitude, azimuth
        image_frame[i, 4:6] = dir, dif  # save direct and diffuse irradiances

    # save this image
    dir_path = dir_npy + file_name + '.npy'
    np.save(dir_path, image_frame)
    return image_frame


"""
# normalize each column
def normalize_column(x):
    # x_normalized = (x - min) / (max - min)
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized, x_min, x_max


# normalize each column of all data: Px, Py, Lx, Ly, LU_AVERAGE, LU_REAL
def normalize_data(X, input_dim, output_dim):
    # normalize it using equation : x - min / max -min
    i = -(input_dim + output_dim)
    mins = []
    maxs = []
    while i <= -1:
        X[:, :, i], x_min, x_max = normalize_column(X[:, :, i])
        mins.append(x_min)
        maxs.append(x_max)
        i += 1
    return X, mins, maxs
"""


# instead of using the min and max value, use fixed value
def normalize_column(x, x_min, x_max):
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized


# normalize each column of all data: Px, Py, Lx, Ly, LU_AVERAGE, LU_REAL
def normalize_data(X, input_dim, output_dim):
    # normalize it using equation : x - min / max -min
    i = -(input_dim + output_dim)
    mins = [0, 0, 0, -125, 0, 1, 0.1, 0, 0]
    # maxs = [229, 114, 70, 125, 950, 660, 3.8, 8.6, 8.6]
    maxs = [229, 114, 70, 125, 950, 660, 9.2, 9.2, 9.2]
    while i <= -1:
        x_min = mins[i]
        x_max = maxs[i]
        X[:, :, i] = normalize_column(X[:, :, i], x_min, x_max)
        i += 1
    return X, mins, maxs


# add average luminance over all images to each pixel
def average_lum(images):
    num_images = images.shape[0]
    lum_avg = images.mean(axis=0)
    for i in range(num_images):
        images[i, :, 6] = lum_avg[:, -1]
    print("average lum image shape is: ", images.shape)
    return images


# read all radiance images from dir_name folder, and save the result image in save_path
def load_data_sample(log_process, log_base, sky_map, input_dim, output_dim, dir_ab4, dir_ab0, dir_sky, npy_dir,
                     save_path, type, gen_txt=False):
    print("start loading data")
    # start record time
    start_time = time.time()

    if os.path.isfile(save_path):
        print("Reading captured image from {:s} ...".format(str(save_path)))
        normalized_image = np.load(save_path)

    else:

        if gen_txt:
            # convert radiance generated pic files to txt files
            pic2txt("pic2txt.bash")

        # read all images of txt formats, organize the data
        # data should be: num_image * 6 (5 first) * (230*348)
        # then, read per image, stack all images
        # np.dstack, store multiple 2d arrays to a 3d array

        result_im = []

        if (os.path.exists(npy_dir)) == False:
            os.makedirs(npy_dir)

        # directly read data from npy directory
        if os.listdir(npy_dir) == "":
            for file in os.listdir(npy_dir):
                if file.endswith(".npy"):
                    file_name_npy = os.path.join(npy_dir, file)
                    im = np.load(file_name_npy)
                    if result_im != []:
                        result_im = np.dstack((result_im, im.T))
                    else:
                        result_im = im.T

        # read from txt folders and transfer to npy file first
        else:

            for file in os.listdir(dir_ab4):
                if file.endswith(".npy"):
                    src_ab4 = dir_ab4 + file[:-4] + ".npy"
                    src_ab0 = dir_ab0 + file[:-4] + "_ab0.npy"
                    src_sky = dir_sky + file[:-4] + ".npy"
                    if (os.path.isfile(src_ab4)) and (os.path.isfile(src_ab0)) and (os.path.isfile(src_sky)):
                        im = read_one_image(log_process, log_base, sky_map, input_dim, output_dim, dir_ab4, dir_ab0,
                                            dir_sky, npy_dir, file)
                        if result_im != []:
                            result_im = np.dstack((result_im, im.T))
                        else:
                            result_im = im.T

        #print(list)
        # change the shape of result_im to match the waldorf format
        # return format of num_im, num_pixels, num_var
        result_im = result_im.T
        print("result im second run is: ", result_im.shape)

        # add average luminance
        result_im = average_lum(result_im)
        print("after add average lum is: ", result_im.shape)

        # result_im = selectSamples(result_im, 5)
        # print("done adding 5 samples")

        # normalize each column

        normalized_image, mins, maxs = normalize_data(result_im, input_dim, output_dim)

        print("final normalized image shape is: ", normalized_image.shape)

        save_dir = save_path.split('/')[0]
        if (os.path.exists(save_dir)) == False:
            os.makedirs(save_dir)
        # save result
        np.save(save_path, normalized_image)

        # print time information
        print("load data, done")
        duration = int(time.time() - start_time)
        minutes, seconds = duration // 60, duration % 60
        print("Time spend on load all captured data: " + str(minutes) + ':' + str(seconds))

        if sky_map:
            # save full normalization information to txt files
            with open("data/processed_data/" + type + "min_max_normalization.txt", "w") as myfile:
                myfile.write('This information is for restore normalization purpose:'
                             'Px min = %.5f, Px max = %.5f, '
                             'Py min = %.5f, Py max = %.5f, '
                             'Altitude min = %.5f, Altitude max =%.5f, '
                             'Azimuth min= %.5f, Azimuth max=%.5f, '
                             'Direct min=%.5f, Direct max=%.5f, '
                             'Diffuse min=%.5f, Diffuse max=%.5f,'
                             'Average luminance min=%.5f, Average luminance max=%.5f, '
                             'Sky panorama luminance min=%.5f, Sky panorama luminance max=%.5f,'
                             'Sun patch luminance min=%.5f, Sun patch luminance max=%.5f,'
                             'Real Luminance min=%.5f, Real Luminance max=%.5f.'
                             % (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2], mins[3], maxs[3],
                                mins[4], maxs[4], mins[5], maxs[5], mins[6], maxs[6], mins[7], maxs[7],
                                mins[8], maxs[8], mins[9], maxs[9]))
                myfile.close()
        else:
            # save full normalization information to txt files
            with open("data/processed_data/" + type + "min_max_normalization.txt", "w") as myfile:
                myfile.write('This information is for restore normalization purpose:'
                             'Px min = %.5f, Px max = %.5f, '
                             'Py min = %.5f, Py max = %.5f, '
                             'Altitude min = %.5f, Altitude max =%.5f, '
                             'Azimuth min= %.5f, Azimuth max=%.5f, '
                             'Direct min=%.5f, Direct max=%.5f, '
                             'Diffuse min=%.5f, Diffuse max=%.5f,'
                             'Average luminance min=%.5f, Average luminance max=%.5f, '
                             'Sun patch luminance min=%.5f, Sun patch luminance max=%.5f,'
                             'Real Luminance min=%.5f, Real Luminance max=%.5f.'
                             % (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2], mins[3], maxs[3],
                                mins[4], maxs[4], mins[5], maxs[5], mins[6], maxs[6], mins[7], maxs[7],
                                mins[8], maxs[8]))
                myfile.close()

        # save normalization information to txt files
        with open("data/processed_data/" + type + "_restoration_min_max.txt", "w") as myfile2:
            myfile2.write(str(mins[input_dim]) + "," + str(maxs[input_dim]))
            myfile2.close()

        with open("data/processed_data/" + type + "_restoration_min_max_parameters.txt", "w") as myfile3:
            myfile3.write(str(mins[2]) + "," + str(maxs[2]) + "," + str(mins[3]) + "," + str(maxs[3])
                          + "," + str(mins[4]) + "," + str(maxs[4]) + "," + str(mins[5]) + "," + str(maxs[5]))
            myfile3.close()

    return normalized_image


# shuffle and split training and validation data
def split_val_train(data, s, Shuffle):
    # shuffle random data
    idx = np.arange(data.shape[0])
    if Shuffle:
        np.random.shuffle(idx)
    n_im = data.shape[0]
    n_val = round(s * n_im)
    return data[idx[:n_val]], data[idx[n_val:]]


def split_input_output(data, input_dim):
    return data[:, :input_dim], data[:, input_dim:]


def hdr_to_image(RGBColor, gamma):
    g = 1.0 / gamma
    RGB = np.power(RGBColor, g)

    return RGB


def plotNormDistributionSun(captured_data, index, type, norm):
    im_sun = captured_data[0, :, index]  # num of image, imx*imy, 9
    im_sun = im_sun.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
    im_sun = np.fliplr(im_sun)
    plt.hist(im_sun.ravel(), bins=256, range=(0, 1.0), fc='k', ec='k')
    print("hist")
    if (os.path.exists("normalilzation/")) == False:
        os.makedirs("normalilzation/")
    plt.savefig("./test_pics/normalization/" + type + "_sun_" + norm + ".png")
    plt.close()


def plotNormDistributionAll(captured_data, index, data_type, norm, im_type):
    ims = captured_data[:, :, index]  # num of image, imx*imy, 9
    ims = ims.reshape(ims.shape[0], IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
    plt.hist(ims.ravel(), bins=256, range=(0, 1), fc='k', ec='k')
    print("hist")
    if (os.path.exists("./test_pics/normalization/")) == False:
        os.makedirs("./test_pics/normalization/")
    plt.savefig("./test_pics/normalization/" + data_type + "_" + im_type + "_include0s_" + norm + ".png")
    plt.close()


def plotNormDistributionAllExclude0s(captured_data, index, data_type, norm, im_type):
    ims = captured_data[:, :, index]  # num of image, imx*imy, 9
    ims = ims.reshape(ims.shape[0], IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
    all_pixel_luminance = ims.ravel()
    not0s = all_pixel_luminance[all_pixel_luminance != 0]
    plt.hist(not0s, bins=256, range=(0, 1.0), fc='k', ec='k')
    print("hist")
    plt.savefig("./test_pics/normalization/" + data_type + "_" + im_type + "_exclude0s_" + norm + ".png")
    plt.close()


def preprocessInput(gamma, log_process, skymap, log_base, input_dim, output_dim, dir_p_ab4, dir_p_ab0, dir_sky, npy_dir,
                    save_p, save_npy, type):
    # load normalized data
    captured_data = load_data_sample(log_process, log_base, skymap, input_dim, output_dim, dir_p_ab4, dir_p_ab0,
                                     dir_sky, npy_dir, save_p, type, gen_txt=False)
    print(captured_data.shape)

    # plot distribution before correction
    plotNormDistributionAll(captured_data, -1, type, "before", "full")
    plotNormDistributionAllExclude0s(captured_data, -1, type, "before", "full")

    plotNormDistributionAll(captured_data, -2, type, "before", "sun")
    plotNormDistributionAllExclude0s(captured_data, -2, type, "before", "sun")

    plotNormDistributionAll(captured_data, -3, type, "before", "sky")
    plotNormDistributionAllExclude0s(captured_data, -3, type, "before", "sky")

    plt.imshow(captured_data[0, :, -1].reshape([IMAGE_SIZE[0], IMAGE_SIZE[1]]))
    plt.savefig("test_pics/test_full.png")
    plt.imshow(captured_data[0, :, -2].reshape([IMAGE_SIZE[0], IMAGE_SIZE[1]]))
    plt.savefig("test_pics/test_sun.png")
    plt.imshow(captured_data[0, :, -3].reshape([IMAGE_SIZE[0], IMAGE_SIZE[1]]))
    plt.savefig("test_pics/test_sky.png")
    plt.imshow(captured_data[0, :, -4].reshape([IMAGE_SIZE[0], IMAGE_SIZE[1]]))
    plt.savefig("test_pics/test_depth.png")

    # add gamma correction
    captured_data[:, :, -1] = hdr_to_image(captured_data[:, :, -1], gamma)  # full
    # captured_data[:, :, -2] = hdr_to_image(captured_data[:, :, -2], gamma) #sun

    if skymap:
        captured_data[:, :, -3] = hdr_to_image(captured_data[:, :, -3], gamma)  # sky

    # plot distribution after correction
    plotNormDistributionAll(captured_data, -1, type, "after", "full")
    plotNormDistributionAllExclude0s(captured_data, -1, type, "after", "full")

    plotNormDistributionAll(captured_data, -2, type, "after", "sun")
    plotNormDistributionAllExclude0s(captured_data, -2, type, "after", "sun")

    plotNormDistributionAll(captured_data, -3, type, "after", "sky")
    plotNormDistributionAllExclude0s(captured_data, -3, type, "after", "sky")

    # save the result
    print("done")
    if (os.path.exists("data/final_data/")) == False:
        os.makedirs("data/final_data/")
    np.save(save_npy, captured_data)
    return


def gen_training_data(VAL_PERCENTAGE, SHUFFLE):
    train = np.load(TRAIN_PROCESS_P)
    test = np.load(TEST_PROCESS_P)
    val, train = split_val_train(train, VAL_PERCENTAGE, SHUFFLE)
    return train, val, test


# main function
def preprocess(log, gamma, skymap, log_base, num_clusters, gen_trainSet,
               latitude, longitude, sm, ab4_dir, ab0_dir, sky_dir):
    if gen_trainSet:
        kmeans(latitude, longitude, sm, ab4_dir, ab0_dir, sky_dir, num_clusters)
    if skymap:
        input_dim = 9
    else:
        input_dim = 8

    output_dim = 1
    preprocessInput(gamma, log, skymap, log_base, input_dim, output_dim, NPY_AB4, NPY_AB0, NPY_SKY, TRAIN_NPY_DIR,
                    TRAIN_SAVE_P, TRAIN_PROCESS_P, "train")
    preprocessInput(gamma, log, skymap, log_base, input_dim, output_dim, TEST_NPY_AB4, TEST_NPY_AB0, TEST_NPY_SKY,
                    TEST_NPY_DIR, TEST_SAVE_P, TEST_PROCESS_P, "test")
