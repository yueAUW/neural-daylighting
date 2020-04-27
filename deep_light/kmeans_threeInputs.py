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



import os.path
import shutil
from shutil import copyfile

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from deep_light import time_to_sun_angles


#####select the training images from the data folder


# read seattle weahter txt data
# return data format as al, az, dir, dif
def readTxt(latitude, longitude, sm):
    # read each line of weather data
    # transfer the time to sun angles
    # save the data
    f = open("testseattle.txt", "r")
    lines = f.readlines()
    data = np.zeros([4709, 7])
    i = 0
    for line in lines:
        month = int(line.splitlines()[0].split(",")[0])
        date = int(line.splitlines()[0].split(",")[1])
        time = float(line.splitlines()[0].split(",")[2])
        dir = int(line.splitlines()[0].split(",")[3])
        dif = int(line.splitlines()[0].split(",")[4])
        al, az = time_to_sun_angles.timeToAltitudeAzimuth(date, month, time, latitude, longitude, sm)
        if (dir > 10) or (dif > 10):
            data[i] = np.array([dir, dif, al, az, month, date, time])
            i = i + 1
    return data


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


def kmeans(latitude, longitude, sm, ab4_dir, ab0_dir, sky_dir, num_clusters):
    data = readTxt(latitude, longitude, sm)
    y_pred = KMeans(n_clusters=num_clusters, init='random', n_init=20).fit(data[:, :4])
    n_nbrs = np.zeros([num_clusters, 7])
    path_ab4 = "data/original_data/kmeans_" + str(num_clusters) + "_npy_ab4/"
    path_ab0 = "data/original_data/kmeans_" + str(num_clusters) + "_npy_ab0/"
    path_sky = "data/original_data/kmeans_" + str(num_clusters) + "_npy_sky/"

    if (os.path.exists(path_ab4)):
        shutil.rmtree(path_ab4)

    if (os.path.exists(path_ab0)):
        shutil.rmtree(path_ab0)

    if (os.path.exists(path_sky)):
        shutil.rmtree(path_sky)

    if (not os.path.exists("data/")):
        os.mkdir("data/")

    if (not os.path.exists("data/original_data/")):
        os.mkdir("data/original_data/")

    os.mkdir(path_ab4)
    os.mkdir(path_ab0)
    os.mkdir(path_sky)

    i = 0
    for value in y_pred.cluster_centers_:
        nearest_neiborgh = findNearest(data[:, :4], value, data)
        n_nbrs[i] = nearest_neiborgh
        i = i + 1

    # plot the kmeans centers and nearest neighbors
    plot_distribution(y_pred, n_nbrs, data, path_ab4, num_clusters)

    # put the data into two folders train and test
    i = 0
    for i in range(num_clusters):
        file_name_npy = "pano_" + str(int(n_nbrs[i][4])) + "_" + str(int(n_nbrs[i][5])) + "_" + str(
            n_nbrs[i][6]) + "_" + str(int(n_nbrs[i][0])) + "_" \
                        + str(int(n_nbrs[i][1]))
        src_ab4 = ab4_dir + file_name_npy + ".npy"
        src_ab0 = ab0_dir + file_name_npy + "_ab0.npy"
        src_sky = sky_dir + file_name_npy + ".npy"

        if (os.path.isfile(src_ab4)) and (os.path.isfile(src_ab0)) and (os.path.isfile(src_sky)):
            dst_ab4 = path_ab4 + file_name_npy + ".npy"
            dst_ab0 = path_ab0 + file_name_npy + "_ab0.npy"
            dst_sky = path_sky + file_name_npy + ".npy"
            copyfile(src_ab4, dst_ab4)
            copyfile(src_ab0, dst_ab0)
            copyfile(src_sky, dst_sky)

        i = i + 1


# These parameters are location related. Right now we use Seattle's parameters.
# latitude = 47
# longitude = 122
# sm = 120
# ab4_dir = "W:/ALL_DATA/AB4_ANNUAL_PANORAMA_npy/"
# ab0_dir = "W:/ALL_DATA/AB0_ANNUAL_PANORAMA_npy/"
# sky_dir = "W:/ALL_DATA/AB4_SKY_ANNUAL_2018-11-30_npy/"
# num_clusters = 250
#

# kmeans(latitude, longitude, sm, ab4_dir, ab0_dir, sky_dir, num_clusters)
