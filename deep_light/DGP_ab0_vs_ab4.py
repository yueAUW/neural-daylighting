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
import time

import numpy as np

from deep_light.pano2fish_lum import save_im_falsecolor, plotDGPAnalysis_whole

IMAGE_SIZE = (460, 230)


# read all radiance images from dir_name folder, and save the result image in save_path
def load_data_sample(dir_ab4, dir_ab0, npy_dir, save_path_full, save_path_sun):
    print("start loading data")
    # start record time
    start_time = time.time()

    result_im_full = []
    result_im_sun = []

    for file in os.listdir(dir_ab4):
        if file.endswith(".npy"):
            im_full = np.load(dir_ab4 + file)
            im_sun = np.load(dir_ab0 + file[:-4] + "_ab0.npy")
            if result_im_full != []:
                result_im_full = np.dstack((result_im_full, im_full.T))
            else:
                result_im_full = im_full.T
            if result_im_sun != []:
                result_im_sun = np.dstack((result_im_sun, im_sun.T))
            else:
                result_im_sun = im_sun.T

    print(list)
    # change the shape of result_im to match the waldorf format
    # return format of num_im, num_pixels, num_var
    result_im_full = result_im_full.T
    result_im_sun = result_im_sun.T
    print("result im second run is: ", result_im_full.shape)

    save_dir_full = save_path_full.split('/')[0]
    if (os.path.exists(save_dir_full)) == False:
        os.makedirs(save_dir_full)
    # save result
    np.save(save_path_full, result_im_full)

    save_dir_sun = save_path_sun.split('/')[0]
    if (os.path.exists(save_dir_sun)) == False:
        os.makedirs(save_dir_sun)
    # save result
    np.save(save_path_sun, result_im_sun)

    # print time information
    print("load data, done")
    duration = int(time.time() - start_time)
    minutes, seconds = duration // 60, duration % 60
    print("Time spend on load all captured data: " + str(minutes) + ':' + str(seconds))


# import ab0 data and ab4 data and compare dgps
def plotAnalysis(sun_ims, full_ims):
    num = int(sun_ims.shape[0])
    print(sun_ims.shape)

    dgps_p = []
    dgps_t = []
    dgps_t_mean = []
    dgps_t_max = []

    for i in range(num):
        n_im = full_ims[i, :, :]
        p_im = sun_ims[i, :, :]

        # calculate the loss and error according to paper definition
        loss = np.sum(np.square(p_im - n_im))
        truth = np.sum(np.square(n_im))
        error = np.divide(loss, truth)
        print("Total loss is: ", loss)
        print("Error rate is: ", error)
        mse = ((p_im - n_im) ** 2).mean(axis=None)
        print("Mse is: ", mse)

        if (os.path.exists("DGPs/") == False):
            os.mkdir("DGPs/")
        p, t, t_mean, t_max = save_im_falsecolor(p_im, n_im, str(i), 3000, "results_3000/", IMAGE_SIZE[0],
                                                 IMAGE_SIZE[1], IMAGE_SIZE[1])
        dgps_p.extend(p)
        dgps_t.extend(t)
        dgps_t_mean.append(t_mean)
        dgps_t_max.append(t_max)
        save_im_falsecolor(p_im, n_im, str(i), 10000, "results_10000/", IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[1])

    np.save("dgps_t.npy", dgps_t)
    np.save("dgps_p.npy", dgps_p)
    plotDGPAnalysis_whole(dgps_t, dgps_p)


TEST_NPY_AB4 = 'test_ab4/'
TEST_NPY_AB0 = 'test_ab0/'
TEST_NPY_DIR = 'test_500_npy/'
TEST_SAVE_FULL_P = 'data_npy/test_full.npy'
TEST_SAVE_SUN_P = 'data_npy/test_sun.npy'
# captured_data = load_data_sample(TEST_NPY_AB4, TEST_NPY_AB0, TEST_NPY_DIR, TEST_SAVE_FULL_P, TEST_SAVE_SUN_P)
sun = np.load("./data_npy/test_sun.npy")
full = np.load("./data_npy/test_full.npy")
plotAnalysis(sun, full)
