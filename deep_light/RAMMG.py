
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




import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from deep_light import plot
from sklearn.metrics import r2_score

IMAGE_SIZE = (230, 115)
EPSILON = 0
def get_contrast_map(im):
    contrast_map = np.zeros((im.shape[0], im.shape[1]), dtype='float32')
    for p in range(im.shape[0] * im.shape[1]):
        i, j = np.unravel_index(p, (im.shape[0], im.shape[1]))
        if i != 0 and i != im.shape[0]-1 and j != 0 and j != im.shape[1] - 1:
            contrast_map[i][j] = get_contrast(im, i, j)
    return contrast_map

def get_contrast(im, i, j):
    l_ij = im[i][j]
    c_4_neibors = abs(l_ij - im[i + 1][j]) + abs(l_ij - im[i][j + 1]) \
                  + abs(l_ij - im[i - 1][j]) + abs(l_ij - im[i][j - 1])
    c_4_corners = abs(l_ij - im[i + 1][j + 1]) + abs(l_ij - im[i + 1][j - 1]) \
                  + abs(l_ij - im[i - 1][j - 1]) + abs(l_ij - im[i - 1][j + 1])
    c_ij = (c_4_neibors + c_4_corners / math.sqrt(2)) / (4 + 2 * math.sqrt(2))
    return c_ij

def angle_correction(py, h):
    return math.cos(math.pi * (py - h / 2) / h) + 0.01

def im_angle(w, h):
    image_frame = np.zeros((w, h), dtype='float32')
    for i in range(w * h):
        index1, index2 = np.unravel_index(i, (w, h))
        image_frame[index1][index2] = angle_correction(index2, h)
    return image_frame

def get_RAMMG(im, levels_start, levels_end, angle_bool):
    w = im.shape[0]
    h = im.shape[1]
    im = im.reshape(w, h)
    #angle correction
    if angle_bool:
        im = im * im_angle(w, h)
    sum_c_mean = 0
    for i in range(levels_start, levels_end):
        im_level = cv2.resize(im.astype(np.float32), (int(h/(2**i)), int(w/(2**i))))
        map_level = get_contrast_map(im_level)
        c_mean = np.mean(map_level)
        sum_c_mean += c_mean
    RAMMG = sum_c_mean / (levels_end - levels_start)
    return RAMMG

def plotRAMMGAnalysis(r_p, r_t, save_path):
    r2 = r2_score(r_t, r_p)
    mse = np.mean(np.square(np.array(r_t) - np.array(r_p)))
    plt.scatter(r_t, r_p, s=10, c='r',
                label="Predicted vs Truth RAMMG(sample size = 200)")
    plt.plot(r_t, r_t, label='1=1 line', c='black',linewidth = 0.5)
    plt.title(("Predicted vs Truth RAMMG, R Square = %.3f, Mse = %.3e" % (r2, mse)), size=6)
    plt.ylabel('Predicted RAMMG')
    plt.xlabel('Ground truth RAMMG')
    plt.legend(fancybox=True)
    plt.savefig(save_path + "RAMMG_ANALYSIS")
    plt.close()

def getAllRAMMGs(t, p, level_start, level_end, angle_bool, save_path):
    r_ts = []
    r_ps = []
    for i in range(t.shape[0]):
        im_t = t[i, :]
        im_p = p[i, :]
        r_t = get_RAMMG(im_t, level_start, level_end, angle_bool)
        r_p = get_RAMMG(im_p, level_start, level_end, angle_bool)
        print(i)
        print(r_t)
        print(r_p)
        if r_t < 500:
            r_ts.append(r_t)
            r_ps.append(r_p)
    np.save(save_path + "truth_RAMMGs.npy", r_ts)
    np.save(save_path + "predicted_RAMMGs.npy", r_ps)
    return r_ts, r_ps




version = ""
result_path = "results/"

t = np.load(result_path + "test_truth.npy")



p = np.load(result_path + "test_prediction.npy")
process_data_path = "data/processed_data" + version + "/"

#im post process, reverse log, normalization, gamma
diff, t, p = plot.postProcessIms(t, p, "test_", t.shape[0], 1.5, process_data_path)
diff, t, p = plot.post_process_log(t, p, True, 10, t.shape[0])

#im apply angle filter

r_ts, r_ps = getAllRAMMGs(t, p, level_start=0, level_end=5, angle_bool=False, save_path = result_path)
#r_ts = np.load("results/truth_RAMMGs.npy")
#r_ps = np.load("results/predicted_RAMMGs.npy")
plotRAMMGAnalysis(r_ps, r_ts, save_path=result_path)
