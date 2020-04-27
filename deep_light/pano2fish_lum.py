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
import sys
import subprocess

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from sklearn.metrics import r2_score

COLORMAP = "jet"
NO_EVALGLARE=False

### Hack to set the path if Radiance is installed on windows
if os.name is 'nt':
    radiance_path="C:/Program Files/Radiance/bin"
    if os.path.exists(radiance_path):
        os.environ['PATH']=radiance_path+os.pathsep+os.environ['PATH']
        sys.path.append(radiance_path)


# normalize fisheye image
# input fisheye location index: i j ((0,0) at left low corner), fisheye image dimension: d_fish
# return normalized (x, y range from -1 to 1) location index: x y ((0,0) at image center)
def normfisheye(i, j, d_fish):
    x = 2 * float(i) / float(d_fish) - 1
    y = 2 * float(j) / float(d_fish) - 1
    return x, y


# transfer noramlized fisheye image to angular fisheye projection
# input normalized fisheye location index: x, y
# output angular index, phy, theta, and r
def projection2angularfisheye(x, y):
    r = math.sqrt(x * x + y * y)
    phi = math.atan2(y, x)
    theta = r * math.pi / 2
    return phi, r, theta


# transfer fisheye dome to unit vector
# input angular index
# return unit vector index px py pz
def fisheye2unitvector(phi, theta):
    px = math.sin(theta) * math.cos(phi)
    py = math.sin(theta) * math.sin(phi)
    pz = math.cos(theta)
    return px, py, pz


# transfer unit vector to longitude and latitude
# input unit vector index px py pz
# return longgitude and latitude
def unitvector2polarcoordinates(px, py, pz, view_offset):
    longitude = math.atan2(pz, px) + view_offset
    latitude = math.atan2(py, math.sqrt(px * px + pz * pz))
    return longitude, latitude


# transfer longgitude and latitude to spherical panorama
# input longitude, latitude, output image width and height
# output spherical panorama location index i, j
def polarcoordinates2sphericalpanorama(longitude, latitude, w, h):
    x = longitude / math.pi
    y = 2 * latitude / math.pi
    # y = math.sin(latitude)
    i = (x + 1) * w / 2
    j = (y + 1) * h / 2
    if i > w:
        i = i - w
    if j > h:
        j = j - h

    return i, j


# given a pixel location on fisheye image (angular projection), find the corresponding location index on spherical panorama
# input: fisheye image pixel i, j, fisheye image width = height = d_fish,
# panorama width and height, view angle (range from -180 ~ + 180)
# output: panorama image pixel m, k
def find_corresponding_pixel_index(i, j, d_fish, w, h, view_offset_angle):
    view_offset = view_offset_angle / 180.0 * math.pi
    x, y = normfisheye(i, j, d_fish)
    phi, r, theta = projection2angularfisheye(x, y)
    px, py, pz = fisheye2unitvector(phi, theta)
    longitude, latitude = unitvector2polarcoordinates(px, py, pz, view_offset)
    m, k = polarcoordinates2sphericalpanorama(longitude, latitude, w, h)
    return int(m) - 1, int(k) - 1


# transfer spherical panoramas to angular fisheye image
# im_in: spherical panorama image
# w: spherical panoram width, h: height
# d_fish: target fisheye image width = height
# angle: view angle range from -180 to 180
# savePath: result fisheye image file path
def sphere2angularfisheye(im_in, w, h, d_fish, angle):
    im_out = np.zeros([d_fish, d_fish])
    for i in range(d_fish):
        for j in range(d_fish):
            m, k = find_corresponding_pixel_index(i, j, d_fish, w, h, angle)
            im_out[i, j] = im_in[m, k]
    return im_out


# crop square image to circle
# im: input square image
# r: output circle image radius
def crop_circle(im, r):
    w = im.shape[0]
    h = im.shape[1]
    for x in range(w):
        for y in range(h):
            if math.sqrt((x - w / 2) * (x - w / 2) + (y - h / 2) * (y - h / 2)) > r:
                im[x, y] = 0
    return im


# save image as txt file
# im: input image
# w: input image width
# h: input image height
def gentxt(im, w, h, path):
    im = im.reshape(w * h)
    im = im / 179
    im_1 = np.array([im, im, im])
    im_2 = im_1.T
    np.savetxt(os.path.join(path + ".txt"), im_2)


# convert txt files to radiance generated pic files
# txt_name : name of the txt file
def txt2hdr(txt_name, fisheye_dimension):
    os.system("pvalue -r -h -d -x " + str(fisheye_dimension) + " -y " + str(
        fisheye_dimension) + " " + txt_name + " >" + txt_name + ".hdr")


# calculate glare index DGP from .pic fisheye file
# im_fisheye: fisheye image
def calc_DGP(path):
    
    global NO_EVALGLARE
    
    if NO_EVALGLARE:
        return "0.0"
    try:
        glare_idx = subprocess.Popen("evalglare -vta -vv 180 -vh 180 -r 0.08 " + path + ".txt.hdr",
                                 stdout=subprocess.PIPE).stdout.read()
        return str(glare_idx).split(' ')[1]
       
    except subprocess.CalledProcessError as e:
        print(e.output)
        return "0.0"
    except FileNotFoundError as e:
        print(e.strerror, '\nIs glare analysis software evalgare installed?')
       
        NO_EVALGLARE=True
        return "0.0"
    
    # glare_im = subprocess.Popen("evalglare -vta -vv 180 -vh 180 -r 0.08 -c check.hdr" + path + ".txt.hdr", stdout=subprocess.PIPE).stdout.read()
    


def calc_DGP_im(spherical_im, view_angle, panorama_w, panorama_h, fisheye_im_dimension, path):
    im = sphere2angularfisheye(spherical_im, panorama_w, panorama_h, fisheye_im_dimension, view_angle)
    im = np.flip(im, axis=0)
    im = np.rot90(im, 1)
    crop_circle(im, fisheye_im_dimension / 2)
    np.save("results/test.npy", im)
    gentxt(im, fisheye_im_dimension, fisheye_im_dimension, path)
    txt2hdr(path + ".txt", fisheye_im_dimension)
    DGP = float(calc_DGP(path))
    print(DGP)
    if (os.path.exists("test.npy")):
        os.remove("test.npy")
    if (os.path.exists("test.txt")):
        os.remove("test.txt")
    return DGP, im


# display image in falsecolor
# im_p : predicted panroama image
# im_t: truth panorama
# pano_path: panorama image save path
# vmax: false color image color bar max value
# savePath: falsecolor image save path
# panorama_w, and panorama_h : width and height of the panorama image
# fisheye_im_dimension: fisheye image width = height
def save_im_falsecolor(im_out_p, im_out_t, pano_path, vmax, savePath, panorama_w, panorama_h, fisheye_im_dimension):
    # setup the figure
    fig = plt.figure(figsize=(30, 10))
    plt.suptitle("DGP and Falsecolor Analysis")
    plt.axis("off")
    DGPs_p = []
    DGPs_t = []
    DGPs_mean = []
    DGPs_max = []
    angle_step = 36
    if (os.path.exists("results/hdrs_t/") == False):
        os.mkdir("results/hdrs_t/")
    if (os.path.exists("results/hdrs_p/") == False):
        os.mkdir("results/hdrs_p/")

    for view_angle in range(-180, 180):
        if view_angle % angle_step == 0:
            DGP_t, im_t = calc_DGP_im(im_out_t, view_angle, panorama_w, panorama_h, fisheye_im_dimension,
                                      "results/hdrs_t/" + pano_path + "_" + str(view_angle))
            DGPs_t.append(DGP_t)
            DGP_p, im_p = calc_DGP_im(im_out_p, view_angle, panorama_w, panorama_h, fisheye_im_dimension,
                                      "results/hdrs_p/" + pano_path + "_" + str(view_angle))
            DGPs_p.append(DGP_p)

            dif = abs(im_t - im_p)
            DGP_dif = abs(DGP_t - DGP_p)

            im_t = np.clip(im_t, 1, vmax)
            im_p = np.clip(im_p, 1, vmax)
            dif = np.clip(dif, 1, vmax)

            colormap = COLORMAP
            fig.add_subplot(3, 10, view_angle / angle_step + 180 / angle_step + 1)
            implot1 = plt.imshow(im_t, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
            plt.title("DGP_truth: %.2f" % (DGP_t))
            plt.axis("off")

            fig.add_subplot(3, 10, view_angle / angle_step + 180 / angle_step + 1 + 10)
            implot2 = plt.imshow(im_p, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
            plt.title("DGP_prediction: %.2f" % (DGP_p))
            plt.axis("off")

            fig.add_subplot(3, 10, view_angle / angle_step + 180 / angle_step + 1 + 20)
            implot3 = plt.imshow(dif, cmap=colormap, norm=LogNorm(vmin=1, vmax=vmax))
            if DGP_dif > 0.1 and DGP_dif < 0.2:
                plt.title("DGP_difference: %.2f" % (DGP_dif), color="red")
            elif DGP_dif > 0.2:
                plt.title("DGP_difference: %.2f" % (DGP_dif), color="green")
            else:
                plt.title("DGP_difference: %.2f" % (DGP_dif))
            plt.axis("off")

    time_name = pano_path
    plt.figtext(.5, .93, (time_name,
                          "DGP truth max: %.2f, prediction max: %.2f" % (
                          np.array(DGPs_t).max(), np.array(DGPs_p).max()),
                          "DGP min: %.2f, prediction min: %.2f" % (np.array(DGPs_t).min(), np.array(DGPs_p).min()),
                          "DGP mean: %.2f, prediction mean: %.2f" % (np.array(DGPs_t).mean(), np.array(DGPs_p).mean()),
                          "DGP MSE %.5f" % (np.mean((np.array(DGPs_t) - np.array(DGPs_p)) ** 2))),
                fontsize=12, ha='center')

    # add color index bar
    cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8])

    if (vmax == 3000):
        formatter = LogFormatter(3, labelOnlyBase=False)
        cb = plt.colorbar(ticks=[1, 4.1, 12.2, 36.5, 109, 325.8, 973.4, 3000], format=formatter, cax=cbaxes)

    else:
        formatter = LogFormatter(3.7, labelOnlyBase=False)
        cb = plt.colorbar(ticks=[1, 3.7, 14, 55, 204, 755, 2794, 10000], format=formatter, cax=cbaxes)

    cb.set_label('Luminance (cd/m2)', rotation=90)
    print(os.path.join(savePath, time_name + ".png"))
    # save the image
    plt.savefig(os.path.join(savePath, time_name + ".png"))
    plt.close()
    plotDGPAnalysis(DGPs_t, DGPs_p, time_name)
    DGPs_mean.append(np.array(DGPs_t).mean())
    DGPs_max.append(np.array(DGPs_t).max())
    return DGPs_p, DGPs_t, DGPs_mean, DGPs_max


def plotDGPAnalysis(DGPs_t, DGPs_p, name):
    plt.scatter(DGPs_t, DGPs_p, s=10, c='r',
                label="Predicted DGP vs Truth DGP (sample size = 200)")
    x = np.arange(0, 1, 0.01)
    plt.plot(x, x, label='1=1 line', c='black', linewidth=0.5)
    plt.plot([0, 0.35], [0.35, 0.35], [0.35, 0.35], [0.35, 0], c='y', linewidth=0.5)
    plt.plot([0, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0], c='y', linewidth=0.5)
    plt.plot([0, 0.45], [0.45, 0.45], [0.45, 0.45], [0.45, 0], c='y', linewidth=0.5)
    plt.title("Predicted DGP versus Ground Truth DGP", size=16)
    plt.xlabel('Ground Truth DGP')
    plt.ylabel('Predicted DGP')
    plt.legend(fancybox=True)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("results/DGPs/" + name + ".png")
    plt.close()


def plotMSE_whole(solid_mse):
    i = np.arange(len(solid_mse))
    plt.scatter(i, solid_mse, s=10, c='r',
                label="Solid Angle Weighted MSE (sample size = 200)")
    plt.figtext(.5, .95, ("Solid Angle MSE"),
                fontsize=12, ha='center')
    plt.title("MSE Distribution", size=16)
    plt.xlabel('time')
    plt.ylabel('Solid Angle Weighted MSE')
    plt.legend(fancybox=True)
    plt.savefig("results/DGPs/MSE.png")
    plt.close()


def plotDGPDistribution(dgps_mse):
    i = np.arange(len(dgps_mse))
    plt.scatter(i, dgps_mse, s=10, c='r',
                label="DGP MSE (sample size = 200)")
    plt.figtext(.5, .95, ("DGP MSE Distribution"),
                fontsize=12, ha='center')
    plt.title("DGP MSE Distribution", size=16)
    plt.xlabel('time')
    plt.ylabel('DGP MSE')
    plt.legend(fancybox=True)
    plt.savefig("results/DGPs/DGP_MSE.png")
    plt.close()


def plotDGPAnalysis_whole(DGPs_t, DGPs_p):
    r2 = r2_score(DGPs_t, DGPs_p)
    plt.scatter(DGPs_t, DGPs_p, s=1, c='r',
                label="Predicted DGP vs Truth DGP (sample size = 200)")
    plt.figtext(.5, .95,
                ("DGP MSE %.5e, R Square = %.3f" % ((np.mean((np.array(DGPs_t) - np.array(DGPs_p)) ** 2)), r2)),
                fontsize=12, ha='center')
    x = np.arange(0, 1, 0.01)
    plt.plot(x, x, label='1=1 line', c='black', linewidth=0.5)
    plt.plot([0, 0.35], [0.35, 0.35], [0.35, 0.35], [0.35, 0], c='y', linewidth=0.5)
    plt.plot([0, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0], c='y', linewidth=0.5)
    plt.plot([0, 0.45], [0.45, 0.45], [0.45, 0.45], [0.45, 0], c='y', linewidth=0.5)
    plt.title("Predicted DGP versus Ground Truth DGP", size=16)
    plt.xlabel('Ground Truth DGP')
    plt.ylabel('Predicted DGP')
    plt.legend(fancybox=True)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("results/DGPs/dgp.png")
    plt.close()


def plotLog10MSEAnalysis(log10_mse):
    plt.hist(log10_mse, bins=100, range=(0, np.array(log10_mse).max()), fc='k', ec='k')
    plt.savefig("results/log10mse.png")
    plt.close()


def plotLog10RERAnalysis(log10_rer):
    plt.hist(log10_rer, bins=100, range=(0, np.array(log10_rer).max()), fc='k', ec='k')
    plt.savefig("results/log10rer.png")
    plt.close()


def plotLog10MSEAnalysis01(log10_mse):
    plt.hist(log10_mse, bins=100, range=(0, 1.0), fc='k', ec='k')
    plt.savefig("results/log10mse_range01.png")
    plt.close()


def plotLog10RERAnalysis01(log10_rer):
    plt.hist(log10_rer, bins=100, range=(0, 1.0), fc='k', ec='k')
    plt.savefig("results/log10rer_range01.png")
    plt.close()


def plot_DGPvsSC_Analysis(DGPs_t, SC_t, type):
    # correlation = pearsonr(DGPs_t, SC_t)
    plt.scatter(DGPs_t, SC_t, s=10, c='r',
                label="DGP vs Spatial Contrast (sample size = 200)")
    x = np.arange(0, 1, 0.01)
    plt.title("DGP vs Normalized Spatial Contrast, Correlation: %.d", size=16)
    plt.xlabel('DGP')
    plt.ylabel('Normalized Spatial Contrast')
    plt.legend(fancybox=True)

    plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.savefig("results/DGPs/dgpVSsc" + type)
    plt.close()


# transfer all panoramas in folder to fisheye images (every 5 degree)
# save the results falsecolor image
# npy_dir: panorama .npy file folder
# vmax: falsecolor image color bar max value
# imPath: save falsecolor image path
# panorama_w: width of panorama
# panorama_h: height of panorama
# fisheye_im_dimension: width and height of fisheye image
def transferAllPanos(npy_dir, vmax, imPath, panorama_w, panorama_h, fisheye_im_dimension):
    dirname = os.path.dirname(__file__)
    npy_dir = os.path.join(dirname, npy_dir)
    imPath = os.path.join(dirname, imPath)
    for file in os.listdir(npy_dir):
        if file.endswith(".npy"):
            file_name_npy = os.path.join(npy_dir, file)
            pano = np.load(file_name_npy)
            save_im_falsecolor(pano, file_name_npy, vmax, imPath, panorama_w, panorama_h, fisheye_im_dimension)


"""
panorama_w = 460
panorama_h = 230
fisheye_im_dimension = 230
vmax = 10000
im_path = "results/"

im_p = np.load("kmeans_200_txt_ab4/pano_2_18_13.5_687_105.npy")
im_t = np.load("kmeans_200_txt_ab0/pano_2_18_13.5_687_105_ab0.npy")
pano_path ="kmeans_200_txt_ab4/pano_2_18_13.5_687_105.npy"


save_im_falsecolor(im_p, im_t, "TEST", vmax, im_path, panorama_w, panorama_h, fisheye_im_dimension)
"""
