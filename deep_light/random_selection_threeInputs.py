
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
from deep_light import time_to_sun_angles
import shutil

from deep_light.genData import get_data_path

####randomly select the test data from the dataset
#### TODO  make a function with source and destination subdirectoies

    
    

def select_test_samples(data_root='./ALL_DATA_FP32',
                LATITUDE=47,
                LONGITUDE=122,
                SM=120,
                NUM_SAMPLES = 500):
    
    
    import os
    
    
    #read seattle weahter txt data
    #return data format as al, az, dir, dif
    def readTxt():
        #read each line
        #transfer the time to sun angles
        #save the data
        f = open("testseattle.txt", "r")
        lines = f.readlines()
        data=np.zeros([4709, 7])
        i =0
        for line in lines:
            month = int(line.splitlines()[0].split(",")[0])
            date = int(line.splitlines()[0].split(",")[1])
            time = float(line.splitlines()[0].split(",")[2])
            dir = int(line.splitlines()[0].split(",")[3])
            dif = int(line.splitlines()[0].split(",")[4])
            al, az = time_to_sun_angles.timeToAltitudeAzimuth(date, month, time, LATITUDE, LONGITUDE, SM)
            if(dir > 10) or (dif > 10):
                data[i]=np.array([dir, dif, al, az, month, date, time])
                i = i + 1
        print(data.shape)
        return data

    # These parameters are location related. Right now we use Seattle's parameters.

    AB4_DIR = data_root + get_data_path('AB4')
    AB0_DIR = data_root + get_data_path('AB0')
    SKY_DIR = data_root + get_data_path('SKY')

    
    data = readTxt()

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)

    n_im = data.shape[0]

    train, test = data[idx[:NUM_SAMPLES]], data[idx[NUM_SAMPLES:]]

    cwd = os.getcwd()
    
    test_all = './data/original_test_all'
    
    if not os.path.exists('./data'):
        os.makedirs("./data")
        
    if not os.path.exists(test_all):
        os.mkdir(test_all)
        
    os.chdir(test_all)
    
    if (os.path.exists("./result_combo_random")):
       shutil.rmtree("./result_combo_random")
    os.makedirs("./result_combo_random")
    fig = plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(train[:, 0], train[:, 1], s=10, color='r', label="train")
    plt.scatter(test[:, 0], test[:, 1], s=1, color='g', label="test")
    plt.title("Sky Direct and Diffuse Irradiances Distribution", size=16)
    plt.xlabel('Direct')
    plt.ylabel('Diffuse')
    plt.legend(fancybox=True)
    plt.savefig('./result_combo_random/sky.png')
    plt.close()

    fig = plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(train[:, 2],train[:, 3], s=10, color='r', label="train")
    plt.scatter(test[:, 2], test[:, 3], s=1, color='g', label="test")
    plt.title("Sun Altitude and Azimuth Distribution", size=16)
    plt.xlabel('Altitude')
    plt.ylabel('Azimuth')
    plt.legend(fancybox=True)
    plt.savefig('./result_combo_random/sun.png')
    plt.close()

    if (os.path.exists("./test_ab0")):
        shutil.rmtree("./test_ab0")

    if (os.path.exists("./test_ab4")):
        shutil.rmtree("./test_ab4")

    if (os.path.exists("./test_sky_ab4")):
        shutil.rmtree("./test_sky_ab4")

    os.makedirs("./test_ab0")
    os.makedirs("./test_ab4")
    os.makedirs("./test_sky_ab4")
    
    os.chdir(cwd)

    #put the data into two folders train and test
    from shutil import copyfile
    import os.path
    i = 0
    bad_samples = 0
    for i in range(NUM_SAMPLES):
        file_name = "pano_" + str(int(train[i][4])) + "_" + str(int(train[i][5])) + "_" + str(train[i][6]) + "_" + str(int(train[i][0])) \
                    + "_" + str(int(train[i][1]))

        src_ab4 = AB4_DIR + file_name + ".npy"
        src_ab0 = AB0_DIR + file_name + "_ab0.npy"
        src_sky = SKY_DIR + file_name + ".npy"
        dst_ab0 = test_all + "/test_ab0/"+ file_name + "_ab0.npy"
        dst_ab4 = test_all + "/test_ab4/" + file_name + ".npy"
        dst_sky = test_all + "/test_sky_ab4/" + file_name + ".npy"

        if (os.path.isfile(src_ab4)) and (os.path.isfile(src_ab0)) and (os.path.isfile(src_sky)):
            copyfile(src_ab4, dst_ab4)
            copyfile(src_ab0, dst_ab0)
            copyfile(src_sky, dst_sky)
        else:
            bad_samples = bad_samples + 1
            print('unable to locate:')
            if not os.path.isfile(src_ab4) : print(src_ab4)
            if not os.path.isfile(src_ab0) : print(src_ab0)
            if not os.path.isfile(src_sky) : print(src_sky)
            
            

        i = i + 1
     
    print('Maps not found = ', bad_samples)
    
    
def sample_consistency(data_root='./ALL_DATA_FP32',
                LATITUDE=47,
                LONGITUDE=122,
                SM=120):
    
    
    import os
    
    
    #read seattle weahter txt data
    #return data format as al, az, dir, dif
    def readTxt():
        #read each line
        #transfer the time to sun angles
        #save the data
        f = open("testseattle.txt", "r")
        lines = f.readlines()
        data=np.zeros([4709, 7])
        i =0
        for line in lines:
            month = int(line.splitlines()[0].split(",")[0])
            date = int(line.splitlines()[0].split(",")[1])
            time = float(line.splitlines()[0].split(",")[2])
            dir = int(line.splitlines()[0].split(",")[3])
            dif = int(line.splitlines()[0].split(",")[4])
            al, az = time_to_sun_angles.timeToAltitudeAzimuth(date, month, time, LATITUDE, LONGITUDE, SM)
            if(dir > 10) or (dif > 10):
                data[i]=np.array([dir, dif, al, az, month, date, time])
                i = i + 1
        print(data.shape)
        return data

    # These parameters are location related. Right now we use Seattle's parameters.

    AB4_DIR = data_root + get_data_path('AB4')
    AB0_DIR = data_root + get_data_path('AB0')
    SKY_DIR = data_root + get_data_path('SKY')

    
    data = readTxt()

    idx = np.arange(data.shape[0])
    

    n_im = data.shape[0]

    train, test = data[idx], data[idx]

    
    test_all = './data/original_test_all'
    
    
    
    import os.path
    i = 0
    bad_samples = 0
    good_samples = 0
    for i in range(data.shape[0]):
        file_name = "pano_" + str(int(train[i][4])) + "_" + str(int(train[i][5])) + "_" + str(train[i][6]) + "_" + str(int(train[i][0])) \
                    + "_" + str(int(train[i][1]))

        src_ab4 = AB4_DIR + file_name + ".npy"
        src_ab0 = AB0_DIR + file_name + "_ab0.npy"
        src_sky = SKY_DIR + file_name + ".npy"
        dst_ab0 = test_all + "/test_ab0/"+ file_name + "_ab0.npy"
        dst_ab4 = test_all + "/test_ab4/" + file_name + ".npy"
        dst_sky = test_all + "/test_sky_ab4/" + file_name + ".npy"

        if (os.path.isfile(src_ab4)) and (os.path.isfile(src_ab0)) and (os.path.isfile(src_sky)):
            good_samples = good_samples + 1
        else:
            bad_samples = bad_samples + 1
            print('unable to locate:')
            if not os.path.isfile(src_ab4) : print(src_ab4)
            if not os.path.isfile(src_ab0) : print(src_ab0)
            if not os.path.isfile(src_sky) : print(src_sky)
            
            

        i = i + 1
     
    print('Maps not found = ', bad_samples)
    

#finish the rest part
