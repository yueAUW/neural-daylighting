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
import fnmatch
import numpy as np

# the default numpy .npy file is a double float64
# copy the data directory and use this to convert .npy files to another type such as float32

def convert_data_depth(base_path, dtype=np.float32):
    for root, dirnames, filenames in os.walk(base_path):
        for filename in fnmatch.filter(filenames, '*.npy'):

            fname = os.path.join(root, filename)
            print('---')
            print('s:', fname)

            data = np.load(fname)
            if not data.dtype == dtype:
                np.save(fname, data.astype(dtype))

