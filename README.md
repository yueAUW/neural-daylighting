Neural daylighting
Computing Long-term Daylighting Simulations from High Dynamic Range Imagery Using Deep Neural Networks.

Yue Liu, Alex Colburn, Mehlika Inanici, IBPSA 2019 Conference, Rome, Italy, September 2-4, 2019.

Paper: http://www.ibpsa.org/proceedings/BS2019/BS2019_210369.pdf

Annual luminance maps provide meaningful evaluations for occupants’ visual comfort, preferences, and perception. However, acquiring luminance maps require labor-intensive and time-consuming simulations or impracticable long-term field measurements. This paper presents a novel method to accelerate annual luminance-based evaluations utilizing a deep neural network (DNN). From a small subset (5%) of high dynamic range (HDR) imagery, our method can predict annual panoramic luminance maps (with 360-degrees horizontal and 180-degrees vertical field of view) within an hour. Unlike the fixed camera viewpoint of perspective or fisheye projections that are commonly used in daylighting evaluations, panoramas allow full degree-of-freedom in camera roll, pitch, and yaw, thus providing a robust source of information for an occupant’s visual experience in a given environment. The DNN predicted high-quality panoramas are validated against Radiance RPICT renderings using a series of quantitative and qualitative metrics. With the developed workflow, practitioners and researchers can incorporate long-term luminance-based metrics over multiple view directions into the design and research process without the lengthy computing processes.

Code prerequisites:
The code in this repo is research code, it is not production quality. It can be used to reproduce the results in the paper.

The current code runs with tensorflow-2.1.0 We highly suggest creating a new conda environment to run the code

conda create deep_light python=3.7
conda install tensorflow-gpu 
conda install numpy matplotlib scikit-learn opencv imageio
optionally:

conda install ipython jupyter spyder
There is also an external dependency on Radiance https://github.com/NREL/Radiance/releases

The command line program evalglare must be on the path.

Once you have data unzipped in ALL_DATA_FP32 run an example session with:

python main.py
Data
There are 3 separate .zip files containing synthetic image data. The .zip files are about 4G is size. The uncompressed zip files are very large due to the image data being stored in numpy serialized arrays, the 16bit floating point format is much smaller that the 32 & 64 bit formats, requiring 20G of uncompressed space. The 16bit format potentially loses some data.
It can be located here: https://drive.google.com/open?id=1a6bxdv3q7G2M3LkVvVR6I72Q-Gyjsp4f

The 32bit float format requires about 35G of disk space, it can be found here: https://drive.google.com/open?id=15JOlC-H8XK_20eIjaSETYwEOGaVUUtWs

The 64bit float format requires about 64G of disk space it can be found here: https://drive.google.com/open?id=1e44yDnP5z3Qkxw64ocEidxZPUH_wxXZ1

It is also possible to mount the zip files via fuze-zip and use the compressed data.
