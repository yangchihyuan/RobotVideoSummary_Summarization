# System setting #
- Ubuntu 16.04
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 1.5.0
- wget 1.17.1
- unzip 6.0
- Python 3.5.2
- python3-tk


# Prerequisite #
- OpenPose's Python API and BODY_COCO model

check the BUILD_PYTHON and DOWNLOAD_BODY_COCO_MODEL in OpenPose's cmake-gui configuration window.

# Required Python packages #
- h5py 2.9.0
- Tensorflow 1.13.0
- sklearnl
- simplejason 3.16.0
- matplotlib
- Pillow 6.1.0
- pytz-2019.1

If you want to generate DNS's results, you need to install those additional packages.
- torch 1.1.0
- torchvision 0.3.0
- [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)
- tabulate 0.8.3
- knapsack 0.0.4 The knapsack.py is implemented in Python 2. In order to be called in Python 3, you need to manually modify its line 77 from "print picks" to "print(picks)"

# Installation #
- clone this repository to your computer
```
git clone https://github.com/yangchihyuan/RobotVideoSummary_Summarization.git
```
- download a zip file containing example frames and a Tensorflow frozen model
```
cd RobotVideoSummary_Summarization
./download_files.sh
```
- run demo.py
```
python3 demo.py
```