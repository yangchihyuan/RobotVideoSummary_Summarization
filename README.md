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

# Instructions #
- clone this repository to your computer
```
git clone https://github.com/yangchihyuan/RobotVideoSummary_Summarization.git
```
- download a sample file containing example frames and a Tensorflow frozen model
```
cd RobotVideoSummary_Summarization
./download_files.sh
```
- run demo.py
Specify your OpenPose/model directory. For example, mine is /home/yangchihyuan/openpose/models/
```
python3 demo.py --OpenPose_model_directory=/home/yangchihyuan/openpose/models/
```
The classified frames files will be avaiable at frames/20190503_classified/, and the extracted image features are saved in features/20190503.h5.
The keyframes selected by 4 different methods are at keyframes/, and the chart showing the timestamps of those keyframes are at charts/20190503. 
The

# Suggestion #
Because I use OpenPose's APIs to classify images, my demo code will run faster if your OpenPose is built with a GPU and run on a GPU.