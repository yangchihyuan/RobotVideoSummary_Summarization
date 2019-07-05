# System setting #
- Ubuntu 16.04
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 1.5.0
- wget 1.17.1
- unzip 6.0
- Python 3.5
- Tensorflow 1.13.0

# Prerequisite #
- OpenPose's Python API and BODY_COCO model
check the BUILD_PYTHON and DOWNLOAD_BODY_COCO_MODEL in OpenPose's cmake-gui configuration window.

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