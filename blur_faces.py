# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        #sys.path.append('../../python');
        sys.path.append('/usr/local/python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="/4t/yangchihyuan/keyframes", required=True, help="image path")
parser.add_argument("--output_path", default="/4t/yangchihyuan/keyframes_blurred", required=True, help="image path")
parser.add_argument("--GaussianKernelWidth", default=21, required=True, type=int, help="Gaussian kernel width")
parser.add_argument("--model_directory", default="/home/yangchihyuan/openpose/models/", required=True, help="OpenPose model directory")
args = parser.parse_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = args.model_directory
params[ "model_pose" ] = "COCO"

listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(args.image_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    for filename in listOfFiles:
        imageToProcess = cv2.imread(filename)
        mask = np.zeros_like(imageToProcess)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

#        print(datum.poseKeypoints.shape)
#        print(datum.poseKeypoints.size)
        if datum.poseKeypoints.size == 1:
            print("no person")
        else:
            nose_x = datum.poseKeypoints[0][0][0]
            nose_y = datum.poseKeypoints[0][0][1]
            print("nose_x nose_y " + str(nose_x) + " " + str(nose_y))

            rightear_x = datum.poseKeypoints[0][16][0]
            rightear_y = datum.poseKeypoints[0][16][0]
            print("rightear_x rightear_y " + str(rightear_x) + " " + str(rightear_y))

            leftear_x = datum.poseKeypoints[0][17][0]
            leftear_y = datum.poseKeypoints[0][17][1]
            print("leftear_x leftear_y " + str(leftear_x) + " " + str(leftear_y))

            if nose_x == 0:
                print("no face")
            else:
                if rightear_x != 0 and leftear_x != 0:
                    roi_width = leftear_x - rightear_x
                elif rightear_x != 0:
                    roi_width = 2*( nose_x - rightear_x )
                elif leftear_x != 0:
                    roi_width = 2*(leftear_x - nose_x)

                blurred_image = cv2.GaussianBlur(imageToProcess, (args.GaussianKernelWidth,args.GaussianKernelWidth), 0)
                mask=cv2.ellipse(mask, center=(nose_x, nose_y), axes=(int(0.5*roi_width),int(0.5*1.3*roi_width)), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
                imageToProcess[np.where(mask==255)]=blurred_image[np.where(mask==255)]
                output_filename_full = str.replace(filename, args.image_path,args.output_path)
                dirname = os.path.dirname(output_filename_full)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                cv2.imwrite( output_filename_full, imageToProcess )

except Exception as e:
    print(e)
    sys.exit(-1)
