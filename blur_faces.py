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
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        #sys.path.append('../../python');
        sys.path.append('/usr/local/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="/4t/yangchihyuan/1226", required=True, help="image path")
parser.add_argument("--GaussianKernelWidth", default=21, required=True, type=int, help="Gaussian kernel width")
parser.add_argument("--model_directory", default="/home/yangchihyuan/openpose/models/", required=True, help="OpenPose model directory")
args = parser.parse_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = args.model_directory
params[ "model_pose" ] = "COCO"

# Add others in path?
'''
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item
'''
# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

mypath = args.image_path
outputpath = args.image_path + "_blurred"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print( file_list)

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    for filename in file_list:
        imageToProcess = cv2.imread(join(mypath,filename))
        mask = np.zeros_like(imageToProcess)
        #mask = np.zeros([imageToProcess.shape[0], imageToProcess.shape[1]])
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

    # Analysize keypoints
    #print(type(datum.faceRectangles))   #list
    #print(str(datum.faceRectangles))    #empty

#        print("Body keypoints: \n" + str(datum.poseKeypoints))    #numpy.ndarray, it becomes an int 2.0 
    # Display Image
#        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
#        cv2.waitKey(0)
        print(datum.poseKeypoints.shape)
        print(datum.poseKeypoints.size)
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

            if nose_x == 0 or rightear_x == 0 or leftear_x == 0:
                print("no face")
            else:
                roi_left = int(rightear_x)
                roi_right = int(leftear_x)
                roi_width = roi_right - roi_left
                roi_bottom = int(nose_y - 0.5 * roi_width)
                if roi_bottom < 0:
                    roi_bottom = 0
                roi_top = int(nose_y + 0.5 * roi_width)
                if roi_top > 479:
                    roi_top = 479

                blurred_image = cv2.GaussianBlur(imageToProcess, (args.GaussianKernelWidth,args.GaussianKernelWidth), 0)
#                print(blurred_image.shape)
 #               cv2.imshow("blurred_image", blurred_image)   #the image is very small. why?
 #               cv2.waitKey(0)

#                print( [roi_left, roi_right, roi_bottom, roi_top])

                mask=cv2.ellipse(mask, center=(nose_x, nose_y), axes=(int(0.5*roi_width),int(0.5*1.3*roi_width)), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
#                cv2.imshow("mask", mask)
#                cv2.waitKey(0)

#                blurred_face = np.bitwise_and(blurred_image, mask)
                imageToProcess[np.where(mask==255)]=blurred_image[np.where(mask==255)]
#                imageToProcess[roi_bottom:roi_top,roi_left:roi_right] = blurred_image[roi_bottom:roi_top,roi_left:roi_right]
#                cv2.imshow("final_image", imageToProcess)
#                cv2.waitKey(0)
                # Display Image
#                cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
#                cv2.waitKey(0)
                cv2.imwrite( join(outputpath,filename), imageToProcess )

except Exception as e:
    print(e)
    sys.exit(-1)
