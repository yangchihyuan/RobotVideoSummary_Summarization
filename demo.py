#Chih-Yuan Yang 2019/7/5
import sys
import cv2
import os
from sys import platform
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
from math import pow, sqrt

# Import Openpose Ubuntu
sys.path.append('/usr/local/python')
from openpose import pyopenpose as op

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/frames/20190503", help="image path")
parser.add_argument("--output_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/frames/20190503_classified", help="image path")
parser.add_argument("--model_directory", default="/home/yangchihyuan/openpose/models/", help="OpenPose model directory")
parser.add_argument("--filelist", default="/home/yangchihyuan/RobotVideoSummary_Summarization/filelist.txt", required=False)
parser.add_argument("--usefilelist", default=False, type=str2bool, required=False, help="if False, all files in the image_path will be used. Otherwise, only files in filelist will be used.")
parser.add_argument("--blurred_var_threshold", default=200, required=False)
parser.add_argument("--distance_center_to_eye_threshold", default=60)
args = parser.parse_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = args.model_directory
params[ "model_pose" ] = "COCO"
#params["render_threshold"] = 0.7

listOfFiles = list()
if args.usefilelist and args.filelist is not None:
    with open(args.filelist ,'r') as f:
        content = f.readlines()
    listOfFiles += [os.path.join(args.image_path, file.strip()) for file in content]
else:
    for (dirpath, dirnames, filenames) in os.walk(args.image_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

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

    basename = os.path.basename(filename)
    print(basename)
    if datum.poseKeypoints.size == 1:
        print("no person")
    else:
        for person_idx in range(0, datum.poseKeypoints.shape[0]):
            nose_x = datum.poseKeypoints[person_idx][0][0]
            nose_y = datum.poseKeypoints[person_idx][0][1]
            nose_prob = datum.poseKeypoints[person_idx][0][2]
            print("nose_x nose_y " + str(nose_x) + " " + str(nose_y) + " prob " + str(nose_prob))

            center_prob = datum.poseKeypoints[person_idx][1][2]

            lefteye_x = datum.poseKeypoints[person_idx][14][0]
            lefteye_y = datum.poseKeypoints[person_idx][14][1]
            lefteye_prob = datum.poseKeypoints[person_idx][14][2]

            righteye_x = datum.poseKeypoints[person_idx][15][0]
            righteye_y = datum.poseKeypoints[person_idx][15][1]
            righteye_prob = datum.poseKeypoints[person_idx][15][2]


            # rightear_x = datum.poseKeypoints[person_idx][16][0]
            # rightear_y = datum.poseKeypoints[person_idx][16][1]
            # rightear_prob = datum.poseKeypoints[person_idx][16][2]
            # print("rightear_x rightear_y " + str(rightear_x) + " " + str(rightear_y) + " prob " + str(rightear_prob))

            # leftear_x = datum.poseKeypoints[person_idx][17][0]
            # leftear_y = datum.poseKeypoints[person_idx][17][1]
            # leftear_prob = datum.poseKeypoints[person_idx][17][2]
            # print("leftear_x leftear_y " + str(leftear_x) + " " + str(leftear_y) + " prob " + str(leftear_prob))

            if center_prob == 0:
                output_directory = os.path.join(args.output_path, "case1_no_center")
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                cv2.imwrite( os.path.join( output_directory, basename ), datum.cvOutputData )
            elif lefteye_prob <= 0.01 and righteye_prob <= 0.01:
                output_directory = os.path.join(args.output_path, "case2_two_eyes_invisible")
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                cv2.imwrite( os.path.join( output_directory, basename ), datum.cvOutputData )
            elif nose_x <= 213 or nose_x >=427:
                output_directory = os.path.join(args.output_path, "case3_not_at_center")
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                cv2.imwrite( os.path.join( output_directory, basename ), datum.cvOutputData )
            else:
                #test the size
                distance_center_to_lefteye = 0
                if nose_prob > 0 and lefteye_prob > 0:
                    distance_center_to_lefteye = sqrt(pow(nose_x - lefteye_x,2) + pow(nose_y - lefteye_y, 2))
                distance_center_to_righteye = 0
                if nose_prob > 0 and righteye_prob > 0:
                    distance_center_to_righteye = sqrt(pow(nose_x - righteye_x,2) + pow(nose_y - righteye_y, 2))
                max_distance = max(distance_center_to_lefteye, distance_center_to_righteye)
                if max_distance < args.distance_center_to_eye_threshold:
                    output_directory = os.path.join(args.output_path, "case4_small_size")
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)
                    cv2.imwrite( os.path.join( output_directory, basename ), datum.cvOutputData )
                

            #check image blur
            image_gray = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2GRAY)
            image_laplacian = cv2.Laplacian(image_gray, -1, ksize=1, scale=1, delta=0)
            var = np.var(image_laplacian)
            print( "var: " + str(var))
            if var < args.blurred_var_threshold:
                print("blurred")
                output_directory = os.path.join(args.output_path, "case6_blurred")
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                cv2.imwrite( os.path.join( output_directory, basename ), datum.cvOutputData )
                continue


            # cv2.imshow("Test", image_laplacian)
            # cv2.waitKey(0)

            # else:
            output_filename_full = str.replace(filename, args.image_path,args.output_path)
            dirname = os.path.dirname(output_filename_full)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            output_directory = os.path.join(args.output_path, "effective")
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            cv2.imwrite( os.path.join( output_directory, basename ), datum.cvOutputData )
