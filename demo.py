import os
from utility.str2bool import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--classify_images", default=True, type=str2bool, required=False)
parser.add_argument("--extract_feature", default=True, type=str2bool, required=False)
parser.add_argument("--run_method3_proposed", default=True, type=str2bool, required=False)
parser.add_argument("--run_method4_DPP", default=True, type=str2bool, required=False)
parser.add_argument("--run_method5_VSUMM", default=True, type=str2bool, required=False)
parser.add_argument("--run_method6_DSN", default=True, type=str2bool, required=False)
parser.add_argument("--generate_chart", default=True, type=str2bool, required=False)
parser.add_argument("--OpenPose_model_directory", default="/home/yangchihyuan/openpose/models/", help="OpenPose model directory")
args = parser.parse_args()


working_directory = os.path.dirname(__file__)

data_name = "20190503"
number_of_keyframes = 8
image_path = os.path.join(working_directory,"frames")
wellposed_image_directory = os.path.join(os.path.join(os.path.join(image_path, data_name+"_classified"),"wellposed"),"original")
feature_path = os.path.join(working_directory,"features")
Charades_model = os.path.join(os.path.join(working_directory,"CharadesWebcam"),"frozen_model.pb")
keyframe_directory = os.path.join(working_directory,"keyframes")
chart_directory = os.path.join(working_directory,"charts")

#classify images
if args.classify_images:
    os.system("python3 classify_images.py" + \
        " --image_directory="+ os.path.join(image_path,data_name) + \
        " --output_directory="+ os.path.join(image_path,data_name+"_classified") + \
        " --model_directory="+ args.OpenPose_model_directory )

#extract_features
if args.extract_feature:
    os.system("python3 extract_feature.py --data_name=" + data_name+ \
        " --wellposed_image_directory="+ wellposed_image_directory + \
        " --feature_path="+ feature_path + \
        " --Charades_model="+ Charades_model )

#run the proposed method
if args.run_method3_proposed:
    os.system("python3 method3_proposed.py --data_name=" + data_name+ \
        " --image_path="+ image_path + \
        " --feature_path="+ feature_path + \
        " --number_of_keyframes="+ str(number_of_keyframes) + \
        " --keyframe_directory="+ keyframe_directory )

#run a compared method DPP
if args.run_method4_DPP:
    os.system("python3 method4_DPP.py --data_name=" + data_name+ \
        " --image_path="+ image_path + \
        " --feature_path="+ feature_path + \
        " --number_of_keyframes="+ str(number_of_keyframes) + \
        " --keyframe_directory="+ keyframe_directory )

#run a compared method VSUMM
if args.run_method5_VSUMM:
    os.system("python3 method5_VSUMM.py --data_name=" + data_name+ \
        " --image_path="+ image_path + \
        " --feature_path="+ feature_path + \
        " --number_of_keyframes="+ str(number_of_keyframes) + \
        " --keyframe_directory="+ keyframe_directory )

#run a compared method VSUMM
if args.run_method6_DSN:
    os.system("python3 method6_DSN.py --data_name=" + data_name+ \
        " --image_path="+ image_path + \
        " --feature_path="+ feature_path + \
        " --number_of_keyframes="+ str(number_of_keyframes) + \
        " --keyframe_directory="+ keyframe_directory )

#generate charts
if args.generate_chart:
    os.system("python3 chart1_histogram.py --data_name=" + data_name+ \
        " --feature_path="+ feature_path + \
        " --keyframe_directory="+ keyframe_directory + \
        " --chart_directory="+chart_directory )
