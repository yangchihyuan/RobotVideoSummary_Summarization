#Utility 1
#Draw a histogram chart, but show different colors for different segments
import os
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import math
import json
import h5py
import numpy as np
import pytz
import argparse
taipei_timezone = pytz.timezone('Asia/Taipei')

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default="20190503")
parser.add_argument("--image_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/frames", help="image path")
parser.add_argument("--feature_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/features", help="image path")
parser.add_argument("--filelist", default="/home/yangchihyuan/RobotVideoSummary_Summarization/filelist.txt", required=False)
parser.add_argument("--number_of_clusters", default=8)
parser.add_argument("--copy_to_directory", default="/home/yangchihyuan/RobotVideoSummary_Summarization/keyframes")
parser.add_argument("--chart_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/charts")

args = parser.parse_args()

feature_filename = os.path.join(args.feature_path,args.data_name+".h5")

#prepare save path
figure_size=(6,4)  #for paper
figure_name_eps=os.path.join(args.chart_path,args.data_name+".eps")
figure_name_png=os.path.join(args.chart_path,args.data_name+".png")
if not os.path.exists(args.chart_path):
    os.makedirs(args.chart_path)


copy_to_directory = os.path.join(os.path.join(args.copy_to_directory,args.data_name), "method3_time_action")
save_clusters_file=os.path.join(copy_to_directory,"clusters.json")


dataset = h5py.File(feature_filename, 'r')
bytes_list = dataset['file_list'][...]
file_list = [n.decode("utf-8") for n in bytes_list]

timestamps = []
for filename in file_list:
    timestamps.append( int(filename[:-4]))
    
timestamp_begin = timestamps[0]
timestamp_end = timestamps[-1]
timestamp_array = np.linspace(timestamp_begin, timestamp_end, 5)
time_string_array = []
for timestamp in timestamp_array:
    value = datetime.fromtimestamp(timestamp/1000, taipei_timezone)       #the default is Greenwish time. I need to control timezone
    time_string_array.append(value.strftime('%H:%M'))

print(time_string_array)

plt.figure(figsize = figure_size)
duration = timestamp_end - timestamp_begin
duration_a_bin = duration / 1000
[n, bins, patches]= plt.hist(timestamps, 1000)
plt.close()

plt.figure(figsize = figure_size)
#load the segment result from method3' json file
with open(save_clusters_file, 'r') as outfile:
    JsonDumpDict = json.load(outfile)

segments = JsonDumpDict['clusters']
dict_keyframe_list={}
dict_keyframe_list["Proposed"] = JsonDumpDict['keyframe_list']
    
#load selected keyframes from method5' json file
method5_result_json_file = os.path.join(os.path.join(os.path.join(args.copy_to_directory,args.data_name), "method5_VSUMM"),"result.json")
with open(method5_result_json_file, 'r') as outfile:
    JsonDumpDict = json.load(outfile)
dict_keyframe_list["VSUMM"] = JsonDumpDict['keyframe_list']

#load selected keyframes from method4' json file
method5_result_json_file = os.path.join(os.path.join(os.path.join(args.copy_to_directory,args.data_name), "method4_DPP"),"result.json")
with open(method5_result_json_file, 'r') as outfile:
    JsonDumpDict = json.load(outfile)
dict_keyframe_list["DPP"] = JsonDumpDict['keyframe_list']


#I need segment data to select colors, 
color_list=['C0','C8']
color_idx = 0
for segment in segments:
    timestamp_head = timestamps[segment[0]]
    timestamp_tail = timestamps[segment[-1]]
    bin_head = int(math.floor( (timestamp_head - timestamp_begin)/duration_a_bin))
    bin_tail = int(math.ceil( (timestamp_tail - timestamp_begin)/duration_a_bin ))
    plt.bar(bins[bin_head:bin_tail],n[bin_head:bin_tail],width=bins[1]-bins[0],align='edge',color=color_list[color_idx])
    color_idx = (color_idx+1) % 2
    print(color_idx)
    
plt.axis([timestamp_begin, timestamp_end, None, None])

plt.xticks( timestamp_array, time_string_array )
plt.xlabel('time')
plt.ylabel('number of effective frames')

#get the list of selected frames
idx =0
#mark_list = ['cs','ro','k^','mv','bx']
mark_list = ['ro','k^','mv','bx']
y_max = max(n)
#print(y_max)
ratio_list = [0.25, 0.2, 0.15, 0.1, 0.05]
height_list = [i * y_max for i in ratio_list]
#print(height_list)
#legends=['VSUMM','DPP','DR-DSN','Proposed']
legends=['VSUMM','DPP','Proposed']
for method in legends:
    #print(directory)
    #output=subprocess.check_output(["ls", directory+"/*.jpg"])
    #print(output) #this is a single string
    #output_parse = output.split('\n')
    #filelist = output_parse[:-1]   #ignore the last empty element
    #print("len", len(output))
    #print("len output_parse", len(output_parse))
    #print(output_parse)
    #print(filelist)
    timestamps_mark = []
    keyframe_list = dict_keyframe_list[method]
    for filename in keyframe_list:
        timestamps_mark.append( int(filename[:-4]))

    print(timestamps_mark)
    y = [height_list[idx]] * len(timestamps_mark)
    l = plt.plot(timestamps_mark,y, mark_list[idx])
    plt.setp(l, markersize=5)
    idx+=1

plt.legend(legends)    
plt.savefig(figure_name_eps,bbox_inches='tight',transparent=True, pad_inches=0)
plt.savefig(figure_name_png, dpi=72*10,bbox_inches='tight',transparent=True, pad_inches=0)
plt.show()