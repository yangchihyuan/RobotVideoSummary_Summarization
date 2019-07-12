#Method 4 DPP
#concatenate timestamps feature_matrix and select keyframes using k-dpp
import os
from dpp import util
from dpp import k_dpp
import matplotlib.image as mpimg
import h5py
import numpy as np
#from datetime import datetime
import matplotlib.pyplot as plt
from shutil import copyfile
import time
import datetime
import json
import argparse

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default="20190503")
parser.add_argument("--image_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/frames", help="image path")
parser.add_argument("--feature_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/features", help="image path")
parser.add_argument("--number_of_clusters", default=8)
parser.add_argument("--copy_to_directory", default="/home/yangchihyuan/RobotVideoSummary_Summarization/keyframes")
args = parser.parse_args()

copy_to_directory = os.path.join(os.path.join(args.copy_to_directory,args.data_name), "method4_DPP")
if not os.path.exists(copy_to_directory):
    os.makedirs(copy_to_directory)
save_result_file=os.path.join(copy_to_directory,"result.json")

#args_do_copy = True

feature_filename = os.path.join(args.feature_path,args.data_name+".h5")
image_directory = os.path.join(os.path.join(os.path.join(args.image_path, args.data_name+"_classified"),"wellposed"),"original")

dataset = h5py.File(feature_filename, 'r')
feature_matrix = dataset['features']['charades_probability'][...]
bytes_list = dataset['file_list'][...]
file_list = [n.decode("utf-8") for n in bytes_list]
number_of_images = feature_matrix.shape[0]
print('number_of_images', number_of_images)

#convert the file_list into a list of timestamps
start_time = time.time()

timestamps = []
idx = 0
for filename in file_list:
    timestamps.append( int(filename[:-4]))
    idx = idx +1

number_of_images = len(timestamps)
duration = timestamps[-1] - timestamps[0]

alpha = 0.5  #the weight of timestamp
feature_matrix_concatenated = np.empty((number_of_images,158), dtype=np.float64)
for idx in range(0,number_of_images):
    feature_matrix_concatenated[idx,0]=float(timestamps[idx]-timestamps[0])/duration*alpha
#    print(feature_matrix_concatenated[idx,0])
    feature_matrix_concatenated[idx,1::]=feature_matrix[idx,:]

#there are too many frames, I prefer to sample them by a step
step = 1   #1 means no subsampling
downsample_set = range(0,number_of_images, step)
feature_matrix_concatenated_downsampled = feature_matrix_concatenated[downsample_set,:]
number_of_images_downsample = len(downsample_set)

#create the distance matrix
#distance_matrix = np.empty((number_of_images_downsample, number_of_images_downsample), dtype=np.float64)
#for idx in range(0,number_of_images_downsample):
#    feature = feature_matrix_concatenated_downsampled[idx,:]
#    tile_feature = np.tile(feature,[number_of_images_downsample,1])
#    diff = tile_feature - feature_matrix_concatenated_downsampled
#    square = diff * diff
#    l2_norm = np.sqrt(np.sum(square,axis=1))
#    distance_matrix[:,idx] = l2_norm
#print(distance_matrix)
#distance_matrix_backup = distance_matrix

#try to reduce half computation load by only computing the lower diagnose
distance_matrix = np.empty((number_of_images_downsample, number_of_images_downsample), dtype=np.float64)
for idx in range(0,number_of_images_downsample-1):
    feature = feature_matrix_concatenated_downsampled[idx,:]
    fillin_range = range(idx+1,number_of_images_downsample)
    range_length = number_of_images_downsample - 1 -idx
    tile_feature = np.tile(feature,[range_length,1])
    diff = tile_feature - feature_matrix_concatenated_downsampled[fillin_range,:]
    square = diff * diff
    l2_norm = np.sqrt(np.sum(square,axis=1))
    distance_matrix[fillin_range,idx] = l2_norm

#fill 0 in the diag
for idx in range(0,number_of_images_downsample):
    distance_matrix[idx,idx] = 0
    
#copy the symmetric elemetns    
for i in range(0,number_of_images_downsample):
    for j in range(i+1,number_of_images_downsample):
        distance_matrix[i,j] = distance_matrix[j,i]
    
#check whether the two matrixes are the same
#diff = distance_matrix_backup - distance_matrix
#print('np.count_nonzero(diff)',np.count_nonzero(diff))
    
D, V = util.decompose_kernel(distance_matrix)
#print(D)
k = 8
Y = k_dpp.k_sample(k, D, V)

elapsed = time.time() - start_time
print('elapsed',elapsed)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

#remove old files
for the_file in os.listdir(copy_to_directory):
    file_path = os.path.join(copy_to_directory, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

#print(Y)
keyframe_list = []
for idx in Y:
    index_global = downsample_set[int(idx)]
    file_name = file_list[index_global]
    keyframe_list.append(file_name)    
#    value = datetime.fromtimestamp(timestamps[index_global]/1000)
#    print('file_name', file_name, 'index_global', index_global, 'time', value.strftime('%H:%M'))
    img=mpimg.imread(os.path.join(image_directory,file_name))
    plt.figure()
    imgplot = plt.imshow(img)
    copyfile(os.path.join(image_directory,file_name), os.path.join(copy_to_directory,file_name))

#save the segment result into a json file
JsonDumpDict = {'keyframe_list':keyframe_list}
with open(save_result_file, 'w') as outfile:
    json.dump(JsonDumpDict, outfile)
