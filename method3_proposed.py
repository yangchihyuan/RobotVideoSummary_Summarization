#Method 3
#compute the mean of the features in each cluster. This is the representative set.
#sort the nested_list_timestamp by the numbers of frames contained in clusters

import os
import sys
import h5py
import numpy as np
from sklearn.cluster import KMeans
from shutil import copyfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json
import time
import datetime
import argparse

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default="20190503")
parser.add_argument("--image_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/frames", help="image path")
parser.add_argument("--feature_path", default="/home/yangchihyuan/RobotVideoSummary_Summarization/features", help="image path")
parser.add_argument("--number_of_clusters", default=8)
parser.add_argument("--copy_to_directory", default="/home/yangchihyuan/RobotVideoSummary_Summarization/keyframes")

args = parser.parse_args()

copy_to_directory = os.path.join(os.path.join(args.copy_to_directory,args.data_name), "method3_time_action")
if not os.path.exists(copy_to_directory):
    os.makedirs(copy_to_directory)
save_clusters_file=os.path.join(copy_to_directory,"clusters.json")

feature_filename = os.path.join(args.feature_path,args.data_name+".h5")
image_directory = os.path.join(os.path.join(os.path.join(args.image_path, args.data_name+"_classified"),"wellposed"),"original")

dataset = h5py.File(feature_filename, 'r')
feature_matrix = dataset['features']['charades_probability'][...]
bytes_list = dataset['file_list'][...]
file_list = [n.decode("utf-8") for n in bytes_list]
number_of_images = int(feature_matrix.shape[0])
feature_dimension = int(feature_matrix.shape[1])
print('number_of_images', number_of_images)

#convert the file_list into a list of timestamps
timestamps = []
idx = 0
for filename in file_list:
    timestamps.append( int(filename[:-4]))
    idx = idx +1

duration = timestamps[-1] - timestamps[0]

start_time = time.time()
time_gap_threshold = 1000  #1 second

def calculate_number_of_cluster(time_gap_threshold):
    print('time_gap_threshold',time_gap_threshold)
    next_timestamp = timestamps[0]
    clusters = []
    cluster = [0]
    for idx in range(1,number_of_images):
        previous_timestamp = next_timestamp
        next_timestamp = timestamps[idx]
        if next_timestamp - previous_timestamp < time_gap_threshold:
            cluster.append(idx)
        else:
            clusters.append(cluster)
            cluster = [idx]
    clusters.append(cluster)   #add the last cluster

    number_of_clusters = len(clusters)
    print('number of clusters',number_of_clusters)
    return number_of_clusters, clusters

#initial calculation
number_of_clusters, clusters = calculate_number_of_cluster(time_gap_threshold)
number_of_clusters_save = number_of_clusters
clusters_save = clusters
#loop
while True:
    if number_of_clusters < args.number_of_clusters:
        time_gap_threshold = time_gap_threshold / 2
    else:
        time_gap_threshold = time_gap_threshold * 2
    
    number_of_clusters, clusters = calculate_number_of_cluster(time_gap_threshold)
    if number_of_clusters < 8:
        break
    else:
        #update
        print('update')
        number_of_clusters_save = number_of_clusters
        clusters_save = clusters
        
#restore        
number_of_clusters = number_of_clusters_save
clusters = clusters_save
        
number_of_indices_list = [len(cluster) for cluster in clusters]
print('number of index in each cluster',number_of_indices_list)
order_by_size = np.argsort(number_of_indices_list)[::-1]
print('order_by_size', order_by_size)

averaged_feature_list = []

#check whether copy_to_directory exists
# if( os.path.exists(copy_to_directory) == False):
#     os.makedirs(copy_to_directory)
# else:
#     #remove old files
#     os.system("rm " + copy_to_directory +"/*")

keyframe_list = []
for idx_cluster in order_by_size[0:8]:
    cluster = clusters[idx_cluster]
    number_of_images = len(cluster)
    feature_sum = np.zeros((number_of_images,feature_dimension), dtype=np.float32)
    feature_in_cluster = feature_matrix[cluster,:]
    averaged_feature = np.mean(feature_in_cluster, axis=0)
    averaged_feature_list.append(averaged_feature)
    diff = feature_in_cluster - np.tile(averaged_feature,(number_of_images,1))
    square = diff * diff
    l2_norm = np.sqrt(np.sum(square,axis=1))
    index_representative = np.argsort(l2_norm)[0]
    index_global = cluster[index_representative]
    print('index_global', index_global)
    file_name = file_list[index_global]
    keyframe_list.append(file_name)
    print('file_name', file_name)
    img=mpimg.imread(os.path.join(image_directory,file_name))
    plt.figure()
    imgplot = plt.imshow(img)
    copyfile(os.path.join(image_directory,file_name), os.path.join(copy_to_directory,file_name))

elapsed = time.time() - start_time
elapsed = str(datetime.timedelta(seconds=elapsed))
print('elapsed',elapsed)
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    
    
#save the segment result into a json file
JsonDumpDict = {'keyframe_list':keyframe_list, 'clusters':clusters}
with open(save_clusters_file, 'w') as outfile:
    json.dump(JsonDumpDict, outfile)

