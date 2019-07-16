#Method 5 VSUMM
from sklearn.cluster import KMeans
import h5py
import os
from shutil import copyfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import datetime
import argparse
import json

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", required=True)
parser.add_argument("--image_path", required=True, help="image path")
parser.add_argument("--feature_path", required=True, help="image path")
parser.add_argument("--number_of_keyframes", type=int, required=True)
parser.add_argument("--keyframe_directory", required=True)
args = parser.parse_args()


keyframe_directory = os.path.join(os.path.join(args.keyframe_directory,args.data_name), "method5_VSUMM")
if not os.path.exists(keyframe_directory):
    os.makedirs(keyframe_directory)
save_result_file=os.path.join(keyframe_directory,"result.json")
figure_name_eps=os.path.join(keyframe_directory,args.data_name+"_distance.eps")
figure_name_png=os.path.join(keyframe_directory,args.data_name+"_distance.png")

feature_filename = os.path.join(args.feature_path,args.data_name+".h5")
image_directory = os.path.join(os.path.join(os.path.join(args.image_path, args.data_name+"_classified"),"wellposed"),"original")

dataset = h5py.File(feature_filename, 'r')
feature_matrix = dataset['features']['HSV_histogram'][...]
bytes_list = dataset['file_list'][...]
file_list = [n.decode("utf-8") for n in bytes_list]
number_of_images = feature_matrix.shape[0]
print('number_of_images', number_of_images)

#check whether keyframe_directory exists
if( os.path.exists(keyframe_directory) == False):
    os.mkdir(keyframe_directory)
else:
    #remove old files
    for the_file in os.listdir(keyframe_directory):
        file_path = os.path.join(keyframe_directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


#compute the frame-wise distance
distance_array = []
feature_next = feature_matrix[0,:]
for idx in range(1,number_of_images):
    feature_pre = feature_next
    feature_next = feature_matrix[idx,:]
    distance = np.linalg.norm(feature_next - feature_pre)
    distance_array.append(distance)
    
plt.plot(distance_array)    
plt.xlabel('frame')
plt.ylabel('distance')
plt.savefig(figure_name_eps, dpi=72*10,bbox_inches='tight',transparent=True, pad_inches=0)
plt.savefig(figure_name_png)
plt.close()
#plt.show()

#distance_array.sort()
#distance_array = distance_array[::-1]
#plt.plot(distance_array)    
#plt.show()
#print(distance_array[15])
    
start_time = time.time()

#initialize the cluster centers
cluster_initial = [math.floor(x) for x in np.linspace(0,16,number_of_images,endpoint=False)]
initial_cluster_feature_matrix = np.empty([args.number_of_keyframes,4096],dtype=np.float64)
for idx in range(0,args.number_of_keyframes):
    indices = [i for i, x in enumerate(cluster_initial) if x == idx]
    features = feature_matrix[indices,:]
    initial_cluster_feature_matrix[idx,:] = np.mean(features, axis=0)

kmeans = KMeans(n_clusters=args.number_of_keyframes, random_state=0, init=initial_cluster_feature_matrix).fit(feature_matrix)
number_of_samples = number_of_images
keyframe_list = []
for cluster_center in kmeans.cluster_centers_:
    diff = feature_matrix - np.tile(cluster_center,(number_of_samples,1))
    square = diff * diff
    l2_norm = np.sqrt(np.sum(square,axis=1))
    index = np.argsort(l2_norm)[0]
    file_name = file_list[index]
    keyframe_list.append(file_name)
    img=mpimg.imread(os.path.join(image_directory,file_name))
#    plt.figure()
#    imgplot = plt.imshow(img)
#    plt.show()
    copyfile(os.path.join(image_directory,file_name), os.path.join(keyframe_directory,file_name))
    
elapsed = time.time() - start_time
print('elapsed',elapsed)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

#save the segment result into a json file
JsonDumpDict = {'keyframe_list':keyframe_list}
with open(save_result_file, 'w') as outfile:
    json.dump(JsonDumpDict, outfile)
