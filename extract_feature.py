#Extract 2:Charades probability.
#This cell has to run in the container of robot because it require tensorflow.
#It uses the full GPU memory and runs very fast
import os
import cv2
import argparse
import sys
import h5py
import numpy as np
import tensorflow as tf

from utility.str2bool import str2bool

import torch

import pretrainedmodels
from pretrainedmodels import utils


# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", required=True)
parser.add_argument("--wellposed_image_directory", required=True, help="image path")
parser.add_argument("--feature_path", required=True, help="path to save extracted features")
parser.add_argument("--filelist", default="/home/yangchihyuan/RobotVideoSummary_Summarization/filelist.txt", required=False)
parser.add_argument("--usefilelist", default=False, type=str2bool, required=False, help="if False, all files in the wellposed_image_directory will be used. Otherwise, only files in filelist will be used.")
parser.add_argument("--Charades_model", required=True, help="path of the Charades frozen model")
args = parser.parse_args()

def prepare_im(image_np):
    # Normalize image and fix dimensions
    image_np = cv2.resize(image_np, dsize=(224,224)).astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np-mean)/std
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np_expanded
    
    
def recognize_activity(image_np, sess, detection_graph):
    image_np_expanded = prepare_im(image_np)
    image_tensor = detection_graph.get_tensor_by_name('input_image:0')
    classes = detection_graph.get_tensor_by_name('classifier/Reshape:0')

    # Actual detection.
    (classes) = sess.run(
        [classes],
        feed_dict={image_tensor: image_np_expanded})
    
    classes = np.exp(np.squeeze(classes))
    classes = classes / np.sum(classes)    #classes means the probability
    return classes

#load tensorflow
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(args.Charades_model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

#load torchvision pretrainedmodels
model_name = 'bninception'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
model.last_linear = utils.Identity()    
tf_img = utils.TransformImage(model)
load_img = utils.LoadImage()


#generate file list
wellposed_image_directory = args.wellposed_image_directory
list_basename = [f for f in os.listdir(wellposed_image_directory) if os.path.isfile(os.path.join(wellposed_image_directory, f))]
list_basename.sort()
number_of_images = len(list_basename)

#extract feature
feature_charades_probability = np.empty((number_of_images, 157), dtype=np.float32)
feature_HSV = np.empty([number_of_images, 4096],dtype=np.float32)
feature_GoogLeNet = np.empty((number_of_images, 1024), dtype=np.float32)

idx = 0
file_list_used = []
for file_name in list_basename:
    print(args.data_name,idx,file_name)
    input_img = cv2.imread(os.path.join(wellposed_image_directory, file_name))
    #extract charades_probability
    probability_array = recognize_activity(input_img, sess, detection_graph)
    feature_charades_probability[idx,:] = probability_array

    #extract HSV
    height = input_img.shape[0]
    width = input_img.shape[1]
    hsv = cv2.cvtColor(input_img,cv2.COLOR_BGR2HSV)
    number_of_bins = [16, 16, 16]
    value_range = [0, 180, 0, 256, 0, 256]
    hist = cv2.calcHist([hsv], [0, 1, 2], None, number_of_bins, value_range) #hist is a 16x16x16 array
    feature_HSV[idx,:] = np.reshape(hist,(1,4096)) / (height*width)  #normalize the distribution

    #extract GoogLeNet features, using torchvision
    input_img = load_img(os.path.join(wellposed_image_directory, file_name))
    input_tensor = tf_img(input_img) 
    input_tensor = input_tensor.unsqueeze(0)
    input = torch.autograd.Variable(input_tensor,requires_grad=False)
    feature = model(input)
    feature_GoogLeNet[idx,:] = feature.detach().numpy()


    idx = idx + 1
    file_list_used.append(file_name)
sess.close()


feature_filename = os.path.join(args.feature_path,args.data_name+".h5")
if not os.path.exists(args.feature_path):
    os.makedirs(args.feature_path)
f = h5py.File(feature_filename, 'w')
f.create_dataset('features/charades_probability', data=feature_charades_probability)
f.create_dataset('features/HSV_histogram', data=feature_HSV)
f.create_dataset('features/GoogLeNet', data=feature_GoogLeNet)
f.create_dataset('n_frames', data=idx)
f.create_dataset('data_name', data=args.data_name)
#convert Unicode to ascii code
asciiList = [n.encode("ascii", "ignore") for n in file_list_used]
f.create_dataset('file_list', data=asciiList)
f.close()