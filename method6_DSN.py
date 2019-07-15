#Method 6: DSN, but modified

import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from shutil import copyfile
import json

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", required=True)
parser.add_argument("--image_path", required=True, help="image path")
parser.add_argument("--feature_path", required=True, help="feature path")
parser.add_argument("--number_of_keyframes", type=int, required=True)
parser.add_argument("--keyframe_directory", required=True)
args = parser.parse_args()

keyframe_directory = os.path.join(os.path.join(args.keyframe_directory,args.data_name), "method6_DSN")
if not os.path.exists(keyframe_directory):
    os.makedirs(keyframe_directory)
#remove old files
for the_file in os.listdir(keyframe_directory):
    file_path = os.path.join(keyframe_directory, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)
save_result_file=os.path.join(keyframe_directory,"result.json")

# args_data_name="1226"
# args_dataset='/4t/yangchihyuan/TransmittedImages/datasets/'+args_data_name+'_effective_GoogLeNet.h5'
# args_image_directory="/4t/yangchihyuan/TransmittedImages/"+args_data_name
# args_keyframe_directory="/4t/yangchihyuan/TransmittedImages/keyframes/method6_DSN/"+args_data_name
# args_save_dir='/4t/yangchihyuan/TransmittedImages/log/'+args_data_name

# Model options
args_input_dim=1024
args_hidden_dim=256
args_num_layers=1
args_rnn_cell='lstm'
# Optimization options
args_lr=1e-05
args_weight_decay=1e-05
args_max_epoch=600
args_stepsize=30
args_gamma=0.1
args_num_episode=1
args_beta=0.01   #the weight of cost #default 0.01
# Misc
args_seed=1
args_gpu='0'
args_use_cpu=False
args_resume=''
args_verbose=True
args_save_results=True

torch.manual_seed(args_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args_gpu
use_gpu = torch.cuda.is_available()

#sys.stdout = Logger(osp.join(args_save_dir, 'log_train.txt'))

if use_gpu:
    print("Currently using GPU {}".format(args_gpu))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args_seed)
else:
    print("Currently using CPU")

feature_filename = os.path.join(args.feature_path,args.data_name+".h5")
image_directory = os.path.join(os.path.join(os.path.join(args.image_path, args.data_name+"_classified"),"wellposed"),"original")

dataset = h5py.File(feature_filename, 'r')
number_of_images = int(dataset['n_frames'][...])   #the original type is 1x1 ndarray
bytes_list = dataset['file_list'][...]
file_list = [n.decode("utf-8") for n in bytes_list]
train_keys=[args.data_name]

print("Initialize model")
model = DSN(in_dim=args_input_dim, hid_dim=args_hidden_dim, num_layers=args_num_layers, cell=args_rnn_cell)
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

optimizer = torch.optim.Adam(model.parameters(), lr=args_lr, weight_decay=args_weight_decay)
if args_stepsize > 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args_stepsize, gamma=args_gamma)

if args_resume:
    print("Loading checkpoint from '{}'".format(args_resume))
    checkpoint = torch.load(args_resume)
    model.load_state_dict(checkpoint)
else:
    start_epoch = 0
    

if use_gpu:
    model = nn.DataParallel(model).cuda()

print("==> Start training")
start_time = time.time()
model.train()
baselines = {key: 0. for key in train_keys} # baseline rewards for videos
#This is a dict
#reward_writers = {key: [] for key in train_keys} # record reward changes for each video
#This is also a dict

#for epoch in range(start_epoch, args_max_epoch):
#    idxs = np.arange(len(train_keys))   #train_keys is a list of video name
#    np.random.shuffle(idxs) # shuffle indices

#    for idx in idxs:
key = train_keys[0]
seq = dataset['features']['GoogLeNet'][...] # sequence of features, (seq_len, dim)
dataset.close()
seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
if use_gpu: seq = seq.cuda()
    
for epoch in range(start_epoch, args_max_epoch):
    probs = model(seq) # output shape (1, seq_len, 1)

    #cost is a torch.Tensor
    #cost = args_beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
    cost = args_beta * (probs.mean() - 8.0/number_of_images)**2 # minimize summary length penalty term [Eq.11]
    #0.5 means the percentage of frames to be selected.
    #in my case. it is 
    m = Bernoulli(probs) #Bernoulli model
    epis_rewards = []
    for _ in range(args_num_episode):   #default value as 5
        actions = m.sample()      #This is a cuda tensor, cannot be detach()
        log_probs = m.log_prob(actions)   #construct a loss function
        reward = compute_reward(seq, actions, use_gpu=use_gpu)
        expected_reward = log_probs.mean() * (reward - baselines[key])  #What is the meaning of the baselines?
        cost -= expected_reward # minimize negative expected reward
#        cost -= reward
        epis_rewards.append(reward.item())
        
#    print('cost',cost)
    optimizer.zero_grad()
    cost.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   
    optimizer.step() #update the model
    baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
#    print('baselines[key]',baselines[key]) #it increases
#    reward_writers[key].append(np.mean(epis_rewards))  #it is the mean of epis_rewards
#    print('len(reward_writers[key])',len(reward_writers[key]))

#    epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
    print("epoch {}/{}\t cost {}\t reward {}\t".format(epoch+1, args_max_epoch, cost, np.mean(epis_rewards)))

probs = torch.Tensor.cpu(probs)
probs_np = probs.detach().numpy()[0]
probs_np_1d = probs_np.ravel()

#Select peak scores and suppress local values
sorted_arg = np.argsort(probs_np_1d)
sorted_arg_descend = sorted_arg[::-1]
available_list = [True] * number_of_images
selected_idx_list = []
for idx in sorted_arg_descend:
    if available_list[idx] == True:
        selected_idx_list.append(idx)
        available_list[idx] = False
#        print('Selected', idx, probs_np_1d[idx])
        if len(selected_idx_list) == args.number_of_keyframes:
            break
        step = 0
        while idx+step+1 < number_of_images:
            if step < 100:
                available_list[idx+step+1] = False
                step = step + 1
            else:
                break
        step = 0
        while idx-step-1 >=0:
            if step < 100:
                available_list[idx-step-1] = False
                step = step + 1
            else:
                break

keyframe_list = []   
for idx in selected_idx_list:
    print(idx, probs_np_1d[idx])
    file_name = file_list[idx]
    keyframe_list.append(file_name)    
    img=mpimg.imread(os.path.join(image_directory,file_name))
#    imgplot = plt.imshow(img)        
#    plt.show()
    copyfile(os.path.join(image_directory,file_name), os.path.join(keyframe_directory,file_name))

#plt.figure()
#plt.plot(probs_np_1d )
#plt.show()

#write_json(reward_writers, osp.join(args_save_dir, 'rewards.json'))
#    evaluate(model, dataset, test_keys, use_gpu)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
model_save_path = osp.join(keyframe_directory, 'model_epoch' + str(args_max_epoch) + '.pth.tar')
save_checkpoint(model_state_dict, model_save_path)
print("Model saved to {}".format(model_save_path))

#save the segment result into a json file
JsonDumpDict = {'keyframe_list':keyframe_list}
with open(save_result_file, 'w') as outfile:
    json.dump(JsonDumpDict, outfile)

