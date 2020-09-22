import os
import sys
import cv2
import subprocess
import math
import numpy as np
import argparse
import imutils
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from shutil import copyfile
from subprocess import call


src = "/media/didpurwanto/DiskD/disertation/datasets/UCF_crimes/untrimmed_ucf_crimes_det_flow/"
output = "/media/didpurwanto/DiskD/disertation/label_crf/"
if not os.path.exists(output):
    os.makedirs(output)
listfile = 'listfile.txt'
bibnumber = []
with open(listfile, "r") as f:
    for line in f:
        bibnumber.append(line.strip())

# configs
config = '../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
model = 'MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'

idx = 0
for vid in bibnumber:
    idx+=1

    vidfile = vid.split(' ')
    vidname = os.path.join(vidfile[0],'i') + '/*.jpg'
    tmp = vidfile[0].split('/')
    output_folder = os.path.join(output,tmp[9])

    
    print('---------------------------------------')
    print(str(idx), 'generating label CRF for', output_folder)    
    # terminal command file
    cmd = 'python3.7 generate_label.py --config ' + config + ' --input ' + vidname +' --output ' + output_folder + ' --opts '+ model  
    os.system(cmd)
    
