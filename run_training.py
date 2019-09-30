"""
Check the correctness of gor on HardNet loss using multiple GPUs
Usage: check_gor_HardNet.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import pandas as pd
import subprocess
import shlex
import argparse
####################################################################
# Parse command line
####################################################################
def usage():
    print >> sys.stderr 
    sys.exit(1)

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

gpu_set = ['0','1']
parameter_set = [
        #'--dataroot ./datasets/horse2zebra/ --name horse_auto --direction A',
        #'--dataroot ./datasets/horse2zebra/ --name zebra_auto --direction B',
        #'--dataroot ./datasets/summer2winter_yosemite/ --name summer_auto --direction A',
        #'--dataroot ./datasets/summer2winter_yosemite/ --name winter_auto --direction B'
       # '--dataroot ./datasets/apple2orange/ --name apple_auto --direction A',
       # '--dataroot ./datasets/apple2orange/ --name orange_auto --direction B',
       # '--dataroot ./datasets/monet2photo/ --name monet_auto --direction A',
       # '--dataroot ./datasets/monet2photo/ --name photo_auto --direction B',
       # '--dataroot ./datasets/cityscapes/ --name cityscapes_auto --direction A',
       # '--dataroot ./datasets/maps/ --name satellite_auto --direction A',
       # '--dataroot ./datasets/facades/ --name facades_auto --direction A',
       # '--dataroot ./datasets/ukiyoe2photo/ --name ukiyoe_auto --direction A',
        '--dataroot ./datasets/cezanne2photo/ --name cezanne_auto --direction A',
        '--dataroot ./datasets/vangogh2photo/ --name vangogh_auto --direction A',
        ]

number_gpu = len(gpu_set)

process_set = []

index = 0
for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
    command = 'python train.py {}  --model auto_gan --upsampling transposed_conv --batch_size 8 --gpu_ids {} --write_dir train_image --niter=50 --niter_decay=50 '\
            .format(parameter, gpu_set[index%number_gpu])# 
    
    print(command)
    p = subprocess.Popen(shlex.split(command))
    process_set.append(p)
     
    if (index+1)%number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()
    
        process_set = []
    
    index+=1
    time.sleep(60)
    
for sub_process in process_set:
    sub_process.wait()

