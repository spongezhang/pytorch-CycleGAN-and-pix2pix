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
parameter_set = ['--dataroot ./datasets/fold6/ --name fold6_auto --direction B',
        '--dataroot ./datasets/fold7/ --name fold7_auto --direction B',
        '--dataroot ./datasets/fold8/ --name fold8_auto --direction B',
        '--dataroot ./datasets/fold9/ --name fold9_auto --direction B']
#                 '--dataroot ./datasets/horse2zebra/ --name zebra_auto_nn --direction B',
#                '--dataroot ./datasets/summer2winter_yosemite/ --name summer_auto_nn --direction A',
#                '--dataroot ./datasets/summer2winter_yosemite/ --name winter_auto_nn --direction B']
#parameter_set = ['--dataroot ./datasets/summer2winter_yosemite/ --name summer_auto_nn_blur --direction A',
#                '--dataroot ./datasets/summer2winter_yosemite/ --name winter_auto_nn_blur --direction B']
#parameter_set = [' ']
number_gpu = len(gpu_set)

datasets = ['']
process_set = []

index = 0
for idx, parameter in enumerate(parameter_set):
    for dataset in datasets:
        print('Test Parameter: {}'.format(parameter))
        command = 'python train.py {}  --model auto_gan --batch_size 8 --gpu_ids {} --write_dir train_image '\
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

