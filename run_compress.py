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

gpu_set = ['1']
parameter_set = ['--jpg_level=90 --random_jpg ']
number_gpu = len(gpu_set)
#--upsampling nearest_neighbor
datasets = [ 
        'horse_auto --subset=trainA', 'horse --subset=trainA' 
        ]

#datasets = [
#        'horse --subset=testA', 'zebra --subset=testB',
#        'summer --subset=testA', 'winter --subset=testB', 
#        'apple --subset=testA', 'orange --subset=testB',
#        'facades --subset=testA', 'cityscapes --subset=testA',
#        'satellite --subset=testA', 'ukiyoe --subset=testA', 
#        'vangogh --subset=testA', 'cezanne --subset=testA',
#        'monet --subset=testA', 'photo --subset=testB']

#datasets = ['monet --subset=testA']
process_set = []

index = 0
for idx, parameter in enumerate(parameter_set):
    for dataset in datasets:
        print('Test Parameter: {}'.format(parameter))
        command = 'python jpg_compress.py --category={} {}'\
                .format(dataset, parameter)# 
    
        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
         
        if (index+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
        
        index+=1
        time.sleep(10)
    
for sub_process in process_set:
    sub_process.wait()

