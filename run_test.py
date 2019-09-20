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
#parameter_set = [' --feature=fft --mode=0 ']
parameter_set = ['fold6/trainB --name fold6_auto ', 'fold7/trainB --name fold7_auto ', 'fold8/trainB --name fold8_auto ', 'fold9/trainB --name fold9_auto ']
#parameter_set = [' ']
number_gpu = len(gpu_set)
#--upsampling nearest_neighbor
datasets = ['']
process_set = []

index = 0
for idx, parameter in enumerate(parameter_set):
    for dataset in datasets:
        print('Test Parameter: {}'.format(parameter))
        command = 'python test.py --dataroot ./datasets/{} --model test --no_dropout --num_test -1 --gpu_ids {} '\
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

