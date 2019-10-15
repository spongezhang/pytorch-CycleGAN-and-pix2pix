#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: run_test.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Tue Oct  1 18:25:31 2019
#
#  Usage: python run_test.py -h
#  Description: test a generator
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

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
parser.add_argument('--dataset', type=str, default='CycleGAN', help='Training dataset select from: CycleGAN and AutoGAN')

args = parser.parse_args()

gpu_set = ['0']


number_gpu = len(gpu_set)

if args.dataset == 'CycleGAN':
    parameter_set = [
        'horse2zebra/trainA --name horse2zebra_pretrained --save_name=zebra ',
        'horse2zebra/testA --name horse2zebra_pretrained --save_name=zebra ',
        'horse2zebra/trainB --name zebra2horse_pretrained --save_name=horse ',
        'horse2zebra/testB --name zebra2horse_pretrained --save_name=horse '

        'summer2winter_yosemite/trainA --name summer2winter_yosemite_pretrained --save_name=winter ',
        'summer2winter_yosemite/testA --name summer2winter_yosemite_pretrained --save_name=winter ',
        'summer2winter_yosemite/trainB --name winter2summer_yosemite_pretrained --save_name=summer ',
        'summer2winter_yosemite/testB --name winter2summer_yosemite_pretrained --save_name=summer ',

        'apple2orange/trainA --name apple2orange_pretrained --save_name=orange ',
        'apple2orange/testA --name apple2orange_pretrained --save_name=orange ',
        'apple2orange/trainB --name orange2apple_pretrained --save_name=apple ',
        'apple2orange/testB --name orange2apple_pretrained --save_name=apple ',

        'facades/trainC --name facades_label2photo_pretrained --save_name=facades ',
        'facades/testC --name facades_label2photo_pretrained --save_name=facades ',

        'cityscapes/trainC --name cityscapes_label2photo_pretrained --save_name=cityscapes ',
        'cityscapes/testC --name cityscapes_label2photo_pretrained --save_name=cityscapes ',

        'maps/testB --name map2sat_pretrained --save_name=satellites ',
        'maps/trainB --name map2sat_pretrained --save_name=satellites ',

        'ukiyoe/trainC --name style_ukiyoe_pretrained --save_name=ukiyoe ',
        'ukiyoe/testC --name style_ukiyoe_pretrained --save_name=ukiyoe ',

        'vangogh/trainC --name style_vangogh_pretrained --save_name=vangogh ',
        'vangogh/testC --name style_vangogh_pretrained --save_name=vangogh ',

        'cezanne/trainC --name style_cezanne_pretrained --save_name=cezanne ',
        'cezanne/testC --name style_cezanne_pretrained --save_name=cezanne ',

        'monet2photo/trainB --name style_monet_pretrained --save_name=monet ',
        'monet2photo/testB --name style_monet_pretrained --save_name=monet ',

        'monet2photo/trainA --name monet2photo_pretrained --save_name=photo ',
        'monet2photo/testA --name monet2photo_pretrained --save_name=photo '
        ]
elif args.dataset == 'AutoGAN':
    parameter_set = [
        'horse2zebra/trainA --name horse_auto --save_name=horse_auto ',
        'horse2zebra/trainB --name zebra_auto --save_name=zebra_auto ',
        'summer2winter_yosemite/trainA --name summer_auto --save_name=summer_auto ',
        'summer2winter_yosemite/trainB --name winter_auto --save_name=winter_auto ',
        'apple2orange/trainA --name apple_auto --save_name=apple_auto ',
        'apple2orange/trainB --name orange_auto --save_name=orange_auto ',
        'facades/trainA --name facades_auto --save_name=facades_auto ',
        'cityscapes/trainA --name cityscapes_auto --save_name=cityscapes_auto ',
        'maps/trainA --name satellite_auto --save_name=satellite_auto ',
        'ukiyoe2photo/trainA --name ukiyoe_auto --save_name=ukiyoe_auto ',
        'vangogh2photo/trainA --name vangogh_auto --save_name=vangogh_auto ',
        'cezanne2photo/trainA --name cezanne_auto --save_name=cezanne_auto ',
        'monet2photo/trainA --name monet_auto --save_name=monet_auto ',
        'monet2photo/trainB --name photo_auto --save_name=photo_auto '
        ]
else:
    print('Not a valid dataset!')
    exit(-1)


number_gpu = len(gpu_set)
process_set = []

index = 0
for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
    command = 'python test.py --dataroot ./datasets/{} --upsampling transposed_conv --model test --no_dropout --num_test -1 --gpu_ids {} '\
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

