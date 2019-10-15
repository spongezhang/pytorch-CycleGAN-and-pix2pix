#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: prepare_data.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Mon Sep 23 18:42:21 2019
#
#  Usage: python prepare_data.py
#  Description: Download all images and prepare for autoGAN training
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


parameter_set = [
        "horse2zebra", 
        "apple2orange",
        "summer2winter_yosemite",
        "monet2photo",
        "cezanne2photo",
        "ukiyoe2photo",
        "vangogh2photo",
        "maps",
        "cityscapes",
        "facades"
        ]

for idx, parameter in enumerate(parameter_set):
    command = './datasets/download_cyclegan_dataset.sh {} '.format(parameter)
    print(command)
    subprocess.call(command,shell=True)
    if parameter in ['cityscapes', 'facades']:
        subprocess.call("mv ./datasets/{}/testB ./datasets/{}/testC".format(parameter, parameter),shell=True)
        subprocess.call("mv ./datasets/{}/trainB ./datasets/{}/trainC".format(parameter, parameter),shell=True)

    mapping_1 = {"monet2photo": 'monet',
            "maps": "satellite",
            "cezanne2photo": "cezanne",
            "ukiyoe2photo": "ukiyoe",
            "vangogh2photo": "vangogh",
            }
    if parameter in mapping_1:
        subprocess.call("mkdir ./datasets/{}".format(mapping_1[parameter]),shell=True)
        subprocess.call("ln -s ./datasets/{}/trainA ./datasets/{}/trainA".format(parameter, mapping_1[parameter]),shell=True)
        subprocess.call("ln -s ./datasets/{}/testA ./datasets/{}/testA".format(parameter, mapping_1[parameter]),shell=True)

    mapping_2 = {"monet2photo": 'photo'}
    if parameter in mapping_2:
        subprocess.call("mkdir ./datasets/{}".format(mapping_2[parameter]),shell=True)
        subprocess.call("ln -s ./datasets/{}/trainB ./datasets/{}/trainB".format(parameter, mapping_2[parameter]),shell=True)
        subprocess.call("ln -s ./datasets/{}/testB ./datasets/{}/testB".format(parameter, mapping_2[parameter]),shell=True)


