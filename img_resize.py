#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: crop_jpg.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 05-02-2019
#  Last Modified: Sat Jul 13 22:37:13 2019
#
#  Usage: python crop_jpg.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================
from PIL import Image
from PIL import ImageFilter
import glob
import random
import cv2
import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='PyTorch Genealogy Classification')
parser.add_argument('--category', type=str,
                    default='horse',
                    help='path to dataset')
parser.add_argument('--subset', type=str,
                    default='testA',
                    help='path to dataset')
parser.add_argument('--resize_size', type=int, default=128, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--random_resize', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
if args.random_resize:
    args.resize_size = 'mixed'

try:
    os.stat('./generated/{}_resize_{}/{}/'.format(args.category, args.resize_size, args.subset))
except:
    os.makedirs('./generated/{}_resize_{}/{}/'.format(args.category, args.resize_size, args.subset))

try:
    os.stat('./datasets/{}_resize_{}/{}/'.format(args.category, args.resize_size, args.subset))
except:
    subprocess.call(["ln", "-s", "{}/".format(args.category), "./datasets/{}_resize_{}".format(args.category, args.resize_size)])
    #os.makedirs('./datasets/{}_resize_{}/{}/'.format(args.category, args.resize_size, args.subset))

resize_list = [256, 200, 150, 128]

for filename in glob.glob('./generated/{}/{}/*.png'.format(args.category, args.subset)):
    img=cv2.imread(filename)
    write_filename = './generated/{}_resize_{}/{}/{}'.format(args.category, args.resize_size, args.subset, filename.split('/')[-1]) 
    if args.random_resize:
        resize_size = random.choice(resize_list)
    else:
        resize_size=args.resize_size
    img = cv2.resize(img, (resize_size, resize_size))
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(write_filename, img)

#for filename in glob.glob('./datasets/{}/{}/*.jpg'.format(args.category, args.subset)):
#    img=cv2.imread(filename)
#    write_filename = './datasets/{}_resize_{}/{}/{}'.format(args.category, args.resize_size, args.subset, filename.split('/')[-1]) 
#    if args.random_resize:
#        resize_size = random.choice(resize_list)
#    else:
#        resize_size=args.resize_size
#    img = cv2.resize(img, (resize_size, resize_size))
#    img = cv2.resize(img, (256, 256))
#    cv2.imwrite(write_filename, img)
