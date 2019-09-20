#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: crop_jpg.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 05-02-2019
#  Last Modified: Sat Jul 13 22:07:07 2019
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
import random


parser = argparse.ArgumentParser(description='PyTorch Genealogy Classification')
parser.add_argument('--category', type=str,
                    default='horse',
                    help='path to dataset')
parser.add_argument('--subset', type=str,
                    default='testA',
                    help='path to dataset')
parser.add_argument('--jpg_level', type=int, default=90, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--random_jpg', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
if args.random_jpg:
    args.jpg_level = 'mixed'

try:
    os.stat('./generated/{}_jpg_{}/{}/'.format(args.category, args.jpg_level, args.subset))
except:
    os.makedirs('./generated/{}_jpg_{}/{}/'.format(args.category, args.jpg_level, args.subset))

try:
    os.stat("./datasets/{}_jpg_{}/{}/".format(args.category, args.jpg_level, args.subset))
except:
    #os.makedirs('./datasets/{}_jpg_{}/{}/'.format(args.category, args.jpg_level, args.subset))
    subprocess.call(["ln", "-s", "{}/".format(args.category), "./datasets/{}_jpg_{}".format(args.category, args.jpg_level)])

quality_list = [100, 90, 70, 50]
for filename in glob.glob('./generated/{}/{}/*.png'.format(args.category, args.subset)):
    img=cv2.imread(filename)
    write_filename = './generated/{}_jpg_{}/{}/{}'.format(args.category, args.jpg_level, args.subset, filename.split('/')[-1]) 
    write_filename = write_filename.replace('.png','.jpg')
    if args.random_jpg:
        jpg_level = random.choice(quality_list)
    else:
        jpg_level=args.jpg_level
    cv2.imwrite(write_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_level])

#for filename in glob.glob('./datasets/{}/{}/*.jpg'.format(args.category, args.subset)):
#    img=cv2.imread(filename)
#    write_filename = './datasets/{}_jpg_{}/{}/{}'.format(args.category, args.jpg_level, args.subset, filename.split('/')[-1]) 
#   # write_filename = write_filename.replace('.png','.jpg')
#    if args.random_jpg:
#        jpg_level = random.choice(quality_list)
#    else:
#        jpg_level=args.jpg_level
#    cv2.imwrite(write_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_level])
