import os
import numpy as np
import cv2
import argparse
from shutil import copyfile

fold_list = ['fold6', 'fold7', 'fold8', 'fold9']
category_list = ['ukiyoe2photo', 'vangogh2photo', 'cezanne2photo', 'monet2photo']

for ind in range(len(fold_list)):
    fold = fold_list[ind]
    category = category_list[ind]
    fold_file = './fold_txt/{}.txt'.format(fold)
    
    pos_count = 0
    neg_count = 0
    with open(fold_file) as f:
        for line in f:
            if 'real' in line:
                filename = line.split('|')[0].split('/')[-1]
                filename = filename.replace('png','jpg')
                try:
                    os.stat('./{}/trainB/{}'.format(category, filename))
                    src_file = './{}/trainB/{}'.format(category, filename)
                    target_file = './{}/trainB/{}'.format(fold, filename)
                    copyfile(src_file, target_file)
                    pos_count+=1
                except:
                    neg_count+=1
                    pass

