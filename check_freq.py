"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    model_path_1 = './checkpoints/horse2zebra_pretrained/latest_net_G.pth'
    #model_path_2 = './checkpoints/cityscapes_label2photo_pretrained/latest_net_G.pth'
    model_path_2 = './checkpoints/orange2apple_pretrained/latest_net_G.pth'

    state_dict = torch.load(model_path_2)
    conv_1_1 = (state_dict['model.19.weight'].numpy()).copy()
    
    fft_img = np.ones((256,256),dtype = np.int)
    low_num = np.sum(fft_img[:24,:])
    low_num += np.sum(fft_img[232:,:])
    low_num += np.sum(fft_img[24:232,:24])
    low_num += np.sum(fft_img[24:232,232:])

    mid_num = np.sum(fft_img[24:54,24:232])
    mid_num += np.sum(fft_img[201:232, 24:232])
    mid_num += np.sum(fft_img[54:201,24:54])
    mid_num += np.sum(fft_img[54:201,201:232])
    
    high_num = np.sum(fft_img[54:201, 54:201])
    
    high_count = 0
    mid_count = 0
    low_count = 0

    for i in tqdm(range(conv_1_1.shape[0])):
        for j in range(conv_1_1.shape[1]):
            #conv_1_1[i,j,:,:] = conv_1_1[i,j,:,:]+np.random.rand(3, 3)*0.07
            zeros_mat = np.zeros((256,256), dtype = np.float)
            zeros_mat[0:3,0:3] = conv_1_1[i,j,:,:]
            fft_img = np.fft.fft2(zeros_mat)
            fft_img = abs(fft_img)
            fft_img = fft_img*fft_img

            low_freq = np.sum(fft_img[:24,:])
            low_freq += np.sum(fft_img[232:,:])
            low_freq += np.sum(fft_img[24:232,:24])
            low_freq += np.sum(fft_img[24:232,232:])

            mid_freq = np.sum(fft_img[24:54,24:232])
            mid_freq += np.sum(fft_img[201:232, 24:232])
            mid_freq += np.sum(fft_img[54:201,24:54])
            mid_freq += np.sum(fft_img[54:201,201:232])
            
            high_freq = np.sum(fft_img[54:201, 54:201])

            low_freq /= low_num
            mid_freq /= low_num
            high_freq /= low_num

            max_freq = max([low_freq,mid_freq,high_freq])
            if high_freq==max_freq:
                high_count+=1
            elif mid_freq==max_freq:
                mid_count+=1
            else:
                low_count+=1
    print(high_count, mid_count, low_count)

