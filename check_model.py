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
    model_path_2 = './checkpoints/cityscapes_label2photo_pretrained/latest_net_G.pth'


    state_dict_1 = torch.load(model_path_1)
    conv_1_1 = (state_dict_1['model.26.weight'].numpy()).copy()
    state_dict_2 = torch.load(model_path_2)
    conv_2_1 = (state_dict_2['model.26.weight'].numpy()).copy()

    conv_1_1 = conv_1_1.transpose((1,0,2,3))
    conv_2_1 = conv_2_1.transpose((1,0,2,3))
    for key in state_dict_1:
        print(key)

    print(state_dict_1['model.19.weight'].numpy().shape)
    print(state_dict_1['model.22.weight'].numpy().shape)
    print(state_dict_1['model.26.weight'].numpy().shape)
    print(conv_1_1.shape)
    print(conv_2_1.shape)

    for i in range(conv_1_1.shape[0]):
        for j in range(conv_1_1.shape[1]):
            sum_val = np.sqrt(np.sum(conv_1_1[i,j,:,:]*conv_1_1[i,j,:,:]))
            conv_1_1[i,j,:,:] = conv_1_1[i,j,:,:]/sum_val

    for i in range(conv_2_1.shape[0]):
        for j in range(conv_2_1.shape[1]):
            #conv_2_1[i,j,:,:] = conv_2_1[i,j,:,:]+np.random.rand(3, 3)*0.07
            sum_val = np.sqrt(np.sum(conv_2_1[i,j,:,:]*conv_2_1[i,j,:,:]))
            conv_2_1[i,j,:,:] = conv_2_1[i,j,:,:]/sum_val
    
    #sampled_channel = [0, 30, 50, 70, 100, 120]
    #sampled_channel = [0, 10, 20, 30, 40, 50, 60]
    sampled_channel = [0,1,2]
    max_corr = np.zeros((conv_1_1.shape[0],len(sampled_channel)))
    for l in range(len(sampled_channel)):
        for k in tqdm(range(conv_1_1.shape[0])):
            for i in range(conv_2_1.shape[0]):
                for j in range(conv_2_1.shape[1]):
                    corr = np.sum(conv_1_1[k,sampled_channel[l],:,:]*conv_2_1[i,j,:,:])
                    if corr>max_corr[k,l]:
                        max_corr[k,l] = corr
    print(np.amin(max_corr,axis = 0))
    print(np.amax(max_corr, axis = 0))

    #his = np.histogram(np.arange(4), bins=np.arange(5), density=True) 
    his = np.histogram(max_corr.flatten(), bins=np.linspace(0.8, 1.0, num=20))
    print(his)
    #print(state_dict['model.19.weight'].numpy()[0,0,:,:])
    
