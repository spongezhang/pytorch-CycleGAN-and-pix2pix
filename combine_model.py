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

    model_path_1 = './checkpoints/summer2winter_yosemite_pretrained/latest_net_G.pth'
    model_path_2 = './checkpoints/horse2zebra_pretrained/latest_net_G.pth'
    save_model_path = './checkpoints/zebra2winter_pretrained/latest_net_G.pth'
    try:
        os.stat('./checkpoints/zebra2winter_pretrained/')
    except:
        os.makedirs('./checkpoints/zebra2winter_pretrained/')
    
    state_dict_1 = torch.load(model_path_1)
    state_dict_2 = torch.load(model_path_2)
    
    state_dict_2['model.19.weight'] = state_dict_1['model.19.weight']
    state_dict_2['model.19.bias'] = state_dict_1['model.19.bias']
    state_dict_2['model.22.weight'] = state_dict_1['model.22.weight']
    state_dict_2['model.22.bias'] = state_dict_1['model.22.bias']
    state_dict_2['model.26.weight'] = state_dict_1['model.26.weight']
    state_dict_2['model.26.bias'] = state_dict_1['model.26.bias']
    torch.save(state_dict_2, save_model_path)

    #print(state_dict['model.19.weight'].numpy().shape)
    #print(state_dict['model.22.weight'].numpy().shape)
    #print(state_dict['model.26.weight'].numpy().shape)
