from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
import sys
import os
import torch
import random

class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
        

        self.dataset_name = opt.name.split('_')[0] 
        self.count = 0
        try:
            os.stat('./generated/{}/test/'.format(self.dataset_name))
        except:
            os.makedirs('./generated/{}/test/'.format(self.dataset_name))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        add_noise_set = set(['module.model.19.weight', 'module.model.22.weight', 'module.model.26.weight'])
        noise_value = random.random()*0.1
        save_dict = {}
        for name, param in self.netG.named_parameters():
            if name in add_noise_set:
                save_dict[name] = torch.tensor(param)
                param.add_(torch.randn(param.size()).cuda() * noise_value)
        self.fake_B = self.netG(self.real_A)  # G(A)
        data = self.fake_B.cpu().numpy()
        data = np.transpose(data, (2, 3, 1, 0))
        #data = data[:,:,:,0]
        #for i in range(data.shape[2]):
        #    image = data[:,:,:]
        #    image = (image-np.amin(np.amin(image)))/(np.amax(np.amax(image))-np.amin(np.amin(image)))
        #    image = image*255
        #    image = image.astype(np.uint8)
        #    cv2.imwrite('./color/all_{}.png'.format(i), image)

        for i in range(data.shape[3]):
            image = data[:,:,:,i]
            image = (image-np.amin(np.amin(image)))/(np.amax(np.amax(image))-np.amin(np.amin(image)))
            image = image*255
            image = image.astype(np.uint8)
            image = image[...,::-1]
            cv2.imwrite('./generated/{}/test/{:04d}.png'.format(self.dataset_name, self.count), image)
            self.count+=1

        for name, param in self.netG.named_parameters():
            if name in add_noise_set:
                param.set_(save_dict[name])

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
