import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import models
import random
from torch.autograd import Variable
import cv2

class classificationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names = ['real_A', 'fake_A']

        self.visual_names = visual_names  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.real_index = 0
        self.fake_index = 0
        if self.isTrain:
            self.model_names = ['D_A']
        else:  # during test time, only load Gs
            self.model_names = ['D_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        with torch.no_grad():
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, 'transposed_conv', opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = torchvision.models.resnet34(pretrained=True)
            num_ftrs = self.netD_A.fc.in_features
            self.netD_A.fc = nn.Linear(num_ftrs, 2)
            self.netD_A.to(self.device)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = nn.CrossEntropyLoss().to(self.device)  # define GAN loss.
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.direction == 'A':
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']
        elif self.opt.direction == 'B':
            self.real_A = input['B'].to(self.device)
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_A = self.netG_A(self.real_A)
        cat_image = torch.cat((self.real_A, self.fake_A), 0)
        real_image = self.real_A.cpu().numpy()
        fake_image = self.fake_A.detach().cpu().numpy()

        all_image = np.transpose(real_image, (2, 3, 1, 0))
        for i in range(real_image.shape[0]):
            image = all_image[:,:,:,i]
            image = (image + 1.0)*0.5
            image = image*255
            image = image.astype(np.uint8)
            image = image[...,::-1]
            cv2.imwrite('./test/real/{:04d}.jpg'.format(self.real_index), image)
            self.real_index+=1

        all_image = np.transpose(fake_image, (2, 3, 1, 0))
        for i in range(real_image.shape[0]):
            image = all_image[:,:,:,i]
            image = (image + 1.0)*0.5
            image = image*255
            image = image.astype(np.uint8)
            image = image[...,::-1]
            cv2.imwrite('./test/fake/{:04d}.png'.format(self.fake_index), image)
            self.fake_index+=1

        all_image = np.zeros((real_image.shape[0]*2, real_image.shape[1], 224, 224), dtype = np.float32)
        for i in range(real_image.shape[0]):
            random_x = random.randint(0,32)
            random_y = random.randint(0,32)
            all_image[i,:,:,:] = real_image[i,:,random_y:(random_y+224),\
                    random_x:(random_x+224)]
            all_image[real_image.shape[0]+i,:,:,:] = fake_image[i,:,random_y:(random_y+224),\
                    random_x:(random_x+224)]

        self.all_image = Variable(torch.from_numpy(all_image)).to(self.device)
        all_label = np.zeros((real_image.shape[0]*2,), dtype = np.int)
        all_label[:real_image.shape[0]] = 1
        self.all_label = Variable(torch.from_numpy(all_label)).to(self.device)
        self.out = self.netD_A(self.all_image)
        _, self.pred = torch.max(self.out,1)
        pred = self.pred.data.cpu().numpy().flatten()
        self.acc = np.sum(all_label == pred)/float(real_image.shape[0]*2)
        self.label = all_label
        self.pred = pred


    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.criterionGAN(self.out, self.all_label)
        self.loss_D_A.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

    def compute_visuals(self):
        print(self.fake_A.shape)

    def get_visuals(self):
        data = self.fake_A.cpu().detach().numpy()
        data = np.transpose(data, (2, 3, 1, 0))
        return data
