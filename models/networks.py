import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_normal_, calculate_gain
import numpy as np
from torchvision import models


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, model = 'resnet', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if model is not 'resnet':
        init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', upsampling_type='transposed_conv', n_downsample=2, init_gain=0.02,  gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                upsampling_type=upsampling_type, n_downsample = n_downsample, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'pggan':     # classify if each pixel is real or fake
        net = Discriminator(input_nc, mbstat_avg='all', resolution=256, fmap_max=512, fmap_base=8192, sigmoid_at_end=False)
    elif netD == 'resnet':     # classify if each pixel is real or fake
        net = models.resnet34(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 1) #nn.Conv2d(in_channels=num_ftrs, out_channels=1, kernel_size=1, stride=1, padding=0) #
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)

class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming
        self.scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        self.scale=self.scale.cuda()
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = self.scale*x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
        return self.__class__.__name__ + param_str

def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0  # default
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal_(layer.weight, a=gain)

def G_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
        to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_batchnorm:
        layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def G_conv_no_Relu(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
        to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if use_batchnorm:
        layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_downsample = 2, n_blocks=6, padding_type='reflect', upsampling_type='transposed_conv'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = n_downsample
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            if upsampling_type == 'transposed_conv':
                mult = 2 ** (n_downsampling - i)
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

            elif upsampling_type == 'nearest_neighbor':
                mult = 2 ** (n_downsampling - i)
                ic = ngf * mult
                oc = int(ngf * mult / 2)
                layers = [nn.Upsample(scale_factor=2, mode='nearest')]
                negative_slope = 0.2
                act = nn.LeakyReLU(negative_slope=negative_slope) 
                iact = 'leaky_relu'
                layers = G_conv(layers, ic, oc, 3, 1, act, iact, negative_slope, False, True, False, True)
                net = G_conv_no_Relu(layers, oc, oc, 3, 1, act, iact, negative_slope, True, True, False, True)
                model.append(net)

        if upsampling_type == 'transposed_conv':
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]
        elif upsampling_type == 'nearest_neighbor':
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

def D_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
        to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = incoming
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_layernorm:
        layers += [LayerNormLayer()]  # TODO: requires incoming layer
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def NINLayer(incoming, in_channels, out_channels, nonlinearity, init, param=None, 
            to_sequential=True, use_wscale=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if not (nonlinearity == 'linear'):
        layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

class DSelectLayer(nn.Module):
    def __init__(self, pre, chain, inputs):
        super(DSelectLayer, self).__init__()
        assert len(chain) == len(inputs)
        self.pre = pre
        self.chain = chain
        self.inputs = inputs
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N  # cur_level: physical index
        if y is not None:
            assert insert_y_at is not None

        max_level, min_level = int(np.floor(self.N-cur_level)), int(np.ceil(self.N-cur_level))
        min_level_weight, max_level_weight = int(cur_level+1)-cur_level, cur_level-int(cur_level)
        
        _from, _to, _step = min_level+1, self.N, 1

        if self.pre is not None:
            x = self.pre(x)

        if max_level == min_level:
            x = self.inputs[max_level](x)
            if max_level == insert_y_at:
                x = self.chain[max_level](x, y)
            else:
                x = self.chain[max_level](x)
        else:
            out = {}
            tmp = self.inputs[max_level](x)
            if max_level == insert_y_at:
                tmp = self.chain[max_level](tmp, y)
            else:
                tmp = self.chain[max_level](tmp)
            out['max_level'] = tmp
            out['min_level'] = self.inputs[min_level](x)
            x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                                out['max_level'] * max_level_weight
            if min_level == insert_y_at:
                x = self.chain[min_level](x, y)
            else:
                x = self.chain[min_level](x)

        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)

        return x

class GDropLayer(nn.Module):
    """
    # Generalized dropout layer. Supports arbitrary subsets of axes and different
    # modes. Mainly used to inject multiplicative Gaussian noise in the network.
    """
    def __init__(self, mode='mul', strength=0.2, axes=(0,1), normalize=False):
        super(GDropLayer, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str

class MinibatchStatConcatLayer(nn.Module):
    """Minibatch stat concatenation layer.
    - averaging tells how much averaging to use ('all', 'spatial', 'none')
    """
    def __init__(self, averaging='all'):
        super(MinibatchStatConcatLayer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8) #Tstdeps in the original implementation

    def forward(self, x):
        shape = list(x.size())
        target_shape = shape.copy()
        vals = self.adjusted_std(x, dim=0, keepdim=True)# per activation, over minibatch dim
        if self.averaging == 'all':  # average everything --> 1 value per minibatch
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)#vals = torch.mean(vals, keepdim=True)

        elif self.averaging == 'spatial':  # average spatial locations
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)  # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':  # no averaging, pass on all information
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':  # EXPERIMENTAL: compute variance (func) over minibatch AND spatial locations.
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)  # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':  # variance of ALL activations --> 1 value per minibatch
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:  # self.averaging == 'group'  # average everything over n groups of feature maps --> n values per minibatch
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1) # feature-map concatanation

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)

class Discriminator(nn.Module):
    def __init__(self, 
                num_channels    = 1,        # Overridden based on dataset.
                resolution      = 32,       # Overridden based on dataset.
                label_size      = 0,        # Overridden based on dataset.
                fmap_base       = 4096,
                fmap_decay      = 1.0,
                fmap_max        = 256,
                mbstat_avg      = 'all',
                mbdisc_kernels  = None,
                use_wscale      = True,
                use_gdrop       = True,
                use_layernorm   = False,
                sigmoid_at_end  = False):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end

        R = int(np.log2(resolution))
        assert resolution == 2**R and resolution >= 4
        gdrop_strength = 0.0

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        # input activation
        iact = 'leaky_relu'
        # output activation
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_iact = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        nins = nn.ModuleList()
        lods = nn.ModuleList()
        pre = None

        nins.append(NINLayer([], self.num_channels, self.get_nf(R-1), act, iact, negative_slope, True, self.use_wscale))

        for I in range(R-1, 1, -1):
            ic, oc = self.get_nf(I), self.get_nf(I-1)
            net = D_conv([], ic, ic, 3, 1, act, iact, negative_slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            lods.append(nn.Sequential(*net))
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            nin = []
            nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
            nins.append(nin)

        net = []
        ic = oc = self.get_nf(1)
        if self.mbstat_avg is not None:
            net += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
            ic += 1
        net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, 
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        net = D_conv(net, oc, self.get_nf(0), 4, 0, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)

        # Increasing Variation Using MINIBATCH Standard Deviation
        if self.mbdisc_kernels:
            net += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

        oc = 1 + self.label_size
        # lods.append(NINLayer(net, self.get_nf(0), oc, 'linear', 'linear', None, True, self.use_wscale))
        lods.append(NINLayer(net, self.get_nf(0), oc, output_act, output_iact, None, True, self.use_wscale))

        self.output_layer = DSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)


