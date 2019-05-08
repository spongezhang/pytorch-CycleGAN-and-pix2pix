from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        #from detector
        parser.add_argument('--dataroot', type=str,
                            default='../datasets/',
                            help='path to dataset')
        parser.add_argument('--log-dir', default='../splice_log/',
                            help='folder to output log')
        parser.add_argument('--training-set', default= 'synthesized_journals_2_train',
                            help='Other options: notredame, yosemite')
        parser.add_argument('--mean-image', type=float, default=0.443728476019,
                            help='mean of train dataset for normalization')
        parser.add_argument('--std-image', type=float, default=0.20197947209,
                            help='std of train dataset for normalization')
        parser.add_argument('--epochs', type=int, default=10, metavar='E',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--train-batch-size', type=int, default=64, metavar='BS',
                            help='input batch size for training (default: 1024)')
        parser.add_argument('--test-batch-size', type=int, default=128, metavar='BST',
                            help='input batch size for testing (default: 1024)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--fliprot', type=str2bool, default=False,
                            help='turns on flip and 90deg rotation augmentation')
        parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                            help='learning rate decay ratio (default: 1e-6')
        parser.add_argument('--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--optimizer', default='sgd', type=str,
                            metavar='OPT', help='The optimizer to use (default: SGD)')
        parser.add_argument('--model', default='cycle_gan', type=str,
                            metavar='OPT', help='The optimizer to use (default: SGD)')
        parser.add_argument('--freq_mode', type=int, default=0, metavar='BST',
                            help='input batch size for testing (default: 1024)')

        # Device options
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--data_augment', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--smooth', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--fft', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--gpu-id', default='0', type=str,
                            help='id(s) for CUDA_VISIBLE_DEVICES')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
