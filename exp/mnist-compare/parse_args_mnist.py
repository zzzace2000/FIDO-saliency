import argparse
import torch

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dropout', type=str, default='gauss',
                    help='choose from ["gauss", "bern", "add_gauss"]')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--estop_num', type=int, default=-1,
                    help='early stopping at which number of alpha. Default as None')
parser.add_argument('--clip_max', type=int, default=100,
                    help='Clip at which number')
parser.add_argument('--vis_method', type=str, default='log_alpha',
                    help='By loss or log_alpha')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--ard_init', type=float, default=-6.,
                    help='ARD initialization')
parser.add_argument('--reg-coef', type=float, default=0.01,
                    help='regularization coefficient')
parser.add_argument('--no-cuda', action='store_false', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epoch-print', type=int, default=10,
                    help='how many epochs to wait before logging training status')
parser.add_argument('--edge', type=int, default=4,
                    help='Output edge*edge grid of images samples')
parser.add_argument('--save-dir', type=str, default='figs/',
                    help='Save directory')
parser.add_argument('--save-tag', type=str, default='0721-gauss',
                    help='Unique tag for output images')

args, _ = parser.parse_known_args()
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print 'cuda:', args.cuda