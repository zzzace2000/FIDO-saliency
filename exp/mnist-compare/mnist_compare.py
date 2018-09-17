import sys
from os import path
sys.path.append(path.dirname(path.dirname(sys.path[1])))

from arch.MnistConvNet import MnistConvNet
from arch.sensitivity.GDNet import GDNet, AdditiveGDNet
from arch.sensitivity.BDNet import BDNet
from exp.loaddata_utils import load_mnist_one_image
import torch
from torch.autograd import Variable
import exp.utils_visualise as utils_visualise
import os
from parse_args_mnist import args
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mnist_compare_utils

# Mnist size
input_size = (28, 28)

# Load the mnist convolution model
trained_classifier = torch.load('../../arch/pretrained/mnist.model')

if args.dropout == 'bern':
    net = BDNet(input_size, trained_classifier, ard_init=args.ard_init,
                lr=args.lr, reg_coef=args.reg_coef, rw_max=30, cuda_enabled=args.cuda,
                estop_num=args.estop_num, clip_max=args.clip_max,
                flip_val=(-0.1307 / 0.3081) # bakground value
                )
else:
    if args.dropout == 'gauss':
        dropout_net = GDNet
    else:
        dropout_net = AdditiveGDNet

    net = dropout_net(input_size, trained_classifier, ard_init=args.ard_init,
                      lr=args.lr, reg_coef=args.reg_coef, rw_max=30, cuda_enabled=args.cuda,
                      estop_num=args.estop_num, clip_max=args.clip_max,
                      )

# Get vector of importance.
# So I need to do them here. Compute all the images and get the maybe just remove top 20% (157)
img_loader = load_mnist_one_image(img_index=100, batch_size=args.batch_size, cuda=args.cuda)
net.fit(img_loader, epochs=args.epochs, epoch_print=args.epoch_print)

rank = -net.get_param().data

# Take out one mnist image and unnormalize it
images, labels = iter(img_loader).next()

flip_log_odd, orig_log_odd, flip_img, orig_img = \
    mnist_compare_utils.calculate_logodds_diff_by_flipping(trained_classifier, images[0, ...],
                                       labels[0], rank, flip_val=(-0.1307 / 0.3081))

def plot_results(img1, img2, img1_title, img2_title):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(img1, cmap='gray', interpolation='nearest')
    plt.colorbar(im)
    plt.title(img1_title)
    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(img2, cmap='gray', interpolation='nearest')
    plt.colorbar(im2)
    plt.title(img2_title)

    # filename = '%s/%s_debug.png' % (args.save_dir, args.save_tag)
    plt.show()
    # plt.savefig(filename, dpi=300)
    # plt.close()

plot_results(orig_img, flip_img, str(orig_log_odd), str(flip_log_odd))


# # Save images
# if not os.path.exists(args.save_dir):
#     os.mkdir(args.save_dir)
#
# filename = '%s/%s_overlayed.png' % (args.save_dir, args.save_tag)
# utils_visualise.save_figs(overlayed_imgs, filename, nrow=args.edge)
#
# import matplotlib.pyplot as plt
# # Debug: see the c vector
# img = overlayed_imgs[0].numpy()[0, ...]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# im = ax1.imshow(img, interpolation='nearest')
# plt.colorbar(im)
# ax2 = fig.add_subplot(122)
# im2 = ax2.imshow(net.eval_reg().data.numpy(), interpolation='nearest')
# plt.colorbar(im2)
#
# filename = '%s/%s_debug.png' % (args.save_dir, args.save_tag)
# # plt.show()
# plt.savefig(filename, dpi=300)
# plt.close()
#
# # filename = '%s/%s_orig.png' % (args.save_dir, args.save_tag)
# # save_fig(orig_imgs, filename)
# # plt.imshow(img, cmap=cm.seismic, vmin=-np.max(np.abs(img)),
# #            vmax=np.max(np.abs(img)), interpolation='nearest')
# # plt.show()
