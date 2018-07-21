import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import datasets, transforms
from arch.Inpainting.InpaintingBase import InpaintingBase, VAEInpaintingBase

from exp.vbd_imagenet.nets.BVLC.BVLC_NET import ImpaintBVLC, VAEImpaintBVLC, BVLC_NET
from exp.vbd_imagenet.nets.GAN.GAN_G import PureGAN_G
import numpy as np
import os


class GAN_Common:
    def set_up_gan_g(self, gan_g_dir):
        # Define generator and load weights
        self.gan_g_model = PureGAN_G()
        self.gan_g_model.load_state_dict(torch.load(os.path.join(gan_g_dir, 'GAN_G.pth'),
                                                    map_location=lambda storage, loc: storage))
        for param in self.gan_g_model.parameters():
            param.requires_grad = False


class AE_GAN(InpaintingBase, GAN_Common):
    def __init__(self, bvlc_dir, gan_g_dir, reg_coef=0., **kwargs):
        super(AE_GAN, self).__init__()

        self.reg_coef = reg_coef

        ## Define encoder
        pretrained_dict = torch.load(os.path.join(bvlc_dir, 'bvlc_reference_pytorch_net.pth'))
        del pretrained_dict['fc7.weight'], pretrained_dict['fc7.bias'], \
            pretrained_dict['fc8.weight'], pretrained_dict['fc8.bias']

        self.bvlc_encoder_net = ImpaintBVLC()
        state = self.bvlc_encoder_net.state_dict()
        state.update(pretrained_dict)
        # Save some storage
        self.bvlc_encoder_net.load_state_dict(state)

        # To save some memories if not used autoencoding loss
        if reg_coef > 0.:
            self.hidden_fixed_encoder = BVLC_NET()
            state = self.hidden_fixed_encoder.state_dict()
            state.update(pretrained_dict)
            self.hidden_fixed_encoder.load_state_dict(state)
            for param in self.hidden_fixed_encoder.parameters():
                param.requires_grad = False

        ## Define generator
        self.set_up_gan_g(gan_g_dir)

    def state_dict(self):
        return self.bvlc_encoder_net.state_dict()

    def load_state_dict(self, state_dict):
        self.bvlc_encoder_net.load_state_dict(state_dict)

    def forward(self, x, mask=None):
        # Encode the parts
        if mask is None:
            mask = Variable(x.data.new(x.size(0), 1, x.size(2), x.size(3)).fill_(0.))

        h = self.bvlc_encoder_net.train_generate_fc6_code(x, mask)

        # GAN generator parts
        x = self.gan_g_model(h)
        # Center crop
        x1 = self.center_crop(x)

        # Change from BGR to RGB. For now pytorch doesn't support ::-1.
        # Look https://github.com/pytorch/pytorch/issues/229
        inv_idx = Variable(x1.data.new([2, 1, 0]).long(), requires_grad=False)
        x = x1.index_select(1, inv_idx)

        # Normalize caffe range to 0 ~ 1
        x = self.normalize_imgs(x)
        # Pytorch normalization
        x = (x - self.pth_mean) / self.pth_std
        return x, x1, h

    @staticmethod
    def center_crop(x, size=224):
        # assert x.size(2) > size and x.size(3) > size

        top_left_coord = ((x.size(2) - size) // 2, (x.size(3) - size) // 2)
        return x[:, :, top_left_coord[0]:(top_left_coord[0] + size),
               top_left_coord[1]:(top_left_coord[1] + size)]

    @staticmethod
    def normalize_imgs(x, input_range=(-120, 120)):
        x[x < input_range[0]] = input_range[0]
        x[x > input_range[1]] = input_range[1]
        return (x - input_range[0]) / (input_range[1] - input_range[0])

    def loss_fn(self, outputs, targets, mask):
        x, x1, h = outputs
        pred_loss = (mask * (x - targets) ** 2).sum()

        reg_loss = 0.
        if self.reg_coef != 0.:
            reg_loss = self.reg_coef * ((self.hidden_fixed_encoder(x1) - h) ** 2).sum()
            self.reg_loss += reg_loss.data[0]

        total_loss = pred_loss + reg_loss

        self.total_loss += total_loss.data[0]
        self.pred_loss += pred_loss.data[0]
        self.num_instances += x.size(0)
        return total_loss / x.size(0)

    def zero_loss_record(self):
        self.total_loss = 0.
        self.pred_loss = 0.
        self.reg_loss = 0.
        self.num_instances = 0

    def report_loss(self):
        return 'loss: {} ({}, {})'.format(self.total_loss / self.num_instances,
                                          self.pred_loss / self.num_instances,
                                          self.reg_loss / self.num_instances)


class VAE_GAN(VAEInpaintingBase, GAN_Common):
    def __init__(self, bvlc_dir, gan_g_dir, clamp=True, num_training=100000, **kwargs):
        super(VAE_GAN, self).__init__(num_training=num_training)
        self.clamp = clamp

        # Define encoder
        self.bvlc_encoder_net = VAEImpaintBVLC()
        state = self.bvlc_encoder_net.state_dict()
        state.update(torch.load(os.path.join(bvlc_dir, 'bvlc_reference_pytorch_net.pth')))
        self.bvlc_encoder_net.load_state_dict(state)

        self.set_up_gan_g(gan_g_dir)

    def forward(self, x):
        # Encode the parts
        mu, logvar = self.bvlc_encoder_net.generate_fc6_code(x)
        z = self.reparametrize(mu, logvar)
        if self.clamp:
            z = self.bound_range_(z)

        # GAN generator parts
        x = self.gan_g_model(z)
        # Center crop
        x = self.center_crop(x)

        # Change from BGR to RGB. For now pytorch doesn't support ::-1.
        # Look https://github.com/pytorch/pytorch/issues/229
        inv_idx = Variable(x.data.new([2, 1, 0]).long(), requires_grad=False)
        x = x.index_select(1, inv_idx)

        # Normalize caffe range to 0 ~ 1
        x = self.normalize_imgs(x)
        return x, mu, logvar

