import torch.nn as nn
from torch.autograd import Variable
import torch
from .Baseline import InpaintTemplate


class InpaintingBase(InpaintTemplate):
    '''
    Implementation of Globally and Locally Consistent Image Completion
    http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
    '''
    def __init__(self):
        super(InpaintingBase, self).__init__()

        pth_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pth_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        self.pth_mean = nn.Parameter(pth_mean, requires_grad=False)
        self.pth_std = nn.Parameter(pth_std, requires_grad=False)

        # Set up loss record system
        self.zero_loss_record()

    def forward(self, x, mask):
        '''
        Output an impaint image that's btw 0 and 1
        :param x: 4d image (3d + 1d mask)
        :return: 3d image
        '''
        raise BaseException('Not implemented forward function')

    def generate_background(self, x, mask):
        '''
        Use to generate whole blurry images with pytorch normalization.
        '''
        outputs = self.forward(Variable(x, volatile=True), Variable(mask, volatile=True))
        return outputs[0].data

    def impute_missing_imgs(self, x, mask):
        '''
        Generate images but replace the part that don't need to impute by original img.
        Used in test time.
        '''
        generated_img = self.generate_background(x, mask)

        if mask.ndimension() == 3:
            mask = mask.unsqueeze(0)

        expand_mask = mask.expand_as(x)
        generated_img[expand_mask == 1] = x[expand_mask == 1]
        return generated_img

    '''
    The following functions are used to train and used in train_gen_model.py
    '''
    def loss_fn(self, outputs, targets, mask):
        loss = ((1. - mask) * (outputs[0] - targets) ** 2).sum()
        self.total_loss += loss.data[0]
        self.num_instances += outputs[0].size(0)
        return loss / outputs[0].size(0)

    def zero_loss_record(self):
        self.total_loss = 0.
        self.num_instances = 0

    def report_loss(self):
        return 'training loss: {}'.format(self.total_loss / self.num_instances)

    def total_avg_loss(self):
        return self.total_loss / self.num_instances


class VAEInpaintingBase(InpaintingBase):
    def __init__(self, num_training=100000):
        super(VAEInpaintingBase, self).__init__()

        print('num_training:', num_training)
        self.num_training = num_training

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = std.data.new(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def zero_loss_record(self):
        self.total_loss = 0.
        self.pred_loss = 0.
        self.reg_loss = 0.
        self.num_instances = 0

    def report_loss(self):
        return 'loss: {} ({}, {})'.format(self.total_loss / self.num_instances,
                                          self.pred_loss / self.num_instances,
                                          self.reg_loss / self.num_instances)
