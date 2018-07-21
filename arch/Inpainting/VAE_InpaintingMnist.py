import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from InpaintingBase import VAEInpaintingBase


class VAE_InpaintingMnist(VAEInpaintingBase):
    '''
    Model architechture follow https://github.com/pytorch/examples/blob/master/vae/main.py
    '''
    def __init__(self, num_training=50000, **kwargs):
        super(VAE_InpaintingMnist, self).__init__(num_training=num_training)

        self.encode = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2, dilation=2),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def decouple(params, channel_num=None):
        if channel_num is None:
            channel_num = params.size(1) / 2

        mu = params[:, :channel_num, :, :]
        logvar = params[:, channel_num:, :, :]
        return mu, logvar

    def forward(self, x, mask):
        x = torch.cat((x * mask, 1. - mask), dim=1)

        z_params = self.encode(x)
        mu, logvar = self.decouple(z_params)

        z = self.reparametrize(mu, logvar)

        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_fn(self, outputs, targets, mask):
        recon_x, mu, logvar = outputs

        recon_loss = nn.BCELoss(size_average=False)(recon_x, targets)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5) / self.num_training

        self.total_loss += (recon_loss + KLD * targets.size(0)).data[0]
        self.pred_loss += recon_loss.data[0]
        self.reg_loss += KLD.data[0] * targets.size(0)
        self.num_instances += targets.size(0)

        return recon_loss / targets.size(0) + KLD


if __name__ == '__main__':
    test = torch.rand(1, 1, 28, 28)
    mask = torch.rand(1, 1, 28, 28).round()

    net = VAE_InpaintingMnist(num_training=50000)

    result = net.forward(Variable(test), Variable(mask))
    print(result[0].size())