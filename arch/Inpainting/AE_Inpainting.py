import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import datasets, transforms
from .InpaintingBase import InpaintingBase, VAEInpaintingBase


class ImpantingModel(InpaintingBase):
    '''
    Implementation of Globally and Locally Consistent Image Completion
    http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
    '''
    def __init__(self, **kwargs):
        super(ImpantingModel, self).__init__()

        self.impant = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Start dilation
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # End Dilation
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Upconvolution
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # Do normal convolution
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        gen_model_inputs = torch.cat((x * mask, 1. - mask), dim=1)

        result = self.impant(gen_model_inputs)
        result = (result - self.pth_mean) / self.pth_std
        return result,

    @staticmethod
    def test_itself():
        model = ImpantingModel()

        test = Variable(torch.rand(1, 4, 256, 256))
        output = model(test)
        print(output.size())
        print(output)


class VAEImpantModel(VAEInpaintingBase):
    def __init__(self, num_training=100000, **kwargs):
        super(VAEImpantModel, self).__init__(num_training=num_training)

        self.encode = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Start dilation
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # End Dilation
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Upconvolution
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # Do normal convolution
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def decouple(params, channel_num=None):
        if channel_num is None:
            channel_num = int(params.size(1) / 2)

        mu = params[:, :channel_num, :, :]
        logvar = params[:, channel_num:, :, :]
        return mu, logvar

    def forward(self, x, mask):
        x = torch.cat((x * mask, 1. - mask), dim=1)

        z_params = self.encode(x)
        mu, logvar = self.decouple(z_params)

        z = self.reparametrize(mu, logvar)

        recon_x = self.decode(z)
        recon_x = (recon_x - self.pth_mean) / self.pth_std
        return recon_x, mu, logvar

    def loss_fn(self, outputs, targets, mask):
        normalized_recon_x, mu, logvar = outputs

        recon_loss = ((1. - mask) * (normalized_recon_x - targets) ** 2).sum()

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5) / self.num_training

        self.total_loss += (recon_loss + KLD * targets.size(0)).data[0]
        self.pred_loss += recon_loss.data[0]
        self.reg_loss += KLD.data[0]
        self.num_instances += targets.size(0)

        return recon_loss / targets.size(0) + KLD


class VAEWithVarImpantModel(VAEImpantModel):
    def __init__(self, num_training=100000, **kwargs):
        super(VAEWithVarImpantModel, self).__init__(num_training=num_training)

        self.decode = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Upconvolution
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # Do normal convolution
            nn.Conv2d(32, 6, kernel_size=3, stride=1, padding=1, dilation=1),
        )

        self.pth_max = nn.Parameter((1. - self.pth_mean.data[0, :, 0, 0]) /
                                    self.pth_std.data[0, :, 0, 0],
                                    requires_grad=False)
        self.pth_min = nn.Parameter((0. - self.pth_mean.data[0, :, 0, 0]) /
                                    self.pth_std.data[0, :, 0, 0],
                                    requires_grad=False)

    def clamp_pth_min_max(self, x):
        result = []
        for i in range(3):
            result.append(x[:, i, :, :].clamp(min=self.pth_min.data[i], max=self.pth_max.data[i]))
        torch.cat(result, dim=1)
        return x

    def forward(self, x, mask):
        x = torch.cat((x * mask, 1. - mask), dim=1)

        z_params = self.encode(x)
        z_mu, z_logvar = self.decouple(z_params)
        z = self.reparametrize(z_mu, z_logvar)

        x_params = self.decode(z)
        x_mu, x_logvar = self.decouple(x_params)
        x_mu = self.clamp_pth_min_max(x_mu)

        recon_x = self.reparametrize(x_mu, x_logvar)
        recon_x = self.clamp_pth_min_max(recon_x)

        return recon_x, x_mu, x_logvar, z_mu, z_logvar

    def loss_fn(self, outputs, targets, mask):
        _, x_mu, x_logvar, z_mu, z_logvar = outputs

        # Calculate unnormalized prediction Gaussian loss
        recon_loss = 0.5 * (1. - mask) * (x_logvar + (targets - x_mu) ** 2 * torch.exp(-x_logvar))
        recon_loss_sum = recon_loss.sum()

        KLD_element = z_mu.pow(2).add_(z_logvar.exp()).mul_(-1).add_(1).add_(z_logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5) / self.num_training

        self.total_loss += (recon_loss_sum + KLD * targets.size(0)).data[0]
        self.pred_loss += recon_loss_sum.data[0]
        self.reg_loss += KLD.data[0] * targets.size(0)
        self.num_instances += targets.size(0)

        return recon_loss_sum / targets.size(0) + KLD


class VAEWithVarImpantModelMean(VAEWithVarImpantModel):
    def generate_background(self, x, mask):
        outputs = self.forward(Variable(x, volatile=True), Variable(mask, volatile=True))
        return outputs[1].data # Return x_mu only

if __name__ == '__main__':
    ImpantingModel.test_itself()