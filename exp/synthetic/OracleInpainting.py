import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from median_pool import MedianPool2d
from datasets import MixtureOfBlocks
from arch.Inpainting.Baseline import InpaintTemplate, BlurryInpainter, LocalMeanInpainter, MeanInpainter


class OracleInpainting(InpaintTemplate):
    """Heuristic oracle inpainting for the mixture of blocks dataset"""
    def __init__(self, mode='median'):
        super(OracleInpainting, self).__init__()
        assert mode.lower() in ['median', 'avg', 'average'], 'unsupported mode'
        filter_class = MedianPool2d if mode.lower() == 'median' else torch.nn.AvgPool2d
        self.filt = filter_class(
                kernel_size=MixtureOfBlocks.block_width,
                stride=MixtureOfBlocks.block_width//2)

    def impute_missing_imgs(self, x, mask):
        # 1) apply mask, apply filter and compute label probs
        label_probs = self.infer_label_probs(x, mask)
        # 2) sample a label
        label_samps = torch.multinomial(label_probs, 1, replacement=True).squeeze().long().detach().data.numpy()
        # 3) generate block according to sampled label
        betas = x.max(-1)[0].max(-1)[0].clamp(0., 1.).squeeze().numpy()  # max pixel value; a crude esitimate of beta
        infill = torch.stack([
            torch.Tensor(
                MixtureOfBlocks.generate_image(beta, label)
                ) for label, beta in zip(label_samps, betas)], 0)
        # 4) return mixture of mask*x and (1-masked)*infill
        return (1. - mask)*x + mask*infill

    def infer_label_probs(self, x, mask, imshape=False):
        xm =  (1. - mask)*x + mask*MixtureOfBlocks.noise_level*torch.randn(*mask.shape)
        logits = self.filt(Variable(xm))
        off = MixtureOfBlocks.offset
        logits = logits[:, :, 1:-1, 1:-1].contiguous()  # cut down to 4x4
        label_probs = F.softmax(logits.view(batch_size, -1), 1).detach()
        # heuristic: draw from only the top two labels
        sorted_probs = torch.sort(label_probs, 1, descending=True)[0]
        label_probs = label_probs*(label_probs > sorted_probs[:, 2, None]).float()  # take top two
        label_probs = label_probs / label_probs.sum(1, keepdim=True)
        if imshape:  # return probs in image shape format; for plotting
            return label_probs.view(*logits.shape)
        else:
            return label_probs


def random_mask(im_shape, p=0.1):
    return torch.Tensor(np.random.binomial(1, p, size=im_shape)).unsqueeze(0)


def random_mask_batch(batch_size, im_shape, p=0.1):
    return torch.stack([random_mask(im_shape, p) for _ in range(batch_size)], 0)


def random_half_block_mask(im_shape, label=None):
    """random half-block split vertically"""
    if label is None:
        label = np.random.choice(range(2*MixtureOfBlocks.num_labels))
    block = np.zeros(im_shape)
    row_start, row_end, col_start, col_end = MixtureOfBlocks._label_to_patch_pixels(
            label, MixtureOfBlocks.block_width, MixtureOfBlocks.block_width//2  # half stride
            )
    block[row_start:row_end, col_start:col_end] += 1.  # the ground truth block
    #offset = np.random.choice((-1, 1))*MixtureOfBlocks.block_width//2
    # posterior conditioned on mask should be bimodal; don't mask edge pixels b/c there is no ambiguity
    offset = (-1 if label % int(MixtureOfBlocks.num_labels ** 0.5) < 2 else 1) * MixtureOfBlocks.block_width//2  
    offset_dim  = 1
    mask = block * np.roll(block, offset, offset_dim)
    mask += (1-block) * np.roll(block, -offset, offset_dim)
    return torch.Tensor(mask).unsqueeze(0)


def random_half_block_mask_batch(batch_size, im_shape, labels=None):
    if labels is None:
        labels = [None for _ in range(batch_size)]
    return torch.stack([random_half_block_mask(im_shape, l) for l in labels], 0)


if __name__ == '__main__':
    import os
    from torchvision.utils import save_image

    dirname = './plots/OracleInpainting'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    from datasets import mixture_of_blocks


    p = 0.2

    num_samples = 512
    batch_size = 64
    seed = 0
    plot = True  # set to true if using the jupyter notebook to viz

    loader = mixture_of_blocks(num_samples, batch_size, seed)
    mode = 'median'
    if mode == 'median':
        filter_class = MedianPool2d
    else:
        filter_class = torch.nn.AvgPool2d
    filt = filter_class(
            kernel_size=MixtureOfBlocks.block_width,
            stride=MixtureOfBlocks.block_width//2)
    f = lambda x: filt(Variable(x + MixtureOfBlocks.noise_level*torch.randn(*x.shape)))  # annoying workaround...

    #inpainter = OracleInpainting(mode)
    #inpainter = LocalMeanInpainter(ndim=1)  # baseline
    inpainter = MeanInpainter()  # baseline


    for i, (x, y) in enumerate(loader):
        im_shape = x.shape[-2:]
        mb = random_mask_batch(batch_size, im_shape, p)  # random mask
        #mb = random_half_block_mask_batch(batch_size, im_shape, labels=y.long().squeeze().numpy())  # "correct" mask
        if isinstance(inpainter, OracleInpainting):
            pl = inpainter.infer_label_probs(x, mb, True).data
        z = inpainter.impute_missing_imgs(x, mb)
        xm = x*(1. - mb)
        cont = lambda t: t[:, :, 1:-1, 1:-1].contiguous()  # cut down to 4x4
        yhat_probs = F.softmax(cont(f(xm)).view(batch_size, -1), 1)
        yhat = torch.max(yhat_probs, 1)[1].data.float().unsqueeze(1)
        acc = torch.sum(yhat.eq(y)) / len(y)
        print(i, 'acc', acc)

        if plot:
            save_image(mb, '{}/masked-blocks-{}-masks.png'.format(dirname, i), 
                    nrow=int(batch_size ** 0.5), pad_value=1., range=[0., 1.])
            save_image(xm, '{}/masked-blocks-{}.png'.format(dirname, i), 
                    nrow=int(batch_size ** 0.5), pad_value=1., range=[0., 1.])
            #save_image(f(xm).data, '{}/masked-blocks-{}-filt.png'.format(dirname, i), 
                    #nrow=int(batch_size ** 0.5), pad_value=1., range=[0., 1.])
            save_image(x, '{}/masked-blocks-{}-x.png'.format(dirname, i), 
                    nrow=int(batch_size ** 0.5), pad_value=1., range=[0., 1.])
            if isinstance(inpainter, OracleInpainting):
                save_image(pl, '{}/masked-blocks-{}-probs.png'.format(dirname, i), 
                        nrow=int(batch_size ** 0.5), pad_value=1., range=[0., 1.])
            save_image(z, '{}/masked-blocks-{}-impute.png'.format(dirname, i), 
                    nrow=int(batch_size ** 0.5), pad_value=1., range=[0., 1.])
            break  # in plot mode we only process one batch

    print('done')

