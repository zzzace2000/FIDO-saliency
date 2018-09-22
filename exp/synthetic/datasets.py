import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


def batch_bernoulli_logp(x, p):
    """x is a batch of data, p is bernoulli probs per dim of x"""
    x_is_im = x.ndimension() == 4  # note that only 1d images are supported
    if x_is_im:
        x = x.squeeze(1)
    px = torch.distributions.Bernoulli(p)
    logp_x = torch.stack([px.log_prob(xx) for xx in x], 0)
    return logp_x.unsqueeze(1) if x_is_im else logp_x


def logsumexp(inputs, dim=None, keepdim=False):  # from https://github.com/pytorch/pytorch/issues/2591
    import torch.nn.functional as F
    from torch.autograd import Variable
    diff = inputs - F.log_softmax(
            Variable(inputs, requires_grad=False), dim=dim
            ).data
    return diff.mean(dim, keepdim=keepdim)


def random_mask(im_shape, p=0.1):  # p is the drop probability; m[i,j] = 0 means drop the pixel
    return torch.Tensor(1. - np.random.binomial(1, p, size=im_shape)).unsqueeze(0)


def random_mask_batch(batch_size, im_shape, p=0.1):
    return torch.stack([random_mask(im_shape, p) for _ in range(batch_size)], 0)


def random_half_block_mask(im_shape, label=None):
    """random half-block split vertically"""
    if label is None:
        label = np.random.choice(range(2*Blocks.num_labels))
    block = np.ones(im_shape)  # mask[i, j] = 1 means keep the pixel
    row_start, row_end, col_start, col_end = Blocks._label_to_patch_pixels(
            label, Blocks.block_width, Blocks.block_width//2  # half stride
            )
    block[row_start:row_end, col_start:col_end] -= 1.  # the ground truth block
    # posterior conditioned on mask should be bimodal; don't mask edge pixels b/c there is no ambiguity
    offset = (-1 if label % int(Blocks.num_labels ** 0.5) < 2 else 1) * Blocks.block_width//2  
    offset_dim  = 1
    mask = block * np.roll(block, offset, offset_dim)
    mask += (1-block) * np.roll(block, -offset, offset_dim)
    return torch.Tensor(mask).unsqueeze(0)


def random_half_block_mask_batch(batch_size, im_shape, labels=None):
    if labels is None:
        labels = [None for _ in range(batch_size)]
    return torch.stack([random_half_block_mask(im_shape, l) for l in labels], 0)


class MixtureDataset(Dataset):
    """labeled dataset drawn from a mixture of components 
    each input is a (binary) greyscale image whose label 
    is the component index"""
    num_labels = 16
    im_shape = (1, 28, 28)

    def __init__(self, num_samples, train=True, p_c=None):
        super(MixtureDataset, self).__init__()
        if p_c is None:
            p_c = torch.ones(self.num_labels)
        assert (p_c >= 0.).all(), 'probs must be >=0'
        self.p_c = p_c / p_c.sum()
        self.logp_bIc = self.compute_logp_bIc()  # CxDxD matrix of ground truth bernoulli probs per class
        samples, labels = self.sample_data(num_samples)
        self.samples, self.labels = torch.Tensor(samples), torch.Tensor(labels).unsqueeze(1)
        self.train = train  # for bookeeping purposes
        self.num_samples = num_samples

    def compute_logp_bIc(self):  # ground truth bernoulli probs per class
        p = 0.95  # hard coded bernoulli prob for inside block
        return torch.stack(torch.Tensor(
            [Blocks.generate_component(
                np.log(p), np.log(1. - p), i
                ) for i in range(Blocks.num_labels)]
            ), 0).squeeze(1)

    @property
    def p_bIc(self):
        return self.logp_bIc.exp()

    @property
    def logp_c(self):
        return (self.p_c + 1e-9).log()

    def sample_data(self, num_samples):
        labels = np.random.choice(range(self.num_labels), ((num_samples, )))
        samples = np.stack(
                [np.random.binomial(1, self.p_bIc[c, :, :])[None, ...] for c in labels],
                0)
        return samples, labels

    def logp_xIc(self, x, c):
        """log p(x|c) where x are batched binary 
        observations and c is a class label"""
        return batch_bernoulli_logp(x, self.p_bIc[c, ...])

    def logp_cIxr(self, x, r):
        """log prob of classes within an observed region x_r,
        i.e., vector log p(c|x_r) where x_r are batched binary
        observations x within the pixel region r (one = "in region")
        and c all possible class labels"""
        sum_pixels = lambda i: i.sum(-1).sum(-1).sum(-1)  # sum X, Y and chan
        batched_logp_c = self.logp_c.unsqueeze(0).repeat(len(x), 1)
        logp_xrIc = torch.stack([
            sum_pixels(self.logp_xIc(x, i) * r)   # r[i,j]=1 means in-region 
            for i in range(Blocks.num_labels)], 1)
        logp_cxr = logp_xrIc + batched_logp_c  # p(x_r, c) at a fixed x_r
        return logp_cxr - logsumexp(logp_cxr, 1, keepdim=True)

    def logp_xmIxnm(self, x, m, apply_mask=False):
        """log p(x_m|x!m) where x_m are binary batched pixel observations 
        x masked by m (zero = "masked") and thus treated as latents, 
        and x_!m are the other pixels, which we observe.  Note that this
        function preserves the shape of x and does not apply the mask"""
        # log p(x_mIc): (all) unobserved pixel probs conditioned on labels
        logp_xmIc = torch.stack([ 
            #self.logp_xIc(x, i)  # mask not applied; do that later
            self.logp_bIc[None, i, ...].repeat(len(x), 1, 1, 1)
            for i in range(Blocks.num_labels)], 1)

        # log p(c|x_!m): label probs conditioned on observed pixels (not in mask)
        logp_cIxnm = self.logp_cIxr(x, m)  # m[i,j]=0 means mask, m[i,j]=condition region
        expand_pixels = lambda i: i[..., None, None, None]  # expand X, Y and chan
        # log p(x_m|x_!m) = log sum_c (log p(x_m|c) + log p(c|x_!m))
        logp_xmIxnm = logsumexp(
                expand_pixels(logp_cIxnm) + logp_xmIc,
                1)
        if apply_mask:
            return logp_xmIxnm * (1. - m)
        else:
            return logp_xmIxnm

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x, y = self.samples[index], self.labels[index]
        return x, y


class Blocks(MixtureDataset):
    block_width = 8  # edge size of block in pixels
    offset = 4  # row/column offset from edges

    @staticmethod
    def generate_component(component_val, background_val, label):
        """
        generates a block whose position is determined by label
        inside the block pixels take component_val; otherwise they take background_val
        we use this to produce B_ic; samples are drawn from p(x|c) = Prod_ij Bernoulli(B_ic)
        """
        im = background_val*np.ones(Blocks.im_shape)
        row_start, row_end, col_start, col_end = Blocks._label_to_patch_pixels(
                label, Blocks.block_width, Blocks.block_width//2
                )
        im[0, row_start:row_end, col_start:col_end] = component_val 
        return im

    @staticmethod
    def _label_to_patch_pixels(label, block_width, stride):
        n_blocks_per_edge = int(Blocks.num_labels ** 0.5)
        row_idx = label // n_blocks_per_edge
        col_idx = label % n_blocks_per_edge
        row_start = row_idx*stride+Blocks.offset
        row_end = row_start+block_width
        col_start = col_idx*stride+Blocks.offset
        col_end = col_start+block_width 
        return row_start, row_end, col_start, col_end


def mixture_of_shapes(num_samples, batch_size, seed=None, 
        use_cuda=torch.cuda.is_available(),
        p_c=None,
        shape_name='Blocks',
        **dataset_kwargs):
    """get a mixture-of-blocks toy dataset loader"""
    if seed is not None:
        torch.manual_seed(seed)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_class = eval(shape_name)
    dataset = dataset_class(num_samples, train=True, p_c=p_c, **dataset_kwargs)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=True, 
            **loader_kwargs)
    return train_loader, dataset


if __name__ == '__main__':
    import os
    from torchvision.utils import save_image

    shape_name = 'Blocks'
    dirname = './plots/{}'.format(shape_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    num_samples = 1024
    batch_size = 16
    seed = 0

    mob, d = mixture_of_shapes(num_samples, batch_size, seed)
    print(d.p_bIc.shape)
    save_image(d.p_bIc.unsqueeze(1), '{}/bIc.png'.format(dirname), nrow=int(Blocks.num_labels ** 0.5))
    for i, (x, y) in enumerate(mob):
        print(i, x.shape, torch.norm(x), y)
        save_image(x, '{}/foo{}.png'.format(dirname, i), nrow=int(batch_size ** 0.5))

    print('done')
