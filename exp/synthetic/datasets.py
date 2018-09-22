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

class MixtureOfBlocks(Dataset):
    im_shape = (1, 28, 28)
    num_labels = 16
    block_width = 8  # edge size of block in pixels
    noise_level = .05
    offset = 4  # row/column offset from edges

    def __init__(self, num_samples, train=True, p_c=None):
        super(MixtureOfBlocks, self).__init__()
        if p_c is None:
            p_c = torch.ones(self.num_labels)
        assert (p_c >= 0.).all(), 'probs must be >=0'
        self.p_c = p_c / p_c.sum()

        self.logp_bIc = self.compute_logp_bIc()  # CxDxD matrix of ground truth bernoulli probs per class
        samples, labels = self.sample_data(num_samples)
        self.samples, self.labels = torch.Tensor(samples), torch.Tensor(labels).unsqueeze(1)
        self.train = train  # for bookeeping purposes
        self.num_samples = num_samples

    @staticmethod
    def generate_component(component_val, background_val, label):
        """
        generates a block whose position is determined by label
        inside the block pixels take component_val; otherwise they take background_val
        we use this to produce B_ic; samples are drawn from p(x|c) = Prod_ij Bernoulli(B_ic)
        """
        im = background_val*np.ones(MixtureOfBlocks.im_shape)
        row_start, row_end, col_start, col_end = MixtureOfBlocks._label_to_patch_pixels(
                label, MixtureOfBlocks.block_width, MixtureOfBlocks.block_width//2
                )
        im[0, row_start:row_end, col_start:col_end] = component_val 
        return im

    @staticmethod
    def _label_to_patch_pixels(label, block_width, stride):
        n_blocks_per_edge = int(MixtureOfBlocks.num_labels ** 0.5)
        row_idx = label // n_blocks_per_edge
        col_idx = label % n_blocks_per_edge
        row_start = row_idx*stride+MixtureOfBlocks.offset
        row_end = row_start+block_width
        col_start = col_idx*stride+MixtureOfBlocks.offset
        col_end = col_start+block_width 
        return row_start, row_end, col_start, col_end

    # TODO write a super class that uses compute_bIc and generate_samples and requires implementing generate_component
    def compute_logp_bIc(self):  # ground truth bernoulli probs per class
        p = 0.95  # hard coded bernoulli prob for inside block
        return torch.stack(torch.Tensor(
            [MixtureOfBlocks.generate_component(
                np.log(p), np.log(1. - p), i
                ) for i in range(MixtureOfBlocks.num_labels)]
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

    def logp_cIxm(self, x, m):
        """log prob of classes within the masked region,
        i.e., vector log p(c|x_m) where x_m are batched 
        binary observations x masked by m and c all 
        possible class labels"""
        sum_pixels = lambda i: i.sum(-1).sum(-1).sum(-1)  # sum X, Y and chan
        u = self.logp_xIc(x, 0).masked_select(m.byte()) 
        batched_logp_c = self.logp_c.unsqueeze(0).repeat(len(x), 1)
        logp_xmIc = torch.stack([
            sum_pixels(self.logp_xIc(x, i) * m) 
            for i in range(MixtureOfBlocks.num_labels)], 1)
        logp_cxm = logp_xmIc + batched_logp_c  # p(xm, c) at a fixed xm
        return logp_cxm - self.logsumexp(logp_cxm, 1, keepdim=True)

    def logp_xmIxnm(self, x, m, apply_mask=False):
        """return matrix of log p(xm|x!m) where xm are binary batched
        pixel observations x masked by m, and x!m are all
        the other pixels. Note that this function preserves the 
        shape of x and does not apply the mask"""
        # log p(x_mIc): (all) unobserved pixel probs conditioned on labels
        logp_xmIc = torch.stack([ 
            #self.logp_xIc(x, i)  # mask not applied; do that later
            self.logp_bIc[None, i, ...].repeat(len(x), 1, 1, 1)
            for i in range(MixtureOfBlocks.num_labels)], 1)

        # log p(c|x_!m): label probs conditioned on pixels not in mask
        logp_cIxnm = self.logp_cIxm(x, 1.-m)  
        expand_pixels = lambda i: i[..., None, None, None]  # expand X, Y and chan
        # log p(x_m|x_!m) = log sum_c (log p(x_m|c) + log p(c|x_!m))
        logp_xmIxnm = self.logsumexp(
                expand_pixels(logp_cIxnm) + logp_xmIc,
                1)
        if apply_mask:
            return logp_xmIxnm * (1. - m)
        else:
            return logp_xmIxnm

    def logsumexp(self, inputs, dim=None, keepdim=False):  # from https://github.com/pytorch/pytorch/issues/2591
        import torch.nn.functional as F
        from torch.autograd import Variable
        diff = inputs - F.log_softmax(
                Variable(inputs, requires_grad=False), dim=dim
                ).data
        return diff.mean(dim, keepdim=keepdim)

    def exact_marginalization(self, x, mask):
        """return p(x_masked|x_not_masked)"""

        return self.logsumexp(
                1/0  # logp(x_masked|c) + logp(c|x_not_masked)
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x, y = self.samples[index], self.labels[index]
        return x, y


def mixture_of_blocks(num_samples, batch_size, seed=None, 
        use_cuda=torch.cuda.is_available(),
        p_c=None):
    """get a mixture-of-blocks toy dataset loader"""
    if seed is not None:
        torch.manual_seed(seed)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = MixtureOfBlocks(num_samples, train=True, p_c=p_c)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=True, 
            **loader_kwargs)
    return train_loader, dataset


if __name__ == '__main__':
    import os
    from torchvision.utils import save_image

    dirname = './plots/MixtureOfBlocks'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    num_samples = 1024
    batch_size = 16
    seed = 0

    mob, d = mixture_of_blocks(num_samples, batch_size, seed)
    print(d.p_bIc.shape)
    save_image(d.p_bIc.unsqueeze(1), '{}/bIc.png'.format(dirname), nrow=int(MixtureOfBlocks.num_labels ** 0.5))
    for i, (x, y) in enumerate(mob):
        print(i, x.shape, torch.norm(x), y)
        save_image(x, '{}/foo{}.png'.format(dirname, i), nrow=int(batch_size ** 0.5))

    print('done')
