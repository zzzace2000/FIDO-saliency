import numpy as np
import torch
from torch.utils.data import Dataset



class MixtureOfBlocks(Dataset):
    im_shape = (1, 28, 28)
    def __init__(self, num_samples, train=True):
        super(MixtureOfBlocks, self).__init__()
        labels = np.random.choice(range(9), ((num_samples, )))
        beta_samples = np.random.beta(5., 1., size=((num_samples, )))  # should probably seed the RNG here...
        samples = np.stack([self.generate_image(b, label) for b, label in zip(beta_samples, labels)], 0)
        self.num_samples = num_samples
        self.labels, self.samples = torch.Tensor(labels).unsqueeze(1), torch.Tensor(samples)
        self.train = train  # for bookeeping purposes

    def generate_image(self, beta_sample, label):
        #n_pixels_per_edge = self.im_shape[1]
        n_blocks_per_edge = 3
        width = 14  # edge size of block in pixels
        stride = width // 2
        row_idx = label // n_blocks_per_edge
        col_idx = label % n_blocks_per_edge
        #im = np.zeros(self.im_shape)  # noiseless
        im = 0.2*np.random.randn(*self.im_shape)  # background noise
        im[0, row_idx*stride:row_idx*stride+width, col_idx*stride:col_idx*stride+width] += beta_sample 
        return im

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x, y = self.samples[index], self.labels[index]
        return x, y



def MaskedMixtureOfBlocks(num_samples):
    """generate training data to train the in-filling network"""
    def __init__(self, num_samples, train=True):
        1/0


def mixture_of_blocks(num_samples, batch_size, seed=None, 
        use_cuda=torch.cuda.is_available()):
    """get a mixture-of-blocks toy dataset loader"""
    if seed is not None:
        torch.manual_seed(seed)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_kwargs = dict()
    train_loader = torch.utils.data.DataLoader(
            MixtureOfBlocks(num_samples, train=True),
            batch_size=batch_size, 
            shuffle=True, 
            **loader_kwargs)
    return train_loader


if __name__ == '__main__':
    import os
    from torchvision.utils import save_image

    dirname = './plots/MixtureOfBlocks'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    num_samples = 1024
    batch_size = 16
    seed = 0

    mob = mixture_of_blocks(num_samples, batch_size, seed)
    for i, (x, y) in enumerate(mob):
        print(i, x.shape, torch.norm(x), y)
        save_image(x, '{}/foo{}.png'.format(dirname, i), nrow=int(batch_size ** 0.5))

    print('done')
    1/0
