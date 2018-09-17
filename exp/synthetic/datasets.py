import numpy as np
import torch
from torch.utils.data import Dataset



class MixtureOfBlocks(Dataset):
    im_shape = (1, 28, 28)
    def __init__(self, num_samples):
        super(MixtureOfBlocks, self).__init__()
        labels = np.random.choice(range(9), ((100, )))
        beta_samples = np.random.beta(5., 1., size=((num_samples, )))  # should probably seed the RNG here...
        samples = np.stack([self.generate_image(b, label) for b, label in zip(beta_samples, labels)], 0)
        self.num_samples = num_samples
        self.labels, self.samples = torch.Tensor(labels).unsqueeze(1), torch.Tensor(samples)

    def generate_image(self, beta_sample, label):
        #n_pixels_per_edge = self.im_shape[1]
        n_blocks_per_edge = 3
        width = 14  # edge size of block in pixels
        stride = width // 2
        row_idx = label // n_blocks_per_edge
        col_idx = label % n_blocks_per_edge
        im = np.zeros(self.im_shape)
        im[0, row_idx*stride:row_idx*stride+width, col_idx*stride:col_idx*stride+width] = beta_sample 
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



if __name__ == '__main__':
    import os
    from torchvision.utils import save_image

    dirname = './plots/MixtureOfBlocks'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    d = MixtureOfBlocks(100)
    dl = iter(d)
    for i in range(20):
        x, y = next(dl)
        print(i, x.shape, torch.norm(x), y)
        save_image(x, '{}/foo{}-y={}.png'.format(dirname, i, int(y)))

    print('done')
    1/0
