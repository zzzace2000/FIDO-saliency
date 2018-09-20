import numpy as np
import torch
from torch.utils.data import Dataset



class MixtureOfBlocks(Dataset):
    im_shape = (1, 28, 28)
    num_labels = 16
    block_width = 8  # edge size of block in pixels
    noise_level = .05
    offset = 4  # row/column offset from edges

    def __init__(self, num_samples, train=True):
        super(MixtureOfBlocks, self).__init__()
        labels = np.random.choice(range(self.num_labels), ((num_samples, )))
        beta_samples = np.random.beta(5., 1., size=((num_samples, )))  # should probably seed the RNG here...
        samples = np.stack([self.generate_image(b, label) for b, label in zip(beta_samples, labels)], 0)
        self.num_samples = num_samples
        self.labels, self.samples = torch.Tensor(labels).unsqueeze(1), torch.Tensor(samples)
        self.train = train  # for bookeeping purposes

    @staticmethod
    def generate_image(beta_sample, label):
        #im = np.zeros(self.im_shape)  # noiseless
        im = MixtureOfBlocks.noise_level*np.random.randn(*MixtureOfBlocks.im_shape)  # background noise
        row_start, row_end, col_start, col_end = MixtureOfBlocks._label_to_patch_pixels(
                label, MixtureOfBlocks.block_width, MixtureOfBlocks.block_width//2
                )
        im[0, row_start:row_end, col_start:col_end] += beta_sample 
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
