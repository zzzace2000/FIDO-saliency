import numpy as np
import os
import torch
from torchvision.utils import save_image
import unittest

from datasets import MixtureOfBlocks, mixture_of_blocks
import OracleInpainter 


class TestMixtureOfBlocks(unittest.TestCase):
    batch_size = 16
    uniform = True
    dirname = './plots/tests'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if uniform:  # uniform mixture
        mob, d = mixture_of_blocks(batch_size, batch_size)
    else:  # random mixture coeffs
        mob, d = mixture_of_blocks(batch_size, batch_size, 
                p_c=torch.randn(MixtureOfBlocks.num_labels).exp())

    def test_valid_probs(self):
        for x, y in self.mob:
            im_shape = x.shape[-2:]
            mb = OracleInpainter.random_mask_batch(self.batch_size, im_shape, 0.1)
            logp_cIxm = self.d.logp_cIxm(x, mb)
            p_cIxm = logp_cIxm.exp()
            self.assertAlmostEqual(
                    torch.norm(p_cIxm.sum(1) - torch.ones(self.batch_size)),
                    0., places=4)

    def test_conditional_probs(self):
        for p_c in [None, torch.randn(MixtureOfBlocks.num_labels).exp()]:
            mob, data = mixture_of_blocks(
                    self.batch_size, self.batch_size, p_c=p_c
                    )
            for x, y in mob:
                im_shape = x.shape[-2:]
                ones_mask = torch.ones_like(x)
                zeros_mask = torch.zeros_like(x)
                logp_cIxm_0 = data.logp_cIxm(x, zeros_mask)
                logp_cIxm_1 = data.logp_cIxm(x, ones_mask)
                cIxm_0 = torch.max(logp_cIxm_0 , 1)[1].float()
                cIxm_1 = torch.max(logp_cIxm_1 , 1)[1].float()
                self.assertAlmostEqual(torch.norm(y.squeeze() - cIxm_1), 0.0)
                self.assertAlmostEqual(
                        torch.norm(logp_cIxm_0.exp() - \
                                data.p_c.unsqueeze(0).repeat(self.batch_size, 1))
                            , 0.0, places=4)
                break   

    def test_qualitative(self):  # this is a qualitative test; look at the plots
        def half_mask(batch_size, im_shape, top=True):
            """mask top or botttom half"""
            mask = torch.zeros(batch_size, 1, *im_shape)
            if top:
                mask[:, :, :im_shape[0]//2, :] = 1.
            else:
                mask[:, :, im_shape[0]//2:, :] = 1.
            return mask

        for x, y in self.mob:
            _, _, *im_shape = x.shape
            for name, m in zip(
                    [0, 1, 'halftop', 'halfbottom', 'random', 'gt'],
                    [torch.zeros_like(x), 
                        torch.ones_like(x),
                        half_mask(self.batch_size, im_shape, True),
                        half_mask(self.batch_size, im_shape, False),
                        OracleInpainter.random_mask_batch(self.batch_size, im_shape),
                        OracleInpainter.random_half_block_mask_batch(self.batch_size, im_shape, y.squeeze().long())]):
                logp_xmInm = self.d.logp_xmIxnm(x, m, False)
                p_xmInm = logp_xmInm.exp()
                save_image(m, 
                        '{}/m-{}.png'.format(self.dirname, name),
                        padding=5,
                        nrow=int(self.batch_size ** 0.5))
                save_image(x , 
                        '{}/x-{}.png'.format(self.dirname, name),
                        padding=5,
                        nrow=int(self.batch_size ** 0.5))
                save_image(p_xmInm , 
                        '{}/xmIxnm-{}.png'.format(self.dirname, name),
                        padding=5,
                        nrow=int(self.batch_size ** 0.5))
                self.assertTrue(True)
            break

if __name__ == '__main__':
    unittest.main()
