import numpy as np
import os
import torch
from torchvision.utils import save_image
import unittest

from datasets import Blocks, mixture_of_shapes, random_mask_batch, random_half_block_mask_batch


class TestBlocks(unittest.TestCase):
    batch_size = 16
    uniform = True
    dirname = './plots/tests'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if uniform:  # uniform mixture
        mob, d = mixture_of_shapes(batch_size, batch_size)
    else:  # random mixture coeffs
        mob, d = mixture_of_shapes(batch_size, batch_size, 
                p_c=torch.randn(Blocks.num_labels).exp())

    def test_valid_probs(self):
        for x, y in self.mob:
            im_shape = x.shape[-2:]
            mb = random_mask_batch(self.batch_size, im_shape, 0.1)
            logp_cIxr = self.d.logp_cIxr(x, mb)
            p_cIxm = logp_cIxr.exp()
            self.assertAlmostEqual(
                    torch.norm(p_cIxm.sum(1) - torch.ones(self.batch_size)),
                    0., places=4)

    def test_conditional_probs(self):
        for p_c in [None, torch.randn(Blocks.num_labels).exp()]:
            mob, data = mixture_of_shapes(
                    self.batch_size, self.batch_size, p_c=p_c
                    )
            for x, y in mob:
                im_shape = x.shape[-2:]
                observed_region = torch.ones_like(x)  # observe all
                latent_region = torch.zeros_like(x)  # don't observe any
                logp_cIxr_observed = data.logp_cIxr(x, observed_region)
                logp_cIxr_latent = data.logp_cIxr(x, latent_region)
                cIxm_observed = torch.max(logp_cIxr_observed , 1)[1].float()
                cIxm_latent = torch.max(logp_cIxr_latent , 1)[1].float()
                self.assertAlmostEqual(torch.norm(y.squeeze() - cIxm_observed), 0.0)
                self.assertAlmostEqual(
                        torch.norm(logp_cIxr_latent.exp() - \
                                data.p_c.unsqueeze(0).repeat(self.batch_size, 1))
                            , 0.0, places=4)
                break   

    def test_qualitative(self):  # this is a qualitative test; look at the plots
        """evaluate p(x_m|x_!m) by looking at plots"""
        def half_mask(batch_size, im_shape, top=True):
            """mask top or botttom half"""
            mask = torch.ones(batch_size, 1, *im_shape)
            if top:
                mask[:, :, :im_shape[0]//2, :] = 0.
            else:
                mask[:, :, im_shape[0]//2:, :] = 0.
            return mask

        for x, y in self.mob:
            _, _, *im_shape = x.shape
            for name, m in zip(
                    [0, 1, 'halftop', 'halfbottom', 'random', 'gt'],
                    [torch.zeros_like(x), 
                        torch.ones_like(x),
                        half_mask(self.batch_size, im_shape, True),
                        half_mask(self.batch_size, im_shape, False),
                        random_mask_batch(self.batch_size, im_shape),
                        random_half_block_mask_batch(self.batch_size, im_shape, y.squeeze().long())]):
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
