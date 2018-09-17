import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
from exp.loaddata_utils import ImageNetLoadClass
from exp.general_utils import Timer
import numpy as np
import os
import torchvision.models as models
from arch.sensitivity.BDNet import BDNet, L1BDNet, OppositeGernarativeL1BDNet, \
    OppositeGernarativeBDNet, GernarativeL1BDNet
import exp.utils_visualise as utils_visualise
from exp.utils_flipping import get_logodds_by_flipping
from collections import OrderedDict
from arch.Inpainting.Baseline import MeanInpainter
from arch.DeepLiftNet import DeepLiftNet
from exp.loaddata_utils import load_mnist_keras_test_imgs
from arch.sensitivity.BBMPNet import BBMP_SDR, BBMP_SSR
from arch.Inpainting.VAE_InpaintingMnist import VAE_InpaintingMnist


def sort_asd_2d(nd_arr):
    ''' (np arr) -> tuple
    Sort 2d array and return an order by ascending order
    :param nd_arr
    :return: a tuple of array of index. first element is index 1, 2nd element is index 2.
    '''
    return np.unravel_index(np.argsort(nd_arr.ravel()), nd_arr.shape)


def repeat_img_in_batch(the_img, the_label, batch_size):
    '''
    Return pytorch loader by repeating one img in batch size.
    :param the_img: pytorch img of size [1, 28, 28]
    :param the_label: integer of class
    :param batch_size: number to get samples in NN
    :return: pytorch loader
    '''

    # Repeat the image "batch_size" times
    repeated_imgs = the_img.unsqueeze(0).expand(batch_size, 1, 28, 28)
    repeated_labels = torch.LongTensor(1).fill_(int(the_label)).expand(batch_size)

    return [(repeated_imgs, repeated_labels)]


def main(args, importance_func, impant_model, interpret_net, the_criteria):
    if args.cuda:
        interpret_net.cuda()
        impant_model.cuda()

    # Load data
    x_test, y_test = load_mnist_keras_test_imgs()
    target_x = x_test[y_test == args.the_digit, ...]
    target_y = y_test[y_test == args.the_digit]

    for img_idx in xrange(args.image_offset, args.image_offset + args.num_imgs):
        the_x = target_x[img_idx]
        the_x = torch.FloatTensor(the_x)

        the_y = target_y[img_idx]

        img_loader = repeat_img_in_batch(the_x, the_y, batch_size=args.batch_size)

        with Timer('evaluating image'):
            tmp = importance_func(interpret_net, impant_model, img_loader)
            imp_vector = tmp[0]

        overlayed_img, clim = utils_visualise.get_overlayed_image(the_x[0, ...], imp_vector)
        file_name = '%s/%d_%d_overlayed.png' % (args.save_dir, args.the_digit, img_idx)

        # Save images
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        utils_visualise.save_figs([overlayed_img], filename=file_name, visualize=False, nrow=1,
                                  clim=clim)

        ### Also do quantitatively evaluation
        # Define the mask generator to quantitatively evaluate the flipping
        def mask_gen_to_flip_top_pixels(importance_2d, num_pixels):
            rank = sort_asd_2d(-importance_2d)

            _, H, W = the_x.size()
            mask = torch.ones(1, 1, H, W)

            for i in xrange(num_pixels[-1]):
                mask[0, 0, rank[0][i], rank[1][i]] = 0.
                if (i + 1) in num_pixels:
                    yield mask

        with Timer('doing quantitative evaluation'):
            # percentage = np.arange(0.01, 0.51, 0.01)
            # num_pixels = [int(p * 28 * 28) for p in percentage]
            num_pixels = range(1, 28 * 28 + 1)
            mask_generator = mask_gen_to_flip_top_pixels(imp_vector, num_pixels=num_pixels)

            orig_odds, all_log_odds = get_logodds_by_flipping(
                mask_generator, interpret_net, impant_model, img_loader,
                batch_size=args.batch_size, num_samples=1, window=1,
                the_log_odds_criteria=the_criteria,
                cuda_enabled=args.cuda)

            print [orig_odds - all_log_odds[i] for i in xrange(len(num_pixels))]

            all_log_odds_dict = OrderedDict()
            for i, p in enumerate(num_pixels):
                all_log_odds_dict[p] = all_log_odds[i]

            # Random sample from vbd to get log odds
            sample_log_odds_diff = []
            if args.importance_method.startswith('vbd'):
                masks = tmp[1]
                _, random_odds_diff = get_logodds_by_flipping(
                    masks, interpret_net, impant_model, img_loader,
                    batch_size=args.batch_size, num_samples=1, window=1,
                    the_log_odds_criteria=the_criteria,
                    cuda_enabled=args.cuda)

                for mask, odd_diff in zip(masks, random_odds_diff):
                    num_removed = (1 - mask).sum()
                    sample_log_odds_diff.append((num_removed, odd_diff))

            torch.save((orig_odds, all_log_odds_dict, the_x, imp_vector, sample_log_odds_diff),
                       '%s/%d_%d_records.th'
                       % (args.save_dir, args.the_digit, img_idx))


def get_impant_model(args):
    if args.gen_model == 'MeanInpainiter':
        impant_model = MeanInpainter()
    elif args.gen_model == 'VAEInpainter':
        impant_model = VAE_InpaintingMnist()
        loaded = torch.load('checkpts/1017-vae-mnist_lr_0.001_epochs_9',
                       map_location=lambda storage, loc: storage)
        impant_model.load_state_dict(loaded['state_dict'])
    impant_model.eval()
    return impant_model

def load_classifier(cuda_enabled=False):
    model = DeepLiftNet()
    model.load_state_dict(torch.load('model/mnist_cnn_allconv_pytorch'))
    model.float()
    model.eval()

    if cuda_enabled:
        model.cuda()
    return model

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Imagenet Example')
    parser.add_argument('--importance-method', type=str, default='p_b',
                        help='choose from ["vbdl1", "vbd", "p_b", "vbd_l1_opposite"]')
    parser.add_argument('--mode', type=str, default='flip_one',
                        help='choose from ["flip_one", "flip_two"]')
    parser.add_argument('--the-digit', type=int, default=8)
    parser.add_argument('--to-digit', type=int, default=-1)
    parser.add_argument('--gen_model', type=str, default='MeanInpainiter',
                        help='Choose from [MeanInpainiter, VAEInpainter]')
    # parser.add_argument('--classifier', type=str, default='alexnet',
    #                     help='Choose from [alexnet, resnet18, vgg19_bn]')
    parser.add_argument('--window', type=int, default=1)
    # parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--reg-coef', type=float, default=0.01,
                        help='regularization coefficient')
    parser.add_argument('--prior', type=float, default=0.999,
                        help='prior probability for reg loss')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--ard_init', type=float, default=0.,
                        help='ARD initialization')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--epoch-print', type=int, default=10,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--save-dir', type=str, default='./result/1005-p_b/',
                        help='Save directory')
    # parser.add_argument('--save-tag', type=str, default='',
    #                     help='Unique tag for output images')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Open verbose or not')
    parser.add_argument('--num-imgs', type=int, default=20,
                        help='number of images to produce')
    # parser.add_argument('--gpu-ids', nargs='+', type=int, default=[],
    #                     help='number of gpus to produce')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--image-offset', type=int, default=0, help='offset for index of image')
    parser.add_argument('--eval-samples', type=int, default=200, help='offset for index of image')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(args.gpu_ids[0])

    print 'args:', args
    print '==================== Start ====================='
    print ''
    return args

if __name__ == '__main__':
    args = parse_args()
    import torch.nn.functional as F

    def log_sum_exp(x, dim):
        x_max = x.max()
        return torch.log(torch.sum(torch.exp(x - x_max), dim=dim)) + x_max

    def flip_one_criteria(outputs, targets):
        log_prob = F.log_softmax(outputs)

        other_log_prob = torch.cat([log_prob[:, :args.the_digit],
                                    log_prob[:, (args.the_digit + 1):]], dim=1)
        tmp = log_sum_exp(other_log_prob, dim=1)
        return -(log_prob[:, args.the_digit] - tmp).mean()

    def flip_two_criteria(outputs, targets):
        # return (outputs[:, args.the_digit] - outputs[:, args.to_digit]).mean()
        return (outputs[:, args.to_digit] - outputs[:, args.the_digit]).mean()

    the_criteria = flip_one_criteria
    if args.mode == 'flip_two':
        the_criteria = flip_two_criteria

    def vbd(interpret_net, impant_model, img_loader):
        bdnet = BDNet
        return _vbd_common(bdnet, interpret_net, impant_model, img_loader)

    def vbd_opposite(interpret_net, impant_model, img_loader):
        bdnet = OppositeGernarativeBDNet
        color_vector, masks = _vbd_common(bdnet, interpret_net, impant_model, img_loader)
        return -color_vector, masks

    def vbd_l1_opposite(interpret_net, impant_model, img_loader):
        bdnet = OppositeGernarativeL1BDNet
        color_vector, masks = _vbd_common(bdnet, interpret_net, impant_model, img_loader)
        return -color_vector, masks

    def vbdl1(interpret_net, impant_model, img_loader):
        bdnet = GernarativeL1BDNet
        return _vbd_common(bdnet, interpret_net, impant_model, img_loader)

    def _vbd_common(bdnet, interpret_net, impant_model, img_loader):
        net = bdnet(dropout_param_size=(1, 28, 28),
                    trained_classifier=interpret_net,
                    generative_model=impant_model,
                    flip_val=0.,
                    ard_init=0., lr=args.lr, reg_coef=args.reg_coef, rw_max=50,
                    loss_criteria=the_criteria,
                    cuda_enabled=args.cuda, verbose=args.verbose, prior_p=args.prior)
        # Train it
        net.fit(img_loader, epochs=args.epochs, epoch_print=args.epoch_print)

        # Visualize by log-odds of dropout probability
        color_vector = -net.logit_p.data[0, ...]

        print 'range: (%.3f, %.3f), shape: %s' % (color_vector.min(), color_vector.max(),
                                                  str(color_vector.size()))
        color_vector = color_vector.cpu().numpy()

        sample_berns = net.sampled_from_logit_p(args.eval_samples)
        return color_vector, sample_berns.data.round()

    def bbmp_ssr(interpret_net, impant_model, img_loader):
        bbmpnet = BBMP_SSR
        return _bbmp_common(bbmpnet, interpret_net, impant_model, img_loader)

    def bbmp_sdr(interpret_net, impant_model, img_loader):
        bbmpnet = BBMP_SDR
        mask_value, = _bbmp_common(bbmpnet, interpret_net, impant_model, img_loader)
        return -mask_value + 1,

    def _bbmp_common(bbmpnet, interpret_net, impant_model, img_loader):
        assert isinstance(impant_model, MeanInpainter), 'BBMP only accept MeanImpaint for now.'

        imgs, targets = iter(img_loader).next()
        new_loader = [(imgs[0:1, ...], targets[0:1])]
        net = bbmpnet(
            mask_value_size=(1, 28, 28),
            trained_classifier=interpret_net,
            loss_criteria=the_criteria,
            ard_init=1., lr=args.lr, reg_coef=args.reg_coef,
            rw_max=1, cuda_enabled=args.cuda,
            flip_value=0.,
            verbose=1)
        net.fit(new_loader, epochs=args.epochs, epoch_print=args.epoch_print)

        # Visualize by mask
        net.mask.data.clamp_(0, 1)
        color_vector = net.mask.data[0, ...]
        print 'range: (%.3f, %.3f), shape: %s' % (color_vector.min(), color_vector.max(),
                                                  str(color_vector.size()))
        color_vector = color_vector.cpu().numpy()
        return color_vector,

    def p_b(interpret_net, impant_model, img_loader):
        # Prevent myself too stupid...
        interpret_net.eval()

        # Take out origisnal imgs and targets
        imgs, targets = iter(img_loader).next()

        # All the inputs dimension
        N, channel, dim1, dim2 = imgs.size()

        width = dim2 - args.window + 1
        height = dim1 - args.window + 1

        # Mask generator
        def mask_generator():
            for i in xrange(height):
                for j in xrange(width):
                    mask = torch.ones(1, 1, dim1, dim2)
                    mask[:, :, i:(i + args.window), j:(j + args.window)] = 0.
                    yield mask

        orig_odds, all_log_odds = get_logodds_by_flipping(
            mask_generator(), interpret_net, impant_model, img_loader,
            the_log_odds_criteria=the_criteria,
            batch_size=args.batch_size, num_samples=1, window=args.window,
            cuda_enabled=args.cuda)

        perturb_rank = np.zeros((dim1, dim2))
        count = np.zeros((dim1, dim2))
        for i in xrange(height):
            for j in xrange(width):
                perturb_rank[i:(i + args.window), j:(j + args.window)] \
                    += (all_log_odds[i * width + j] - orig_odds)
                count[i:(i + args.window), j:(j + args.window)] += 1
        return perturb_rank / count,

    # Load which method to interpret importance
    importance_func = eval(args.importance_method)

    # Load which impanting model you want to use
    impant_model = get_impant_model(args)

    # Load the classifier
    interpret_net = load_classifier()
    # func = getattr(models, args.classifier)
    # interpret_net = func(pretrained=True)
    # interpret_net.eval()

    main(args, importance_func, impant_model, interpret_net, the_criteria)
