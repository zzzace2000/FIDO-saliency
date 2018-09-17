import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(os.getcwd())))

from arch.sensitivity.GDNet import GDNet
from arch.sensitivity.BDNet import BDNet, IsingBDNet, IsingSoftPenaltyBDNet, \
    ImageWindowBDNet, OppositeGernarativeL1BDNet
from exp.loaddata_utils import load_mnist_keras_test_imgs
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mnist_compare_utils
from torch.utils.data import DataLoader, TensorDataset
from arch.DeepLiftNet import DeepLiftNet
from torchvision.utils import make_grid
import argparse
import pkgutil
import exp.utils_visualise as utils_visualize
from scipy.stats import rankdata

torch.manual_seed(1)

def repeat_img_in_batch(the_img, the_label, batch_size):
    '''
    Return pytorch loader by repeating one img in batch size.
    :param the_img: numpy img of size [1, 28, 28]
    :param the_label: integer of class
    :param batch_size: number to get samples in NN
    :return: pytorch loader
    '''
    the_img = torch.FloatTensor(the_img)

    # Repeat the image "batch_size" times
    repeated_imgs = the_img.unsqueeze(0).expand(batch_size, 1, 28, 28)
    repeated_labels = torch.LongTensor(1).fill_(int(the_label)).expand(batch_size)

    return [(repeated_imgs, repeated_labels)]
    # train_loader = torch.utils.data.DataLoader(
    #     TensorDataset(repeated_imgs, repeated_labels),
    #     batch_size=batch_size, shuffle=False)
    # return train_loader

def load_classifier(cuda_enabled=False):
    model = DeepLiftNet()
    model.load_state_dict(torch.load('model/mnist_cnn_allconv_pytorch'))
    model.float()
    model.eval()

    if cuda_enabled:
        model.cuda()
    return model

def visualize_imgs(imgs, nrow):
    imgs = torch.from_numpy(imgs)
    grid = make_grid(imgs, nrow=nrow)

    def pytorch_to_np(pytorch_image):
        return pytorch_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    ndarr = pytorch_to_np(grid)
    im = plt.imshow(ndarr, interpolation='nearest')
    plt.show()

def train_log_odds_diff(rank_func, classifier, dataset, from_digit, to_digit,
                        visualize=False, top_number=-1, batch_size=128, cuda_enabled=False):
    x, y = dataset

    # Get those images that correpsond to the from_digit class
    target_x = x[y == from_digit, ...]
    target_y = y[y == from_digit][:top_number]

    if top_number > 0:
        target_x = target_x[:top_number, ...]
        target_y = target_y[:top_number]

    # Set up pytorch data and model
    diff = []
    overlayed_imgs = []
    ranks = []
    for the_x, the_y in zip(target_x, target_y):
        loader = repeat_img_in_batch(the_x, the_y, batch_size=batch_size)

        rank = rank_func(classifier, loader)

        # Rank log odds diff
        the_img = torch.from_numpy(the_x)
        log_odds, order, flipped_img = mnist_compare_utils.cal_logodds_diff_btw_two_class(
            classifier, the_img, from_digit=from_digit, to_digit=to_digit, importance_2d=rank,
            flip_percentage=0.20, flip_val=0., cuda_enabled=cuda_enabled)

        diff.append(log_odds[-1] - log_odds[0])
        # ranks.append(rank.numpy())

        if visualize:
            # plt.imshow(flipped_img, interpolation='nearest')
            # plt.colorbar()
            # plt.show()

            # img = utils_visualize.overlay(the_x[0, ...], rank.numpy())
            img, clim = utils_visualize.overlay(the_x[0, ...], flipped_img)
            overlayed_imgs.append(torch.from_numpy(img))

    return diff, overlayed_imgs, ranks


def main(rank_func, from_digit=8, to_digit=3, top_n=2, cuda_enabled=False, visualize=False):
    classifer = load_classifier(cuda_enabled=cuda_enabled)
    X_test, y_test = load_mnist_keras_test_imgs()

    return train_log_odds_diff(rank_func, classifer, (X_test, y_test), from_digit, to_digit,
                               top_number=top_n, batch_size=64,
                               cuda_enabled=cuda_enabled, visualize=visualize)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Deeplift comparison')
    parser.add_argument('--model', type=str, default='vbd_opposite',
                        choices=['vbd', 'vgd', 'p_b', 'p_g', 'ising_vbd', 'ising_soft_vbd', 'vbd_window'],
                        help='choose from ["vbd_rank_func", "bern", "add_gauss"]')
    parser.add_argument('--l1_reg_coef', type=float, default=0.1, help='Only use in IsingBDNet')
    parser.add_argument('--l2_reg_coef', type=float, default=0., help='Only use in IsingBDNet')
    parser.add_argument('--window', type=int, default=2, help='Perturbation size. Used in p_b or vbd_window')
    parser.add_argument('--from-digit', type=int, default=8,
                        help='mask from some digits')
    parser.add_argument('--to-digit', type=int, default=3,
                        help='masked to some digits')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--top_n', type=int, default=1, help='-1 means whole test sets')
    parser.add_argument('--no-cuda', action='store_false', default=True,
                        help='disables CUDA training')
    parser.add_argument('--visualize', action='store_false', default=True)
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print 'cuda:', args.cuda

    # CS server settings
    if args.cuda and pkgutil.find_loader('gpu_lock') is not None:
        import gpu_lock
        board = gpu_lock.obtain_lock_id()
        torch.cuda.set_device(board)
        print 'start using gpu device: %d' % board

    return args

def run(args):
    def log_odds_criteria(outputs, targets):
        # It needs to return the things needs to be minimized.
        return (outputs[:, args.from_digit] - outputs[:, args.to_digit]).mean()

    def vbd_opposite(classifier, loader):
        input_size = (1, 28, 28)
        vbdnet = OppositeGernarativeL1BDNet(input_size, trained_classifier=classifier, ard_init=0.,
                                            lr=0.01, reg_coef=0., rw_max=30, cuda_enabled=args.cuda,
                                            loss_criteria=log_odds_criteria,
                                            verbose=args.verbose)
        vbdnet.fit(loader, epochs=200, epoch_print=10)

        # The smaller the dropout rate is, it's less important.
        rank = vbdnet.logit_p.data[0, ...]
        return rank

    def vbd(classifier, loader):
        input_size = (1, 28, 28)
        vbdnet = BDNet(input_size, trained_classifier=classifier, ard_init=0.,
                       lr=0.01, reg_coef=1E-7, rw_max=30, cuda_enabled=args.cuda,
                       estop_num=10, clip_max=100,
                       flip_val=0., loss_criteria=log_odds_criteria,
                       flip_train=False, verbose=args.verbose,
                       )
        vbdnet.fit(loader, epochs=1000, epoch_print=10)

        # The smaller the dropout rate is, it's less important.
        rank = vbdnet.logit_p.data[0, ...]
        return rank

    def vbd_window(classifier, loader):
        input_size = (1, 28, 28)
        rank = ImageWindowBDNet.fit_multiple_windows(
            loader, epochs=1000, epoch_print=10, dropout_param_size=input_size, trained_classifier=classifier,
            loss_criteria=log_odds_criteria, ard_init=0., lr=0.01, reg_coef=0., rw_max=30,
            cuda_enabled=args.cuda, verbose=args.verbose, estop_num=None, clip_max=100, flip_val=0.,
            flip_train=False, window_size=args.window)

        return rank

    def _ising_common(classifier, loader, model):
        input_size = (1, 28, 28)
        vbdnet = model(input_size, trained_classifier=classifier, ard_init=0.,
                        lr=0.01, reg_coef=0., rw_max=30, cuda_enabled=args.cuda,
                        estop_num=10, clip_max=100,
                        flip_val=0., loss_criteria=log_odds_criteria,
                        flip_train=False, verbose=args.verbose, l1_reg_coef=args.l1_reg_coef,
                        l2_reg_coef=args.l2_reg_coef
                        )
        vbdnet.fit(loader, epochs=1000, epoch_print=10)

        # The smaller the dropout rate is, it's
        rank = vbdnet.logit_p.data[0, ...]
        return rank

    def ising_vbd(classifier, loader):
        return _ising_common(classifier, loader, IsingBDNet)

    def ising_soft_vbd(classifier, loader):
        return _ising_common(classifier, loader, IsingSoftPenaltyBDNet)

    def vgd(classifier, loader, vd_model=GDNet):
        input_size = (1, 28, 28)
        gauss_net = vd_model(input_size, trained_classifier=classifier, ard_init=-6.,
                             lr=0.03, reg_coef=0., rw_max=30, cuda_enabled=args.cuda,
                             estop_num=1., clip_max=100.,
                             loss_criteria=log_odds_criteria,
                             verbose=args.verbose
                             )
        gauss_net.fit(loader, epochs=500, epoch_print=10)
        return gauss_net.log_alpha.data[0, ...]

    def p_b(classifier, loader):
        def perturb_by_binary(feature_idx, old_val):
            return torch.zeros(old_val.size())

        classifier.set_criteria(log_odds_criteria)

        return -mnist_compare_utils.perturb_2d(classifier, loader, perturb_by_binary, window=args.window,
                                               cuda_enabled=args.cuda)

    def p_g(classifier, loader):
        def perturb_by_multiply_gauss(feature_idx, old_val, var=0.5):
            return old_val + old_val * var * torch.normal(torch.zeros(*old_val.size()), 1)

        classifier.set_criteria(log_odds_criteria)

        return -mnist_compare_utils.perturb_2d(classifier, loader, perturb_by_multiply_gauss, num_samples=10,
                                               window=args.window, cuda_enabled=args.cuda)

    rank_func = eval(args.model)

    diff, overlayed_imgs, _ = main(rank_func, args.from_digit, args.to_digit, args.top_n,
                                       cuda_enabled=args.cuda, visualize=args.visualize)
    print diff

    if args.visualize:
        utils_visualize.save_figs(overlayed_imgs, filename='', visualize=True, nrow=8)
    else:
        torch.save(diff, 'result/deeplift-%d-%d-%s-%d.pkl' % (args.from_digit, args.to_digit, args.model, args.window))

    return diff, overlayed_imgs

if __name__ == '__main__':
    args = parse_args()
    run(args)
