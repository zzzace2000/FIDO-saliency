import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.getcwd(), 'generative_inpainting'))

import argparse
from exp.loaddata_utils import ImageNetLoadClass
from exp.general_utils import Timer
import numpy as np
import os
from arch.sensitivity.BDNet import OppositeGernarativeL1BDNet, GernarativeL1BDNet
import exp.utils_visualise as utils_visualise
from exp.utils_flipping import get_logodds_by_flipping

from arch.sensitivity.BBMPNet import BBMP_SDR_Generative, BBMP_SSR_Generative
import torch.nn.functional as F
import tensorflow as tf
import visdom
import utils_model


def main(args, importance_func, impant_model, interpret_net):
    if args.cuda:
        interpret_net.cuda()
        impant_model.cuda()

    # Load data
    load_helper = ImageNetLoadClass(imagenet_folder=args.data_dir,
                                    dataset=args.dataset)

    for img_idx in range(args.image_offset, args.image_offset + args.num_imgs):
        print('img_idx:', img_idx)

        img_loader, img_filename, gt_class_name, pred_class_name, classifier_output = \
            load_helper.get_imgnet_one_image_loader(trained_classifier=interpret_net,
                                                    img_index=img_idx,
                                                    batch_size=args.batch_size,
                                                    target_label='gt',
                                                    cuda=args.cuda)

        output_file = '%s/%s_records.th' % (args.save_dir, img_filename)
        if not args.overwrite and os.path.exists(output_file):
            print('Already evaluate for the img idx %d' % img_idx)
            continue

        # Take out one mnist image and unnormalize it
        images, targets = next(iter(img_loader))
        orig_img = images[0:1, ...]
        unnormalized_img = load_helper.unnormalize_imagenet_img(orig_img)[0, ...]

        if args.visdom_enabled:
            visdom.Visdom().image(load_helper.unnormalize_imagenet_img(img_loader[0][0][0]))

        with Timer('evaluating image'):
            impant_model.reset()
            imp_vector = importance_func(interpret_net, impant_model, img_loader)

        overlayed_img, clim = utils_visualise.get_overlayed_image(unnormalized_img, imp_vector)

        if args.visdom_enabled:
            visdom.Visdom().image(overlayed_img)

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        # Visualize images
        # file_name = '%s/%s_overlayed.png' % (args.save_dir, img_filename)
        # utils_visualise.plot_orig_and_overlay_img(unnormalized_img, overlayed_img,
        #                                           bbox_coord=coord_arr[0][1],
        #                                           file_name=file_name,
        #                                           gt_class_name=gt_class_name,
        #                                           pred_class_name=pred_class_name,
        #                                           clim=clim,
        #                                           visualize=args.visualize)

        torch.save({
            'unnormalized_img': unnormalized_img,
            'imp_vector': imp_vector,
            'img_idx': img_idx,
            'classifier_output': classifier_output,
            'gnd_truth_label': targets[0],
        }, output_file)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Imagenet Example')
    parser.add_argument('--importance-method', type=str, default='vbd_sdr',
                        help='choose from ["p_b", "vbd_ssr", "vbd_sdr", "bbmp_ssr", "bbmp_sdr"]')
    parser.add_argument('--classifier', type=str, default='alexnet',
                        help='Choose from [alexnet, resnet18, vgg19_bn]')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--prior', type=float, default=0.5,
                        help='prior probability for reg loss')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--ard_init', type=float, default=0.,
                        help='ARD initialization')
    parser.add_argument('--reg-coef', type=float, default=0.01,
                        help='regularization coefficient')
    parser.add_argument('--tv_coef', type=float, default=0.,
                        help='regularization coefficient')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--epoch-print', type=int, default=1,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default='../imagenet/',
                        help='data directory')
    parser.add_argument('--gen-model-path', type=str, default=None, help='data directory')
    parser.add_argument('--gen-model-name', type=str,
                        default='ImpantingModel',
                        help='choose from [ImpantingModel, VAEImpantModel, VAEWithVarImpantModel,'
                             'VAEWithVarImpantModelMean, MeanInpainter, LocalMeanInpainter]')
    parser.add_argument('--dataset', type=str, default='valid/',
                        help='Choose from train/ or valid/')
    parser.add_argument('--save-dir', type=str, default='./imgs/hole_model_0.01/',
                        help='Save directory')
    parser.add_argument('--save-tag', type=str, default='',
                        help='Unique tag for output images')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Open verbose or not')
    parser.add_argument('--num-imgs', type=int, default=1,
                        help='number of images to produce')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[],
                        help='number of gpus to produce')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--image-offset', type=int, default=0, help='offset for index of image')
    parser.add_argument('--bvlc_dir', type=str, default='nets/BVLC/',
                        help='bvlr directory')
    parser.add_argument('--gan_g_dir', type=str, default='nets/GAN/',
                        help='gan generator directory')
    parser.add_argument('--eval-samples', type=int, default=1,
                        help='number of samples in evaluation')
    parser.add_argument('--dropout_param_size', nargs='+', type=int, default=[56, 56],
                        help='Dropout parameter size')
    parser.add_argument('--rw_max', type=int, default=30, help='annealing')
    parser.add_argument('--visdom_enabled', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.set_random_seed(args.seed)

    # If use bbmp, batch size needs to be 1
    if args.importance_method.startswith('bbmp'):
        args.batch_size = 1

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    if args.cuda and torch.cuda.current_device() != args.gpu_ids[0]:
        torch.cuda.set_device(args.gpu_ids[0])

    print('args:', args)
    print('==================== Start =====================')
    print('')
    return args


def log_sum_exp(x, dim):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), dim=dim)) + x_max


def log_odds_loss(outputs, targets):
    log_prob = F.log_softmax(outputs, dim=1)

    if targets.data[0] == 0:
        other_log_prob = log_prob[:, (targets.data[0] + 1):]
    elif targets.data[0] == log_prob.size(1) - 1:
        other_log_prob = log_prob[:, :targets.data[0]]
    else:
        other_log_prob = torch.cat([log_prob[:, :targets.data[0]],
                                    log_prob[:, (targets.data[0] + 1):]], dim=1)
    tmp = log_sum_exp(other_log_prob, dim=1)
    return -(log_prob[:, targets.data[0]] - tmp).mean()


if __name__ == '__main__':
    args = parse_args()
    def vbd_ssr(interpret_net, impant_model, img_loader):
        bdnet = GernarativeL1BDNet
        return _vbd_shared(bdnet, interpret_net, impant_model, img_loader)

    def vbd_sdr(interpret_net, impant_model, img_loader):
        bdnet = OppositeGernarativeL1BDNet
        color_vector = _vbd_shared(bdnet, interpret_net, impant_model, img_loader)
        return -color_vector

    def _vbd_shared(bdnet, interpret_net, impant_model, img_loader):
        net = bdnet(dropout_param_size=(1, args.dropout_param_size[0], args.dropout_param_size[1]),
                    trained_classifier=interpret_net, generative_model=impant_model,
                    loss_criteria=log_odds_loss,
                    ard_init=args.ard_init, lr=args.lr, reg_coef=args.reg_coef,
                    tv_coef=args.tv_coef, rw_max=args.rw_max,
                    cuda_enabled=args.cuda, verbose=args.verbose, prior_p=args.prior,
                    upsample_to_size=(224, 224), visdom_enabled=args.visdom_enabled)
        # Train it
        net.fit(img_loader, epochs=args.epochs, epoch_print=args.epoch_print)

        # Visualize the keep probability
        keep_probability = net.get_importance_vector()

        print('range: (%.3f, %.3f), shape: %s' % (keep_probability.min(), keep_probability.max(),
                                                  str(keep_probability.size())))

        color_vector = (keep_probability - 0.5).cpu().numpy()
        # sample_berns = net.sampled_from_logit_p(args.eval_samples)
        return color_vector

    def bbmp_ssr(interpret_net, impant_model, img_loader):
        bbmpnet = BBMP_SSR_Generative
        return _bbmp_common(bbmpnet, interpret_net, impant_model, img_loader)

    def bbmp_sdr(interpret_net, impant_model, img_loader):
        bbmpnet = BBMP_SDR_Generative
        return _bbmp_common(bbmpnet, interpret_net, impant_model, img_loader)

    def _bbmp_common(bbmpnet, interpret_net, impant_model, img_loader):
        imgs, targets = next(iter(img_loader))
        new_loader = [(imgs[0:1, ...], targets[0:1])]
        net = bbmpnet(dropout_param_size=(1, args.dropout_param_size[0], args.dropout_param_size[1]),
                      trained_classifier=interpret_net, generative_model=impant_model, loss_criteria=log_odds_loss,
                      ard_init=1., lr=args.lr, reg_coef=args.reg_coef, tv_coef=args.tv_coef, rw_max=1,
                      cuda_enabled=args.cuda, verbose=args.verbose,
                      upsample_to_size=(224, 224), visdom_enabled=args.visdom_enabled)
        net.fit(new_loader, epochs=args.epochs, epoch_print=args.epoch_print)

        # Visualize by mask
        keep_probability = net.get_importance_vector()
        print('range: (%.3f, %.3f), shape: %s' % (keep_probability.min(), keep_probability.max(),
                                                  str(keep_probability.shape)))

        color_vector = (keep_probability - 0.5).cpu().numpy()
        return color_vector

    def p_b(interpret_net, impant_model, img_loader):
        # Prevent myself too stupid...
        interpret_net.eval()

        # Take out origisnal imgs and targets
        imgs, targets = next(iter(img_loader))
        orig_img = imgs[0:1, ...]
        orig_target = targets[0]

        # All the inputs dimension
        N, channel, dim1, dim2 = imgs.size()

        width = dim2 - args.window + 1
        height = dim1 - args.window + 1

        # Mask generator
        def mask_generator():
            for i in range(height):
                for j in range(width):
                    mask = torch.ones(1, 1, dim1, dim2)
                    mask[:, :, i:(i + args.window), j:(j + args.window)] = 0.
                    yield mask

        orig_odds, all_log_odds = get_logodds_by_flipping(
            mask_generator(), interpret_net, impant_model, img_loader,
            batch_size=args.batch_size, num_samples=args.num_samples, window=args.window,
            cuda_enabled=args.cuda)

        perturb_rank = np.zeros((dim1, dim2))
        count = np.zeros((dim1, dim2))
        for i in range(height):
            for j in range(width):
                perturb_rank[i:(i + args.window), j:(j + args.window)] \
                    += (orig_odds - all_log_odds[i * width + j])
                count[i:(i + args.window), j:(j + args.window)] += 1
        return perturb_rank / count

    # Load which method to interpret importance
    importance_func = eval(args.importance_method)

    # Load which impanting model you want to use
    impant_model = utils_model.get_impant_model(args.gen_model_name, args.batch_size, args.gen_model_path)
    interpret_net = utils_model.get_pretrained_classifier(args.classifier)

    main(args, importance_func, impant_model, interpret_net)
