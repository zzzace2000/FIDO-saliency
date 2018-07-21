import os
import sys

import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
from exp.loaddata_utils import ImageNetLoadClass
from exp.general_utils import Timer
import numpy as np
import os
import torch.optim as optim
import pkgutil
from torch.optim.lr_scheduler import ReduceLROnPlateau


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def random_masks(N, C, H, W):
    mask = torch.rand(N, 1, H, W).round()
    return mask

def eightth_masks(N, C, H, W):
    size = np.random.randint(H / 8 - H / 32, H / 8 + 1, size=2)
    loc1 = np.random.randint(0, H - size[0] + 1, size=16)
    loc2 = np.random.randint(0, W - size[1] + 1, size=16)

    mask = (torch.ones(N, 1, H, W))
    for i in range(16):
        mask[:, :, loc1[i]:(loc1[i] + size[0]), loc2[i]:(loc2[i] + size[1])] = 0.
    return mask

def quarter_masks(N, C, H, W):
    size = np.random.randint(H / 4 - H / 16, H / 4 + 1, size=2)
    loc1 = np.random.randint(0, H - size[0] + 1, size=4)
    loc2 = np.random.randint(0, W - size[1] + 1, size=4)

    mask = (torch.ones(N, 1, H, W))
    for i in range(4):
        mask[:, :, loc1[i]:(loc1[i] + size[0]), loc2[i]:(loc2[i] + size[1])] = 0.
    return mask

def half_masks(N, C, H, W):
    # Corrupt the inputs by randomly having a image that covers 1/4
    size = np.random.randint(H / 2 - H / 8, H / 2 + 1, size=2)
    loc1 = np.random.randint(0, H - size[0] + 1, size=N)
    loc2 = np.random.randint(0, W - size[1] + 1, size=N)

    mask = (torch.ones(N, 1, H, W))
    for i in range(N):
        mask[i, :, loc1[i]:(loc1[i] + size[0]), loc2[i]:(loc2[i] + size[1])] = 0.
    return mask


def grad_clamp(parameters, clip=5):
    for p in parameters:
        if p.requires_grad:
            p.grad.data.clamp_(-clip, clip)

def main(args):
    with Timer('loading imagenet'):
        load_helper = ImageNetLoadClass(imagenet_folder=args.path, dataset='train/')
        loader = load_helper.get_imgnet_loader(batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)

    # Model's parameter
    start_epoch = -1

    the_gen_model = eval(args.gen_model)
    num_training = len(loader.dataset.imgs)
    model = the_gen_model(num_training=num_training, bvlc_dir=args.bvlc_dir, gan_g_dir=args.gan_g_dir,
                          reg_coef=args.reg_coef,
                          # clamp=args.clamp,
                          # bound=os.path.join(args.gan_g_dir, 'fc6.txt')
                          )

    if args.pretrained_path is not None:
        state_dict = model.state_dict()
        state_dict.update(torch.load(args.pretrained_path)['state_dict'])
        model.load_state_dict(state_dict)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.2, min_lr=5E-7, verbose=True)
    loss_records = {}

    # Load all the stuffs!
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            current_dev_name = 'cuda:%d' % torch.cuda.current_device()
            map_location = {}
            for i in range(4):
                map_location['cuda:%d' % i] = current_dev_name

            checkpoint = torch.load(
                args.resume, map_location=map_location)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'loss_records' in checkpoint:
                loss_records = checkpoint['loss_records']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        model.cuda()

    parr_model = model
    if len(args.gpu_ids) > 1:
        parr_model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    total_batch = len(loader)
    mask_dice_arr = [
        half_masks,
        # quarter_masks, eightth_masks,
        random_masks]

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.zero_loss_record()

        # with Timer('training an epoch'):
        for batch_idx, (inputs, _) in enumerate(loader):
            # Roll the dice to random get mask!
            mask_func = mask_dice_arr[np.random.randint(0, len(mask_dice_arr))]
            mask = mask_func(*inputs.size())

            inputs = Variable(inputs)
            mask = Variable(mask)
            if args.cuda:
                inputs = inputs.cuda(async=True)
                mask = mask.cuda(async=True)

            # corrupted_inputs = inputs * mask
            # import exp.utils_visualise as utils_visualize
            # plotted_imgs = load_helper.unnormalize_imagenet_img(corrupted_inputs.data[0:9, ...])
            # utils_visualize.save_figs(plotted_imgs, nrow=3, visualize=True)
            # corrupted_inputs = torch.cat((corrupted_inputs, 1. - mask), dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # outputs = parr_model.pth_normalized_forward(corrupted_inputs)
            outputs = parr_model.forward(inputs, mask)

            the_loss = model.loss_fn(outputs, inputs, mask)
            the_loss.backward()

            # Before optimizing grad, do grad clipping
            grad_clamp(model.parameters())

            optimizer.step()

            if batch_idx % args.batch_print == (args.batch_print - 1):
                print('batch: [%d / %d], %s' % (batch_idx, total_batch, model.report_loss()))

        print('epoch: [%d / %d], %s' % (epoch, args.epochs, model.report_loss()))
        loss_records[epoch] = model.total_avg_loss()

        # Reduce learning rate if epoch loss is not improving
        scheduler.step(model.total_avg_loss())

        if epoch % args.save_freq == (args.save_freq - 1):
            if not os.path.exists('checkpts3/'):
                os.mkdir('checkpts3')

            name = 'checkpts3/{}_lr_{}_epochs_{}'.format(args.identifier, args.lr, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss_records': loss_records,
            }, name)

            torch.save(loss_records, 'checkpts3/{}_lr_{}.loss'.format(args.identifier, args.lr))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Generate Multivariate Gaussian Parameters')
    parser.add_argument('--gen-model', type=str, default='ImpantingModel',
                        help='choose from [ImpantingModel, VAEImpantModel, AE_GAN, VAE_GAN,'
                             'VAEWithVarImpantModel]')
    parser.add_argument('--bvlc_dir', type=str, default='nets/BVLC/',
                        help='bvlr directory')
    parser.add_argument('--gan_g_dir', type=str, default='nets/GAN/',
                        help='gan generator directory')
    parser.add_argument('--pretrained-path', type=str, default=None, help='pretrained ImpaintGAN')
    parser.add_argument('--reg-coef', type=float, default=4)
    # parser.add_argument('--no-clamp', action='store_true', default=False)
    parser.add_argument('--path', type=str, default='../imagenet/', help='the imagenet folder path')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--batch-print', type=int, default=1)
    # parser.add_argument('--shrink', action='store_true', default=False)

    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='train/', help='choose from [train, valid, test]')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[],
                        help='number of gpus to produce')
    parser.add_argument('--identifier', type=str, default='test')

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    # args.clamp = (not args.no_clamp)

    if args.cuda and pkgutil.find_loader('gpu_lock') is not None:
        # for i in range(args.num_gpus):
        #     board = gpu_lock.obtain_lock_id()
        #     if board < 0:
        #         exit('No GPU available now!')
        #     args.gpu_ids.append(board)
        print('gpu current device:', torch.cuda.current_device())
        if len(args.gpu_ids) > 1:
            print('start using gpu device:', args.gpu_ids)
            torch.cuda.set_device(args.gpu_ids[0])

    print('args:', args)
    print('==================== Start =====================')
    print('')

    main(args)
