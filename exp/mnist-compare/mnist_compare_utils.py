import sys
from os import path
sys.path.append(path.dirname(path.dirname(sys.path[1])))

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy

def sort_asd_2d(nd_arr):
    ''' (np arr) -> tuple
    Sort 2d array and return an order by ascending order
    :param nd_arr
    :return: a tuple of array of index. first element is index 1, 2nd element is index 2.
    '''
    return np.unravel_index(np.argsort(nd_arr.ravel()), nd_arr.shape)


def unnormalize(arr):
    return arr * 0.3081 + 0.1307

# Get that vector and rank them to plot the box plot. Compare Max Welling and my method
def calculate_logodds_diff_by_flipping(net, image, label, importance_2d,
                                       flip_percentage=1., flip_val=(-0.1307 / 0.3081)):
    ''' (net, 3d torch array, int, 2d torch array, float) -> float
    Calculate the drop of log odds ratio when flipping the pixel value by the given order of importance_2d to reset image.
    It would flip until certain ratio and report the drop of log odds ratio.
    :param image: The original image.
    :param importance_2d: The rank of image pixels.
    :param flip_percentage: Flip pixels until some percentage.
    :return:
        log_odds, order, flip_image, flip_image[0, ...].numpy()
    '''

    def get_log_odds(image):
        image = Variable(image)
        orig_pred = net(image.unsqueeze(0))
        orig_log_prob = F.log_softmax(orig_pred)[0, label]
        print(orig_log_prob)
        return (orig_log_prob - torch.log(1 - torch.exp(orig_log_prob))).data[0]

    return _flip(net, image, importance_2d, get_log_odds, flip_percentage, flip_val)

def cal_logodds_diff_btw_two_class(net, image, from_digit, to_digit, importance_2d,
                                   flip_percentage=1., flip_val=(-0.1307 / 0.3081),
                                   cuda_enabled=False):
    def get_log_odds(image):
        image = Variable(image)
        if cuda_enabled:
            image = image.cuda(async=True)

        orig_pred = net(image.unsqueeze(0))
        from_log_prob = F.log_softmax(orig_pred)[0, from_digit]
        to_log_prob = F.log_softmax(orig_pred)[0, to_digit]

        return (to_log_prob - from_log_prob).data[0]

    return _flip(net, image, importance_2d, get_log_odds, flip_percentage, flip_val, exact=True)


def _flip(net, image, importance_2d, diff_func, flip_percentage=1.,
          flip_val=(-0.1307 / 0.3081), exact=False):
    if hasattr(importance_2d, 'cpu'):
        importance_2d = importance_2d.cpu().numpy()

    order = sort_asd_2d(-importance_2d)

    num_flip = int(np.prod(image.size()) * flip_percentage) + 1

    flip_image = image.clone()
    log_odds = [diff_func(flip_image)]

    # Only calculate the last image that flips everything
    if exact:
        for idx in range(num_flip):
            flip_image[0, order[0][idx], order[1][idx]] = flip_val
        log_odds.append(diff_func(flip_image))
        return log_odds, order, flip_image[0, ...].numpy()

    # Flip every one pixel and record log odds change each time. Suitable for debug.
    for idx in range(num_flip):
        flip_image[0, order[0][idx], order[1][idx]] = flip_val

        log_odd = diff_func(flip_image)
        log_odds.append(log_odd)

    return log_odds, order, flip_image[0, ...].numpy()

def perturb(model, loader, perturb_val_func, num_samples=1):
    ''' (net, loader, func) -> numpy 1d array
    Perturb the input value to rank which feature is more important.
    :param model: Classifier model. Needs to have 'evaluate_loader' method
    :param loader: Data loader.
    :param perturb_val_func: Generate the perturbed tensor by 2 params:
        feature_idx: the feature index.
        size: number of samples to generate in a torch floatTensor 1d array
    :return: the perturb rank. The higher, the more important.
    '''

    assert hasattr(model, 'evaluate_loader'), 'Need to have this func!'

    print(loader.dataset.data_tensor.size())
    N, P = loader.dataset.data_tensor.size()
    orig_loss, _ = model.evaluate_loader(loader)

    perturb_rank = np.zeros(P)
    for i in range(P):
        copy_loader = copy.deepcopy(loader)
        if num_samples > 1:
            copy_loader.dataset.data_tensor = \
                copy_loader.dataset.data_tensor.repeat(num_samples, 1)

        old_val = copy_loader.dataset.data_tensor[:, i]
        copy_loader.dataset.data_tensor[:, i] = perturb_val_func(i, old_val)

        loss, acc = model.evaluate_loader(copy_loader)

        perturb_rank[i] = loss.data[0] - orig_loss.data[0]

    return perturb_rank

def perturb_2d(model, loader, perturb_val_func, num_samples=1, window=1, cuda_enabled=False):
    ''' (net, loader, func) -> numpy 2d array
    Perturb the input value to rank which feature is more important.
    :param model: Classifier model. Needs to have 'evaluate_loader' method
    :param loader: Data loader.
    :param perturb_val_func: Generate the perturbed tensor by 2 params:
        feature_idx: the feature index.
        size: number of samples to generate in a torch floatTensor 1d array
    :param window: how many to perturb
    :return: the perturb rank. The higher, the more important.
    '''

    assert hasattr(model, 'evaluate_loader'), 'Need to have this func!'
    model.eval()

    # print(loader.dataset.data_tensor.size())
    N, channel, dim1, dim2 = loader.dataset.data_tensor.size()
    orig_loss, _ = model.evaluate_loader(loader, cuda=cuda_enabled)

    copy_loader = copy.deepcopy(loader)

    perturb_rank = np.zeros((dim1, dim2))
    count = np.zeros((dim1, dim2))
    for i in range(dim1 - window + 1):
        for j in range(dim2 - window + 1):
            copy_loader.dataset.data_tensor = copy.deepcopy(loader.dataset.data_tensor)
            if num_samples > 1:
                copy_loader.dataset.data_tensor = \
                    copy_loader.dataset.data_tensor.repeat(num_samples, 1)

            old_val = copy_loader.dataset.data_tensor[:, :, i:(i + window), j:(j + window)]
            copy_loader.dataset.data_tensor[:, :, i:(i + window), j:(j + window)] = \
                perturb_val_func((i, j), old_val)

            loss, acc = model.evaluate_loader(copy_loader, cuda=cuda_enabled)

            perturb_rank[i:(i + window), j:(j + window)] += (loss.data[0] - orig_loss.data[0])
            count[i:(i + window), j:(j + window)] += 1

    return torch.from_numpy(perturb_rank / count)