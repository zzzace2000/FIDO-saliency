import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys, gzip
import pkgutil
import os
from PIL import Image
import copy


def load_mnist_loader(data_dir='../data', batch_size=128, cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def load_mnist_one_image(img_index=None, batch_size=128, cuda=True):
    if img_index is None:
        img_index = np.random.randint(60000)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    trained_sets = datasets.MNIST('../data', train=True, transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    # This way the image would go through transform
    the_img, _ = trained_sets[img_index]
    # This way the_label would remain IntTensor so could expand dim
    the_label = trained_sets.train_labels[img_index:(img_index + 1)]

    # Repeat the image "batch_size" times
    repeated_imgs = the_img.unsqueeze(0).expand(batch_size, 1, 28, 28)
    repeated_labels = the_label.expand(batch_size)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(repeated_imgs, repeated_labels),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader

def load_mnist_keras_loader(batch_size=128, cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    def compose_loader(x, y, batch_size):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        train_loader = torch.utils.data.DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size, shuffle=True, **kwargs)
        return train_loader

    (X_train, y_train), (X_test, y_test) = load_mnist_keras_imgs()

    train_loader = compose_loader(X_train, y_train, batch_size=batch_size)
    test_loader = compose_loader(X_test, y_test, batch_size=batch_size)
    return train_loader, test_loader

def load_mnist_keras_imgs():
    if pkgutil.find_loader('keras') is not None:
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = load_data()

    X_train = (X_train[:,None,:,:] / 255.).astype('float32')
    X_test = (X_test[:,None,:,:] / 255.).astype('float32')

    return (X_train, y_train), (X_test, y_test)


def load_mnist_keras_test_imgs():
    _, (X_test, y_test) = load_mnist_keras_imgs()
    print('load mnist test image. shape: %s, min: %f, max: %f' % (
        X_test.shape, X_test.min(), X_test.max()))

    return X_test, y_test


def load_data(path='../data/mnist.pkl.gz'):
    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')
    import cPickle
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')

    f.close()
    return data  # (X_train, y_train), (X_test, y_test)


class ImageNetLoadClass:
    def __init__(self, imagenet_folder='imagenet/', dataset='train/', resize=224):
        self.imagenet_folder = imagenet_folder
        self.dataset = dataset
        self.resize = resize # For the different input size for inception_v3

        self.id_name_dict = self._get_id_to_class_dict()

        self.img_folder = self._get_image_folder()

        the_img, _ = self.img_folder[0]
        self.num_channels = the_img.size(0)
        self.img_height = the_img.size(1)
        self.img_width = the_img.size(2)

        self.coord_dict = {} # map from ILSVRC2012_val_00037956 to ('n03995372', [85 1 499 272])
        with open(os.path.join(imagenet_folder, 'LOC_val_solution.csv')) as fp:
            fp.readline()
            for line in fp:
                line = line.strip().split(',')
                filename = line[0]
                tmp = line[1].split(' ')

                arr = []
                for i in range(len(tmp) // 5):
                    theclass = tmp[i * 5]
                    coord = np.array(tmp[(i*5 + 1):(i*5 + 5)], dtype=int).reshape((2, 2))
                    arr.append([theclass, coord])
                self.coord_dict[filename] = arr
        # import pdb; pdb.set_trace()

    def translate_class_id_to_name(self, id):
        if isinstance(id, str):
            id = int(id)

        assert 0 <= id < len(self.img_folder.classes), 'id %d out of range!' % id

        long_name = self.id_name_dict[self.img_folder.classes[id]]
        short_name = long_name.split(',')[0]
        return short_name

    def get_bounding_box_by_img_idx(self, img_idx, test=False):
        '''
        :param img_idx:
        :return: Return an array of tuple. Each image may have multiple bbox.
            Ex: [(cls_name, coordinates)]
            coordinates is a 2x2 array with [[x_left_top, y_left_top],
                                             [x_right_botton, y_right_bottom]]
            Example code:
                the_class_name, gnd_truth_coord = coordinate_arr[0]
                gnd_truth_label = loadhelper.img_folder.class_to_idx[the_class_name]

                coord = coord.astype(int)
                result = np.zeros((224, 224))
                result[coord[0, 1]:coord[1, 1], coord[0, 0]:coord[1, 0]] = 1
        '''
        img_path, _ = self.img_folder.imgs[img_idx]

        # Take the bounding box:
        filename_identifier = img_path.split('/')[-1].split('.')[0]
        if filename_identifier not in self.coord_dict:
            print('Not found bounding box!')
            return None

        thecoord_arr = self.coord_dict[filename_identifier]

        img = Image.open(img_path)
        if test:
            self._draw_bounding_box(img, thecoord_arr[0][1])

        w, h = img.size
        new_coord_arr = []
        for i, (theclass, coord) in enumerate(thecoord_arr):
            new_coord = self._handle_coord_transform(coord, w, h)
            if new_coord is None:
                continue

            new_coord_arr.append((theclass, new_coord))

        if test:
            self._draw_bounding_box(self.img_folder[img_idx][0], new_coord_arr[0][1])
        return new_coord_arr

    def get_npy_bounding_box_by_img_idx(self, img_idx):
        '''
        Take all the boxes that is our ground truth class and produce a numpy array with 1 as bbox.
        '''
        img_path, label = self.img_folder.imgs[img_idx]

        bboxes = self.get_bounding_box_by_img_idx(img_idx)

        # Take all the boxes that is our ground truth class
        result_box = np.zeros((224, 224))
        for the_class_name, gnd_truth_coord in bboxes:
            gnd_truth_label = self.img_folder.class_to_idx[the_class_name]
            if gnd_truth_label == label:
                gnd_truth_coord = gnd_truth_coord.astype(int)
                result_box[gnd_truth_coord[0, 1]:gnd_truth_coord[1, 1],
                    gnd_truth_coord[0, 0]:gnd_truth_coord[1, 0]] = 1

        if result_box.sum() == 0:
            return None

        return result_box

    def _handle_coord_transform(self, coord, img_width, img_height):
        '''
        Transform bounding box coordinate into right coordinate
        :param coord: [N x 2] coordinate. Each represents [x, y] coordinate
        :return:
        '''
        # Handle Resize(256)
        w, h = img_width, img_height
        if h > w:
            new_h, new_w = 256 * h / w, 256
        else:
            new_h, new_w = 256, 256 * w / h
        new_h, new_w = int(new_h), int(new_w)

        new_coord = copy.copy(coord)
        new_coord = new_coord * [new_w / w, new_h / h]

        # Handle Center Crop(224)
        top = int(round((new_h - 224) / 2.))
        left = int(round((new_w - 224) / 2.))

        new_coord = new_coord - [left, top]
        new_coord = np.round(new_coord).astype(int)
        new_coord[new_coord < 0] = 0
        new_coord[new_coord > 224] = 224

        # After transforming, no bounding box! (some bugs in gnd truth)
        if new_coord[0, 1] == new_coord[1, 1] or new_coord[0, 0] == new_coord[1, 0]:
            return None

        return new_coord

    def _draw_bounding_box(self, image, coord):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from exp.utils_visualise import plot_pytorch_img
        fig, ax = plt.subplots()

        ax.add_patch(
            patches.Rectangle(
                coord[0, :], coord[1, 0] - coord[0, 0], coord[1, 1] - coord[0, 1],
                color='red',
                fill=False  # remove background
            )
        )

        if isinstance(image, torch.Tensor):
            print(coord, image.shape)
            # image[:, coord[0, 1]:coord[1, 1], coord[0, 0]:coord[1, 0]] = 0
            fig = plot_pytorch_img(image, ax=ax)
        else:
            fig = ax.imshow(image)
        plt.show()

    def _get_id_to_class_dict(self):
        the_dict = {}
        with open(os.path.join(self.imagenet_folder, 'class_names.txt')) as fp1, \
            open(os.path.join(self.imagenet_folder, 'synsets.txt')) as fp2:
            for name, id in zip(fp1, fp2):
                name = name.strip()
                id = id.strip()
                the_dict[id] = name
        return the_dict

    def _get_image_folder(self):
        # Data loading code
        arr = [
            transforms.Resize(int(self.resize / 224 * 256)),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]

        transform = transforms.Compose(arr)

        img_folder = datasets.ImageFolder(
            os.path.join(self.imagenet_folder, self.dataset), transform)
        return img_folder

    @staticmethod
    def get_min_max_val(height, width):
        '''
        The 0 to 1 ranges but by this normalization!
        '''
        min_max_vals = np.zeros((2, 3, height, width))
        min_max_vals[0, 0, ...] = -0.485 / 0.229
        min_max_vals[1, 0, ...] = (1. - 0.485) / 0.229
        min_max_vals[0, 1, ...] = -0.456 / 0.224
        min_max_vals[1, 1, ...] = (1. - 0.456) / 0.224
        min_max_vals[0, 2, ...] = -0.406 / 0.225
        min_max_vals[1, 2, ...] = (1. - 0.406) / 0.225
        return min_max_vals

    def get_imgnet_loader(self, batch_size, shuffle=True, num_workers=8):
        train_loader = torch.utils.data.DataLoader(
            self.img_folder, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return train_loader

    @staticmethod
    def unnormalize_imagenet_img(images):
        '''
        Unnormalized these images. If images value after normalization larger than 0-1,
        print a warning and clip the values.
        :param images: 4d pytorch array.
        :return: images that's unnormalized.
        '''

        result_images = images.clone()
        is_three_dim = (len(result_images.shape) == 3)
        if is_three_dim:
            result_images = result_images.unsqueeze(0)

        result_images[:, 0, :, :] = result_images[:, 0, :, :] * 0.229 + 0.485
        result_images[:, 1, :, :] = result_images[:, 1, :, :] * 0.224 + 0.456
        result_images[:, 2, :, :] = result_images[:, 2, :, :] * 0.225 + 0.406
        if is_three_dim:
            result_images = result_images[0]
        return result_images

    @staticmethod
    def normalize_imagenet_img(images):
        result_images = images.clone()
        result_images[:, 0, :, :] = (result_images[:, 0, :, :] - 0.485) / 0.229
        result_images[:, 1, :, :] = (result_images[:, 1, :, :] - 0.456) / 0.224
        result_images[:, 2, :, :] = (result_images[:, 2, :, :] - 0.406) / 0.225
        return result_images

    @staticmethod
    def make_predictions(trained_classifier, images, cuda=True):
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = Variable(images, volatile=True)
        if cuda:
            images = images.cuda(async=True)

        outputs = trained_classifier.forward(images)
        pred_val, pred_pos = outputs.data.max(dim=1)
        return pred_pos[0], outputs.data.cpu()

    def get_imgnet_one_image_loader(self, trained_classifier, img_index=None, batch_size=128,
                                    cuda=True, target_label='gt'):
        if img_index is None:
            img_index = np.random.randint(len(self.img_folder))

        the_img, gt_label = self.img_folder[img_index]
        pred_label, classifier_output = self.make_predictions(trained_classifier, the_img, cuda)

        if target_label == 'gt':
            the_target_label = gt_label
        elif target_label == 'pred':
            the_target_label = pred_label
        elif isinstance(target_label, int):
            the_target_label = target_label
        else:
            raise Exception('Error! Not known label!')

        # Repeat the image "batch_size" times
        repeated_imgs = the_img.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        repeated_labels = torch.LongTensor(1).fill_(the_target_label).repeat(batch_size)
        if cuda:
            repeated_imgs = repeated_imgs.pin_memory()
            repeated_labels = repeated_labels.pin_memory()

        # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        # train_loader = torch.utils.data.DataLoader(
        #     TensorDataset(repeated_imgs, repeated_labels),
        #     batch_size=batch_size, shuffle=False, **kwargs)

        train_loader = [(repeated_imgs, repeated_labels)]

        img_path = self.img_folder.imgs[img_index][0]
        img_filename = os.path.basename(img_path)
        img_filename = img_filename.split('.')[0]

        pred_img_class_name = self.translate_class_id_to_name(pred_label)
        gt_img_class_name = self.translate_class_id_to_name(gt_label)

        return train_loader, img_filename, gt_img_class_name, pred_img_class_name, classifier_output


if __name__ == '__main__':
    # Test bounding box transformation
    loadhelper = ImageNetLoadClass('imagenet/', dataset='valid/')
    # loadhelper.get_bounding_box_by_img_idx(139, test=True)
    # for img_idx in range(10, 20):
        # loadhelper.get_bounding_box_by_img_idx(img_idx, test=True)
        # loadhelper.get_npy_bounding_box_by_img_idx(img_idx)

    loadhelper.translate_class_id_to_name(100)

    # loadhelper.get_bounding_box_by_img_idx(422, test=True)
    for img_idx in range(1000):
        result = loadhelper.get_npy_bounding_box_by_img_idx(img_idx)
        if result is None:
            print('Found!')

