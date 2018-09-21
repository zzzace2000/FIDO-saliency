import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from datasets import mixture_of_blocks, MixtureOfBlocks


class Squeeze(nn.Module):
    def forward(self, x):
        x = x.view(len(x), -1)
        return x

def get_default_train_args():
    return dict(
        im_shape = (28, 28),
        num_epochs = 1,
        learning_rate = 0.001 ,
        checkpoint_filename = 'foo.pth',
    )

def get_dataset_and_classifier(num_examples, batch_size, seed=None, train_args=None, neural_net=True, verbose=False):
    """returns MixureOfBlocks dataset and trained classifier"""
    train_args = train_args or get_default_train_args()
    loader, dataset = mixture_of_blocks(num_examples, batch_size, seed)
    if neural_net:  # neural net
        classifier = nn.Sequential(
                Squeeze(),
                nn.Linear(int(np.prod(train_args['im_shape'])), 400),
                nn.LeakyReLU(),
                nn.Linear(400, 200),
                nn.LeakyReLU(),
                nn.Linear(200, 50),
                nn.LeakyReLU(),
                nn.Linear(50, MixtureOfBlocks.num_labels),
                )
    else:  # logistic regression
        classifier = nn.Sequential(Squeeze(), nn.Linear(
            int(np.prod(train_args['im_shape'])), MixtureOfBlocks.num_labels
            ))
  
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
            classifier.parameters(), 
            train_args['learning_rate']
            )

    for e in range(train_args['num_epochs']):
        for i, (x, y) in enumerate(loader):
            #x = Variable(x.view(len(x), -1), requires_grad=False)
            x = Variable(x, requires_grad=False)
            y = Variable(y.long().squeeze(), requires_grad=False)
            yhat = torch.max(classifier(x), 1)[1]
            acc = yhat.eq(y).float().sum() / len(y)
            optimizer.zero_grad()
            loss = loss_fn(classifier(x), y)
            loss.backward()
            optimizer.step()
            if verbose:
                print(e, i, *loss.data.numpy(), *acc.data.numpy())

    print('done training {}'.format('neural net' if neural_net else 'logistic regression'))
    torch.save(classifier.state_dict(), train_args['checkpoint_filename'])

    return classifier, dataset

def get_imgnet_one_image_loader(
        dataset, trained_classifier, 
        img_index=None, batch_size=128, 
        cuda=True, target_label='gt'):
    """in the spirit of ImageNetLoadClass.get_imgnet_one_image_loader"""

    def _make_predictions(trained_classifier, images, cuda=True):
        images = Variable(images.view(len(images), -1), volatile=True)
        if cuda:
            images = images.cuda(async=True)
        # TODO: run classifier forward on squeezed image but produce batch of not-squeezed images
        logits = trained_classifier.forward(images)
        pred_val, pred_pos = logits.data.max(dim=1)
        pred_label, classifier_output = pred_pos[0], logits.data.cpu()
        return pred_label, classifier_output

    if img_index is None:
        img_index = np.random.randint(len(dataset))

    the_img = dataset.samples[img_index]
    gt_label = int(dataset.labels[img_index].numpy())
    pred_label, classifier_output = _make_predictions(trained_classifier, the_img, cuda)
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
    #if cuda:
        #repeated_imgs = repeated_imgs.data.pin_memory()
        #repeated_labels = repeated_labels.data.pin_memory()

    train_loader = [(repeated_imgs, repeated_labels)]

    img_filename = 'MixtureOfBlocks_{}'.format(img_index)

    pred_img_class_name = str(pred_label)
    gt_img_class_name = str(gt_label)

    return train_loader, img_filename, gt_img_class_name, pred_img_class_name, classifier_output



if __name__ == '__main__':
    c, d = get_dataset_and_classifier(1000, 10, 7)
    img_loader, img_filename, gt_class_name, pred_class_name, classifier_output = \
        get_imgnet_one_image_loader(d, c, cuda=False)
    for x, y in img_loader:
        print(x.shape, x.mean(-1).mean(-1).mean(-1), torch.var(y.float()))
        print(c(Variable(x)))



