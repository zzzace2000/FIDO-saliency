import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_logodds_by_flipping(mask_generator, interpret_net, impant_model, img_loader,
                            batch_size, the_log_odds_criteria=None, num_samples=1, window=1,
                            cuda_enabled=False):
    def log_sum_exp(x, dim):
        x_max = x.max()
        return torch.log(torch.sum(torch.exp(x - x_max), dim=dim)) + x_max

    def log_odds_criteria(outputs, targets):
        log_prob = F.log_softmax(outputs, dim=1)

        if targets == 0:
            other_log_prob = log_prob[:, (targets + 1):]
        elif targets == log_prob.size(1) - 1:
            other_log_prob = log_prob[:, :targets]
        else:
            other_log_prob = torch.cat([log_prob[:, :targets], log_prob[:, (targets + 1):]], dim=1)

        tmp = log_sum_exp(other_log_prob, dim=1)
        return (log_prob[:, targets] - tmp).mean()

    # Prevent myself too stupid...
    interpret_net.eval()

    if the_log_odds_criteria is None:
        the_log_odds_criteria = log_odds_criteria

    images, targets = next(iter(img_loader))

    _, _, dim1, dim2 = images.size()

    # Get the original logodds
    # the_img = Variable(image, volatile=True)
    if cuda_enabled:
        images = images.cuda()
        targets = targets.cuda()

    output = interpret_net(Variable(images[0:1, ...], volatile=True))
    orig_odds = the_log_odds_criteria(output, targets[0])

    # Calculate how many pixel I can progress each time
    actual_batch_size = batch_size - batch_size % num_samples
    # print('actual batch size:', actual_batch_size)

    num_pixels_progress = actual_batch_size // num_samples

    masks = images.new(batch_size, 1, dim1, dim2)
    mask_generator = iter(mask_generator)

    all_log_odds = []
    non_empty = True
    while non_empty:
        masks.fill_(1.)

        num_masks = 0
        for i in range(int(num_pixels_progress)):
            try:
                the_mask = next(mask_generator)
            except StopIteration:
                non_empty = False
                break

            if the_mask.ndimension() == 3:
                the_mask = the_mask.unsqueeze(0)
            expand_mask = the_mask.expand(num_samples, 1, the_mask.shape[2], the_mask.shape[3])
            masks[i * num_samples:(i + 1) * num_samples, ...] = expand_mask
            num_masks += 1

        if num_masks == 0:
            break

        the_masks = masks
        if cuda_enabled:
            the_masks.cuda()

        generated_img = impant_model.impute_missing_imgs(images, the_masks)
        outputs = interpret_net.forward(Variable(generated_img, volatile=True))

        for i in range(num_masks):
            avg_odds = the_log_odds_criteria(outputs[i * num_samples:(i + 1) * num_samples, :],
                                             targets[0])
            all_log_odds.append(avg_odds.data[0])

    return orig_odds.data[0], all_log_odds
