import torch
from arch.Inpainting.Baseline import BlurryInpainter, LocalMeanInpainter, MeanInpainter, \
    RandomColorWithNoiseInpainter

from arch.Inpainting.AE_Inpainting import ImpantingModel
from arch.Inpainting.AE_Inpainting import VAEImpantModel, VAEWithVarImpantModelMean
import torchvision.models as models


def load_vbd_result(path):
    '''
    For backward compatibility. To read things in a dictionary instead of a tuple
    '''
    result = torch.load(path)
    if isinstance(result, dict):
        return result

    unnormalized_img, imp_vector, img_idx = result
    return dict(unnormalized_img=unnormalized_img, imp_vector=imp_vector, img_idx=img_idx)


def get_pretrained_classifier(classifier_name, cuda_enabled=False):
    func = getattr(models, classifier_name)
    interpret_net = func(pretrained=True)
    interpret_net.eval()
    if cuda_enabled:
        interpret_net.cuda()

    return interpret_net


def get_impant_model(gen_model_name, batch_size=None, gen_model_path=None, cuda_enabled=False):
    if gen_model_name == 'MeanInpainter' or gen_model_name == 'LocalMeanInpainter' \
            or gen_model_name == 'BlurryInpainter' or gen_model_name == 'RandomColorWithNoiseInpainter':
        impant_model_obj = eval(gen_model_name)
        impant_model = impant_model_obj()
    elif gen_model_name == 'CAInpainter':
        from generative_inpainting.CAInpainter import CAInpainter
        impant_model = CAInpainter(
            batch_size, checkpoint_dir='./generative_inpainting/model_logs/release_imagenet_256/')
    else:
        if gen_model_name == 'VAEWithVarImpantModelMean' and gen_model_path is None:
            gen_model_path = 'checkpts2/0928-VAE-Var-hole_lr_0.0002_epochs_7'
        impant_model_obj = eval(gen_model_name)
        impant_model = impant_model_obj()

        states = torch.load(gen_model_path, map_location=lambda storage, loc: storage)
        state_dict = impant_model.state_dict()
        state_dict.update(states['state_dict'])
        impant_model.load_state_dict(state_dict)

    impant_model.eval()
    if cuda_enabled:
        impant_model.cuda()
    return impant_model
