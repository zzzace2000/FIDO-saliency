# Explaining Image Classifiers by Counterfactual Generation

[ArXiv](https://arxiv.org/abs/1807.08024) | [BibTex](#citing)

This is the code for reproducing the paper result. We show that using generative model could help explain neural network compared to other heuristics approach.

## Run
0. Requirements:
    * Install python3
    * Install pytorch 0.3 version (It does not support 0.4)
    * Put an imagenet folder under exp/imagenet/
        - You can download it from [official websites](http://www.image-net.org/challenges/LSVRC/2012/index)
    * Install Visdom if you want visualization, and see [here](#visdom).
    * If you want to replicate VAE experiments, please download the pretrained model from [here](https://drive.google.com/file/d/0B-d9idOJBwD7WDhnTWJSZ285N0k/view?usp=sharing) and put under exp/vbd_imagenet/checkpts2/
    * If you want to replicate Contextual Attention GAN experiments, you need to setup the pre-trained GAN from this [repo](https://github.com/zzzace2000/generative_inpainting). Put this repo under exp/vbd_imagenet/generative_inpainting/
1. Running examples:
    * Run FIDO with SDR objectives (vbd_sdr), and with Local inpainter (LocalMeanInpainter)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u vbd_imagenet.py --importance-method vbd_sdr --classifier resnet50 --dropout_param_size 56 56 --epochs 300 --lr 0.05 --reg-coef 1e-3 --batch-size 8 --num-samples 1 --gen-model-name LocalMeanInpainter --save-dir ./imgs/0317_local_vbd_sdr_1E-3_56_val_resnet/ --num-imgs 50 --dataset valid/ --data-dir ../imagenet/ --gpu-ids 0
    ```
    * Without GPU (add --no-cuda) 
    ```bash
    python -u vbd_imagenet.py --importance-method vbd_ssr --classifier resnet50 --dropout_param_size 56 56 --epochs 300 --lr 0.05 --reg-coef 1e-3 --batch-size 8 --num-samples 1 --gen-model-name LocalMeanInpainter --save-dir ./imgs/0317_local_vbd_sdr_1E-3_56_val_resnet/ --num-imgs 50 --dataset valid/ --data-dir ../imagenet/ --no-cuda
    ```
    * Run FIDO with SSR objectives (vbd_ssr), and with Contextual Attention Inpainter (CAInpainter)
    ```bash
    python -u vbd_imagenet.py --importance-method vbd_sdr --classifier resnet50 --dropout_param_size 56 56 --epochs 300 --lr 0.05 --reg-coef 1e-3 --batch-size 8 --num-samples 1 --gen-model-name CAInpainter --save-dir ./imgs/0317_local_vbd_sdr_1E-3_56_val_resnet/ --num-imgs 50 --dataset valid/ --data-dir ../imagenet/ --no-cuda
    ```
    * Run BBMP with SSR objectives under random inpainting
    ```bash
    python -u vbd_imagenet.py --importance-method bbmp_ssr --classifier resnet50 --dropout_param_size 56 56 --epochs 300 --lr 0.05 --reg-coef 5e-3 --gen-model-name RandomColorWithNoiseInpainter --save-dir ./imgs/0317_random_bbmp_ssr_5E-3_56_val_resnet/ --num-imgs 50 --dataset valid/ --data-dir ../imagenet/ --no-cuda
    ```
    
2. Still have questions?
    * If you still have questions, please first search over closed issues. If the problem is not solved, please open a new issue.

## Visdom

Visualization on [Visdom](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for the saliency map during training is supported. Run visdom server and then add argument --visdom_enabled to view.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.

## Citing
```
@article{chang2018explaining,
  title={Explaining Image Classifiers by Adaptive Dropout and Generative In-filling},
  author={Chang, Chun-Hao and Creager, Elliot and Goldenberg, Anna and and Duvenaud, David},
  journal={arXiv preprint arXiv:1807.08024},
  year={2018}
}
```
