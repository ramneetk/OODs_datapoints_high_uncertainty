# Detecting OODs as datapoints with High Uncertainty
This code is for the [paper](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-081.pdf), "Detecting OODs as datapoints with High Uncertainty", presented at ICML 2021 Workshop on uncertainty & Robustness in Deep Learning. 

It is built on top of the [deep_Mahalanobis_detector code](https://github.com/pokaxpoka/deep_Mahalanobis_detector).

## Requirements
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries Pytorch package to be installed:

1. [Pytorch](https://pytorch.org/)
2. [scipy](https://github.com/scipy/scipy)
3. [scikit-learn](https://scikit-learn.org/stable/)

## Downloading OOD datasets (from deep_Mahalanobis_detector code directory)
mkdir data and download the following OOD datasets in ./data
1. [Tiny ImageNet (resized)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
2. [LSUN (resized)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

## Downloading pre-trained models
mkdir pre_trained and download the .pth files in ./pre_trained from this [link.](https://drive.google.com/drive/folders/1yuiTOgKgPsLGNJwoRckSAHXBZ_AOBlGI?usp=sharing)

## Baseline and ODIN (from deep_Mahalanobis_detector code directory) CHECK FIRST
1. Baseline - python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type resnet34 --gpu 0
2. ODIN - after modifying T and M

## Mahalanobis results


## Generating OOD and ID features
1. Edit configs/generation_config.yaml for command line arguments pretrained_model_path, net_type, out_dist_list etc. in model_params, gpu no. in trainer_params and outf as output directory for storing the generated features in logging_params
2. python generation.py

## Training logistic regression on OOD and ID features for OOD detection
1. Edit configs/regression_config.yaml with dataset_list as the ID dataset, out_dist_list in model_params and outf in the logging_params as the same outf in generation_config.yaml
2. python regression.py
