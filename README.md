# Detecting OODs as datapoints with High Uncertainty
This code is for the [paper](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-081.pdf), "Detecting OODs as datapoints with High Uncertainty", presented at ICML 2021 Workshop on uncertainty & Robustness in Deep Learning. 

It is built on top of the [deep_Mahalanobis_detector code](https://github.com/pokaxpoka/deep_Mahalanobis_detector).

## Requirements
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries Pytorch package to be installed:

1. [Pytorch](https://pytorch.org/)
2. [scipy](https://github.com/scipy/scipy)
3. [scikit-learn](https://scikit-learn.org/stable/)

## Downloading OOD datasets
1. [Tiny ImageNet (resized)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
2. [LSUN (resized)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
