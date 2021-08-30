# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import torch
import sklearn.datasets as sklearn_datasets
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
#from models import DataGenerator 

torch.manual_seed(25)
np.random.seed(1000)

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], -1

    def __len__(self):
        return self.data_tensor.size(0)

def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getPlaces365(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'places365-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.Places365(
                root=data_root, split='train-standard', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.Places365(
                root=data_root, split='val', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        ############################### TO BE REMOVED ###################################
        # correct_pred = np.load("models/correct_pred.npz") 
        # pred = correct_pred["predictions"] 
        # correct_pred_index = np.where(pred==True)[0]
        # class_indices = correct_pred_index
        #################################################################################
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getSTL10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'stl10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=data_root, split='train', download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=data_root, split='test', download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_indices(dataset,class_num, num_oods):
    indices =  []
    count = 0
    for i in range(len(dataset.targets)):
        for j in range(len(class_num)):
            if dataset.targets[i] == class_num[j] and count < num_oods:
                indices.append(i)
                count = count + 1
    return indices

# for getting a subset of CIFAR100 containing data for specific classes with the class ids specified in idx
def getSubsetCIFAR100(batch_size, TF, class_labels, num_oods, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    dataset = datasets.CIFAR100(root=data_root, train=True, download=True,transform=TF)
    #print(dataset.classes, dataset.class_to_idx)
    class_indices = get_indices(dataset, class_labels, num_oods)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=False, sampler = torch.utils.data.sampler.SubsetRandomSampler(class_indices), **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=False, sampler = torch.utils.data.sampler.SubsetRandomSampler(class_indices), **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getGerman(batch_size, TF, data_root="datasets/GTSRB-Train/Final_Training/Images", train=True, val=True, **kwargs):
        
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []

    if train:
        dataset = GermanTrafficData(root="datasets/GTSRB-Train/Final_Training/Images", img_size=32, train=True)        

        train_loader = DataLoader(dataset,
                            batch_size= batch_size,
                            shuffle = False,
                            drop_last=False)

        ds.append(train_loader)

    if val:
        dataset = GermanTrafficData(root="datasets/GTSRB-Test/Final_Test/Images", img_size=32, train=False)        

        test_loader = DataLoader(dataset,
                            batch_size= batch_size,
                            shuffle = False,
                            drop_last=False)

        ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getAdvCIFAR10(batch_size, data_root):

    cifar10_adv_data = torch.load(data_root)

    dataset = CustomTensorDataset(cifar10_adv_data)

    test_loader = DataLoader(dataset,
                            batch_size= batch_size,
                            shuffle = False,
                            drop_last=False)
    return test_loader

def getAdvSVHN():
    pass

def getAdvMNIST():
    pass

def getCIFAR10InconrrectlyPred(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        ############################### TO BE REMOVED ###################################
        correct_pred = np.load("models/correct_pred.npz") 
        pred = correct_pred["predictions"] 
        incorrect_pred_index = np.where(pred==False)[0]
        class_indices = incorrect_pred_index
        #################################################################################
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False,  sampler = torch.utils.data.sampler.SubsetRandomSampler(class_indices), **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'stl10':
        train_loader, test_loader = getSTL10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        print("target_data_type == svhn")
        train_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'subset_cifar100':
        train_loader, test_loader = getSubsetCIFAR100(batch_size=batch_size, TF=input_TF, class_labels=kwargs['idx'], num_oods=kwargs['num_oods'], data_root=dataroot, num_workers=1)
        # for idx, (data, target) in enumerate(test_loader):
    elif data_type == 'german':
        train_loader, test_loader = getGerman(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)       
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        train_loader = test_loader
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
        train_loader = test_loader
    elif data_type == 'mnist':
        print("in-dist= ", data_type)
        # download and create datasets
        dataroot = os.path.expanduser(os.path.join(dataroot, 'mnist_data'))
        train_dataset = datasets.MNIST(dataroot, 
                               train=True, 
                               transform=input_TF,
                               download=True)

        test_dataset = datasets.MNIST(dataroot, 
                               train=False, 
                               transform=input_TF)
        
        train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=False)

        test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=batch_size, 
                          shuffle=False)

    elif data_type == 'toy_data': # half_moon dataset

        data,target = sklearn_datasets.make_moons(200*2,noise=.05, random_state=200, shuffle=False)
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        train_loader = DataGenerator(data,target,batch_size=batch_size)

        test_data,test_target = sklearn_datasets.make_moons(200*2,noise=0.05,random_state=500, shuffle=False)
        test_data = test_data.astype(np.float32)
        test_data = torch.from_numpy(test_data)
        test_target = torch.from_numpy(test_target)
        test_loader = DataGenerator(test_data,test_target,batch_size=batch_size)

    # elif data_type == 'blob': # blob dataset as OODs for half_moon toy dataset. This is called from ADV_Samples.py for LID scores
    #     test_data, test_target = sklearn_datasets.make_blobs(n_samples=[120,60,30],centers = [[0,2],[1.95,2],[0.5,0.25]],cluster_std=[0.1,0.03,0.02],shuffle=False,random_state=200)
    #     test_data = test_data.astype(np.float32)
    #     test_data = torch.from_numpy(test_data)
    #     test_target = torch.from_numpy(test_target)
    #     test_loader = DataGenerator(test_data,test_target,batch_size=batch_size)
    #     train_loader = test_loader

    return train_loader, test_loader

def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot, **kwargs):
    print("data_type: ", data_type)
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    if data_type == 'incorrect_pred_cifar10':
        _, test_loader = getCIFAR10InconrrectlyPred(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'adv_cifar10':
        # adv examples are generated after applying the required transformation
        test_loader = getAdvCIFAR10(batch_size=batch_size, data_root="adv_output/DenseNet3_svhn/adv_data_DenseNet3_svhn_FGSM.pth")
    elif data_type == 'adv_svhn':
        test_loader = getAdvSVHN()
    elif data_type == 'adv_mnist':
        test_loader = getAdvMNIST()
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'places365':
        _, test_loader = getPlaces365(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'stl10':
        _, test_loader = getSTL10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'subset_cifar100':
        print("kwargs ", kwargs)
        _, test_loader = getSubsetCIFAR100(batch_size=batch_size, TF=input_TF, class_labels=kwargs['idx'], num_oods=kwargs['num_oods'], data_root=dataroot, num_workers=1)
        # for idx, (data, target) in enumerate(test_loader):
    elif data_type == 'german':
        _, test_loader = getGerman(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)       
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'kmnist':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'kmnist_data'))
        test_dataset = datasets.KMNIST(root=dataroot, 
                                train=False,
                                download=True, 
                                transform=input_TF)

        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=batch_size, 
                                shuffle=False)
    elif data_type == 'fashion-mnist':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'fashion-mnist_data'))
        test_dataset = datasets.FashionMNIST(dataroot, 
                                train=False, 
                                download=True,
                                transform=input_TF)

        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=batch_size, 
                                shuffle=False)
    elif data_type == 'blob': # blob dataset as OODs for half_moon toy dataset
        #test_data, test_target = sklearn_datasets.make_blobs(n_samples=[200,200,20,30],centers = [[2.4,0.5],[1,-0.99],[1.8,0.3],[0.5,0.4]],cluster_std=[0.05,0.03,0.02,0.05],shuffle=True,random_state=200) #sec 4 orig
        # test_data, test_target = sklearn_datasets.make_blobs(n_samples=[100,200,200],centers = [[0.4,0.4],[1.7,0.8],[-1,1.5]],cluster_std=[0.0125,0.02,0.02],shuffle=True,random_state=200) #sec 3 and sec 4- PREV FINAL
        test_data, test_target = sklearn_datasets.make_blobs(n_samples=[100,200,200],centers = [[0.4,0.4],[1.7,0.9],[-1,1.5]],cluster_std=[0.0125,0.015,0.02],shuffle=True,random_state=200) #sec 3 and sec 4- NEW FINAL
        #test_data, test_target = sklearn_datasets.make_blobs(n_samples=[200,200,200,200],centers = [[2.4,0.5],[-1,1.5],[0.5,0.4],[1,-0.8]],cluster_std=[0.05,0.02,0.03,0.03],shuffle=True,random_state=200) #sec 4 new
        test_data = test_data.astype(np.float32)
        test_data = torch.from_numpy(test_data)
        test_target = torch.from_numpy(test_target)
        test_loader = DataGenerator(test_data,test_target,batch_size=batch_size)
    elif data_type == 'grid':
        x_min,x_max = -3,3
        y_min,y_max = -3,3
        X,Y = np.arange(x_min,x_max,.1),np.arange(y_min,y_max,.1)
        XX,YY = np.meshgrid(X,Y)
        test_data = (np.c_[XX.ravel(),YY.ravel()]).astype(np.float32)
        test_data = torch.from_numpy(test_data)
        test_target = torch.zeros(len(test_data))
        test_loader = DataGenerator(test_data,test_target,batch_size=batch_size)

    return test_loader


