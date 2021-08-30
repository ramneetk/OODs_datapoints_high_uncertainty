# python resnet.py --optim $$ --dataAug $$ --device $cuda:<>$ --niter $$

'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import os
from datetime import datetime 

#import argparse

#parser = argparse.ArgumentParser()

#parser.add_argument('--optim', default='SGD', help='ADAM/SGD')
#parser.add_argument('--dataAug', type=int, default=1, help='0 for no data augmentation and 1 for data augmentation with transformations')
#parser.add_argument('--device', default='cuda:0')
#parser.add_argument('--niter', type=int, default=350, help="No. of training epochs")
#parser.add_argument('--resnetType', type=int, default=18, help='18/34/50/101/152')

#args = parser.parse_args()
#print(args)

#device = args.device

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        ####### Features from here
        y = self.linear(out)
        return y
    
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        #out_list.append(out)
        out = self.layer1(out)
        #out_list.append(out)
        out = self.layer2(out)
        #out_list.append(out)
        out = self.layer3(out)
        #out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate
    
def ResNet18(num_c=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_c)

def ResNet34(num_c=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())

# def train(train_loader, model, criterion, optimizer, device):
#     '''
#     Function for the training step of the training loop
#     '''

#     model.train()
#     running_loss = 0
    
#     for X, y_true in train_loader:

#         optimizer.zero_grad()
        
#         X = X.to(device)
#         y_true = y_true.to(device)
    
#         # Forward pass
#         y_hat = model(X) 
#         loss = criterion(y_hat, y_true) 
#         running_loss += loss.item() * X.size(0)

#         # Backward pass
#         loss.backward()
#         optimizer.step()
        
#     epoch_loss = running_loss / len(train_loader.dataset)
#     return model, optimizer, epoch_loss

# def validate(valid_loader, model, criterion, device):
#     '''
#     Function for the validation step of the training loop
#     '''
   
#     model.eval()
#     running_loss = 0
    
#     for X, y_true in valid_loader:
    
#         X = X.to(device)
#         y_true = y_true.to(device)

#         # Forward pass and record loss
#         y_hat = model(X) 
#         loss = criterion(y_hat, y_true) 
#         running_loss += loss.item() * X.size(0)

#     epoch_loss = running_loss / len(valid_loader.dataset)
        
#     return model, epoch_loss

# def get_accuracy(model, data_loader, device):
#     '''
#     Function for computing the accuracy of the predictions over the entire data_loader
#     '''
    
#     # correct_pred will store 1 for correct pred and 0 for wring pred
#     correct_pred = []

#     correct = 0 
#     n = 0
    
#     with torch.no_grad():
#         model.eval()
#         for X, y_true in data_loader:

#             X = X.to(device)
#             y_true = y_true.to(device)


#             output = model(X)
#             pred = output.data.max(1)[1]

#             equal_flag = pred.eq(y_true.to(device)).cpu()
#             correct += equal_flag.sum()
#             correct_pred.append(equal_flag.numpy().flatten())
#             n += y_true.size(0)

#     correct_pred = np.concatenate(correct_pred)
#     #print("correct_pred shape: ", correct_pred.shape)
#     np.savez("correct_pred.npz", predictions=correct_pred)
#     return correct.float() / n

# def training_loop(model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs, device, print_every=1):
#     '''
#     Function defining the entire training loop
#     '''
    
#     # set objects for storing metrics
#     best_loss = 1e10
#     train_losses = []
#     valid_losses = []
 
#     # Train model
#     for epoch in range(0, epochs):
#         # training
#         model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
#         train_losses.append(train_loss)

#         # validation
#         with torch.no_grad():
#             model, valid_loss = validate(valid_loader, model, criterion, device)
#             valid_losses.append(valid_loss)

#         if epoch % print_every == (print_every - 1):
            
#             train_acc = get_accuracy(model, train_loader, device=device)
#             valid_acc = get_accuracy(model, valid_loader, device=device)
                
#             print(f'{datetime.now().time().replace(microsecond=0)} --- '
#                   f'Epoch: {epoch}\t'
#                   f'Train loss: {train_loss:.4f}\t'
#                   f'Valid loss: {valid_loss:.4f}\t'
#                   f'Train accuracy: {100 * train_acc:.2f}\t'
#                   f'Valid accuracy: {100 * valid_acc:.2f}')
        
#         if scheduler:
#             scheduler.step()
    
#     return model

# def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
#     data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     ds = []
#     if train:
#         train_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(
#                 root=data_root, train=True, download=True,
#                 transform=TF[0]),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         ds.append(train_loader)
#     if val:
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(
#                 root=data_root, train=False, download=True,
#                 transform=TF[1]),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         ds.append(test_loader)
#     ds = ds[0] if len(ds) == 1 else ds
#     return ds

# def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
#     data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     def target_transform(target):
#         new_target = target - 1
#         if new_target == -1:
#             new_target = 9
#         return new_target

#     ds = []
#     if train:
#         train_loader = torch.utils.data.DataLoader(
#             datasets.SVHN(
#                 root=data_root, split='train', download=True,
#                 transform=TF[0],
#             ),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         ds.append(train_loader)

#     if val:
#         test_loader = torch.utils.data.DataLoader(
#             datasets.SVHN(
#                 root=data_root, split='test', download=True,
#                 transform=TF[1],
#             ),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         ds.append(test_loader)
#     ds = ds[0] if len(ds) == 1 else ds
#     return ds

# def get_acc_pretrained_model():
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     _, test_loader = getCIFAR10(batch_size=100, TF=[transform_train, transform_test], data_root='./data', num_workers=1)

#     if args.resnetType == 18:
#         model = ResNet18()
#     elif args.resnetType == 34:
#         model = ResNet34()
#     elif args.resnetType == 50:
#         model = ResNet50()
#     elif args.resnetType == 101:
#         model = ResNet101()
#     elif args.resnetType == 152:
#         model = ResNet152()
    
#     model=model.to(device)
#     model.load_state_dict(torch.load('../pre_trained/resnet{}_dataAug_{}_{}_cifar10.pth'.format(args.resnetType, args.dataAug, args.optim), map_location = device))

#     get_accuracy(model, test_loader, device)
#     print("Test accuracy of Resnet18 on CIFAR10 is: ", get_accuracy(model, test_loader, device))

# if __name__ == "__main__":

#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])    

#     if args.dataAug == 1:
#         train_loader, test_loader = getCIFAR10(batch_size=100, TF=[transform_train, transform_test], data_root='./data', num_workers=1)
#     else:
#         train_loader, test_loader = getCIFAR10(batch_size=100, TF=[transform_test, transform_test], data_root='./data', num_workers=1)
    
#     if args.resnetType == 18:
#         model = ResNet18()
#     elif args.resnetType == 34:
#         model = ResNet34()
#     elif args.resnetType == 50:
#         model = ResNet50()
#     elif args.resnetType == 101:
#         model = ResNet101()
#     elif args.resnetType == 152:
#         model = ResNet152()

#     model=model.to(device)

#     if args.optim == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr=0.1,
#                         momentum=0.9, weight_decay=5e-4)
#         scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)
#     else:
#         optimizer = optim.Adam(model.parameters())
#         scheduler = None

#     criterion = nn.CrossEntropyLoss()
#     #model = training_loop(model, criterion, optimizer, scheduler, train_loader, test_loader, epochs=args.niter, device=device)
#     #torch.save(model.state_dict(),'../pre_trained/resnet{}_dataAug_{}_{}_cifar10.pth'.format(args.resnetType, args.dataAug, args.optim))

#     # for getting accuracy on test dataset
#     get_acc_pretrained_model()
