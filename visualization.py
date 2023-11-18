#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
# import umap
# import umap.umap_ as umap
from sklearn import datasets
from sklearn import manifold

import hydra
import torch
import copy

from torchvision import transforms as transforms
from dataset.cifar import *
from dataset.svhn import *
from dataset.stl10 import *
from models.WideResNet import *
from models.shakeshake import *

# def log_params_from_omegaconf_dict(params):
#     for param_name, element in params.items():
#         _explore_recursive(param_name, element)

@hydra.main('config/visualization.yaml') 
def main(cfg):
    # log_params_from_omegaconf_dict(cfg)

    device = torch.device('cuda')
    # load model
    model = WideResNet(depth=cfg.model.depth, num_classes=10, widen_factor=cfg.model.widen_factor, drop_rate=0.0).to(device)
    # model.load_state_dict(copy.deepcopy(torch.load(cfg.PATH)))
    model = torch.load(cfg.PATH)
    model.eval()

    test_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])

    if cfg.data.name == "cifar10":
        test_set = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif cfg.data.name == "cifar100":
        test_set = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    elif cfg.data.name == "svhn":
        test_set = SVHN(root='./data', split='test', download=True, transform=test_transform)
    elif cfg.data.name == "stl10":
        test_set = STL10(root='./data', split='test', download=True, transform=test_transform)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.data.batch_size, shuffle=False)

    X, Y = None, None
    with torch.no_grad():
        for batch_num, (data, _, target) in enumerate(test_loader):
            data, target = data[0].to(device), target.to(device)
            output = model(data)
            if X == None:
                X = output.cpu()
                Y = target.cpu()
            else:
                X = torch.cat([X, output.cpu()])
                Y = torch.cat([Y, target.cpu()])

    X, y = X.data, Y.data

    # dimension reduction
    # mapper = umap.UMAP(random_state=0)
    # embedding = mapper.fit_transform(X)

    # # 次元削減する
    mapper = manifold.TSNE(random_state=0)
    embedding = mapper.fit_transform(X)

    # plot results at 2d dim space
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    for n in np.unique(y):
        plt.scatter(embedding_x[y == n],
                    embedding_y[y == n],
                    label=n)
    # グラフを表示する
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig("foo.png")


if __name__ == '__main__':
    main()