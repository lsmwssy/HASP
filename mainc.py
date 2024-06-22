from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.common import logger
from utils.custom_dset import CustomDset
# from utils.analytics import draw_roc, draw_roc_for_multiclass

# import train_test_splitter
from trainc import train_model
from testc import test

# from Net import Net, Cnn_With_Clinical_Net  #for VIT
# from myNetc4 import Net, Cnn_With_Clinical_Net
from myNet import Net  #for EFF
from myNetc4 import Cnn_With_Clinical_Net


plt.ion()   

# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(224),
#         # transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(30),  # 随机旋转角度范围为±30度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print("using {} device.".format(device))

def generative_model(model, k, cnv=True):
    # image_datasets = {x: CustomDset(os.getcwd()+f'/data/FOXA1ONE_10_TILE_new/{x}_{k}.csv',data_transforms[x]) for x in ['train']}
    # image_datasets = {x: CustomDset(os.getcwd()+f'/data/FOXA1ONE_20_TILE_new/{x}_{k}.csv',data_transforms[x]) for x in ['train']}
    image_datasets = {x: CustomDset(os.getcwd() + f'/data/FOXA1ONE_40_TILE_new/{x}_{k}.csv', data_transforms[x]) for x in ['train']}
    # image_datasets = {x: CustomDset(os.getcwd() + f'/data/isup_fx/FOXA1_CPGEA_isup45_40/{x}_{k}.csv', data_transforms[x]) for x in ['train']}


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                    shuffle=True, num_workers=4) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes

    logger.info(f'model {model} / 第 {k+1} 折')

    # available_policies = {"resnet18": models.resnet18,"resnet50": models.resnet50, "resnet101": models.resnet101, "vgg16": models.vgg16, "vgg19": models.vgg19,
    #         "alexnet": models.alexnet, "inception": models.inception_v3}
    available_policies = {"resnet18": models.resnet18, "resnet50": models.resnet50, "resnet101": models.resnet101,
                          "vgg16": models.vgg16, "vgg19": models.vgg19,
                          "alexnet": models.alexnet, "inception": models.inception_v3,
                          "shufflenetv2": models.shufflenet_v2_x2_0, "densenet": models.densenet169,
                          "efficientnet": models.efficientnet_b3, "vision_transformer": models.vit_b_16}

    # model_ft = available_policies[model](weights=True)
    model_ft = available_policies[model](pretrained=True)

    if cnv:
        model_ft = Cnn_With_Clinical_Net(model_ft)
    else:
        model_ft = Net(model_ft)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model_ft, tb = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
        dataset_sizes, num_epochs=30, cnv=cnv)
    tb.close()      # tb：tensorboard

    # save_model = os.getcwd() + f'/results/clinicATMNI_10x_res50_isup_sslstm_30/{model}_{k}'
    # save_model = os.getcwd() + f'/results/clinicATMNI_20x_res50_isup_sslstm_30/{model}_{k}'
    save_model = os.getcwd() + f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/{model}_{k}'

    if cnv:
        save_model = save_model + '_cnv'
    save_model = save_model + '.pkl'

    torch.save(model_ft, save_model)




def main(ocs, classification, K, cnv):
     
    # train_test_splitter.main("/media/zw/Elements1/tiles_cn", "/home/xisx/tmbpredictor/labels/uteri.csv")



    # # resnet50
    for k in range(K):
        k+=2
        generative_model("resnet50", k, cnv=cnv)
        path = os.getcwd() + f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/resnet50_{k}'

        if cnv:
            path = path + '_cnv'
        model_ft = torch.load(path + '.pkl')
        test(model_ft, "resnet50", k)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--classification', type=int, default=2)  
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--cnv', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')  # new
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    origirn_classfication_set = None


    main(origirn_classfication_set, args.classification, args.K, args.cnv)
