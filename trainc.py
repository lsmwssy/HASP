import torch
import time
import copy
import sys

from utils.common import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
# from ignite.handlers import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tb = SummaryWriter('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/runs/')


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10, cnv=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if cnv:
        cnv_feature=pd.read_csv('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/label/clinic_CPGEA_NEW_an.csv')
        # cnv_feature = pd.read_csv('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/label/clinic_CPGEA_ETS_i.csv')
        peoples=[i for i in cnv_feature.TCGA_ID]
        # print('people',peoples)
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()
        cnv_features = min_max_scaler.fit_transform(features)

    # new
    # 使用早停策略进行早停
    # early_stopping = EarlyStopping(patience=10, score_function=lambda engine: -engine.state.metrics['val_loss'])
    # early_stopping = ReduceLROnPlateau(optimizer, mode='min', patience=5,factor=0.5, verbose=True)
    # new

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        phase = 'train'
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs_, labels_, names_, img in dataloaders[phase]:
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            # print('names',names_)  #WSI
            # print('img', img)      #patch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                if cnv:
                    X_train_minmax = [cnv_features[:,peoples.index(i)] for i in names_]
                    outputs_ = model(inputs_, torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
                else:
                    outputs_ = model(inputs_)
                _, preds = torch.max(outputs_, 1)
                loss = criterion(outputs_, labels_)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs_.size(0)
            running_corrects += torch.sum((preds == labels_.data).int())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        scheduler.step()

        # # new
        # # 在每个 epoch 结束后调用 evaluate_model 函数
        # validation_loss = evaluate_model(model, criterion, dataloaders['train'])  # 使用训练集作为“伪验证集”
        # # early_stopping(validation_loss, model)
        # early_stopping.step(validation_loss)
        # # 判断是否需要早停
        # # if early_stopping.early_stop:
        # #     logger.info("Early stopping triggered.")
        # #     break
        # if early_stopping.num_bad_epochs > early_stopping.patience:
        #     logger.info("Early stopping triggered.")
        #     break
        # # new

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        tb.add_scalar("Train/Loss", epoch_loss, epoch)
        tb.add_scalar("Train/Accuracy", epoch_acc, epoch)
        tb.flush()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // 3600, (time_elapsed-time_elapsed // 3600) * 60))
    logger.info('Best train Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, tb


# def train_model(k, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10, cnv=True):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     if cnv:
#         cnv_feature = pd.read_csv('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/label/clinic_CPGEA_NEW_i.csv')
#         peoples = [i for i in cnv_feature.TCGA_ID]
#         features = [cnv_feature[i] for i in cnv_feature.columns[1:]]
#         min_max_scaler = preprocessing.MinMaxScaler()
#         cnv_features = min_max_scaler.fit_transform(features)
#
#     for epoch in range(num_epochs):
#         logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
#
#         phase = 'train'
#         model.train()
#
#         running_loss = 0.0
#         running_corrects = 0
#
#         # Iterate over data
#         for inputs_, labels_, names_, _ in dataloaders[phase]:
#             inputs_ = inputs_.to(device)
#             labels_ = labels_.to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward
#             # track history if only in train
#             with torch.set_grad_enabled(phase == 'train'):
#                 if cnv:
#                     X_train_minmax = [cnv_features[:, peoples.index(i)] for i in names_]
#                     outputs_ = model(inputs_, torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
#                 else:
#                     outputs_ = model(inputs_)
#                 _, preds = torch.max(outputs_, 1)
#                 loss = criterion(outputs_, labels_)
#
#                 # Calculate AUC
#                 probs = torch.softmax(outputs_, dim=1)
#                 # auc = roc_auc_score(labels_.cpu().numpy(), probs.cpu().numpy()[:, 1])
#                 auc = roc_auc_score(labels_.detach().cpu().numpy(), probs.detach().cpu().numpy()[:, 1])
#
#                 # backward + optimize only if in training phase
#                 loss.backward()
#                 optimizer.step()
#
#             # statistics
#             running_loss += loss.item() * inputs_.size(0)
#             running_corrects += torch.sum((preds == labels_.data).int())
#
#         epoch_loss = running_loss / dataset_sizes[phase]
#         epoch_acc = running_corrects / dataset_sizes[phase]
#         epoch_auc = auc
#
#         scheduler.step()
#
#         # 构建保存权重的文件名
#         checkpoint_name = f'resnet50_{k}_epoch_{epoch}_loss_{epoch_loss:.4f}_acc_{epoch_acc:.4f}_auc_{epoch_auc:.4f}.pkl'
#         save_dir = os.getcwd() + '/results/allchep/resnet50_isup_hasp_30_64/'
#         checkpoint_path = os.path.join(save_dir, checkpoint_name)
#
#         # 保存权重
#         torch.save(model.state_dict(), checkpoint_path)
#         model.load_state_dict(torch.load(checkpoint_path))
#         logger.info('{} Loss: {:.4f} Acc: {:.4f} Auc: {:.4f}'.format(
#             phase, epoch_loss, epoch_acc, epoch_auc))
#         tb.add_scalar("Train/Loss", epoch_loss, epoch)
#         tb.add_scalar("Train/Accuracy", epoch_acc, epoch)
#         tb.add_scalar("Train/Auc", epoch_auc, epoch)
#         tb.flush()
#
#     return model, tb


# 添加了一个新的函数 evaluate_model，用于在每个 epoch 结束后计算验证集上的损失
def evaluate_model(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    total_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels, _, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_corrects / total_samples

    return average_loss
