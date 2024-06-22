import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from sklearn import metrics
import os
from utils.custom_dset import CustomDset
from utils.common import logger
import csv
import sys
from sklearn import preprocessing
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data_transforms = {
#     'test': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(30),  # 随机旋转角度范围为±30度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def test(model, model_name, k=0, K=2, types=0, cnv=True):
    model.eval()
    model = model.to(device)

    if cnv:
        cnv_feature=pd.read_csv('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/label/clinic_CPGEA_NEW_an.csv')
        peoples=[i for i in cnv_feature.TCGA_ID]
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()
        cnv_features = min_max_scaler.fit_transform(features)
    
    # testset = CustomDset(os.getcwd() + f'/data/FOXA1ONE_10_TILE_new/test_{k}.csv'.format(k), data_transforms['test'])
    # testset = CustomDset(os.getcwd() + f'/data/FOXA1ONE_20_TILE_new/test_{k}.csv'.format(k), data_transforms['test'])
    testset = CustomDset(os.getcwd() + f'/data/FOXA1ONE_40_TILE_new/test_{k}.csv'.format(k), data_transforms['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=False, num_workers=4)

    ############################保存每张patch的分数###########################################################################
    #
    # # 1. 将每一张 patch 的信息存储到一个列表中
    # patch_info_list = []
    #
    # with torch.no_grad():
    #     for data in testloader:
    #         images, _, names_, images_names = data
    #         X_train_minmax = [cnv_features[:, peoples.index(i)] for i in names_]
    #         outputs = model(images.to(device), torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
    #         probability = F.softmax(outputs, dim=1).data.squeeze()
    #         probs = probability.cpu().numpy()
    #
    #         for i in range(len(images_names)):
    #             patch_name = images_names[i]
    #             prob_0 = probs[i, 0] if probs.ndim == 2 else probs[0]
    #             prob_1 = probs[i, 1] if probs.ndim == 2 else probs[1]
    #
    #             # 提取 WSI 名称
    #             wsi_name = names_[i]
    #
    #             # 存储每一张 patch 的信息
    #             patch_info_list.append({
    #                 'wsi_name': wsi_name,
    #                 'patch_name': patch_name,
    #                 'prob_0': prob_0,
    #                 'prob_1': prob_1
    #             })
    #
    # # 2. 根据 WSI 的名称，将所有的 patch 信息分组
    # grouped_patch_info = {}
    # for patch_info in patch_info_list:
    #     wsi_name = patch_info['wsi_name']
    #     if wsi_name not in grouped_patch_info:
    #         grouped_patch_info[wsi_name] = {'patch_names': [], 'prob_0': [], 'prob_1': []}
    #     grouped_patch_info[wsi_name]['patch_names'].append(patch_info['patch_name'])
    #     grouped_patch_info[wsi_name]['prob_0'].append(patch_info['prob_0'])
    #     grouped_patch_info[wsi_name]['prob_1'].append(patch_info['prob_1'])
    #
    # # 3. 遍历分组后的结果，将每一组的信息写入到一个 CSV 文件中
    # output_folder = '/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/results/allchep/resnet50_mix_64/patches_score'  # 请替换为你的输出文件夹路径
    # os.makedirs(output_folder, exist_ok=True)
    #
    # for wsi_name, patch_info_group in grouped_patch_info.items():
    #     output_csv_path = os.path.join(output_folder, f'{wsi_name}_predictions.csv')
    #     df = pd.DataFrame(patch_info_group)
    #     df.to_csv(output_csv_path, index=False)

    ############################原来的测试代码###########################################################################
    person_prob_dict = dict()
    with torch.no_grad():
        for data in testloader:
            images, labels, names_, images_names = data
            if cnv:
                X_train_minmax = [cnv_features[:,peoples.index(i)] for i in names_]
                outputs = model(images.to(device), torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
            else:
                outputs = model(images.to(device))
            probability = F.softmax(outputs, dim=1).data.squeeze()
            probs = probability.cpu().numpy()
            for i in range(labels.size(0)):
                p = names_[i]
                if p not in person_prob_dict.keys():
                    person_prob_dict[p] = {
                        'prob_0': 0,
                        'prob_1': 0,
                        'label': labels[i].item(),
                        'img_num': 0}
                if probs.ndim == 2:
                    person_prob_dict[p]['prob_0'] += probs[i, 0]
                    person_prob_dict[p]['prob_1'] += probs[i, 1]
                    person_prob_dict[p]['img_num'] += 1
                else:
                    person_prob_dict[p]['prob_0'] += probs[0]
                    person_prob_dict[p]['prob_1'] += probs[1]
                    person_prob_dict[p]['img_num'] += 1

    keys = []
    y_true = []
    y_pred = []
    score_list = []
    score_list0 = []
    score_list1 = []
    # print(names_)

    total = len(person_prob_dict)
    correct = 0
    for key in person_prob_dict.keys():
        keys.append(key)
        predict = 0
        if person_prob_dict[key]['prob_0'] < person_prob_dict[key]['prob_1']:
            predict = 1
        if person_prob_dict[key]['label'] == predict:
            correct += 1
        y_true.append(person_prob_dict[key]['label'])
        score_list.append([person_prob_dict[key]['prob_0'] / person_prob_dict[key]["img_num"],person_prob_dict[key]['prob_1'] / person_prob_dict[key]["img_num"]])
        score_list0.append(person_prob_dict[key]['prob_0'] / person_prob_dict[key]["img_num"])
        score_list1.append(person_prob_dict[key]['prob_1'] / person_prob_dict[key]["img_num"])
        y_pred.append(predict)
        open(f'{model_name}_confusion_matrix_classification_{types}.txt', 'a+').write(
            str(person_prob_dict[key]['label']) + "\t" + str(predict) + '\n')

    # MINE
    final_df = pd.DataFrame({'SLIDE_ID': keys, 'y_true': y_true, 'y_pred': y_pred, 'score_list0': score_list0, 'score_list1': score_list1})

    # final_df.to_csv(os.path.join('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/results/clinicATMNI_10x_res50_isup_sslstm_30',f'fold{k}.csv'))
    # final_df.to_csv(os.path.join('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/results/clinicATMNI_20x_res50_isup_sslstm_30',f'fold{k}.csv'))
    # final_df.to_csv(os.path.join('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/results/clinicATMNI_40x_res50_isup_ss2lstm_30_001',f'fold{k}.csv'))
    final_df.to_csv(os.path.join('/mnt/ai2022_tr/lsm/pathology/genemutation/MMDL-master/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an',f'fold{k}.csv'))


    np.save(os.getcwd()+f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/y_true_{k}.npy', np.array(y_true))
    np.save(os.getcwd()+f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/y_pred_{k}.npy', np.array(y_pred))
    np.save(os.getcwd()+f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/score_{k}.npy', np.array(score_list))
    np.save(os.getcwd() + f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/score0_{k}.npy', np.array(score_list0))
    np.save(os.getcwd() + f'/results/FOXA1_DB_multimodel/FOXA1_WSI_Clinic_HASP_res50_40x_an/score1_{k}.npy', np.array(score_list1))
    logger.info('Accuracy of the network on test images: %d %%' % (100 * correct / total))





    
