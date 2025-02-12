from torch.utils.data import Dataset
from PIL import Image
import os
import pathlib
from torch import from_numpy
import re
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomDset(Dataset):
    def __init__(self,
                #  img_path,
                 csv_path,
                 img_transform=None,
                 target_transform=None,
                 loader=default_loader):
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            # mine
            img_list = [i[:-3:] for i in lines]
            # mine
            self.img_list = [
                # os.path.join(img_path, re.split(',|\\n', i)[0]) for i in lines
                # re.split(',|\\n', i)[0] for i in lines
                # i[:-3:] for i in lines
                i.strip('"') for i in img_list
            ]
            print(self.img_list[2])
            # self.label_list = [int(re.split(',|\\n', i)[1]) for i in lines]
            self.label_list = [int(re.split(',|\\n', i)[2]) for i in lines]
            # print(self.label_list)
            self.classes = sorted(list(set([re.split(',|\\n', i)[2] for i in lines])))
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = self.loader(img_path)
        people = pathlib.Path(img_path).parent.stem
        name = pathlib.Path(img_path).stem
        if self.img_transform is not None:
            img = self.img_transform(img)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        # label = Tensor(label)
        return img, label, people, name

    def __len__(self):
        return len(self.label_list)




#new
class CustomDsetco(Dataset):
    def __init__(self,
                 csv_path,
                 target_transform=None):
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            # self.clinical_data_list = [[float(val) for val in re.split(',|\\n', i)[0:-2]] for i in lines]  # 假设倒数第一个是标签，倒数第二个是名称
            self.clinical_data_list = [int(re.split(',|\\n', i)[1]) for i in lines]
            self.label_list = [int(re.split(',|\\n', i)[2]) for i in lines]
            self.name_list = [str(re.split(',|\\n', i)[0]) for i in lines]
            # self.classes = sorted(list(set([re.split(',|\\n', i)[2] for i in lines])))
        self.target_transform = target_transform

    def __getitem__(self, index):
        clinical_data = self.clinical_data_list[index]
        label = self.label_list[index]
        name = self.name_list[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return clinical_data, label, name

    def __len__(self):
        return len(self.label_list)