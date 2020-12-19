import os
import glob
import random

from torch.utils.data import Dataset
import torch
import nibabel as nib
import numpy as np


class MySet(Dataset):
    def __init__(self, data_list):
        """
        the dataset class receive a list that contain the data item, and each item
        is a dict with two item include data path and label path. as follow:
        data_list = [
        {
        "data":　data_path_1,
        "label": label_path_1,
        },
        {
        "data": data_path_2,
        "label": label_path_2,
        }
        ...
        ]
        """
        self.data_list = data_list

    def __getitem__(self, item):
        data_dict = self.data_list[item]

        data_path = data_dict["data"]
        mask_path = data_dict["label"]

        data = np.load(data_path)[0]

        mask = np.load(mask_path)

        data = self.normalize(data)
        data = data[np.newaxis, :, :, :]
        mask = mask.astype(np.float32)

        mask_tensor = torch.from_numpy(mask)
        data_tensor = torch.from_numpy(data)

        return data_tensor, mask_tensor

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data))/(np.max(data) - np.min(data))
        return data

    def __len__(self):
        return len(self.data_list)


def create_list(data_path, ratio=0.8):
    """
    this function is create the data list and the data is set as follow:
    --data
        --data_1
            image.nii
            label.nii
        --data_2
            image.nii
            label.nii
        ...
    if u use your own data, u can rewrite this function
    """
    filedirTrain = 'C:/Users/Viraj/Documents/UIUC/Sem 3/CS 446/FA20_CS446_Project_Data/data_pub/train/'
    size = int(len(glob.glob1(filedirTrain,"*.npy"))/2)

    train_list = list()
    for i in range(size):
        train_list.append({'data': os.path.join(filedirTrain, '%03d_imgs.npy'%(i+1)), 'label': os.path.join(filedirTrain, '%03d_seg.npy'%(i+1))})

    filedirVal = 'C:/Users/Viraj/Documents/UIUC/Sem 3/CS 446/FA20_CS446_Project_Data/data_pub/validation/'
    size = int(len(glob.glob1(filedirVal,"*.npy"))/2)
    test_list = list()
    for i in range(size):
        test_list.append({'data': os.path.join(filedirVal, '%03d_imgs.npy'%(i+1)), 'label': os.path.join(filedirVal, '%03d_seg.npy'%(i+1))})
    random.shuffle(train_list)
    return train_list, test_list
