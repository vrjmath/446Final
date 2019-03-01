import os
import glob
import random

from torch.utils.data import Dataset
import torch
import nibabel as nib
import numpy as np


class MySet(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, item):
        data_dict = self.data_list[item]
        data_path = data_dict["data"]
        mask_path = data_dict["label"]

        data = nib.load(data_path)
        data = data.get_fdata()

        mask = nib.load(mask_path)
        mask = mask.get_fdata()

        data = self.normalize(data)
        data = data[np.newaxis, :, :, :]
        mask = mask.astype(np.float32)
        # mask = mask[np.newaxis, :, :, :]

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
    data_list = glob.glob(os.path.join(data_path, '*'))

    label_name = 'label.nii'
    data_name = 'image.nii'

    data_list.sort()
    list_all = [{'data': os.path.join(path, data_name), 'label': os.path.join(path, label_name)} for path in data_list]

    cut = int(ratio * len(list_all))
    train_list = list_all[:cut]
    test_list = list_all[cut:]

    random.shuffle(train_list)

    return train_list, test_list
