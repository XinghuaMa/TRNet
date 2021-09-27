import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils import data
from sklearn.model_selection import train_test_split

from config import opt


class cubic_sequence_data(data.Dataset):
    def __init__(self, raw_dataset_root, num_fold=0, pattern='train'):

        cross_val = opt.num_cross_fold
        raw_dataset = np.load(raw_dataset_root)
        split_label = int(opt.cube_side_length ** 3)

        eval_begin, eval_end = max(0, int(raw_dataset.shape[0] / cross_val * num_fold)), min(raw_dataset.shape[0], int(
            raw_dataset.shape[0] / cross_val * (num_fold + 1)))

        if pattern=='train':
            front_data, behind_data = raw_dataset[:eval_begin], raw_dataset[eval_end:]
            self.data_array = np.concatenate((front_data, behind_data), axis=0)
        elif pattern=='eval':
            self.data_array = raw_dataset[eval_begin: eval_end]
        else :
            self.data_array = raw_dataset

        self.sequence_array, self.label_array = self.data_array[:, :, :split_label], self.data_array[:, :, split_label]
        return

    def __getitem__(self, index):
        sequence_image = torch.tensor(self.sequence_array[index], dtype=torch.float32)
        sequence_image = sequence_image.view(opt.cubic_sequence_length, opt.cube_side_length, opt.cube_side_length,
                                             opt.cube_side_length)
        sequence_label = torch.tensor(self.label_array[index], dtype=torch.long)
        return sequence_image, sequence_label

    def __len__(self):
        return len(self.data_array)
