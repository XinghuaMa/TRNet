import os
import cv2
import nrrd
import torch
import random
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from matplotlib import pyplot as plt
from skimage.transform import resize
from torchvision import transforms as T

from config import opt
import data_maker as dm

def number_parameters(Net, type_size=8):
    para = sum([np.prod(list(p.size())) for p in Net.parameters()])
    return para / 1024 * type_size / 1024

class Counter:
    def __init__(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return

    def updata(self, value, num_updata=1):
        self.count += num_updata
        self.sum += value * num_updata
        self.avg = self.sum / self.count
        return

    def clear(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return


def get_root_file(root_file):
    list_file = os.listdir(root_file)
    list_file.sort(key=lambda x: str(x.split('.')[0]))
    return list_file


def create_root(root_name):
    if os.path.exists(root_name):
        os.makedirs(root_name)
    return


def read_txt_file(txt_root):
    txt_file = open(txt_root).read()
    data_list = []
    for row in txt_file.split("\n"):
        if row != '':
            data_list += [row.split(" ")]
    return data_list


def add_window(image, WL=200, WW=1000):
    WLL = WL - (WW / 2)
    image = (image - WLL) / WW * 255
    image[image < 0] = 0
    image[image > 255] = 255
    image2 = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
    image2[:, :, :] = image[:, :, :]
    return image2


def read_dicom(root_dicom, dim_change=False, window_add=False):
    list_dicom = get_root_file(root_dicom)
    num_dicom = len(list_dicom)
    dicom_data = np.zeros([num_dicom, 512, 512])
    for i in range(num_dicom):
        file_root = root_dicom + list_dicom[i]
        dcm = pydicom.dcmread(file_root)
        matrix = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        dicom_data[i, :, :] = matrix
    if dim_change:
        dicom_data = dicom_data.transpose((2, 1, 0))
    if window_add:
        dicom_data = add_window(dicom_data)
    return dicom_data


def info_data(matrix_data):
    type_data = type(matrix_data)
    if type(matrix_data) is torch.Tensor:
        if matrix_data.is_cuda:
            matrix_data = matrix_data.detach().cpu()
        matrix_data = np.array(matrix_data)
    print('data_type/dtype/shape: ', type_data, matrix_data.dtype, matrix_data.shape)
    print('min/max: ', np.min(matrix_data), np.max(matrix_data), 'num</=/>zero: ', np.sum(matrix_data < 0),
          np.sum(matrix_data == 0), np.sum(matrix_data > 0))
    return


def connect_volume_views(volume_views):
    ret_volume_views = volume_views[0]
    for i in range(1, 4):
        ret_volume_views = np.concatenate((ret_volume_views, volume_views[i]), axis=0)
    return ret_volume_views


def check_cubic_sequence(cubic_sequence):
    ret_cubic_sequence = connect_volume_views(longitudinal_mpr(cubic_sequence[0]))
    for i in range(1, cubic_sequence.shape[0]):
        tmp_volume_views = connect_volume_views(longitudinal_mpr(cubic_sequence[i]))
        ret_cubic_sequence = np.concatenate((ret_cubic_sequence, tmp_volume_views), axis=1)
    return ret_cubic_sequence


def get_world_position(dicom_root):
    return pydicom.dcmread(dicom_root)[0X0020, 0X0032].value
    # dicom_position = pydicom.dcmread(dicom_root)[0X0020, 0X0032].value
    # return np.array([dicom_position[0], dicom_position[1], dicom_position[2]])


def world2voxel(centerpoint, world_position, label_threshold=0):
    return np.array([int((float(centerpoint[2]) - world_position[2]) / (world_position[2] * 2)),
                     int((float(centerpoint[1]) - world_position[1]) / (world_position[0] * 2)),
                     int((float(centerpoint[0]) - world_position[0]) / (world_position[0] * 2)),
                     int(1 if int(centerpoint[6]) > label_threshold else 0)], dtype=np.int16)


def deduplication(center_info):
    ret_center_info, last_point = [center_info[0]], [center_info[0, 0], center_info[0, 1], center_info[0, 2]]
    for i in range(1, center_info.shape[0]):
        now_point = [center_info[i, 0], center_info[i, 1], center_info[i, 2]]
        if last_point != now_point:
            ret_center_info += [center_info[i]]
            last_point = now_point
    return np.array(ret_center_info, dtype=np.int16)


def calculate_distance(point1, point2):
    tmp1 = (point1[0] - point2[0]) * (point1[0] - point2[0])
    tmp2 = (point1[1] - point2[1]) * (point1[1] - point2[1])
    tmp3 = (point1[2] - point2[2]) * (point1[2] - point2[2])
    return (tmp1 + tmp2 + tmp3) ** 0.5


def check_neighbor(point1, point2):
    move = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
                     [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]],
                    dtype=int)
    for j in range(26):
        if point1[0] - point2[0] == move[0, j] and point1[1] - point2[1] == move[1, j] and point1[2] - point2[2] == \
                move[2, j]:
            return True
    return False


def connection_point(point1, point2):
    move = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
                     [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]],
                    dtype=int)
    add_point, tmp_bestp = [], point1
    while (check_neighbor(tmp_bestp, point2) == False):
        tmp_direction, tmp_mindis = -1, -1
        for i in range(26):
            tmp_point = [tmp_bestp[0] + move[0, i], tmp_bestp[1] + move[1, i], tmp_bestp[2] + move[2, i]]
            tmp_distance = calculate_distance(tmp_point, point2)
            if tmp_mindis == -1 or tmp_distance < tmp_mindis:
                tmp_direction, tmp_mindis = i, tmp_distance
        tmp_bestp = [tmp_bestp[0] + move[0, tmp_direction], tmp_bestp[1] + move[1, tmp_direction],
                     tmp_bestp[2] + move[2, tmp_direction]]
        add_point += [tmp_bestp]
    return add_point


def center_continuity(center_info):
    last_point = center_info[0].tolist()
    ret_center_info = [last_point]
    for i in range(1, center_info.shape[0]):
        now_point = center_info[i].tolist()
        if not check_neighbor(now_point[:3], last_point[:3]):
            add_point = connection_point(last_point[:3], now_point[:3])
            for tmp_point in add_point:
                ret_center_info += [[tmp_point[0], tmp_point[1], tmp_point[2], max(now_point[3], last_point[3])]]
        ret_center_info += [now_point]
        last_point = now_point
    return np.array(ret_center_info, dtype=np.int16)


def voxel_position(centerline, world_position):
    aaa = np.array(centerline, dtype=np.float)
    ret_centerline = np.zeros([len(centerline), 4], dtype=np.int16)
    for i in range(len(centerline)):
        ret_centerline[i] = world2voxel(centerline[i], world_position)
    return center_continuity(deduplication(ret_centerline))


def check_continuity(center_info):
    last_point = center_info[0].tolist()
    for i in range(1, center_info.shape[0]):
        now_point = center_info[i].tolist()
        if not check_neighbor(now_point[:3], last_point[:3]):
            return False
        last_point = now_point
    return True

def check_rationality(dicom, center_info):
    if not check_continuity(center_info):
        return False
    if np.max(center_info[:, 0]) >= dicom.shape[0] or np.max(center_info[:, 1]) >= dicom.shape[1] or np.max(
            center_info[:, 2]) >= dicom.shape[2]:
        return False
    if np.sum(center_info[:, 3]) == 0:
        return False
    return True

def longitudinal_mpr(mpr_volume):
    ret_mpr_image = np.zeros([4, mpr_volume.shape[0], mpr_volume.shape[1]], dtype=np.uint8)
    for i in range(4):
        if i == 0 :
            ret_mpr_image[i] = mpr_volume[:, int(mpr_volume.shape[1] / 2), :]
        elif i == 1 :
            ret_mpr_image[i] = mpr_volume[:, :, int(mpr_volume.shape[2] / 2)]
        else :
            select_x, selece_y = 0, 0 if i == 2 else mpr_volume.shape[2] - 1
            for j in range(mpr_volume.shape[1]):
                ret_mpr_image[i, :, j] = mpr_volume[:, select_x, selece_y]
                select_x += 1
                selece_y = selece_y + 1 if i == 2 else selece_y - 1
    return ret_mpr_image