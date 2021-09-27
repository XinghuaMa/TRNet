import random
import numpy as np
import cv2

from config import opt

def random_move(len_mpr, now_point) :
    move = np.array([[1, -1, 0, 0, 0, 0],
                     [0, 0, 1, -1, 0, 0],
                     [0, 0, 0, 0, 1, -1]],
                    dtype=np.int16)
    now_point[0] = min(len_mpr, max(0, now_point[0] - 1))
    ret_point, num_move = [now_point[0], now_point[1], now_point[2]], random.randint(0, 3)
    for i in range(num_move) :
        opt_move = random.randint(0, 5)
        for j in range(3) :
            ret_point[j] += move[j, opt_move]
    ret_point[0] = min(len_mpr, max(0, ret_point[0] - 1))
    return ret_point

def get_voxel_cude(mpr_volume, position) :
    cude_size_sub, cude_size_add = (opt.cube_side_length - 1) / 2, (opt.cube_side_length + 1) / 2
    ret_voxel_cude = np.zeros([opt.cube_side_length, opt.cube_side_length, opt.cube_side_length], dtype=mpr_volume.dtype)
    cupl, voxl = np.zeros([3], dtype=np.int), np.zeros([3], dtype=np.int)
    cupr, voxr = cupl.copy(), voxl.copy()
    for i in range(3) :
        if position[i] - cude_size_sub < 0:
            voxl[i], cupl[i] = cude_size_sub - position[i], 0
        else :
            voxl[i], cupl[i] = 0, position[i] - cude_size_sub
        if position[i] + cude_size_add > mpr_volume.shape[i] :
            voxr[i], cupr[i] = cude_size_sub - position[i] + mpr_volume.shape[i], mpr_volume.shape[i]
        else :
            voxr[i], cupr[i] = opt.cube_side_length, position[i] + cude_size_add
    ret_voxel_cude[voxl[0]:voxr[0], voxl[1]:voxr[1], voxl[2]:voxr[2]] = mpr_volume[cupl[0]:cupr[0], cupl[1]:cupr[1],
                                                                        cupl[2]:cupr[2]]
    return ret_voxel_cude

def cube_spin(select_cude) :
    spin_num = random.randint(0, 3)
    for i in range(spin_num):
        select_cude = np.rot90(select_cude.swapaxes(0, 2)).swapaxes(0, 2)
    return select_cude

from skimage.exposure import rescale_intensity

def image_transforms(image_data):
    image_data = image_data.astype(np.float64)

    p5, p95 = np.percentile(image_data, 5), np.percentile(image_data, 95)
    image_data = rescale_intensity(image_data, out_range=(p5, p95))

    # normalization operation
    image_mean = np.mean(image_data)
    image_std = np.std(image_data)
    image_data = (image_data - image_mean) / image_std
    return image_data

def sqcuence_maker(mpr_volume, mpr_label, mpr_proportion, interval, voxel_bias = 0) :
    mpr_volume = image_transforms(mpr_volume)
    # mpr_volume = feat.add_window(mpr_volume)

    cubic_sequence, label_sequence = np.zeros(
        [opt.cubic_sequence_length, opt.cube_side_length, opt.cube_side_length, opt.cube_side_length],
        dtype=np.float64), np.zeros([opt.cubic_sequence_length, 1], dtype=np.float64)
    for i in range(opt.cubic_sequence_length):
        orig_point = [int(round(interval * i, 0)) + voxel_bias, int(mpr_volume.shape[1] / 2 + 1),
                     int(mpr_volume.shape[2] / 2 + 1)]
        now_point = random_move(mpr_volume.shape[0], orig_point)
        cubic_sequence[i] = cube_spin(get_voxel_cude(mpr_volume, now_point))
        label_sequence[i] = mpr_label[now_point[0]]
    return cubic_sequence, label_sequence