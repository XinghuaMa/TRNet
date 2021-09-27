import random
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import dataset.dataset as dataset
from config import opt
import features as feat
import data_maker as dm
from method.model import transformer_network
import method.evaluate as evalu


def test_sequence(mpr_volume, mpr_label):
    mpr_volume = dm.image_transforms(mpr_volume)
    interval = mpr_volume.shape[0] / opt.cubic_sequence_length

    cubic_sequence, label_sequence, point_list = np.zeros(
        [opt.cubic_sequence_length, opt.cube_side_length, opt.cube_side_length, opt.cube_side_length],
        dtype=np.float64), np.zeros([opt.cubic_sequence_length], dtype=np.int16), np.zeros(
        [opt.cubic_sequence_length], dtype=np.int16)

    for i in range(opt.cubic_sequence_length):
        now_point = [int(round(interval * i, 0)), int(mpr_volume.shape[1] / 2 + 1),
                     int(mpr_volume.shape[2] / 2 + 1)]

        now_point[0] = min(mpr_volume.shape[0], max(0, now_point[0] - 1))

        cubic_sequence[i], label_sequence[i], point_list[i] = dm.get_voxel_cude(mpr_volume, now_point), mpr_label[
            now_point[0]], now_point[0]

    return cubic_sequence, label_sequence, point_list

def draw_pred_image(mpr_image, pred, label, point_list):
    concat_image = mpr_image[0]
    for i in range(1, mpr_image.shape[0]):
        concat_image = np.concatenate((concat_image, mpr_image[1]), axis=1)
    concat_image = concat_image.T

    pred_image = cv2.cvtColor(np.zeros([concat_image.shape[0] + 20, concat_image.shape[1]], dtype=np.uint8),
                             cv2.COLOR_GRAY2RGB)
    pred_image[:concat_image.shape[0]] = cv2.cvtColor(concat_image, cv2.COLOR_GRAY2RGB)
    pred_image[concat_image.shape[0]:concat_image.shape[0] + 10] = [144,238,144]
    pred_image[concat_image.shape[0] + 10: concat_image.shape[0] + 20] = [139, 139, 144]
    for i in range(opt.cubic_sequence_length) :
        left_bound = 0 if i == 0 else max(0, int(round((point_list[i] + point_list[i - 1]) / 2, 0)))
        right_bound = pred_image.shape[1] if i == opt.cubic_sequence_length - 1 else min(pred_image.shape[1], int(
            round((point_list[i] + point_list[i + 1]) / 2, 0)))
        for j in range(left_bound, right_bound):
            if label[i] == 1:
                pred_image[concat_image.shape[0]:concat_image.shape[0] + 10, j] = [0,0,255]
            if pred[i] == 1:
                pred_image[concat_image.shape[0] + 10:concat_image.shape[0] + 20, j] = [0, 255, 255]

    pred_image = cv2.resize(pred_image, (400, 200), )
    cv2.imwrite('image save path', pred_image)
    return

def predict(model, mpr_volume, mpr_label):

    mpr_image = feat.longitudinal_mpr(feat.add_window(mpr_volume))

    cubic_sequence, label_sequence, point_list = test_sequence(mpr_volume, mpr_label)

    Input = torch.tensor(cubic_sequence, dtype=torch.float32).requires_grad_()
    Input = Input.view(-1, opt.cubic_sequence_length, opt.cube_side_length, opt.cube_side_length, opt.cube_side_length)

    if opt.use_gpu:
        Input = Input.cuda()

    pred = model(Input).view(opt.cubic_sequence_length, 2)

    pred = torch.argmax(pred, dim=1)

    if opt.use_gpu:
        pred = pred.cpu()

    pred = pred.detach().numpy()

    draw_pred_image(mpr_image, pred, label_sequence, point_list)
    return

def test_check():
    model = transformer_network(in_channels=opt.in_channels, num_levels=opt.num_levels, f_maps=opt.f_maps,
                                dim_hidden=opt.dim_hidden, num_heads=opt.num_heads, dim_head=opt.dim_head,
                                num_encoders=opt.num_encoders, num_linear=opt.num_linear, num_class=opt.num_class)



    if opt.use_gpu:
        model = model.cuda()

    root_mpr_volume, root_mpr_label = 'MPR image path to be detected', 'MPR image label path to be detected'
    mpr_volume, mpr_label = np.load(root_mpr_volume), np.load(root_mpr_label)

    predict(model, mpr_volume, mpr_label)
    return

def test_quantity(test_dataset_root, num_indexes=7):

    model = transformer_network(in_channels=opt.in_channels, num_levels=opt.num_levels, f_maps=opt.f_maps,
                                dim_hidden=opt.dim_hidden, num_heads=opt.num_heads, dim_head=opt.dim_head,
                                num_encoders=opt.num_encoders, num_linear=opt.num_linear, num_class=opt.num_class)

    if opt.use_gpu:
        model = model.cuda()

    test_dataset = dataset.cubic_sequence_data(test_dataset_root, pattern='test')
    test_dataLoader = DataLoader(test_dataset, batch_size=opt.batch_size,  shuffle=True)

    index_counter = []
    for i in range(num_indexes):
        index_counter += [feat.Counter()]

    model.eval()
    for batch_id, (sequence_image, sequence_label) in tqdm(enumerate(test_dataLoader),
                                                           total=int(len(test_dataset) / opt.batch_size)):
        Input, target = sequence_image.requires_grad_(), sequence_label

        if opt.use_gpu:
            Input, target = Input.cuda(), target.cuda()

        pred = model(Input)
        pred, target = pred.view(-1, 2), target.view(-1)
        indexes = evalu.get_index(pred, target, num_index=num_indexes)

        for j in range(num_indexes):
            index_counter[j].updata(indexes[j])

    ret_index = []
    for i in range(num_indexes):
        ret_index += [index_counter[i].avg]
    print(ret_index)
    return







