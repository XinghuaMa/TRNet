from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from method.model import transformer_network
from config import opt
import dataset.dataset as dataset
import features as feat
import data_maker as dm
import method.evaluate as eva


class Loss_Saver:
    def __init__(self):
        self.loss_list, self.last_loss = [], 0.0
        return

    def updata(self, value):
        if not self.loss_list:
            self.loss_list += [value]
            self.last_loss = value
        else:
            update_val = self.last_loss * 0.9 + value * 0.1
            self.loss_list += [update_val]
            self.last_loss = update_val
        return

    def loss_drawing(self):
        print(self.loss_list)
        return

def train(train_num_fold):
    model = transformer_network(in_channels=opt.in_channels, num_levels=opt.num_levels, f_maps=opt.f_maps,
                                dim_hidden=opt.dim_hidden, num_heads=opt.num_heads, dim_head=opt.dim_head,
                                num_encoders=opt.num_encoders, num_linear=opt.num_linear, num_class=opt.num_class)
    if opt.use_gpu:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    losssaver = Loss_Saver()

    train_dataset = dataset.cubic_sequence_data(opt.raw_dataset_root, num_fold=train_num_fold, pattern='train')
    train_dataLoader = DataLoader(train_dataset, batch_size=opt.batch_size,  shuffle=True)

    eval_dataset = dataset.cubic_sequence_data(opt.raw_dataset_root, num_fold=train_num_fold, pattern='eval')
    eval_dataLoader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False)

    for epoch in range(opt.max_epoch):

        model.train()
        epoch_loss = feat.Counter()
        for batch_id, (sequence_image, sequence_label) in tqdm(enumerate(train_dataLoader),
                                                               total=int(len(train_dataset) / opt.batch_size)):
            Input, target = sequence_image.requires_grad_(), sequence_label

            if opt.use_gpu:
                input, target = Input.cuda(), target.cuda()

            pred = model(Input)
            pred, target = pred.view(-1, 2), target.view(-1)

            loss = loss_fn(pred, target)
            epoch_loss.updata(float(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch:{epoch} loss:{epoch_loss.avg}')
        losssaver.updata(epoch_loss.avg)

        model.eval()
        eva.print_evaluate_index(model, eval_dataLoader, num_indexes=opt.num_indexes)

    return
