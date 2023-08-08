import csv
from scipy import stats
import torch.nn as nn
import torch
import os
import math
import time
import pandas as pd
# from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms


def writer_train():
    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    writer_train_dir = os.path.join('./runs', 'train', start_time)
    writer_t = SummaryWriter(writer_train_dir, comment='MyNet')
    return writer_t
def writer_finetune():
    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    writer_test_dir = os.path.join('./runs', 'finetune', start_time)
    writer_v = SummaryWriter(writer_test_dir, comment='MyNet')
    return writer_v
def writer_test():
    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    writer_test_dir = os.path.join('./runs', 'test', start_time)
    writer_v = SummaryWriter(writer_test_dir, comment='MyNet')
    return writer_v


# def writer_test(index):
#
#     start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
#     index = 'noise' + str(index)
#     writer_test_dir = os.path.join('./runs', 'test', index,  start_time)
#     writer_v = SummaryWriter(writer_test_dir, comment='MyNet')
#     return writer_v


def get_PLCC(y_pred, y_val):
    return stats.pearsonr(y_pred, y_val)[0]


def get_SROCC(y_pred, y_val):
    return stats.spearmanr(y_pred, y_val)[0]

def model_save():
    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    model_save_dir = os.path.join('model-save', start_time)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    return model_save_dir


def save_pred_label(csv_path, list1, list2):
    infos = zip(list1, list2)
    # with open(csv_path, encoding='utf-8', newline='') as f:
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        for info in infos:
            writer.writerow(info)


# get a batch iterator
def get_training_batch(train_loader):
    while True:
        for sequence in train_loader:
            yield sequence


# get a batch iterator
def get_testing_batch(test_loader):
    while True:
        for sequence in test_loader:
            yield sequence


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)  #x的y次幂  #取整除，向下取
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

# class DotProductSimilarity(nn.Module):
#     """
#     This similarity function simply computes the dot product between each pair of vectors, with an
#     optional scaling to reduce the variance of the output elements.
#     """
#     def __init__(self):
#         super(DotProductSimilarity, self).__init__()
#         self.scale_output = False
#
#     def forward(self, tensor_1, tensor_2):
#         # tensor_1 = torch.tensor(tensor_1)
#         # tensor_2 = torch.tensor(tensor_2)
#         result = (tensor_1 * tensor_2).sum(dim=-1)
#         if self.scale_output:
#             # TODO why allennlp do multiplication at here ?
#             result /= math.sqrt(tensor_1.size(-1))
#         return result

class DotProductSimilarity(nn.Module):

    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        tensor_1 = torch.tensor(tensor_1).cuda()

        tensor_2 = torch.tensor(tensor_2).cuda()
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result

class ProjectedDotProductSimilarity(nn.Module):
    """
    This similarity function does a projection and then computes the dot product. It's computed
    as ``x^T W_1 (y^T W_2)^T``
    An activation function applied after the calculation. Default is no activation.
    """

    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        self.activation = torch.nn.ReLU()
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class TriLinearSimilarity(nn.Module):

    def __init__(self, input_dim, activation=None):
        super(TriLinearSimilarity, self).__init__()
        self.weight_vector = nn.Parameter(torch.Tensor(3 * input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        combined_tensors = torch.cat([tensor_1, tensor_2, tensor_1 * tensor_2], dim=-1)
        result = torch.matmul(combined_tensors, self.weight_vector) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    a = torch.Tensor([[1, 2, 3, 4, 5, 9], [1, 2, 3, 4, 5, 9]])
    b = torch.Tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

    index = 2
    index = 'noise'+str(index)
    path = writer_test(index)
    os.mkdir(path)