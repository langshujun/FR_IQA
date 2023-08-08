import copy

import numpy as np
import torch.optim as optim
from utils import exp_lr_scheduler
from function import load_data, computeSpearman
from torch.autograd import Variable
from utils import writer_test
from scipy import stats
from utils import exp_lr_scheduler
import torch
import numpy as np
import torch.optim as optim
import copy
from function import load_data
from torch.autograd import Variable
use_gpu = True
import time
from scipy import stats
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from sklearn.model_selection import train_test_split
from utils import writer_train
import os
import torch.nn as nn


def finetune_test(model,  criterion, list_noise_test):
    epochs = 64
    max_pearsonr = 0
    max_spearmanr = 0
    writer_t = writer_test()
    for epoch in range(epochs):
        print("\033[1;33m ------------ TID 2013 test epoch %2d --------------- \033[0m" % epoch)
        f_all_predict = np.array([])
        f_all_label = np.array([])
        all_predict = np.array([])
        all_label = np.array([])
        for index in list_noise_test:
            print("\033[1;33m ------------ finetune stage %2d --------------- \033[0m" % index)
            temp_model = model
            optimizer = optim.Adam([{'params': temp_model.parameters(), 'lr': 1e-2}], lr=1e-4)
            temp_model.cuda()
            optimizer = exp_lr_scheduler(optimizer, epoch)
            # noise
            test_finetune, test_valid = load_data('test', 'tid2013', index)
            finetune_loss = 0
            finetune_predict = np.array([])
            fine_tune_label = np.array([])
            for idx, data in enumerate(test_finetune):
                temp_model.train()
                dis_inputs = data['dis_image']
                ref_inputs = data['ref_image']
                batch_size = dis_inputs.size()[0]  # 得到一个batch_size的图片个数
                f_labels = data['rating'].view(batch_size, -1)  # 将label的形状调整为(batch_size, 1)
                dis_inputs, ref_inputs, f_labels = Variable(dis_inputs.float().cuda()), \
                                                   Variable(ref_inputs.float().cuda()), \
                                                   Variable(f_labels.float().cuda())
                # if use_gpu:
                #     try:
                #         dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                #                                                      Variable(ref_inputs.float().cuda()), Variable(
                #             f_labels.float().cuda())
                #     except:
                #         print(dis_inputs, ref_inputs, f_labels)
                # else:
                #     dis_inputs, ref_inputs, f_labels = Variable(dis_inputs.float().cuda()), \
                #                                      Variable(ref_inputs.float().cuda()), \
                #                                      Variable(f_labels.float().cuda())

                f_outputs = temp_model(dis_inputs, ref_inputs)
                # # f_outputs = Variable(f_outputs.float().cuda())
                loss = criterion(f_outputs, f_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                finetune_loss += loss.item()

            print("\033[1;31m ------------in %2d, we are in test--------------- \033[0m" % (epoch))
            predict = np.array([])
            label = np.array([])
            for idx, data in enumerate(test_valid):
                temp_model.eval()
                temp_model.cuda()
                dis_inputs = data['dis_image']
                ref_inputs = data['ref_image']
                batch_size = dis_inputs.size()[0]  # 得到一个batch_size的图片个数
                labels = data['rating'].view(batch_size, -1)  # 将label的形状调整为(batch_size, 1)
                if use_gpu:
                    try:
                        dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                                                                     Variable(ref_inputs.float().cuda()), Variable(
                            labels.float().cuda())
                    except:
                        print(dis_inputs, ref_inputs, labels)
                else:
                    dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                                                     Variable(ref_inputs.float().cuda()), \
                                                     Variable(labels.float().cuda())

                outputs = temp_model(dis_inputs, ref_inputs)

                predict = np.append(predict, outputs.cpu().detach().numpy())
                label = np.append(label, labels.cpu().numpy())

            all_predict = np.append(all_predict, predict)
            all_label = np.append(all_label, label)

        pe = stats.pearsonr(all_predict, all_label)[0]
        sp = stats.spearmanr(all_predict, all_label)[0]

        writer_t.add_scalar('plcc', pe, epoch)
        writer_t.add_scalar('srocc', sp, epoch)
        writer_t.close()

        if pe > max_pearsonr:
            max_pearsonr = pe
        if sp > max_spearmanr:
            max_spearmanr = sp
        print('new plcc {:4f}, best plcc {:4f}'.format(pe, max_pearsonr))
        print('new srocc {:4f}, best srocc {:4f}'.format(sp, max_spearmanr))


# if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = SiameseNetwork().to(device)
    # optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-2}], lr=1e-4)
    # criterion = nn.L1Loss()
    # list_noise = list(range(24))  # list_noise = [0, 1, ..., 23]
    # list_noise_train, list_noise_test = train_test_split(list_noise, train_size=0.8)
    # finetune_test(model, criterion, list_noise_test)


