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
from m_1_test import finetune_test
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from sklearn.model_selection import train_test_split
from utils import writer_train, writer_finetune, writer_test, model_save
import os
import torch.nn as nn


def train_validate_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    # model = nn.DataParallel(model)
    # optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-5}], lr=1e-8)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-2}], lr=1e-4)
    # criterion = nn.L1Loss()
    criterion = ContrastiveLoss()
    meta_model = copy.deepcopy(model)
    temp_model = copy.deepcopy(model)
    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    writer_t = writer_train()
    writer_f = writer_finetune()
    writer_te = writer_test()
    model_save_dir = model_save()
    task_num = 1
    noise_num1 = 25
    noise_num2 = 24
    epochs = 128

    # list_noise = list(range(noise_num2))  # list_noise = [0, 1, ..., 23] [0, 1, 2, 3, 4]
    # list_noise_train, list_noise_test = train_test_split(list_noise, train_size=0.8)
    list_noise_train = list(range(noise_num1))
    list_noise_test = list(range(noise_num2))
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = exp_lr_scheduler(optimizer, epoch)  #在这里设置下降率

        print('\033[1;33m'
              '----------------------------------------'
              'TID 2013 train phase Epoch: %d'
              '----------------------------------------'
              ' \033[0m' % (epoch + 1))
        count = 0
        predict = np.array([])
        label = np.array([])
        for index in list_noise_train:
            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            dataloader_support, dataloader_query = load_data('train', 'KADID10k', index)
            if dataloader_support == 0:
                continue
            dataiter = iter(enumerate(dataloader_query))

            # model.train()
            # print('------------------------- train supprt set noise %2d---------------------' % index)
            for batch_idx, data in enumerate(dataloader_support):
                model.train()
                dis_inputs = data['dis_image']
                ref_inputs = data['ref_image']
                batch_size = dis_inputs.size()[0]  # 得到一个batch_size的图片个数
                labels = data['rating'].view(batch_size, -1)  # 将label的形状调整为(batch_size, 1)
                # labels = (labels - 0.5) / 5.0    #kadid10k才有
                if use_gpu:
                    try:
                        dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                                                         Variable(ref_inputs.float().cuda()), Variable(
                            labels.float().cuda())
                    except:
                        print(dis_inputs, ref_inputs, labels)
                else:
                    dis_inputs, ref_inputs, labels = Variable(dis_inputs), Variable(ref_inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(dis_inputs, ref_inputs).view(batch_size, -1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                idx, data_val = next(dataiter)
                if idx >= len(dataloader_query) - 1:
                    dataiter = iter(enumerate(dataloader_query))  # 又重新开始迭代验证集中的batch
                # print('------------------------- train query set noise %2d-----------------' % index)
                dis_inputs_val = data_val['dis_image']
                ref_inputs_val = data_val['ref_image']
                batch_size1 = dis_inputs_val.size()[0]
                labels_val = data_val['rating'].view(batch_size1, -1)

                # labels = (labels - 0.5) / 5.0  # kadid10k才有
                if use_gpu:
                    try:
                        dis_inputs_val, ref_inputs_val, labels_val = Variable(dis_inputs_val.float().cuda()), \
                                                                     Variable(ref_inputs_val.float().cuda()), Variable(
                            labels_val.float().cuda())
                    except:
                        print(dis_inputs_val, ref_inputs_val, labels_val)
                else:
                    dis_inputs_val, ref_inputs_val, labels_val = Variable(dis_inputs_val), Variable(
                        ref_inputs_val), Variable(labels_val)
                optimizer.zero_grad()
                outputs_val = model(dis_inputs_val, ref_inputs_val).view(batch_size1, -1)  # 模型验证中，获得预测结果

                predict = np.append(predict, outputs_val.cpu().detach().numpy())
                label = np.append(label, labels_val.cpu().numpy())

                loss_val = criterion(outputs_val, labels_val)
                loss_val.backward()
                optimizer.step()

                if batch_idx % 40 == 0:
                    print('epoch {:2d}, time {:s}, loss {:f}'.format(epoch+1, start_time, loss.item()))
                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param1 = dict(meta_model.named_parameters())
                name_to_param2 = dict(temp_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param2[name].data
                    name_to_param1[name].data.add_(diff / task_num)
                count += 1


        train_plcc = stats.pearsonr(predict, label)[0]
        train_srocc = stats.spearmanr(predict, label)[0]
        train_krocc = stats.kendalltau(predict, label)[0]
        epoch_loss = running_loss / count

        torch.save(meta_model.state_dict(), os.path.join(model_save_dir, 'fullmodel_{}.pth'.format(epoch)))

        writer_t.add_scalar('loss', epoch_loss, epoch)
        writer_t.add_scalar('plcc', train_plcc, epoch)
        writer_t.add_scalar('srocc', train_srocc, epoch)
        print("current loss: {}, count: {}".format(epoch_loss, count))
        print("plcc: {}, srocc: {}".format(train_plcc, train_srocc))

        print('\033[1;33m'
              '----------------------------------------'
              'TID 2013 test phase Epoch: %d'
              '----------------------------------------'
              ' \033[0m' % (epoch + 1))
        f_pe, f_sp, pe, sp = finetune_test(meta_model, criterion, list_noise_test)
        # pe, sp = finetune_test(meta_model, criterion, list_noise_test)
        writer_f.add_scalar('f_plcc', f_pe, epoch)
        writer_f.add_scalar('f_srocc', f_sp, epoch)
        writer_te.add_scalar('t_plcc', pe, epoch)
        writer_te.add_scalar('t_srocc', sp, epoch)

    writer_t.close()
    print('-------------END----------------------')


if __name__ == '__main__':
    train_validate_model()

