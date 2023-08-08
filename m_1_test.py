from utils import writer_test
from utils import exp_lr_scheduler
import numpy as np
import torch.optim as optim
from function import load_data
from torch.autograd import Variable
use_gpu = True
from scipy import stats

def finetune_test(model, criterion, list_noise_test):
    all_f_predict = np.array([])
    all_f_label = np.array([])
    all_predict = np.array([])
    all_label = np.array([])
    for index in list_noise_test:
        temp_model = model
        # optimizer = optim.Adam([{'params': temp_model.parameters(), 'lr': 1e-5}], lr=1e-8)
        optimizer = optim.Adam([{'params': temp_model.parameters(), 'lr': 1e-2}], lr=1e-4)
        temp_model.cuda()
        test_finetune, test_valid = load_data('test', 'tid2013', index)
        finetune_loss = 0

        ######################### 开始finetune ##########################
        f_predict = np.array([])
        f_label = np.array([])
        for idx, data in enumerate(test_finetune):
            temp_model.train()
            dis_inputs = data['dis_image']
            ref_inputs = data['ref_image']
            batch_size = dis_inputs.size()[0]  # 得到一个batch_size的图片个数
            labels = data['rating'].view(batch_size, -1)  # 将label的形状调整为(batch_size, 1)
            dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                                               Variable(ref_inputs.float().cuda()), \
                                               Variable(labels.float().cuda())
            predicts = temp_model(dis_inputs, ref_inputs)
            loss = criterion(predicts, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            finetune_loss += loss.item()

            f_predict = np.append(f_predict, predicts.cpu().detach().numpy())
            f_label = np.append(f_label, labels.cpu().numpy())
        all_f_predict = np.append(all_f_predict, f_predict)
        all_f_label = np.append(all_f_label, f_label)

        ########################## 开始test #############################
        predict = np.array([])
        label = np.array([])
        for idx, data in enumerate(test_valid):
            temp_model.eval()
            temp_model.cuda()
            dis_inputs = data['dis_image']
            ref_inputs = data['ref_image']
            batch_size = dis_inputs.size()[0]  # 得到一个batch_size的图片个数
            labels = data['rating'].view(batch_size, -1)  # 将label的形状调整为(batch_size, 1)
            dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                                             Variable(ref_inputs.float().cuda()), \
                                             Variable(labels.float().cuda())
            outputs = temp_model(dis_inputs, ref_inputs)

            predict = np.append(predict, outputs.cpu().detach().numpy())
            label = np.append(label, labels.cpu().numpy())
        all_predict = np.append(all_predict, predict)
        all_label = np.append(all_label, label)

    f_pe = stats.pearsonr(all_f_predict, all_f_label)[0]
    f_sp = stats.spearmanr(all_f_predict, all_f_label)[0]
    print('new f_plcc {:4f}, new f_srocc {:4f}'.format(f_pe, f_sp))
    pe = stats.pearsonr(all_predict, all_label)[0]
    sp = stats.spearmanr(all_predict, all_label)[0]
    print('new t_plcc {:4f}, new t_srocc {:4f}'.format(pe, sp))
    return f_pe, f_sp, pe, sp
    # return pe, sp