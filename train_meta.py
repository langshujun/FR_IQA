import copy
import os
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
use_gpu = True
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from SiameseNetwork import ContrastiveLoss
from function import Rescale, RandomCrop, RandomHorizontalFlip
from scipy.stats import spearmanr
from dataset import ImageRatingsDataset


def train_model():
    epochs = 35
    noise_num1 = 24
    task_num = 5
    criterion = ContrastiveLoss()

    model = SiameseNetwork()
    loss = ContrastiveLoss()
    # ignored_params = list(map(id, model.net.parameters()))   #list(map(squre, [1,2]))  结束是：[1,4]
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())  #返回不在ignored_params和model.parameters中的参数
    # optimizer = optim.Adam([
    #     {'params': base_params},
    #     {'params': model.net.parameters(), 'lr': 1e-2}], lr=1e-4)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-2}], lr=1e-4)
    model.cuda()
    meta_model = copy.deepcopy(model)
    temp_model = copy.deepcopy(model)

    spearman = 0

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = exp_lr_scheduler(optimizer, epoch)
        list_noise = list(range(noise_num1))  #list_noise = [0, 1, ..., 23]
        np.random.shuffle(list_noise)  #打乱数组list_noise中的顺序

        print('############# TID 2013 train phase epoch %2d ###############' % epoch)
        count = 0
        for index in list_noise:
            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            dataloader_train, dataloader_valid = load_data('train', 'tid2013', index)
            if dataloader_train == 0:
                continue
            dataiter = iter(enumerate(dataloader_valid))

            # model.train()
            for batch_idx, data in enumerate(dataloader_train):
                model.train()
                dis_inputs = data['dis_image']
                ref_inputs = data['ref_image']
                batch_size = dis_inputs.size()[0]   #得到一个batch_size的图片个数
                labels = data['rating'].view(batch_size, -1) #将label的形状调整为(batch_size, 1)
                # labels = (labels - 0.5) / 5.0    #kadid10k才有
                if use_gpu:
                    try:
                        dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), \
                                                         Variable(ref_inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(dis_inputs, ref_inputs, labels)
                else:
                    dis_inputs, ref_inputs, labels = Variable(dis_inputs), Variable(ref_inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(dis_inputs, ref_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                idx, data_val = next(dataiter)
                if idx >= len(dataloader_valid)-1:
                    dataiter = iter(enumerate(dataloader_valid))  #又重新开始迭代验证集中的batch
                dis_inputs_val = data_val['dis_image']
                ref_inputs_val = data_val['ref_image']
                batch_size1 = dis_inputs_val.size()[0]
                labels_val = data_val['rating'].view(batch_size1, -1)
                # labels = (labels - 0.5) / 5.0  # kadid10k才有
                if use_gpu:
                    try:
                        dis_inputs_val, ref_inputs_val, labels_val = Variable(dis_inputs_val.float().cuda()), \
                                                                     Variable(ref_inputs_val.float().cuda()), Variable(labels_val.float().cuda())
                    except:
                        print(dis_inputs_val, ref_inputs_val, labels_val)
                else:
                    dis_inputs_val, ref_inputs_val, labels_val = Variable(dis_inputs_val), Variable(ref_inputs_val), Variable(labels_val)
                optimizer.zero_grad()
                outputs_val = model(dis_inputs_val, ref_inputs_val)  # 模型验证中，获得预测结果
                loss_val = criterion(outputs_val, labels_val)
                loss_val.backward()
                optimizer.step()

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
        epoch_loss = running_loss / count
        print('current loss=', epoch_loss)

        print('############# test phase epoch %2d ###############' % epoch)
        dataloader_train, dataloader_valid = load_data('test', 0)
        model.eval()
        model.cuda()
        sp = computeSpearman(dataloader_valid, model)[0]
        if sp > spearman:
            spearman = sp
        print('new srocc {:4f}, best srocc {:4f}'.format(sp, spearman))
    torch.save(model.cuda(),
               'model_IQA/meta_best.pt')


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_valid):
            dis_inputs = data['dis_image']
            ref_inputs = data['ref_image']
            batch_size = dis_inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)
            if use_gpu:
                try:
                    dis_inputs, ref_inputs, labels = Variable(dis_inputs.float().cuda()), Variable(dis_inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(dis_inputs, ref_inputs, labels)
            else:
                dis_inputs, ref_inputs, labels = Variable(dis_inputs), Variable(ref_inputs), Variable(labels)
            outputs_a = model(dis_inputs, ref_inputs)
            ratings.append(labels.float())
            predictions.append(outputs_a.float())

    ratings_i = np.vstack(ratings)  #按垂直方向（行顺序）堆叠数组
    predictions_i = np.vstack(predictions)
    a = ratings_i[:, 0]
    b = predictions_i[:, 0]
    sp = spearmanr(a, b)
    return sp


def load_data(mod = 'train', dataset = 'tid2013', worker_idx = 0):
    if dataset == 'tid2013':
        data_dir = os.path.join('csv')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'tid2013_image_labeled_by_per_noise.csv'), sep=',')  #阅读csv文件
        image_path = 'data/img/'
    elif dataset == 'kadid10k':
        data_dir = os.path.join('csv')
    workers_fold = "noise/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)

    worker = worker_orignal['noise'].unique()[worker_idx]  #unique--去除noise列中的重复元素，若worker_idx=0，则此时worker=1
    print("----worker number: %2d---- %s" % (worker_idx, worker))
    if mod == 'train':
        percent = 0.8
        images = worker_orignal[worker_orignal['noise'].isin([worker])][['dis_image', 'ref_image', 'dmos']] #noise=1对应的所有信息
        train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
        train_path = workers_fold + "train_scores_" + str(worker) + ".csv"
        test_path = workers_fold + "test_scores_" + str(worker) + ".csv"

        train_dataframe.to_csv(train_path, sep=',', index=False)
        valid_dataframe.to_csv(test_path, sep=',', index=False)

        output_size = (224, 224)

        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path,
                                                        transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(output_size=output_size),
                                                                                      transforms.ToTensor(),
                                                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                                                           [0.229, 0.224, 0.225])
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path,
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      transforms.ToTensor(),
                                                                                      transforms.Normalize(
                                                                                          [0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225])
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=32,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=32,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)
    else:
        cross_data_path = 'LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                          root_dir='/home/hancheng/IQA/iqa-db/LIVE_WILD/images',
                                                          transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                        transforms.ToTensor(),
                                                                                        transforms.Normalize(
                                                                                            [0.485, 0.456, 0.406],
                                                                                            [0.229, 0.224, 0.225])
                                                                                        ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size=32,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    return dataloader_train, dataloader_valid





