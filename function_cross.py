import os
from skimage import transform
import numpy as np
import random
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
use_gpu = True
from dataset import ImageRatingsDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from scipy.stats import spearmanr, pearsonr
from torch.autograd import Variable


class Rescale(object):  #改变图像的大小，使长度和宽度之间存在倍数关系
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))  #判断out_size是否是int或者tuple类型
        self.output_size = output_size

    def __call__(self, sample):
        dis_image, ref_image, rating = sample['dis_image'],  sample['ref_image'], sample['rating']
        h, w = dis_image.shape[:2] #对应维度的0维和1维
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        dis_image = transform.resize(dis_image, (new_h, new_w))

        return {'dis_image': dis_image, 'ref_image':ref_image, 'rating': rating}


class RandomCrop(object):  #随机划分图片，取图片中的一小块
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        dis_image, ref_image, rating = sample['dis_image'], sample['ref_image'], sample['rating']
        h, w = dis_image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        dis_image = dis_image[top: top + new_h, left: left + new_w]

        return {'dis_image': dis_image, 'ref_image': ref_image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        dis_image, ref_image, rating = sample['dis_image'], sample['ref_image'], sample['rating']
        if random.random() < self.p:
            dis_image = np.flip(dis_image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'dis_image': dis_image, 'ref_image': ref_image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        dis_image, ref_image, rating = sample['dis_image'], sample['ref_image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}


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
            # else:
            #     dis_inputs, ref_inputs, labels = Variable(dis_inputs), Variable(ref_inputs), Variable(labels)
            outputs_a = model(dis_inputs, ref_inputs)
            ratings.append(labels.detach().cpu().numpy())
            predictions.append(outputs_a.detach().cpu().numpy())

    ratings_i = np.vstack(ratings)  #按垂直方向（行顺序）堆叠数组
    predictions_i = np.vstack(predictions)

    a = ratings_i.squeeze(-1)
    b = predictions_i.squeeze(-1)
    sp = spearmanr(a, b)
    pe = pearsonr(a, b)
    return pe, sp


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod, dataset, worker_idx):   #worker_idx表示的是失真类型
    if dataset == 'tid2013':
        data_dir = os.path.join('csv')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'TID_2013.csv'), sep=',')  #阅读csv文件
        dis_image_path = 'data/TID_2013/dis_images/'
        ref_image_path = 'data/TID_2013/ref_images/'

    if dataset == 'LIVE':
        data_dir = os.path.join('csv')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'LIVE.csv'), sep=',')  #阅读csv文件
        dis_image_path = 'data/LIVE/'
        ref_image_path = 'data/LIVE/'

    workers_fold1 = "./noise/train_noise/"
    if not os.path.exists(workers_fold1):
        os.makedirs(workers_fold1)
    workers_fold2 = "./noise/test_noise/"
    if not os.path.exists(workers_fold2):
        os.makedirs(workers_fold2)
    worker = worker_orignal['noise'].unique()[worker_idx]  #unique--去除noise列中的重复元素，若worker_idx=0，则此时worker=1

    if mod=='train':
        percent = 0.8
        images = worker_orignal[worker_orignal['noise'].isin([worker])][
            ['dis_image', 'ref_image', 'rating']]  # noise=1对应的所有信息
        train_support, train_query = train_test_split(images, train_size=percent)
        support_path = workers_fold1 + "train_support_scores_" + str(worker) + ".csv"
        query_path = workers_fold1 + "train_query_scores_" + str(worker) + ".csv"

        train_support.to_csv(support_path, sep=',', index=False)
        train_query.to_csv(query_path, sep=',', index=False)

        output_size = (224, 224)

        transformed_dataset_support = ImageRatingsDataset(csv_file=support_path,
                                                          dis_root_dir=dis_image_path,
                                                          ref_root_dir=ref_image_path
                                                          )
        transformed_dataset_query = ImageRatingsDataset(csv_file=query_path,
                                                        dis_root_dir=dis_image_path,
                                                        ref_root_dir=ref_image_path
                                                        )
        dataloader_support = DataLoader(transformed_dataset_support, batch_size=16,
                                        shuffle=True, num_workers=0, collate_fn=my_collate)
        dataloader_query = DataLoader(transformed_dataset_query, batch_size=16,
                                      shuffle=True, num_workers=0, collate_fn=my_collate)
        # dataloader_train = 0
        return dataloader_support, dataloader_query

    if mod == 'test':
        percent = 0.2
        images = worker_orignal[worker_orignal['noise'].isin([worker])][['dis_image', 'ref_image', 'rating']]
        test_finetune, test_valid = train_test_split(images, train_size=percent)
        finetune_path = workers_fold2 + "test_finetune_scores_" + str(worker) + ".csv"
        valid_path = workers_fold2 + "test_valid_scores_" + str(worker) + ".csv"
        test_finetune.to_csv(finetune_path, sep=',', index=False)
        test_valid.to_csv(valid_path, sep=',', index=False)

        output_size = (224, 224)

        transformed_dataset_finetune = ImageRatingsDataset(csv_file=finetune_path,
                                                           dis_root_dir = dis_image_path,
                                                           ref_root_dir =ref_image_path,
                                                           # transform=transforms.Compose([
                                                           #                              transforms.ToPILImage(),
                                                           #                              Rescale(output_size=(256, 256)),
                                                           #                              RandomHorizontalFlip(0.5),
                                                           #                              RandomCrop(
                                                           #                                  output_size=output_size),
                                                           #                              transforms.ToTensor(),
                                                           #                              transforms.Normalize(
                                                           #                                  [0.485, 0.456, 0.406],
                                                           #                                  [0.229, 0.224, 0.225])
                                                           #                              ])
                                                           )
        transformed_dataset_valid = ImageRatingsDataset(csv_file=valid_path,
                                                        dis_root_dir=dis_image_path,
                                                        ref_root_dir=ref_image_path
                                                        )
        dataloader_finetune = DataLoader(transformed_dataset_finetune, batch_size=8,
                                        shuffle=True, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=8,
                                      shuffle=True, num_workers=0, collate_fn=my_collate)
        # dataloader_train = 0
        return dataloader_finetune, dataloader_valid


