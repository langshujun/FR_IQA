import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class MyDataset(Dataset):
    def __init__(self, filepath, root_dir):  #标签地址，图片地址
        self.root_dir = root_dir  # 读取数据文件夹地址
        self.csv = np.loadtxt(filepath, str, delimiter=',', skiprows=1)  # 读取csv文件

    def __getitem__(self, index):
        csv = self.csv
        transform = transforms.Compose([transforms.ToTensor()])
        ref_name = csv[index][0]
        dis_name = csv[index][1]
        q = float(csv[index][2])
        ref_path = os.path.join(self.root_dir, ref_name)
        dis_path = os.path.join(self.root_dir, dis_name)
        ref = Image.open(ref_path).resize((224, 224))
        ref = transform(ref)
        dis = Image.open(dis_path).resize((224, 224))  # 读取图像，转换为三维矩阵
        dis = transform(dis)
        sample ={'ref': ref, 'dis': dis, 'label': q}
        return sample

    def __len__(self):
        return self.csv.shape[0]


# 改进后得dataset，只需要传入一个字典即可
class MyDataset_1(Dataset):
    def __init__(self, filepath, root_dir):
        self.imgAndScore = getImgAndScore(filepath, root_dir)  # 键值对

    def __getitem__(self, index):
        return self.imgAndScore[index]['ref'], self.imgAndScore[index]['dis'], self.imgAndScore[index]['label']

    def __len__(self):
        return len(self.imgAndScore)


def getImgAndScore(filepath, root_dir):
    # 函数返回一个字典 index ：{"img":img, "score":score}
    imgAndScore = {}
    csv = np.loadtxt(filepath, str, delimiter=',', skiprows=1)  # 读取csv文件
    train_transforms = transforms.Compose([
                                           transforms.Resize((224, 224)),
                                           transforms.Normalize(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])   #transforms.RandomResizedCrop(224),
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    # 统计最终有多少张图片，顺便当字典中的键
    count = 0
    for index in range(len(csv)):
        ref_name = csv[index][0]
        # 图片名称
        dis_name = csv[index][1]
        # 分数
        # q = Xy[index][4]
        q = float(csv[index][2])
        # 读图片
        ref_path = os.path.join(root_dir, ref_name)
        dis_path = os.path.join(root_dir, dis_name)
        ref = Image.open(ref_path).convert('RGB').resize((224, 224))
        dis = Image.open(dis_path).convert('RGB').resize((224, 224))  # 读取图像

        # # 随机裁剪图片，range里得个数为定义裁剪得图片个数
        for j in range(3):
            # 生成裁剪图片，根据自己的需求进行更改，224就是从图片中裁剪出224 * 224的图片，padding用于填充，默认为0
            ref = train_transforms(ref)
            dis = train_transforms(dis)
            # 放到字典里，img放图片，score放分数
            imgAndScore[count] = {"ref": ref, "dis": dis, "label": q}
            # 数量加一
            count += 1
    return imgAndScore


class ImageRatingsDataset(Dataset):
    def __init__(self, csv_file, dis_root_dir, ref_root_dir, transform=None):
        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.ref_root_dir = ref_root_dir
        self.dis_root_dir = dis_root_dir
        # self.transform = transform
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToPILImage(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])
                                                                                      ])

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        dis_name = str(os.path.join(self.dis_root_dir, str(self.images_frame.iloc[idx, 0])))
        ref_name = str(os.path.join(self.ref_root_dir, str(self.images_frame.iloc[idx, 1])))
        transform = self.transform
        dis = Image.open(dis_name).resize((256, 256)) #.convert('RGB')
        ref = Image.open(ref_name).resize((256, 256))  #.convert('RGB')

        if dis.mode == 'P':
            dis = dis.convert('RGB')
        if ref.mode == 'p':
            ref = ref.convert('RGB')
        dis_imge = np.asarray(dis)
        dis_imge = transform(dis_imge)
        ref_imge = np.asarray(ref)
        ref_imge = transform(ref_imge)
        # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        rating = self.images_frame.iloc[idx, 2]  # 图片对应的dmos值
        sample = {'dis_image': dis_imge, 'ref_image': ref_imge,  'rating': rating}
        # if self.transform:
        #     sample = self.transform(sample)
        return sample