# import cv2
# import os
#
#
# def read_path(file_pathname):
#     for filename in os.listdir(file_pathname):
#         img = cv2.imread(file_pathname+'/'+filename)
#         print(filename)
#         denoise_2 = cv2.fastNlMeansDenoisingColored(img, None, 10, 15, 7, 21)
#         cv2.imwrite('F:/test/new'+'/'+filename, denoise_2)
#
#
# read_path("F:/test/old")

# import torch
# from skimage.metrics import structural_similarity as ssim
# x = torch.randn(2, 2000).numpy()
# y = torch.randn(2, 2000).numpy()
#
# for i in range(2):
#     print(ssim(x[i], y[i]))
#
# similarity2 = ssim(x.numpy(), y.numpy())
import os
import time

import torch
import numpy as np

# tensor = torch.randn(3, 3, 3)
# print(tensor)
# mean = torch.from_numpy(np.array([0.5, 0.4, 0.3]))
# std = torch.from_numpy(np.array([0.3, 0.2, 0.1]))
# tensor = tensor.sub_(mean)
# print(tensor)
# tensor = tensor.div_(std)
# print(tensor)
epoch = 0
index = 9
from scipy.stats import spearmanr, pearsonr
# a = np.array([[1], [2], [3], [4], [5], [6]]).squeeze(-1)
# b = np.array([[1], [2], [3], [4], [5], [6]]).squeeze(-1)

# pe = pearsonr(a, b)
# pe = pearsonr(a, b)
# print(pe)
#
# content = torch.load('/home/wpl/LSJ/full_base/data/pth/Conformer_tiny_patch16.pth')
# print(len(content))
# log1 = open("/home/wpl/LSJ/full_base/data/log1.txt", mode = "a+", encoding = "utf-8")
# for key,value in content.items():
#     print(key, value.size(), sep=" ", file = log1)
# log1.close()

print('---------------------------------')

# ll_content = torch.load('/home/wpl/LSJ/full_base/model-save/2022_11_22_16_45_19/fullmodel_0.pth')
base_pth = torch.load('data/pth/fullmodel_101.pth')
print(len(base_pth))
print(base_pth.keys())

# log2 = open("/home/wpl/LSJ/full_base/data/log2.txt", mode = "a+", encoding = "utf-8")
# for key,value in base_pth.items():
#     print(key, value.size(), sep=" ", file = log2)
# log2.close()


