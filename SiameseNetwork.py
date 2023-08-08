import os

import numpy as np
import torchvision.transforms

from conformer import Conformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import ProjectedDotProductSimilarity as dot
from utils import TriLinearSimilarity as tri
from skimage.metrics import structural_similarity as ssim
import conformer_weight



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.Conformer = Conformer()
        self.FC = nn.Sequential(
                                # nn.Linear(in_features=8000, out_features=4000, bias=False),
                                # nn.ReLU(inplace=True),
                                nn.Linear(in_features=4000, out_features=2000, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=2000, out_features=512, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=64, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=64, out_features=1, bias=False),
                                )
        # self.ProjectedDotProductSimilarity = ProjectedDotProductSimilarity(2000, 2000, 512)
        # self.TriLinearSimilarity = TriLinearSimilarity(2000)
        self.dot = dot(2000, 2000, 1000)
        self.tri  = tri(2000)
        # conformer_weight.init(self)

    def forward_once(self, x):
        output = self.Conformer(x)
        return output

    def forward_tiwce(self, picture1, picture2):
        output1 = self.forward_once(picture1)
        output2 = self.forward_once(picture2)
        return output1, output2

    def forward(self, x, y):
        vector1, vector2 = self.forward_tiwce(x, y) #torch.size([2, 2000])  torch.size([2, 2000])
        fuse = torch.cat((vector1, vector2), 1)
        w = self.FC(fuse)
        # similarity1 = self.ProjectedDotProductSimilarity(vector1, vector2)
        # similarity2 = self.TriLinearSimilarity(vector1, vector2)
        # similarity2 = []
        # B = vector1.shape[0]
        # for i in range(B):
        #     x = vector1[i].view(-1).detach().cpu().numpy()
        #     y = vector2[i].view(-1).detach().cpu().numpy()
        #     similarity2.append(torch.unsqueeze(torch.tensor(ssim(x, y)), dim=0).cuda())
        #     # similarity2.append(self.dot(torch.tensor((x, y)), -1).unsqueeze(0).cuda())
        #
        # similarity2 = torch.cat(similarity2, dim=0)
        # similarity2 = similarity2.unsqueeze(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # vector1 = F.normalize(vector1, dim=1)
        # vector2 = F.normalize(vector2, dim=1)
        Bilinear = nn.Bilinear(2000, 2000, 1).to(device)
        similarity1 = Bilinear(vector1, vector2)
        # similarity2 = tri(vector1, vector2)
        similarity2 = self.dot(vector1, vector2).unsqueeze(0).permute(1, 0)

        # print('1',similarity1)
        # print('2', similarity2)
        # print(similarity1, similarity2)
        # print(similarity1.shape, similarity2.shape)
        # similarity1 = torch.cosine_similarity(vector1, vector2)
        # similarity1 = torch.unsqueeze(similarity1, 0)  #使用cos_ssimilarity时需要
        # similarity1 = similarity1.permute(1, 0)
        # similarity2 = ssim(vector1.view(-1).detach().cpu().numpy(), vector2.view(-1).detach().cpu().numpy())
        # similarity2 = torch.tensor(similarity2)
        # similarity2 = torch.unsqueeze(similarity2, 0)

        score = (torch.mul(w, similarity1) + torch.mul((1 - w), similarity2))
        # score = torch.mean(score.squeeze(1), 0)
        print('1', torch.mul(w, similarity1))
        print('2', torch.mul((1 - w), similarity2))
        print('s', score)
        return score



class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, score, label):
        # euclidean_distance = F.pairwise_distance(output1, output2) #计算特征图之间的像素级欧式距离，两个特征间维度相同
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # loss_contrastive = torch.mean(label * torch.pow((1 - score), 2) +
        #                               (1 - label) * torch.pow(torch.clamp(self.margin - (1 - score), min=0.0), 2))
        loss_contrastive = torch.mean(label * pow((1 - score), 2) + (1 - label) * pow(score, 2))
        return loss_contrastive


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    result = SiameseNetwork()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result.to(device)
    summary(result, [[3, 224, 224], [3, 224, 224]])