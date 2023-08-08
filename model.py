import os
import torchvision
import torch
import torch.nn as nn
from torchsummary import summary
from SiameseNetwork import SiameseNetwork


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.SiameseNetwork = SiameseNetwork

        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)

        self.aff_fusion = TAFF(512, 4)
        self.EDN = RCF()
        self.LTE = LTE()
        self.dsn = nn.Conv2d(640, 512, 1)
        self.dLET = nn.Conv2d(256, 512, 1)
        self.pooling = nn.Conv2d(512, 512, 9, 7)
        self.fc = nn.Sequential(nn.Linear(in_features=25088, out_features=4098, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=4098, out_features=512, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=128, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=128, out_features=1, bias=False)
                                )
        # init_weight.init(self)

    def resnet(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)  #2*2048*7*7
        x = self.conv(x)   #2*512*7*7
        # x = self.resnet50.avgpool(x)   #2*512*1*1
        return x


    def forward(self, x):
        edn = self.EDN(x)  # 2*640*7*7
        edn1 = self.dsn(edn)  #2*512*7*7
        q1 = torch.flatten(edn1, 1)  #[2, 25088]
        q1 = self.fc(q1)

        lte = self.LTE(x)  #2*256*56*56
        dLET1 = self.dLET(lte)  #2*512*65*56
        dLET2 = self.pooling(dLET1)  #2*512*7*7
        q2 = torch.flatten(dLET2, 1)  # [2, 25088]
        q2 = self.fc(q2)

        x_resnet = self.resnet(x)
        q3 = torch.flatten(x_resnet, 1)  # [2, 25088]
        q3 = self.fc(q3)

        lte_edn = self.aff_fusion(edn1, dLET2)
        res_lte_edn= self.aff_fusion(lte_edn, x_resnet)

        addition = edn1 + dLET2 + x_resnet
        concatenation = torch.cat((edn1, dLET2,x_resnet),1)
        two_module = self.aff_fusion(dLET2, x_resnet)

        x0 = torch.flatten(res_lte_edn, 1) #2*25088   x_resnet:2*512
        q = self.fc(x0)   #torch.size([2, 1])
        return torch.cat((q1, q2, q3, q), 1)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = MyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, (3, 224, 224))