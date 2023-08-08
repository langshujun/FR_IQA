import numpy as np
import torch


def init(model):
    extra_standard = ['trans_1.attn.qkv.bias', 'conv_trans_2.trans_block.attn.qkv.bias', 'conv_trans_3.trans_block.attn.qkv.bias',
              'conv_trans_4.trans_block.attn.qkv.bias', 'conv_trans_5.trans_block.attn.qkv.bias', 'conv_trans_6.trans_block.attn.qkv.bias',
              'conv_trans_7.trans_block.attn.qkv.bias', 'conv_trans_8.trans_block.attn.qkv.bias', 'conv_trans_9.trans_block.attn.qkv.bias',
              'conv_trans_10.trans_block.attn.qkv.bias', 'conv_trans_11.trans_block.attn.qkv.bias', 'conv_trans_12.trans_block.attn.qkv.bias'
              ]
    extra_my = ['fc.0.weight']

    base_pth = torch.load('data/pth/fullmodel_0.pth')
    log1 = open("data/log1.txt", mode = "a+", encoding = "utf-8")
    base_content = []
    for key, value in base_pth.items():
        if key.split('.', 1)[0] == 'Conformer' and key.split('.', 1)[1] not in extra_my:
            print(key, value.size(), sep=" ", file=log1)
            base_content.append(key)
    log1.close()


    cotent_pth = torch.load('data/pth/Conformer_tiny_patch16.pth')
    log2 = open("data/log2.txt", mode="a+", encoding="utf-8")
    pth = []
    for key, value in cotent_pth.items(): #key是网络层的名字，value是网络层对应的参数
        if key not in extra_standard:
            pth.append(key)
            print(key, value.size(), sep=" ", file=log2)
    log2.close()

    # print(base_pth[base_content[1]])
    # g更改base_pth参数之前
    for key, value in enumerate(pth):
        base_pth[base_content[key]] = cotent_pth[pth[key]]
    # 更改base_pth参数之后
    # print(base_pth[base_content[1]])
    model.load_state_dict(base_pth)




