import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import MyDataset
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from utils import save_pred_label
from opts import parse_opts
from train import train_validate_model
from eval import val_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    opt = parse_opts()

    model_type = opt.model_type

    image_dir_train = opt.image_dir_train
    image_dir_test = opt.image_dir_test

    label_train_path = opt.label_train_path
    label_test_path = opt.label_test_path

    learning_rate = opt.learning_rate
    weight_decay = opt.weight_decay
    epoch_nums = opt.epoch_nums
    batch_size = opt.batch_size

    start_time = opt.start_time
    model_save_dir = opt.save_model
    model_load_path = opt.load_model
    writer_t_dir = opt.writer_t_dir
    writer_v_dir = opt.writer_v_dir
    save_checkpoint = opt.save_checkpoint

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    my_model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()

    if model_type == 'train':
        #加载数据集
        print('load data')
        train_dataset = MyDataset(label_train_path, image_dir_train)
        val_dataset = MyDataset(label_test_path, image_dir_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=True)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

        if model_type == 'retrain':
            # 加载模型
            if not model_load_path:
                raise Exception('no model_load_path')
            model = torch.load(model_load_path)
            my_model.load_state_dict(model['net'])
            optimizer.load_state_dict(model['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = model['epoch'] + 1
        else:
            start_epoch = 0

        writer_t = SummaryWriter(writer_t_dir, comment='MyNet')
        writer_v = SummaryWriter(writer_v_dir, comment='MyNet')

        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        if not os.path.isdir(model_save_dir + '/checkpoint'):
            os.mkdir(model_save_dir + '/checkpoint')

        print('train model')

        train_validate_model(my_model, device, criterion, optimizer, train_loader, val_loader,
                         start_epoch, epoch_nums, save_checkpoint, model_save_dir,
                         start_time, writer_t, writer_v)
        print('done')


    if model_type == 'predict':
        val_dataset = MyDataset(image_dir_test, label_test_path)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

        if not model_load_path:
            raise Exception('no model_load_path')
        print('load model:', model_load_path)
        model = torch.load(model_load_path)
        my_model.load_state_dict(model['net'])
        val_loss, val_plcc, val_srocc, val_pred, val_label = val_model(my_model, device, criterion, val_loader)
        save_pred_label('predict.csv', val_pred, val_label)
