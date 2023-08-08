import numpy as np
import torch
from scipy import stats
from eval import val_model
from utils import get_PLCC, get_SROCC
from utils import save_pred_label
from utils import get_training_batch


def train_validate_model(my_model, device, criterion, optimizer, train_loader, val_loader,
                         start_epoch, epoch_nums, save_checkpoint, model_save_dir,
                         start_time, writer_t, writer_v):
    interval = 1

    # training_batch_generator = get_training_batch(train_loader)
    for epoch in range(start_epoch, start_epoch + epoch_nums):
        my_model.train()
        predict = np.array([])
        label = np.array([])
        train_epoch_loss = 0
        total_train_step = 0
        print("-----------第 {} 轮训练开始-----------".format(epoch + 1))
        # 训练步骤开始
        for i, inputs in enumerate(train_loader):
            # 优化器优化模型
            optimizer.zero_grad()
            # 载入图片和标签

            ref = inputs[0].to(device, non_blocking=True)
            dist = inputs[1].to(device, non_blocking=True)
            score = my_model(ref, dist).to(device)
            qy = inputs[2].to(device, non_blocking=True)

            predict = np.append(predict, score.cpu().detach().numpy())
            print('aaa', predict)
            label = np.append(label, qy.cpu().numpy())

            loss = criterion(score, qy)

            if i % 80 == 0:
                print('\033[1;33m'
                      '----------------------------------------'
                      'Epoch: %d / %d    Done: %d / %d    %s'
                      '----------------------------------------'
                      ' \033[0m' % (epoch + 1, epoch_nums, i + 1, len(train_loader), start_time))
                print('\033[1;31m loss: ', loss.item(), '\033[0m')
                print('\033[1;34m output, label: ', torch.cat((score, qy), 1).data, '\033[0m')

            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

            torch.manual_seed(2)
            writer_t.add_scalar('loss', train_epoch_loss / (i + 1), epoch)
            # next(training_batch_generator)

        plcc = stats.pearsonr(label, predict)[0]
        srocc = stats.pearsonr(label, predict)[0]

        writer_t.add_scalar('plcc', plcc, epoch)
        writer_t.add_scalar('srocc', srocc, epoch)

        # save checkpoint
        if save_checkpoint:
            if epoch % interval == interval - 1:
                # my_model = my_model.to(torch.device('cpu'))
                ckpt = {
                    'state_dict': my_model.state_dict()
                }
                torch.save(ckpt, model_save_dir + '/checkpoint/BIQAModel_{}.pth'.format(epoch))
                # my_model = my_model.to('cuda:0')

        ####################################################
        print('start')
        val_loss, val_plcc, val_srocc, predict, label = val_model(my_model, device, criterion, val_loader)

        writer_v.add_scalar('loss', val_loss, epoch)
        writer_v.add_scalar('plcc', val_plcc, epoch)
        writer_v.add_scalar('srocc', val_srocc, epoch)
        print('\033[1;31m val_plcc: ', val_plcc, '\033[0m')
        print('\033[1;31m val_srocc: ', val_srocc, '\033[0m')
        ####################################################
    writer_t.close()
    writer_v.close()

    # save model
    final_model = {
        'state_dict': my_model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'epoch': start_epoch + epoch_nums
    }
    torch.save(final_model, model_save_dir + '/final_model.pth')
    print('done')
