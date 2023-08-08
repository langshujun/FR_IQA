import torch
import numpy as np
from scipy import stats


def val_model(my_model, device, criterion, val_loader, show_detail=True):  #criterion--损失函数
    my_model.eval()
    with torch.no_grad():

        predict = np.array([])
        label = np.array([])
        val_loss = 0
        # 预测步骤开始
        # for dist, qy in val_loader:
        # testing_batch_generator = get_testing_batch(val_loader)
        for i, inputs in enumerate(val_loader):
            ref = inputs['ref'].to(device, non_blocking=True)
            dist = inputs['dis'].to(device, non_blocking=True)
            score = my_model(ref, dist).to(device)
            qy = inputs['label'].to(device, non_blocking=True)
            qy = qy.transpose(-1, 0).unsqueeze(-1)

            loss = criterion(score, qy)

            predict = np.append(predict, score.cpu().detach().numpy())
            label = np.append(label, qy.cpu().numpy())

            if (i+1) % 40 == 0:
                if show_detail:
                    print('\033[1;33m'
                          '----------------------------------------'
                          'Validating: %d / %d'
                          '----------------------------------------'
                          ' \033[0m' % (i + 1, len(val_loader)))
                    print('\033[1;31m loss: ', loss.item(), '\033[0m')
                    print('\033[1;34m score: ', score, '\033[0m')
                    print('\033[1;34m MOS: ', qy, '\033[0m')


            val_loss += loss.item()
            # next(testing_batch_generator)
            # val_loss = np.append(val_loss, loss.cpu().numpy())

        val_plcc = stats.pearsonr(predict, label)[0]
        val_srocc = stats.spearmanr(predict, label)[0]
        val_loss = val_loss / (i+1)
        print("plcc: {}, srocc: {}, loss: {}".format(val_plcc, val_srocc, loss))
        return val_loss, val_plcc, val_srocc, predict, label

