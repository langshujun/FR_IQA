import argparse
import os
import time

LIVE2005_ref = 'G:/python/base/data/LIVE2005/databaserelease2/ALL'
LIVE2005_dir = 'G:/python/base/data/LIVE2005/databaserelease2/ALL'
LIVE2005_train_path = 'G:/python/base/data/LIVE_ALL_train.csv'
LIVE2005_test_path = 'G:/python/base/data/LIVE_ALL_test.csv'

CSIQ_ref = 'G:/python/base/data/CSIQ/dst_imgs'
CSIQ_dir = 'F:/FF_data/CSIQ'
CSIQ_train_path = 'F:/FF_data/CSIQmos_train.csv '
CSIQ_test_path = 'F:/FF_data/CSIQmos_test.csv'

TID2013_ref = 'G:/python/base/data/TID_2013'
TID2013_dir = 'G:/python/base/data/TID_2013'
TID2013_train_path = 'G:/python/base/data/TID_train.csv'
TID2013_test_path = 'G:/python/base/data/TID_test.csv'

kadid10k_ref = 'G:/python/base/data/kadid10k/images'
kadid10k_dir = 'G:/python/base/data/kadid10k/images'
kadid10k_train_path = 'G:/python/base/data/kadid10k_train.csv'
kadid10k_test_path = 'G:/python/base/data/kadid10k_test.csv'


CLIVE_ref = 'G:/python/base/data/CLIVE/Images/'
CLIVE_dir = 'G:/python/base/data/CLIVE/Images/'
CLIVE_train_path = 'G:/python/base/data/CLIVE.csv'
CLIVE_test_path = 'G:/python/base/data/CLIVE.csv'

KonIQ_ref = 'G:/python/base/data/KonIQ/'
KonIQ_dir = 'G:/python/base/data/KonIQ/'
KonIQ_train_path = 'G:/python/base/data/KonIQ.csv'
KonIQ_test_path = 'G:/python/base/data/KonIQ_test.csv'


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='train', type=str, help='train retrain or predict')
    parser.add_argument('--image_dir_train', default=CSIQ_dir, type=str, help='Path to input images about train')
    parser.add_argument('--image_dir_test', default=CSIQ_dir, type=str, help='Path to input images about test')

    parser.add_argument('--label_train_path', default=CSIQ_train_path, type=str,
                        help='Path to input train database score')
    parser.add_argument('--label_test_path', default=CSIQ_test_path, type=str,
                        help='Path to input info')

    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='L2 regularization')
    parser.add_argument('--epoch_nums', default=32, type=int, help='epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    parser.add_argument('--start_time', default=start_time, type=str,
                        help='start time of this process')
    parser.add_argument('--save_model', default='./model-save/' + start_time, type=str,
                        help='path to save the model')
    parser.add_argument('--save_checkpoint', default=True, type=bool, help='')
    parser.add_argument('--load_model', default='', type=str, help='path to load checkpoint')
    parser.add_argument('--writer_t_dir', default='./runs/' + start_time + '_train', type=str, help='batch size to train')
    parser.add_argument('--writer_v_dir', default='./runs/' + start_time + '_val', type=str, help='batch size to train')

    args = parser.parse_args()

    if not os.path.isdir('data/model-save/'):
        os.mkdir('data/model-save/')
    if not os.path.isdir('data/runs/'):
        os.mkdir('data/runs/')

    return args


if __name__ == '__main__':
    args = parse_opts()
    print(args)
