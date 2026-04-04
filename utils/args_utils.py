import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train Change Detection Models')

    parser.add_argument('--data_dir', default=r'./data/datasets/LEVIR_CD_256/', type=str, help='where for data.')
    parser.add_argument('--result_dir', default=r'./LEVIR_CD_result_1/', type=str, help='where to write.')
    parser.add_argument('--data', default=r'LEVIR_CD', type=str, help='which dataset')

    parser.add_argument('--train_batchsize', default=8, type=int, 
                        help="batchsizefor train。"
                        "训练模式下，BatchNorm 需要计算每个通道的均值和方差，用于归一化。"
                        "计算方差需要至少 2 个样本（batch_size >= 2）"
                        "训练时使用 batch_size >= 2")
    parser.add_argument('--val_batchsize', default=1, type=int, help='batchsize for validation')
    parser.add_argument('--lr',  default=1e-4,type=float , help='initial learning rate for adam')
    parser.add_argument('--num_epochs', default= 100, type=int, help='train epoch number')
    parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
    parser.add_argument('--model', default=r'SFEARNet', type=str, help='which model')
    parser.add_argument('--lr_decline', default=r'ReduceLROnPlateau', type=str, help='ReduceLROnPlateau')

    parser.add_argument('--weight_decay', default=1e-3,type=float ,   help='weight_decay')

    parser.add_argument('--lamda', default=1,type=float ,   help='edge_loss')

    parser.add_argument('--seed', default=0, type=int, help='random seed for reproducibility')

    parser.add_argument('--print_every_batches', default=100, type=int, help='print metrics every N batches during training')
    parser.add_argument('--max_test_batches', default=0, type=int,
                        help='max batches per epoch for quick test; 0 means use full loader')

    return parser.parse_args()


