from pprint import pformat

from tqdm import tqdm
from data.Dataset import Dataset
from utils import get_args, set_seed, create_loggerGt, prepare_train_log_dirs
from utils.stages import train_epoch, validate_epoch
import os
from torch.utils.data import DataLoader
import pandas as pd
from loss.Iouloss import IoULoss
from loss.ContrastiveLoss import ContrastiveLoss
import torch
from model.sfearnet import SFEARNet
from torch.optim import lr_scheduler


from tensorboardX import SummaryWriter

from loss.Diceloss import Diceloss

# def seed_torch(seed=1): # 已经用了更高效的种子指定函数。
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)


args = get_args()

# set seed
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_decay = args.weight_decay
beta = args.beta
alpha = args.alpha
gamma = args.gamma

if beta + gamma > 1.0:
    raise ValueError(
        f"beta + gamma must be <= 1.0, got beta={beta}, gamma={gamma}")

model = SFEARNet(2, phi='b0', pretrained=True).to(device, dtype=torch.float)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=weight_decay)

lr_scheduler_model = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=6,
    min_lr=1e-20
)

criterion_iou = IoULoss()
criterion_ce = torch.nn.CrossEntropyLoss()
criterion_dice_edge = Diceloss()  # 边缘损失函数
in_channels = model.in_channels[3]
criterion_contrast = ContrastiveLoss(
    in_channels=in_channels, margin=3.0)  # 对比损失函数

# Move criteria to device
criterion_iou = criterion_iou.to(device)
criterion_ce = criterion_ce.to(device)
criterion_dice_edge = criterion_dice_edge.to(device)
criterion_contrast = criterion_contrast.to(device)
# 创建实验目录:
# logs/<data>/<model>/<超参摘要>/run_<时间>
run_dir, timeLocal = prepare_train_log_dirs(args=args, base_dir="logs")

# 创建 logger
logger = create_loggerGt(dirLog=run_dir,
                         name="training",
                         t=timeLocal)
logger.info(f"实验参数：{pformat(vars(args))}")
writer = SummaryWriter(os.path.join(run_dir, "TF_log"))
save_model_dir = os.path.join(run_dir, "best_model")
save_csv_dir = os.path.join(run_dir, "csv")

data_dir = args.data_dir
Batch_Size = args.train_batchsize

train_dir_A = os.path.join(data_dir, 'train/A/')
train_dir_B = os.path.join(data_dir, 'train/B/')
train_label_dir = os.path.join(data_dir, 'train/label/')
train_edge_dir = os.path.join(data_dir, 'train/edge/')

val_dir_A = os.path.join(data_dir, 'val/A/')
val_dir_B = os.path.join(data_dir, 'val/B/')
val_label_dir = os.path.join(data_dir, 'val/label/')
val_edge_dir = os.path.join(data_dir, 'val/edge/')

max_IOU = 0.0

if __name__ == '__main__':
    train_dataset = Dataset(train_dir_A, train_dir_B,
                            train_label_dir, train_edge_dir, is_train=True)
    val_dataset = Dataset(val_dir_A, val_dir_B,
                          val_label_dir, val_edge_dir, is_train=False)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(
        train_dataset, batch_size=Batch_Size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset, batch_size=Batch_Size, shuffle=False, **loader_kwargs)

    batch_train_records = []
    batch_val_records = []
    epoch_train_records = []
    epoch_val_records = []

    for epoch_id in range(1, args.num_epochs + 1):
        # 训练阶段
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_iou, criterion_ce, criterion_dice_edge, criterion_contrast,
            alpha, beta, gamma, device, epoch_id, args, logger, writer, batch_train_records
        )

        # 记录训练epoch指标
        epoch_train_records.append([
            epoch_id, train_metrics['loss'], train_metrics['oa'], train_metrics['pa'],
            train_metrics['iou'], train_metrics['recall'], train_metrics['f1'], train_metrics['kappa']
        ])

        # 验证阶段
        val_metrics = validate_epoch(
            model, val_loader, criterion_iou, criterion_ce, criterion_dice_edge, criterion_contrast,
            alpha, beta, gamma, device, epoch_id, args, logger, writer, batch_val_records
        )

        # 记录验证epoch指标
        epoch_val_records.append([
            epoch_id, val_metrics['loss'], val_metrics['oa'], val_metrics['pa'],
            val_metrics['iou'], val_metrics['recall'], val_metrics['f1'], val_metrics['kappa']
        ])

        # 学习率调度
        lr = optimizer.param_groups[0]['lr']
        lr_scheduler_model.step(val_metrics['loss'])
        writer.add_scalar('lr', lr, epoch_id)

        # 记录epoch总结信息
        logger.info(f"Epoch [{epoch_id}/{args.num_epochs}], "
                    f"train_loss: {train_metrics['loss']:.4f}, "
                    f"val_loss: {val_metrics['loss']:.4f}, "
                    f"train_IOU: {train_metrics['iou']:.4f}, "
                    f"val_IOU: {val_metrics['iou']:.4f}")

        # 保存最佳模型
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir, exist_ok=True)

        if val_metrics['iou'] > max_IOU:
            max_IOU = val_metrics['iou']
            torch.save(model, os.path.join(save_model_dir, 'best_model.pth'))
            logger.info(f'model saved, max_IOU={max_IOU}')

    os.makedirs(save_csv_dir, exist_ok=True)
    pd.DataFrame(epoch_train_records, columns=['epoch', 'loss', 'oa', 'pa', 'iou', 'recall', 'F1', 'kappa']).to_csv(
        os.path.join(save_csv_dir, 'train.csv'), index=False)
    pd.DataFrame(epoch_val_records, columns=['epoch', 'loss', 'oa', 'pa', 'iou', 'recall', 'F1', 'kappa']).to_csv(
        os.path.join(save_csv_dir, 'val.csv'), index=False)
    pd.DataFrame(batch_train_records).to_csv(os.path.join(
        save_csv_dir, 'train_batches.csv'), index=False)
    pd.DataFrame(batch_val_records).to_csv(os.path.join(
        save_csv_dir, 'val_batches.csv'), index=False)

    del model
