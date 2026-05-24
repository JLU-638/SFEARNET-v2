import torch
from tqdm import tqdm
from Metric import SegmentationMetric
import logging
from tensorboardX import SummaryWriter

def _normalize_ce(loss_ce: torch.Tensor) -> torch.Tensor:
    # Map cross-entropy loss to [0, 1]
    return 1.0 - torch.exp(-loss_ce)


def train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion_iou: torch.nn.Module,
                criterion_ce: torch.nn.Module,
                criterion_dice_edge: torch.nn.Module,
                criterion_contrast: torch.nn.Module,
                alpha: float,
                beta: float,
                gamma: float,
                device: torch.device,
                epoch_id: int,
                args,
                logger: logging.Logger,
                writer: SummaryWriter,
                batch_train_records: list) -> dict:
    """
    执行一个训练epoch

    :param model: 训练的模型
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param criterion_iou: IOU损失函数
    :param criterion_ce: 交叉熵损失函数
    :param criterion_dice_edge: Dice边缘损失函数
    :param criterion_contrast: 对比损失函数
    :param alpha: segmentation loss weight for CE vs IoU
    :param beta: contrastive loss weight
    :param gamma: edge loss weight
    :param device: 设备
    :param epoch_id: 当前epoch编号
    :param args: 参数对象
    :param logger: 日志记录器
    :param writer: TensorBoard写入器
    :param batch_train_records: 批次记录列表
    :return: 包含训练loss和各项指标的字典
    """
    if not (0.0 <= beta <= 1.0 and 0.0 <= gamma <= 1.0 and beta + gamma <= 1.0):
        raise ValueError(f"beta and gamma must satisfy 0 <= beta <=1, 0 <= gamma <=1 and beta+gamma <=1; got beta={beta}, gamma={gamma}")

    model.train()
    train_metric = SegmentationMetric(numClass=2)
    train_loss_sum = 0.0
    train_batch_count = 0
    max_test_batches = args.max_test_batches

    for idx_batch_train, (image_A, image_B, label, edge, img_ids) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch_id}/{args.num_epochs}")):
        if max_test_batches > 0 and idx_batch_train >= max_test_batches:
            break
        train_batch_count += 1
        image_A, image_B, label, edge = image_A.to(device), image_B.to(device), label.to(device), edge.to(device)

        model_x = model(image_A, image_B)
        label_pred = model_x[0]
        #损失函数分为三部分，1是语义分割（ce，iou）,2是边缘检测（dice_edge），3是特征级别，对比损失（contrast）。alpha控制ce和iou的权重，beta控制contrast的权重，gamma控制edge的权重。总损失是三部分的加权和，并且保证在[0,1]之间。
        loss_ce = criterion_ce(label_pred, label)
        loss_iou = criterion_iou(label_pred, label, 2)
        loss_dice_edge = criterion_dice_edge(model_x[1], edge, 2)
        loss_contrast = criterion_contrast(model_x[2], model_x[3], label)

        loss_ce_norm = _normalize_ce(loss_ce)
        loss_iou_norm = torch.clamp(loss_iou, 0.0, 1.0)
        loss_edge_norm = torch.clamp(loss_dice_edge, 0.0, 1.0)
        loss_contrast_norm = torch.clamp(loss_contrast, 0.0, 1.0)

        loss_seg = alpha * loss_ce_norm + (1.0 - alpha) * loss_iou_norm
        loss = (1.0 - beta - gamma) * loss_seg + beta * loss_contrast_norm + gamma * loss_edge_norm
        loss = torch.clamp(loss, 0.0, 1.0)

        train_loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        label_pred1 = torch.argmax(label_pred, dim=1)
        train_metric.addBatch(label_pred1, label)

        # 记录本batch指标
        batch_train_records.append({
            'epoch': epoch_id,
            'batch': idx_batch_train,
            'img_id': img_ids,  # 记录图像ID列表（batch）
            'loss': loss.item(),
            'iou': train_metric.IntersectionOverUnion(),
            'oa': train_metric.OverallAccuary(),
            'pa': train_metric.PixelAccuary(),
            'recall': train_metric.recall(),
            'F1': train_metric.F1(),
            'kappa': train_metric.kappa(),
        })

        # 每隔print_every_batches批次记录一次中间指标
        if (idx_batch_train + 1) % args.print_every_batches == 0:
            current_avg_loss = train_loss_sum / (idx_batch_train + 1)
            current_iou = train_metric.IntersectionOverUnion()
            current_oa = train_metric.OverallAccuary()
            current_pa = train_metric.PixelAccuary()
            current_recall = train_metric.recall()
            current_f1 = train_metric.F1()
            current_kappa = train_metric.kappa()
            logger.info(f"Epoch [{epoch_id}/{args.num_epochs}], Batch [{idx_batch_train+1}/{len(train_loader)}], "
                       f"avg_loss: {current_avg_loss:.4f}, iou: {current_iou:.4f}, oa: {current_oa:.4f}, "
                       f"pa: {current_pa:.4f}, recall: {current_recall:.4f}, F1: {current_f1:.4f}, kappa: {current_kappa:.4f}")

        # if idx_batch_train + 1 >= 20:
        #     break  # 快速测试时只跑20个批次

    # 计算epoch指标
    train_epoch_loss = train_loss_sum / max(train_batch_count, 1)
    train_once_OA = train_metric.OverallAccuary()
    train_once_PA = train_metric.PixelAccuary()
    train_once_IOU = train_metric.IntersectionOverUnion()
    train_once_recall = train_metric.recall()
    train_once_F1 = train_metric.F1()
    train_once_kappa = train_metric.kappa()

    # 记录到TensorBoard
    writer.add_scalar('train_loss_epoch', train_epoch_loss, epoch_id)
    writer.add_scalar('train_OA_epoch', train_once_OA, epoch_id)
    writer.add_scalar('train_pa_epoch', train_once_PA, epoch_id)
    writer.add_scalar('train_IOU_epoch', train_once_IOU, epoch_id)
    writer.add_scalar('train_recall_epoch', train_once_recall, epoch_id)
    writer.add_scalar('train_F1_epoch', train_once_F1, epoch_id)
    writer.add_scalar('train_kappa_epoch', train_once_kappa, epoch_id)

    
    return {
        'loss': train_epoch_loss,
        'oa': train_once_OA,
        'pa': train_once_PA,
        'iou': train_once_IOU,
        'recall': train_once_recall,
        'f1': train_once_F1,
        'kappa': train_once_kappa
    }


def validate_epoch(model: torch.nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   criterion_iou: torch.nn.Module,
                   criterion_ce: torch.nn.Module,
                   criterion_dice_edge: torch.nn.Module,
                   criterion_contrast: torch.nn.Module,
                   alpha: float,
                   beta: float,
                   gamma: float,
                   device: torch.device,
                   epoch_id: int,
                   args,
                   logger: logging.Logger,
                   writer: SummaryWriter,
                   batch_val_records: list) -> dict:
    """
    执行一个验证epoch

    :param model: 验证的模型
    :param val_loader: 验证数据加载器
    :param criterion_iou: IOU损失函数
    :param criterion_ce: 交叉熵损失函数
    :param criterion_dice_edge: Dice边缘损失函数
    :param criterion_contrast: 对比损失函数
    :param alpha: segmentation loss weight for CE vs IoU
    :param beta: contrastive loss weight
    :param gamma: edge loss weight
    :param device: 设备
    :param epoch_id: 当前epoch编号
    :param args: 参数对象
    :param logger: 日志记录器
    :param writer: TensorBoard写入器
    :param batch_val_records: 批次记录列表
    :return: 包含验证loss和各项指标的字典
    """
    if not (0.0 <= beta <= 1.0 and 0.0 <= gamma <= 1.0 and beta + gamma <= 1.0):
        raise ValueError(f"beta and gamma must satisfy 0 <= beta <=1, 0 <= gamma <=1 and beta+gamma <=1; got beta={beta}, gamma={gamma}")

    model.eval()
    val_metric = SegmentationMetric(numClass=2)
    val_loss_sum = 0.0
    val_batch_count = 0
    max_test_batches = args.max_test_batches

    with torch.no_grad():
        for idx_batch_val, (image_A, image_B, label, edge, img_ids) in enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch_id}/{args.num_epochs}")):
            if max_test_batches > 0 and idx_batch_val >= max_test_batches:
                break
            val_batch_count += 1
            image_A, image_B, label, edge = image_A.to(device), image_B.to(device), label.to(device), edge.to(device)

            model_x = model(image_A, image_B)
            label_pred = model_x[0]

            loss_ce = criterion_ce(label_pred, label)
            loss_iou = criterion_iou(label_pred, label, 2)
            loss_dice_edge = criterion_dice_edge(model_x[1], edge, 2)
            loss_contrast = criterion_contrast(model_x[2], model_x[3], label)

            loss_ce_norm = _normalize_ce(loss_ce)
            loss_iou_norm = torch.clamp(loss_iou, 0.0, 1.0)
            loss_edge_norm = torch.clamp(loss_dice_edge, 0.0, 1.0)
            loss_contrast_norm = torch.clamp(loss_contrast, 0.0, 1.0)

            loss_seg = alpha * loss_ce_norm + (1.0 - alpha) * loss_iou_norm
            loss = (1.0 - beta - gamma) * loss_seg + beta * loss_contrast_norm + gamma * loss_edge_norm
            loss = torch.clamp(loss, 0.0, 1.0)
            val_loss_sum += loss.item()

            label_pred1 = torch.argmax(label_pred, dim=1)
            val_metric.addBatch(label_pred1, label)

            # 记录本batch指标
            batch_val_records.append({
                'epoch': epoch_id,
                'batch': idx_batch_val,
                'img_id': img_ids,  # 记录图像ID列表（batch）
                'loss': loss.item(),
                'iou': val_metric.IntersectionOverUnion(),
                'oa': val_metric.OverallAccuary(),
                'pa': val_metric.PixelAccuary(),
                'recall': val_metric.recall(),
                'F1': val_metric.F1(),
                'kappa': val_metric.kappa(),
            })

            # 每隔print_every_batches批次记录一次中间指标
            if (idx_batch_val + 1) % args.print_every_batches == 0:
                current_avg_loss = val_loss_sum / (idx_batch_val + 1)
                current_iou = val_metric.IntersectionOverUnion()
                current_oa = val_metric.OverallAccuary()
                current_pa = val_metric.PixelAccuary()
                current_recall = val_metric.recall()
                current_f1 = val_metric.F1()
                current_kappa = val_metric.kappa()
                logger.info(f"Val Epoch [{epoch_id}/{args.num_epochs}], Batch [{idx_batch_val+1}/{len(val_loader)}], "
                           f"avg_loss: {current_avg_loss:.4f}, iou: {current_iou:.4f}, oa: {current_oa:.4f}, "
                           f"pa: {current_pa:.4f}, recall: {current_recall:.4f}, F1: {current_f1:.4f}, kappa: {current_kappa:.4f}")

    # 计算epoch指标
    val_epoch_loss = val_loss_sum / max(val_batch_count, 1)
    val_once_OA = val_metric.OverallAccuary()
    val_once_PA = val_metric.PixelAccuary()
    val_once_IOU = val_metric.IntersectionOverUnion()
    val_once_recall = val_metric.recall()
    val_once_F1 = val_metric.F1()
    val_once_kappa = val_metric.kappa()

    # 记录到TensorBoard
    writer.add_scalar('val_loss_epoch', val_epoch_loss, epoch_id)
    writer.add_scalar('val_OA_epoch', val_once_OA, epoch_id)
    writer.add_scalar('val_pa_epoch', val_once_PA, epoch_id)
    writer.add_scalar('val_IOU_epoch', val_once_IOU, epoch_id)
    writer.add_scalar('val_recall_epoch', val_once_recall, epoch_id)
    writer.add_scalar('val_F1_epoch', val_once_F1, epoch_id)
    writer.add_scalar('val_kappa_epoch', val_once_kappa, epoch_id)

    return {
        'loss': val_epoch_loss,
        'oa': val_once_OA,
        'pa': val_once_PA,
        'iou': val_once_IOU,
        'recall': val_once_recall,
        'f1': val_once_F1,
        'kappa': val_once_kappa
    }