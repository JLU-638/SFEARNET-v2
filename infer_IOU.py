# coding=utf-8
import os
import datetime

import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from model.sfearnet import SFEARNet
from data.Dataset import Dataset
from Metric import SegmentationMetric


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############需要改的部分

save_dir = './vis_beta_alg/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_classes=2

re=[]
if __name__ == '__main__':
    ###############加载数据
    # DATA_DIR = opt.data_dir  # 根据自己的路径来设置
    DATA_DIR = "data/datasets/LEVIR_CD_256"
    # print(DATA_DIR)
    test_dir_A = os.path.join(DATA_DIR, 'test/A')
    test_dir_B = os.path.join(DATA_DIR, 'test/B')
    test_dir_label = os.path.join(DATA_DIR, 'test/label')
    test_dir_edge = os.path.join(DATA_DIR, 'test/edge')


    test_dataset = Dataset(test_dir_A, test_dir_B, test_dir_label,test_dir_edge,is_train=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = torch.load('/home/fengruyue/lvguixin/SFEARNet_test/logs/model_SFEARNet_data_LEVIR_CD_lr_0.0001_bs_8_wd_0.001_lam_1.0_ep_100_seed_0/run_2026-04-28_21-05-24/best_model/best_model.pth').to(device)

    model.eval()
    test_SegmentationMetric = SegmentationMetric(numClass=n_classes)
    IOU_result=[]
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for  image_A, image_B, label,edge,img_id in test_bar:
            image_A, image_B, label,edge = image_A.to(device), image_B.to(device), label.to(device),edge.to(device)
            img_id_str=",".join(img_id)
            print(img_id_str)
            #break
            model_x = model(image_A, image_B)
            prob=model_x[0]
            label_pred1 = torch.argmax(prob, dim=1)
            #print(label_pred1.size())
            #compute IOU by own
            label_iou = np.array(label.cpu())
            label_pre_iou = np.array(label_pred1.cpu())
            #61乘法是否应该
            #这两个值是整数还是浮点数 type
            print(f"type of label: {type(label_iou)}, type of label_pred1: {type(label_pre_iou)}")
            intersection = np.sum(label_iou * label_pre_iou)
            union = np.sum(label_iou) + np.sum(label_pre_iou) - intersection
            #真实标签和预测标签都是纯黑时候，算交并比分母为0，所以强行指定IOU为1
            if union == 0:
                sample_IOU = 1.0
            else:
                sample_IOU = intersection / (union + 1e-8)  # +1e-8 防止除0
            IOU_result.append({
                "img_id":img_id_str,
                "sample":sample_IOU
                })
  
            file_path = save_dir + img_id_str.split('.')[-2].split('\\')[-1] + '.png'

            cd_pred1 = label_pred1.unsqueeze(1)

            cd_preds = label_pred1.data.cpu().numpy()
            cd_preds = cd_preds.squeeze() * 255

            cv2.imwrite(file_path, cd_preds)
        IOU_result.sort(key=lambda x:x['sample'], reverse=False)
        print(IOU_result)
        for i in IOU_result:
            re.append([i["img_id"], i["sample"]])

        data = pd.DataFrame(data=re, index=None, columns=['img_id', 'IOU'])
        # #print(data)
        data.to_csv(save_dir+'infer.csv')






