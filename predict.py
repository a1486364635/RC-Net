#encoding:utf-8
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from PIL import Image
from loss import softdiceloss
from my_dataset import MyDataSet, testDataSet
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import imageio
import copy
import numpy as np
from measures import *
from models import Rcnet
import xlwt


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_size = args.image_size
    test_images_path = args.test_images_path
    test_labels_path = args.test_labels_path

    images_name = [name for name in os.listdir(test_images_path)]

    data_transform = transforms.Compose([transforms.Resize(img_size),
                                         transforms.ToTensor()])
    # data_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = testDataSet(images_path=test_images_path,
                             labels_path=test_labels_path,
                             transform=data_transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=1)

    model = Rcnet(n_channels=1, n_classes=1).to(device)
    weights_path = args.weights_path
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    # loss_function = torch.nn.BCEWithLogitsLoss()
    loss_function1 = torch.nn.BCELoss()
    loss_function2 = softdiceloss()
    accu_loss = torch.zeros(1).to(device)   # 累计损失
    sample_num = 0
    F1 = 0.    # F1 score
    F11 = []     # 记录每张图像的F1 score
    iou = 0.
    IoU = []    # 记录每张图像的IoU
    recall = 0.
    Recalls = []  # 记录每张图像的Recall
    precision = 0.
    Precisions = []  # 记录每张图像的Precision
    names = []   # 记录每张图像的name
    test_loader = tqdm(test_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            images, labels, name = data
            names.append(name)
            sample_num += images.shape[0]
            images_color = torch.cat([images,images,images], dim=1)
            images_color = images_color.permute(0,2,3,1)
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            pred = torch.sigmoid(pred)
            loss1 = loss_function1(pred, labels)
            loss2 = loss_function2(pred, labels)
            loss = loss1+loss2
            pred = pred >=0.5
            pred = pred.float()
            F11.append(get_F1(pred, labels, threshold=0.3))   # F1
            IoU.append(get_JS(pred, labels, threshold=0.3))   # IoU
            Recalls.append(get_sensitivity(pred, labels, threshold=0.3))   # Recall
            Precisions.append(get_precision(pred, labels, threshold=0.3))   # Precision
            F1 += get_F1(pred, labels, threshold=0.3)  # 所有图像F1总和
            iou += get_JS(pred, labels, threshold=0.3)
            recall += get_sensitivity(pred, labels, threshold=0.3)
            precision += get_precision(pred, labels, threshold=0.3)
            pred1 = pred.data.cpu().numpy().squeeze()
            images = images.data.cpu().numpy().squeeze()
            labels = labels.data.cpu().numpy().squeeze()
            image_color = images_color.data.cpu().numpy().squeeze()
            img_c = np.zeros((256,512,3))
            img_r = copy.deepcopy(images)
            img_g = copy.deepcopy(images)
            img_b = copy.deepcopy(images)
            
            for i in list(range(img_r.shape[0])):   # 将gt和pred一起显示在原图像中，gt为红色，pred为绿色
                for j in list(range(img_r.shape[1])):
                    if labels[i,j]==1 and pred1[i,j]!=1:
                        img_r[i,j] = 1
                        img_g[i,j] = 0
                        img_b[i,j] = 0
                    elif labels[i,j]==1 and pred1[i,j]==1:
                        img_r[i,j] = 1
                        img_g[i,j] = 1
                        img_b[i,j] = 0
                    elif labels[i,j]!=1 and pred1[i,j]==1:
                        img_r[i,j] = 0
                        img_g[i,j] = 1
                        img_b[i,j] = 0
            img_c[:,:,0] = img_r
            img_c[:,:,1] = img_g
            img_c[:,:,2] = img_b
            temp = np.ones((5,img_r.shape[1]))
            temp1 = np.ones((5,img_r.shape[1],3))
            save_binary_path = "./{} results/binary/".format(args.model_name)
            save_color_path = "./{} results/color/".format(args.model_name)
            save_fuse_path = "./{} results/fuse/".format(args.model_name)
            if not os.path.exists(save_binary_path):
                os.makedirs(save_binary_path)
            if not os.path.exists(save_color_path):
                os.makedirs(save_color_path)
            if not os.path.exists(save_fuse_path):
                os.makedirs(save_fuse_path)

            imageio.imsave(save_binary_path + images_name[step], pred1)
            imageio.imsave(save_fuse_path + images_name[step], np.concatenate([labels,temp,pred1],axis=0))
            imageio.imsave(save_color_path + images_name[step], np.concatenate([image_color,temp1,img_c],axis=0))

            accu_loss += loss.detach()
            test_loader.desc = "[sample {}] loss: {:.6f}".format(step, accu_loss.item()/sample_num)
        print("average F1 score: %.6f, average IoU: %.6f, average recall: %.6f, average precision: %.6f" % (F1/sample_num, iou/sample_num, recall/sample_num, precision/sample_num))
        # print("F1 score: ", F11)
        # print(names)
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet('sheet1')
    col = ('name', 'F1score', 'Recall', 'Precision')
    for j in range(len(col)):
        sheet1.write(0,j,col[j])   # 写入表头
    for i in range(len(names)):
        sheet1.write(i+1, 0, names[i])   # 第一列为图片名字
        sheet1.write(i+1, 1, F11[i])   # 第二列为Fscore
        sheet1.write(i+1, 2, Recalls[i])   # 第三列为Recall
        sheet1.write(i+1, 3, Precisions[i])   # 第四列为Precision
    save_path = "./{} results/result.xls".format(args.model_name)
    workbook.save(save_path)
    return accu_loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--image_size', type=int, default=(256,512))
    parser.add_argument('--test_images_path', type=str, default="/home/datacenter/ssd2/data_zb/811/test/image/")
    parser.add_argument('--test_labels_path', type=str, default="/home/datacenter/ssd2/data_zb/811/test/lab/")
    parser.add_argument('--save_color_path', type=str, default='/home/zhangbo/testcode/HRD_20221018/results/color/')
    parser.add_argument('--save_binary_path', type=str, default='/home/zhangbo/testcode/HRD_20221018/results/binary/')
    parser.add_argument('--save_fuse_path', type=str, default='/home/zhangbo/testcode/HRD_20221018/results/fuse/')

    parser.add_argument('--weights_path', type=str, default="./checkpoints/Rcnet.pth")    # 训练权重载入路径
    parser.add_argument('--model_name', type=str, default='Rcnet', help='UNet, Rcnet')   # 测试模型选择

    opt = parser.parse_args()

    main(opt)
