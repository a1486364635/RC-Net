import os 
import torch
import sys
from tqdm import tqdm
from loss import softdiceloss

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = softdiceloss()
    loss_function1 = torch.nn.BCELoss()
    loss_function2 = softdiceloss()
    accu_loss = torch.zeros(1).to(device)   # 累计损失
    optimizer.zero_grad()
    sample_num = 0.
    data_loader = tqdm(data_loader, file=sys.stdout)   # 进度条
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        
        pred = torch.sigmoid(model(images))     # 预测
        loss1 = loss_function1(pred, labels)
        loss2 = loss_function2(pred, labels)
        loss = loss1+loss2
        # loss = loss_function1(pred, labels)
        loss.backward()
        # clip_gradient(optimizer, 0.5)  # 防止梯度爆炸
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.6f}".format(epoch, accu_loss.item() / sample_num)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item()/sample_num


def eval(model, data_loader, device, epoch):
    # loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = softdiceloss()
    loss_function1 = torch.nn.BCELoss()
    loss_function2 = softdiceloss()
    model.eval()
    accu_loss = torch.zeros(1).to(device)   # 累计损失
    sample_num = 0.
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            images, labels = images.to(device), labels.to(device)
            pred = torch.sigmoid(model(images))
            loss1 = loss_function1(pred, labels)
            loss2 = loss_function2(pred, labels)
            loss = loss1+loss2
            # loss = loss_function1(pred, labels)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.6f}".format(epoch, accu_loss.item()/sample_num)
    return accu_loss.item()/sample_num

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)    # 将梯度限制在(-grad_clip, grad_clip)内

def update_lr(optimizer,lr,index,factor=False):
    if factor:
        new_lr = lr/index
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('update learning rate: %f -> %f' % (lr,new_lr))
    return new_lr