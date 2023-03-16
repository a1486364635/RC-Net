import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from my_dataset import MyDataSet
from models import Rcnet
from utils import train_one_epoch, eval, update_lr
from torch.utils.tensorboard import SummaryWriter

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./checkpoints") is False:
        os.makedirs("./checkpoints")
    
    tb_writer = SummaryWriter(comment=args.model_name)

    train_images_path = args.train_images_path
    train_labels_path = args.train_labels_path
    val_images_path = args.val_images_path
    val_labels_path = args.val_labels_path

    img_size = args.image_size
    data_transform = {
        "train": transforms.Compose([transforms.Resize(img_size),
                                     transforms.RandomHorizontalFlip(),   # 随机水平翻转
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize(img_size),
                                   transforms.ToTensor()])}

                        # transforms.RandomRotation(20),   # 在[-20，20]之间旋转
    # 实例化训练集
    train_dataset = MyDataSet(images_path=train_images_path,
                              labels_path=train_labels_path,
                              transform=data_transform["train"])
    # 实例化验证集
    val_dataset = MyDataSet(images_path=val_images_path,
                            labels_path=val_labels_path,
                            transform=data_transform["val"])
    
    batch_size = args.batch_size
    nw = args.num_workers
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,   # 验证集 batch_size = 1
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw)

    model = Rcnet(in_channels=1, out_channels=1).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # optimizer = optim.AdamW(pg, lr=args.lr)
    best_loss = 100.
    best_epoch = 0
    no_optim = 0
    lr = args.lr   # 初始学习率

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)
        # validate
        val_loss = eval(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)
        if val_loss < best_loss:
            no_optim = 0
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "./checkpoints/{}.pth".format(args.model_name))  # 保存验证集中loss最低的模型参数
        else:
            no_optim += 1   # 记录损失不下降的epoch数
        if no_optim > 20:   # 如果连续20epoch损失不下降
            if lr < 5e-7:   # 最小学习率为5e-7
                break
            model.load_state_dict(torch.load("./checkpoints/{}.pth".format(args.model_name)))
            lr = update_lr(optimizer, lr, 2, factor=True)   # 更新学习率,学习率变为原来的一半

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        # if (epoch > 129 and epoch % 10 == 0) or epoch==args.epochs-1:
        # if epoch==args.epochs-1:   #  只保存最后一个epoch的训练结果
        #     torch.save(model.state_dict(),"./checkpoints/{}-{}.pth".format(args.model_name ,epoch))
    print('finish!!! best epoch: {}'.format(best_epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--train_images_path', type=str, default="/home/datacenter/ssd2/data_zb/811/train/image/")
    parser.add_argument('--train_labels_path', type=str, default="/home/datacenter/ssd2/data_zb/811/train/lab/")
    parser.add_argument('--val_images_path', type=str, default="/home/datacenter/ssd2/data_zb/811/valid/image/")
    parser.add_argument('--val_labels_path', type=str, default="/home/datacenter/ssd2/data_zb/811/valid/lab/")
    parser.add_argument('--image_size', type=int, default=(256,512))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model_name', type=str, default='Rcnet', help='UNet, Rcnet')

    opt = parser.parse_args()   

    main(opt)

