from psutil import net_connections
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch

import albumentations as A

from torch.utils.tensorboard import SummaryWriter
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import torch.nn as nn

from models.u_net import UNet



import torch.optim as optim
# from mymodel import UNET
from utils import (
    # loat_checkpoint,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_train_accuracy,
    check_val_accuracy,
    save_predictions_as_imgs,
    adjust_learning_rate,
    write_txt_log,
)


# from models.github_unet.unet_model import UNet as net

# 在labeltrans.py中有预测标签的转换列表，需要根据自己的数据集进行修改

torch.backends.cudnn.benchmark = False

# Hyperparameters etc. 超参数
# 必须修改的参数 soft在84行
MODEL_NAME = "classification-2"  # 本次训练模型名称（改）
LOAD_MODEL = False  # 加载预训练模型
LOAD_MODEL_PATH = r"C:\Users\User\Desktop\底\底微萌\CBAM ASPP RES\CBAM ASPP RES\checkpoint\classification-1\epoch_150.pth"  # 预训练模型参数(改)
START_EPOCH = 0  # 如果不是继续训练，设为0
dataset_dir = "datasets"
TRAIN_IMG_DIR = os.path.join(dataset_dir, "train_image")
TRAIN_MASK_DIR = os.path.join(dataset_dir, "train_label")
VAL_IMG_DIR = os.path.join(dataset_dir, "test_image")
VAL_MASK_DIR = os.path.join(dataset_dir, "test_label")

LEARNING_RATE = 0.001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
BATCH_SIZE = 16  # batch_size （内存不足就改）
IN_CHENNELS = 3
OUT_CHENNELS = 6  # （类别要改）
NUM_EPOCH = 150  # epoch
NUM_WORKERS = 0  # 线程
IMAGE_HEIGHT = 256  # 图像h
IMAGE_WIDTH = 256  # 图像w
PIN_MEMORY = True  # 锁页内存
# MODEL_NAME = "test"
SAVE_MODEL_PATH = os.path.join("cheakpoint", MODEL_NAME)  # (不必修改)模型参数保存位置
LOG_PATH = os.path.join("log", MODEL_NAME)  # (不必修改)日志记录位置
log_txt_path = os.path.join(LOG_PATH, f"{MODEL_NAME}.txt")  # (不必修政)日志txt记录位置


def train_fn(epoch_num, loader, model, optimizer, loss_fn, scaler, model_choice=None):
    loop = tqdm(loader, desc=f"epoch{epoch_num}")

    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward 正向传播
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # predictions = torch.softmax(predictions, dim=1)
            # print('predictions.shape:',predictions.shape)
            # print('targets.long().shape:',targets.long().shape)
            loss = loss_fn(predictions, targets.long())

        # backwoed 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop 更新进度条
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    print(f"train_epoch_loss:{total_loss / len(loader):.2f}", end="    ")
    return round(total_loss / len(loader), 2)


def main(model_choice=None):
    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),    # 重新设置大小
            # A.Rotate(limit=35, p=0.5),  # 旋转
            # A.HorizontalFlip(p=0.5),    # 水平翻转
            # A.VerticalFlip(p=0.5),      # 垂直翻转
            A.Normalize(  # 归一化
                mean=[0.0, 0.0, 0.0],  # 三波段要改
                std=[1.0, 1.0, 1.0],
                max_pixel_value=10000.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            A.Normalize(  # 归一化
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=10000.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNet(IN_CHENNELS, OUT_CHENNELS).to(DEVICE)

    if LOAD_MODEL:
        cheakpoint = torch.load(LOAD_MODEL_PATH)
        load_checkpoint(cheakpoint, model)
        print(f"加载模型:{LOAD_MODEL_PATH}")
        del cheakpoint
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=95)

    train_lorder, val_lorder = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(LOG_PATH)
    write_txt_log(log_txt_path, "epoch\ttrain_loss\tval_loss\ttrain_acc\tval_acc\n")

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCH):
        adjust_learning_rate(optimizer, epoch, LEARNING_RATE)
        train_loss = train_fn(epoch + 1, train_lorder, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(epoch + 1, checkpoint, SAVE_MODEL_PATH)

        # 每5个epoch计算一次训练集精度
        if (epoch + 1) % 5 == 0:
            _, train_acc = check_train_accuracy(train_lorder, model, loss_fn, device=DEVICE)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
        # check accuracy 估计验证集的loss和准确率
        else:
            val_loss, val_acc = check_val_accuracy(val_lorder, model, loss_fn, device=DEVICE)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/test', val_acc, epoch + 1)

        if (epoch + 1) % 5 == 0:
            write_txt_log(log_txt_path, f"{epoch + 1}, {train_loss}, {val_loss}, {train_acc}, {val_acc}")
        else:
            write_txt_log(log_txt_path, f"{epoch + 1}, {train_loss}, {val_loss}, 0, {val_acc}")
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('lr/train', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)
    writer.close()


import os

if __name__ == "__main__":
    for path in [SAVE_MODEL_PATH, LOG_PATH]:
        os.makedirs(path, exist_ok=True)

    main()
