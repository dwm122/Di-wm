from cv2 import transform
import torch
import torchvision
from dataset import CarvanaDataset
from labeltrans import gray2rgb
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os


def save_checkpoint(epoch_num, state, filepath):
    torch.save(state, f"{filepath}/epoch_{epoch_num}.pth")


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_val_accuracy(loader, model, loss_fn, device="cuda"):
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y.long())
            total_loss += loss
            preds = torch.argmax(preds, 1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
    t_loss = round(float(total_loss / len(loader)), 2)
    t_acc = round(float(num_correct / num_pixels * 100), 2)
    print(
        f"val_loss:{t_loss}    val_accuracy: {t_acc}"
    )
    # print(f"Dice score:{dice_score/len(loader)}")
    model.train()
    return t_loss, t_acc


def check_train_accuracy(loader, model, loss_fn, device="cuda"):
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y.long())
            total_loss += loss
            preds = torch.argmax(preds, 1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
    t_loss = round(float(total_loss / len(loader)), 2)
    t_acc = round(float(num_correct / num_pixels * 100), 2)

    print(
        f"\ntrain_loss:{t_loss}    train_accuracy: {t_acc}"
    )
    # print(f"Dice score:{dice_score/len(loader)}")
    model.train()
    return t_loss, t_acc


# 生成预测图像
def save_predictions_as_imgs(
        epoch_num, model, transform, in_folder=r"predict\img", out_folder=r"predict\pre", device="cuda"
):
    out_folder = f"{out_folder}/epoch{epoch_num}"
    files = os.listdir(in_folder)
    if len(files) == 0:
        pass
    model.eval()
    imgs = []
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    with torch.no_grad():
        for file in files:
            imgs = []
            img = Image.open(os.path.join(in_folder, file))
            img = np.array(img)
            img = transform(image=img)["image"]
            img = np.array(img, dtype=np.float32)
            imgs.append(img)
            x = torch.tensor(np.array(imgs), dtype=torch.float32)
            x = x.cuda()
            preds = model(x)
            preds = torch.argmax(preds, 1)
            preds = np.array(preds.cpu())
            pred = gray2rgb(preds[0])
            img = Image.fromarray(np.array(pred, dtype=np.uint8))
            img.save(os.path.join(out_folder, file))
    model.train()


def adjust_learning_rate(optimizer, epoch, start_lr):
    if epoch % 1 == 0 and epoch != 0:  # epoch != 0 and
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.95
            print(param_group["lr"])


def write_txt_log(data_path, text):
    with open(data_path, "a") as f:
        f.write(f"{text}\n")
        f.close


def check_accuracy():
    return None


def epoch_num():
    return None