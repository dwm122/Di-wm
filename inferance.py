from PIL import Image
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import tif_read

from models.github_unet import UNet as net

# from models.github_unet.unet_model import UNet as net

transform = A.Compose(
    [
        # A.Resize(height=128,width=128),
        A.Normalize(  # 归一化
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=10000.0,
        ),
    ]
)


def get_test(path, batch_size):
    files = os.listdir(path)
    for i in range(0, len(files), batch_size):
        imgs = []
        for ind in range(i, min(i + batch_size, len(files))):
            img = tif_read(os.path.join(path, files[ind]))
            img = np.array(img).transpose(1, 2, 0)
            img = transform(image=np.array(img))
            imgs.append(img["image"].transpose(2, 0, 1))
        yield torch.tensor(np.array(imgs)), files[i:min(i + batch_size, len(files))]


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"/root/autodl-tmp/mycode/ASPP-unet/cheakpoint/classification-8/epoch_200.pth"
# test_dir = r"dataset\test_image"
# save_dir = r"dataset\pred\nosoft"
test_dir = r"/root/autodl-tmp/mycode/ASPP-unet/dataset/test_image"
save_dir = r"/root/autodl-tmp/mycode/ASPP-unet/dataset/result/val8"
batch_size = 8
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

net = net(3, 7)
net.to(device=device)
cheak_point = torch.load(model_path)["state_dict"]
net.load_state_dict(cheak_point)

total_num = len(os.listdir(test_dir))

net.eval()
for x, files in tqdm(get_test(test_dir, batch_size), total=total_num // batch_size + 1):
    with torch.no_grad():
        x = x.to(device)
        pres = net(x)
        pres = torch.argmax(pres, 1)
    pres = np.array(pres.cpu(), dtype=np.uint8)
    for pre, file in zip(pres, files):
        img = Image.fromarray(pre)
        img.save(os.path.join(save_dir, file))
