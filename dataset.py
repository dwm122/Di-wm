import os
from PIL import Image
from osgeo import gdal
from torch.utils.data import Dataset
import numpy as np


def tif_read(image_path):
    dataset = gdal.Open(image_path)
    return dataset.ReadAsArray()


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(mask_dir)
        self.label_type = self.labels[0].split(".")[-1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_file = self.images[index].split(".")[:-1]
        mask_file = ".".join(mask_file + [self.label_type])
        # mask_path = os.path.join(self.mask_dir, '_groundtruth_(1)_'+mask_file.replace("original_",""))
        mask_path = os.path.join(self.mask_dir, mask_file)
        image = np.array(tif_read(img_path), dtype=np.int16).transpose(1, 2, 0)
        mask = np.array(tif_read(mask_path), dtype=np.int64)
        # print('image.shape:',image.shape)
        # print('mask.shape:',mask.shape)
        # image = np.array(Image.open(img_path))
        # mask = np.array(Image.open(mask_path),dtype=np.int64)

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask