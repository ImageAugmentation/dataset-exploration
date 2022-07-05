import os

import numpy as np
import torch
import torchvision.transforms as transforms
from IPython.core.display_functions import display
from PIL import Image

import config
from tools import image_utils


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size: int = 256, img_root='image_2', mask_root='semantic'):
        self.root = root
        self.image_size = image_size
        self.img_root = img_root
        self.mask_root = mask_root
        self.imgs = list(
            sorted([file for file in os.listdir(os.path.join(root, self.img_root)) if file.endswith(".png")]))
        self.masks = list(
            sorted([file for file in os.listdir(os.path.join(root, self.mask_root)) if file.endswith(".png")]))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.img_root, self.imgs[idx])
        mask_path = os.path.join(self.root, self.mask_root, self.masks[idx])

        img = Image.open(img_path)
        print(img.size)
        mask = Image.open(mask_path)

        CAR_LABEL = 26
        ADJUSTMENT_FACTOR = 10
        fake_image = image_utils.adjust_color(transforms.ToTensor()(img), transforms.ToTensor()(mask), CAR_LABEL,
                                              ADJUSTMENT_FACTOR)

        ###########
        input_image = np.asarray(fake_image)
        target_image = np.asarray(img)

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]
        ###########

        return input_image, target_image

    def __len__(self):
        return len(self.imgs)
