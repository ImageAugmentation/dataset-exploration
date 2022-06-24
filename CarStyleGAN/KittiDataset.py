import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from tools import image_utils


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, img_root='image_2', mask_root='semantic'):
        self.root = root
        self.transforms = transforms
        self.img_root = img_root
        self.mask_root = mask_root
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.img_root))))
        self.masks = list(sorted(os.listdir(os.path.join(root, self.mask_root))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.img_root, self.imgs[idx])
        mask_path = os.path.join(self.root, self.mask_root, self.masks[idx])

        img = Image.open(img_path)
        # display(img)
        mask = Image.open(mask_path)

        if self.transforms is not None:
            img_tensor = self.transforms(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        # img.size = WxH
        # img_tensor.shape = CxHxW
        if img.size[0] != img_tensor.shape[2] or img.size[1] != img_tensor.shape[1]:
            mask = transforms.Compose([transforms.CenterCrop(tuple(img_tensor.shape[1:])), transforms.ToTensor()])(mask)
        else:
            mask = transforms.ToTensor()(mask)

        fake_image = image_utils.adjust_color(img_tensor, mask, 26, 10)

        return img_tensor, fake_image

    def __len__(self):
        return len(self.imgs)
