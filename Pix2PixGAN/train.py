import os
import sys

# pytorch package
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
# root package
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")))

import multiprocessing
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import KittiDataset as dataset
from discriminator import Discriminator
from generator import Generator
from utils import save_checkpoint, load_checkpoint, save_some_examples

torch.backends.cudnn.benchmark = True


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = dataset.KittiDataset(root=config.TRAIN_DIR,
                                 img_root='image_2',
                                 mask_root='semantic')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = dataset.KittiDataset(root=config.VAL_DIR,
                               img_root='image_2',
                               mask_root='semantic')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Training is started...")
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=config.TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=config.VAL_DIR)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--max_epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--load_model", type=bool, default=False)

    hparams = parser.parse_args()

    config.TRAIN_DIR = hparams.train_dir
    config.VAL_DIR = hparams.val_dir
    config.LEARNING_RATE = hparams.lr
    config.BATCH_SIZE = hparams.batch_size
    config.IMAGE_SIZE = hparams.img_size
    config.NUM_EPOCHS = hparams.max_epochs
    config.NUM_WORKERS = hparams.num_workers
    config.LOAD_MODEL = hparams.save_model
    config.SAVE_MODEL = hparams.load_model

    main()
