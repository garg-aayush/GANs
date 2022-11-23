import torch
from utils.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader
from tqdm import tqdm
from torchvision.utils import save_image

import config
from datasets.dataset import MapDataset
from models.generator import Generator
from models.discriminator import Discriminator

def train(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):

    for idx, (x, y) in enumerate(tqdm(loader, leave=True)):
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


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE_loss = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    
    # train dataloader
    train_dataset = MapDataset(root_dir=config.TRAIN_ROOT_DIR)
    train_loader = Dataloader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # val dataloader
    val_dataset = MapDataset(root_dir=config.VAL_ROOT_DIR)
    val_loader = Dataloader(val_dataset, batch_size=1, shuffle=False)

    # for fp16 training
    g_scalar = torch.cuda.amp.GradScalar()
    d_scalar = torch.cuda.amp.GradScalar()
    
    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE_loss, g_scalar, d_scalar)

        if config.SAVE_MODEL and epoch % config.MODEL_SAVE_FREQ == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        

        save_some_examples(gen, val_loader, epoch, folder=config.VAL_SAVE_DIR)

if __name__ == "__main__":
    main()
