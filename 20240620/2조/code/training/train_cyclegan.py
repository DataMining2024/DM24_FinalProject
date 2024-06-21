import os
import torch
import itertools
from torch.utils.data import DataLoader
from datasets.dataset import MonetDataset
from models.cyclegan import Generator, Discriminator
from utils.image_utils import save_checkpoint, save_sample_images
from torchvision import transforms

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {checkpoint_path}, epoch {epoch}")
        return epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

def train_cyclegan(config):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    monet_dataset = MonetDataset(config['data']['monet_dir'], config['data']['photo_dir'], transform=transform)
    dataloader = DataLoader(monet_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G_XtoY = Generator().to(device)
    G_YtoX = Generator().to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)

    learning_rate = config['training']['learning_rate']
    g_optimizer = torch.optim.Adam(itertools.chain(G_XtoY.parameters(), G_YtoX.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    d_X_optimizer = torch.optim.Adam(D_X.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_Y_optimizer = torch.optim.Adam(D_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()

    checkpoint_dir = config['training']['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = 0
    if os.path.isdir(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('G_XtoY_')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            start_epoch = load_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint), G_XtoY, g_optimizer)
            load_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint.replace('G_XtoY', 'G_YtoX')), G_YtoX, g_optimizer)
            load_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint.replace('G_XtoY', 'D_X')), D_X, d_X_optimizer)
            load_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint.replace('G_XtoY', 'D_Y')), D_Y, d_Y_optimizer)

    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        for i, (real_X, real_Y) in enumerate(dataloader):
            real_X = real_X.to(device)
            real_Y = real_Y.to(device)

            try:
                g_optimizer.zero_grad()

                fake_Y = G_XtoY(real_X)
                fake_X = G_YtoX(real_Y)

                loss_GAN_XtoY = criterion_GAN(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y)))
                loss_GAN_YtoX = criterion_GAN(D_X(fake_X), torch.ones_like(D_X(fake_X)))

                rec_X = G_YtoX(fake_Y)
                rec_Y = G_XtoY(fake_X)

                loss_cycle_XYX = criterion_cycle(rec_X, real_X)
                loss_cycle_YXY = criterion_cycle(rec_Y, real_Y)

                loss_G = loss_GAN_XtoY + loss_GAN_YtoX + 10 * (loss_cycle_XYX + loss_cycle_YXY)
                loss_G.backward()
                g_optimizer.step()

                d_X_optimizer.zero_grad()
                loss_D_X = (criterion_GAN(D_X(real_X), torch.ones_like(D_X(real_X))) +
                            criterion_GAN(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X)))) * 0.5
                loss_D_X.backward()
                d_X_optimizer.step()

                d_Y_optimizer.zero_grad()
                loss_D_Y = (criterion_GAN(D_Y(real_Y), torch.ones_like(D_Y(real_Y))) +
                            criterion_GAN(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))) * 0.5
                loss_D_Y.backward()
                d_Y_optimizer.step()

                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                          f"Loss G: {loss_G.item():.4f}, Loss D_X: {loss_D_X.item():.4f}, Loss D_Y: {loss_D_Y.item():.4f}")
            except Exception as e:
                print(f"Error at Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}]: {e}")
                continue

        save_checkpoint(G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_X_optimizer, d_Y_optimizer, epoch, checkpoint_dir)
        save_sample_images(G_XtoY, G_YtoX, real_X, epoch, config['training']['sample_dir'])