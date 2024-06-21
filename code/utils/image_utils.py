import os
import torch
from torchvision.utils import save_image

def save_checkpoint(G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_X_optimizer, d_Y_optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': G_XtoY.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict()
    }, os.path.join(checkpoint_dir, f'G_XtoY_{epoch+1}.pth'))
    torch.save({
        'epoch': epoch,
        'model_state_dict': G_YtoX.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict()
    }, os.path.join(checkpoint_dir, f'G_YtoX_{epoch+1}.pth'))
    torch.save({
        'epoch': epoch,
        'model_state_dict': D_X.state_dict(),
        'optimizer_state_dict': d_X_optimizer.state_dict()
    }, os.path.join(checkpoint_dir, f'D_X_{epoch+1}.pth'))
    torch.save({
        'epoch': epoch,
        'model_state_dict': D_Y.state_dict(),
        'optimizer_state_dict': d_Y_optimizer.state_dict()
    }, os.path.join(checkpoint_dir, f'D_Y_{epoch+1}.pth'))

def save_sample_images(G_XtoY, G_YtoX, real_X, epoch, sample_dir):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    fake_Y = G_XtoY(real_X)
    save_image(fake_Y, os.path.join(sample_dir, f'fake_Y_{epoch+1}.png'))

def save_generated_images(fake_Y, batch_idx, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_image(fake_Y, os.path.join(output_dir, f'generated_{batch_idx}.png'))