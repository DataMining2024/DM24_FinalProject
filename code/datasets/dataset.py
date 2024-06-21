import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MonetDataset(Dataset):
    def __init__(self, monet_dir, photo_dir, transform=None):
        self.monet_images = glob.glob(os.path.join(monet_dir, '*.jpg'))
        self.photo_images = glob.glob(os.path.join(photo_dir, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return max(len(self.monet_images), len(self.photo_images))

    def __getitem__(self, idx):
        monet_image = Image.open(self.monet_images[idx % len(self.monet_images)])
        photo_image = Image.open(self.photo_images[idx % len(self.photo_images)])

        if self.transform:
            monet_image = self.transform(monet_image)
            photo_image = self.transform(photo_image)

        return monet_image, photo_image