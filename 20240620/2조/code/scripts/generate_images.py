import glob
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from models.cyclegan import Generator
from PIL import Image

def generate_images(config):
    input_dir = config['generation']['input_dir']
    output_dir = config['generation']['output_dir']
    model_path = config['generation']['model_path']
    direction = config['generation']['direction']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)

    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint['model_state_dict'])
    G.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_files = glob.glob(os.path.join(input_dir, '*.jpg'))

    for idx, image_file in enumerate(image_files):
        image = Image.open(image_file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if direction == 'XtoY':
                fake_image = G(image)
            else:
                raise ValueError("Unsupported direction: {}".format(direction))

        save_image(fake_image, os.path.join(output_dir, f'generated_{idx+1}.png'))