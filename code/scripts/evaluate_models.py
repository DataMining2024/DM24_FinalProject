import glob
import os
import torch
from utils.fid import get_activations, calculate_mifid, InceptionV3FeatureExtractor
from torchvision import transforms
from PIL import Image

def evaluate_models(config):
    mifid_score = evaluate_mifid(config)
    print(f"MiFID Score: {mifid_score}")

def evaluate_mifid(config):
    real_img_dir = config['evaluation']['real_img_dir']
    generated_img_dir = config['evaluation']['generated_img_dir']
    batch_size = config['evaluation']['batch_size']
    epsilon = config['evaluation'].get('epsilon', 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real_img_paths = glob.glob(os.path.join(real_img_dir, '*.jpg'))
    generated_img_paths = glob.glob(os.path.join(generated_img_dir, '*.png')) 

    model = InceptionV3FeatureExtractor().to(device)

    activations_real = get_activations(real_img_paths, model, batch_size, device=device)
    activations_generated = get_activations(generated_img_paths, model, batch_size, device=device)

    mifid_score = calculate_mifid(activations_real, activations_generated, epsilon)
    return mifid_score