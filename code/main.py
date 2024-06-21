import argparse
import json
from training.train_cyclegan import train_cyclegan
from scripts.generate_images import generate_images
from scripts.evaluate_models import evaluate_models

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monet Style Transfer")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate', 'evaluate'], help='Mode to run: train, generate, evaluate')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'train':
        train_cyclegan(config)
    elif args.mode == 'generate':
        generate_images(config)
    elif args.mode == 'evaluate':
        evaluate_models(config)