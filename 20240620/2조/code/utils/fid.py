import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_distances

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(inception.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x

def get_activations(image_paths, model, batch_size=50, dims=2048, device='cuda'):
    model.eval()

    transform = transforms.Compose([
        Resize((299, 299)),  # InceptionV3 requires 299x299 images
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    pred_arr = np.empty((len(image_paths), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID calculation produces singular product; adding %s to diagonal of cov estimates" % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_mifid(activations_real, activations_generated, epsilon=0.1):
    mu_real = np.mean(activations_real, axis=0)
    sigma_real = np.cov(activations_real, rowvar=False)
    mu_gen = np.mean(activations_generated, axis=0)
    sigma_gen = np.cov(activations_generated, rowvar=False)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    # Cosine distances
    cosine_dist = cosine_distances(activations_generated, activations_real)

    # Minimum cosine distances for each generated image
    min_cosine_distances = np.min(cosine_dist, axis=1)

    # Mean of minimum distances
    mean_min_distance = np.mean(min_cosine_distances)

    # Apply threshold epsilon
    if mean_min_distance <= epsilon:
        weight = 1 / mean_min_distance
    else:
        weight = 1

    mifid_score = fid_score * weight
    return mifid_score