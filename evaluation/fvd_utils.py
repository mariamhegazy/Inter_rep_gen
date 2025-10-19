"""
Frechet Video Distance (FVD). Matches the original tensorflow implementation from
https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
up to the upsampling operation. Note that this tf.hub I3D model is different from the one released in the I3D repo.
"""

import copy
import os
from functools import lru_cache

# Total frames per batch budget used by the reference code
NUM_FRAMES_IN_BATCH = {128: 128, 256: 128, 512: 64, 1024: 32}

import numpy as np
import torch
from scipy import linalg


@lru_cache()
def load_fvd_model(device="cuda"):
    """
    Loads the pre-trained I3D model (TorchScript) for FVD computation.
    Assumes the model is downloaded manually to ~/.cache/i3d/i3d_torchscript.pt.
    """
    # model_path = os.path.join(
    #     os.environ.get("I3D_HOME", os.path.expanduser("~/.cache/i3d")),
    #     "i3d_torchscript.pt",
    # )
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Expected I3D TorchScript model at: {model_path}")
    model_path = (
        "/capstor/store/cscs/swissai/a144/mariam/checkpoints/i3d_torchscript.pt"
    )
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def get_fvd_logits(videos, i3d, device="cuda", batch_size=4):
    """
    Extracts embeddings from a batch of videos using the I3D model.

    Args:
        videos (np.ndarray): shape [B, T, H, W, C], pixel range [0, 255]
        i3d (torch.jit.ScriptModule): loaded I3D model
        device (str): device to run on
        batch_size (int): mini-batch size for feeding into I3D

    Returns:
        torch.Tensor: embeddings of shape [B, D]
    """
    # assert (
    #     videos.ndim == 5 and videos.shape[-1] == 3
    # ), "Expected videos of shape [B, T, H, W, C]"
    B = videos.shape[0]
    embeddings = []

    for i in range(0, B, batch_size):
        batch = videos[i : i + batch_size]  # [b, T, H, W, C]
        batch = torch.tensor(batch).permute(0, 4, 1, 2, 3).float()  # [b, C, T, H, W]
        # batch = batch / 127.5 - 1.0  # scale to [-1, 1]
        batch = torch.tensor(batch).to(device)

        with torch.no_grad():
            # emb = i3d(batch)  # [b, D]
            emb = i3d(  # ⬅️ old:  i3d(batch)
                x=batch,
                rescale=True,
                resize=False,
                return_features=True,
            )
        embeddings.append(emb)

    return torch.cat(embeddings, dim=0)  # [B, D]


def frechet_distance(mu1, mu2, sigma1=None, sigma2=None, eps=1e-6):
    """
    Compute the Frechet Distance (FID/FVD) between two Gaussian distributions.

    Args:
        mu1, mu2 (Union[Tensor, np.ndarray]): feature means
        sigma1, sigma2 (optional): feature covariances
            If None, computed from data directly (assumes inputs are embeddings).

    Returns:
        float: the Frechet Distance
    """
    if sigma1 is None and sigma2 is None:
        assert isinstance(mu1, torch.Tensor)
        assert isinstance(mu2, torch.Tensor)
        mu1 = mu1.cpu().numpy()
        mu2 = mu2.cpu().numpy()
        sigma1 = np.cov(mu1, rowvar=False)
        sigma2 = np.cov(mu2, rowvar=False)
        mu1 = np.mean(mu1, axis=0)
        mu2 = np.mean(mu2, axis=0)
    else:
        mu1 = mu1.cpu().numpy() if isinstance(mu1, torch.Tensor) else mu1
        mu2 = mu2.cpu().numpy() if isinstance(mu2, torch.Tensor) else mu2
        sigma1 = sigma1.cpu().numpy() if isinstance(sigma1, torch.Tensor) else sigma1
        sigma2 = sigma2.cpu().numpy() if isinstance(sigma2, torch.Tensor) else sigma2

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
