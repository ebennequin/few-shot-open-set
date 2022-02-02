import torch
import numpy as np


def trivial(images):
    return images


def cutmix(images, alpha=1.0):

    indices = torch.randperm(images.size(0))
    shuffled_images = images[indices]
    # shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = images.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    images[:, :, y0:y1, x0:x1] = shuffled_images[:, :, y0:y1, x0:x1]
    # targets = (targets, shuffled_targets, lam)

    return images