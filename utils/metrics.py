import torch
import cv2
import numpy as np
import math
import torch
from kornia.metrics import ssim as kornia_ssim

def get_bbox(masks):
    # pytorch image of shape [batch_size, 1, x, y]
    batch_size = masks.shape[0]
    w, h = masks.shape[-2], masks.shape[-1]
    bboxes = torch.zeros_like(masks)
    for bi in range(batch_size):
        mask = masks[bi, ...]
        wl = 0
        wh = w-1
        for i in range(w):
            if mask[0, i, :].sum().item() > 0:
                wl = i
                break
        for i in range(w-1, -1, -1):
            if mask[0, i, :].sum().item() > 0:
                wh = i
                break
        hl = 0
        hh = h-1
        for i in range(h):
            if mask[0, :, i].sum().item() > 0:
                hl = i
                break
        for i in range(h-1, -1, -1):
            if mask[0, :, i].sum().item() > 0:
                hh = i
                break
        bboxes[bi, :, wl:wh+1, hl:hh+1] = 1.
    return bboxes

def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    _mse = mse(image_pred, image_gt, valid_mask, reduction)
    max_pixel = 255.
    return 20 * torch.log10(max_pixel / torch.sqrt(_mse))

def ssim(image_pred, image_gt, valid_mask=None, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    assert reduction == "mean"
    ssim_map = kornia_ssim(image_pred, image_gt, 3)
    return torch.mean(ssim_map)
    # from kornia.losses import ssim as dssim
    # dssim_ = dssim(image_pred, image_gt, 3, reduction)  # dissimilarity in [0, 1]
    # return 1 - 2 * dssim_  # in [-1, 1]

if __name__ == '__main__':
    device = torch.device('cuda:0')
    import sys
    sys.path.append('/data/sunfanyun/ActiveNeRF/')
    from car_main import evaluate
    import torch.nn.functional as F
    generated_images = torch.rand(1, 3, 1048, 1048)*2-1
    real_images = torch.rand(1, 3, 1048, 1048)*2-1

    print(generated_images.min(), generated_images.max())
    print(real_images.min(), real_images.max())
    print(evaluate(generated_images.to(device), real_images.to(device)))

    generated_images = F.interpolate(generated_images, size=(512, 512), mode='nearest')
    real_images = F.interpolate(real_images, size=(512, 512), mode='nearest')
    print(generated_images.min(), generated_images.max())
    print(real_images.min(), real_images.max())
    print(evaluate(generated_images.to(device), real_images.to(device)))

    generated_images = F.interpolate(generated_images, size=(1048, 1048), mode='nearest')
    real_images = F.interpolate(real_images, size=(1048, 1048), mode='nearest')
    print(generated_images.min(), generated_images.max())
    print(real_images.min(), real_images.max())
    print(evaluate(generated_images.to(device), real_images.to(device)))
