import sys
import pickle
import functools
import torch
from configs import global_config


def mask_out(images, instance_masks):
    """ input images shape
    # [batch_size, 3, 480, 480]
    """
    assert images.shape[0] == instance_masks.shape[0]
    images = (images + 1) * instance_masks - 1
    return images

def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def load_tuned_G(run_id, type, full_path=None):
    if full_path is None:
        new_G_path = f'{global_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    else:
        new_G_path = full_path

    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G
