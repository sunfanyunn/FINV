"""Project given image to the latent space of pretrained network pickle."""
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and any modifications thereto.  Any use, reproduction, disclosure or
# and proprietary rights in and to this software, related documentation
# distribution of this software and related documentation without an express
# from camera_utils import LookAtPoseSampler #, FOV_to_intrinsics
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from  numpy import asarray
from PIL import Image
from matplotlib.pyplot import imshow
from my_camera_utils import get_int_ext
from pose_transform import get_scan2cad_poses
from torch_utils import misc
from tqdm import tqdm
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from typing import List, Optional, Tuple, Union
import PIL.Image
import click
import copy
import dnnlib
import legacy
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import os
import re
import sys
import torch
import torch.nn.functional as F
import trimesh


def generate_mesh(G, ws, output_path=None):
    ws = ws.repeat([ws.shape[0], G.backbone.mapping.num_ws, 1])
    v_list, f_list = G.my_generate_3d_mesh(ws)
    v_list, f_list =v_list[0].cpu().numpy(), f_list[0].cpu().numpy()

    v_list[..., 0] -= (v_list[..., 0].max() + v_list[..., 0].min())/2
    v_list[..., 2] -= (v_list[..., 2].max() + v_list[..., 2].min())/2
    v_list[..., 1] = v_list[..., 1] - v_list[..., 1].min()

    scale_factor =  v_list[..., 1].max() - v_list[..., 1].min()
    v_list /= scale_factor

    mesh = trimesh.Trimesh(v_list, f_list)
    if output_path:
        mesh.export(output_path)

def generate_new_image(G, ws, save_path=None):
    imgs = []
    tmp_imgs = []
    angle_p = -0.2
    for rotation in np.arange(0, 180, 60):
        camera_params = get_int_ext(rotation, 90, use_random_label=True)
        camera_params = torch.from_numpy(camera_params.reshape(1, -1)).to(device)

        img = G.synthesis(ws.repeat([1, G.backbone.mapping.num_ws, 1]), camera_params, noise_mode='const', force_fp32=True)['image']
        tmp_imgs.append(img)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)

    img = torch.cat(imgs, dim=2)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_path)
    return tmp_imgs[-1], camera_params


def run(start_w, targets, camera_params, out_prefix='./out/', num_steps=500):
    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image.
    target_images = []
    for idx, target in enumerate(targets):
        target_image = target.to(device).to(torch.float32)
        target_image = ((target_image + 1) * (255 / 2)).clamp(0, 255.)
        target_images.append(target_image)

        # target_image = F.interpolate(target_image, size=(128, 128), mode='area')
        im = PIL.Image.fromarray(target_image.permute(1,2,0).cpu().numpy().astype(np.uint8), 'RGB')
        im.save(out_prefix + 'target{}.png'.format(idx))

    target_images = torch.stack(target_images, dim=0)
    camera_params = torch.cat(camera_params, dim=0)
    target_image_size = target_images.shape[-1]
    batch_size = target_images.shape[0]

    print(target_images.shape, camera_params.shape)

    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    assert target_features.requires_grad is False
    assert camera_params.requires_grad is False
    # camera_params.requires_grad = True


    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=first_inv_lr)
    # optimizer = torch.optim.Adam([w_opt, camera_params] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 # lr=first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf.requires_grad = False
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # torch.autograd.set_detect_anomaly(True)
    for step in range(num_steps):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([batch_size, G.backbone.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, camera_params.to(device), noise_mode='const', force_fp32=True)
        synth_images = synth_images['image']
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = ((synth_images + 1) * (255 / 2)).clamp(0, 255)
        synth_images = F.interpolate(synth_images, size=(target_image_size, target_image_size), mode='area')

        #if synth_images.shape[2] > 256:
        #    synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        #assert target_images.shape == synth_images.shape
        #dist = (target_images/255 - synth_images/255).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if step % image_log_step == 0:
            print(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
            with torch.no_grad():
                im = PIL.Image.fromarray(synth_images[0,...].permute(1,2,0).detach().cpu().numpy().astype(np.uint8), 'RGB')
                im.save(out_prefix + 'synth_{}.png'.format(step))
                generate_new_image(G, w_opt.detach(), out_prefix + '_{}.png'.format(step))
                generate_mesh(G, w_opt.detach(), out_prefix + '_{}.ply'.format(step))

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ## Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    return w_opt.detach()


if __name__ == '__main__':
    out_dir = sys.argv[1]
    trunc = 0.7
    shapes = True
    seeds=[]
    network_pkl = '/raid/sun/pretrained_models/Jun_shapenet_chairs.pkl'
    #network_pkl = '/raid/sun/pretrained_models/shapenetcars128-64.pkl'
    truncation_cutoff=14
    truncation_psi=1
    shape_format='.mrc'
    reload_modules=True
    # fov_deg=45 #18.837
    shape_res = 512

    num_steps=500
    w_avg_samples = int(sys.argv[2])
    pose_rotation = int(sys.argv[3])
    initial_learning_rate=0.01
    initial_noise_factor=0.05
    lr_rampdown_length=0.25
    lr_rampup_length=0.05
    #first_inv_lr=0.01
    first_inv_lr=5e-3
    noise_ramp_length=0.75
    regularize_noise_weight=1e5
    verbose=False
    use_wandb=False
    initial_w=None
    image_log_step=100

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('out_dir/{}'.format(pose_rotation), exist_ok=True)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    from pose_transform import setup, get_image_camera_params
    renderer, train_dataset = setup()

    target_images, camera_params = [], []
    frame_idxs = [100, 90, 80, 70, 60]
    for obs_id, frame_idx in enumerate(frame_idxs):
        out_prefix = out_dir + '/{}_'.format(obs_id)

        instance_mask, target_image, camera_param = \
            get_image_camera_params(renderer, train_dataset, rotation=pose_rotation*2*np.pi/360, obj_id=4, frame_idx=frame_idx)
        camera_param = camera_param.to(device)
        target_image = torch.from_numpy(target_image*2-1).to(device).permute(2,0,1)
        assert camera_param.shape[-1] == 25
        assert target_image.shape[0] == 3
        target_images.append(target_image)
        camera_params.append(camera_param)

        if obs_id == 0:
            print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
            # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
            z_samples = np.random.randn(w_avg_samples, G.z_dim)
            print(z_samples.shape)
            w_samples = G.mapping(torch.from_numpy(z_samples).to(device),
                                    torch.zeros_like(camera_params[0]).repeat([w_avg_samples, 1]).to(device),
                                    truncation_psi=truncation_psi,
                                    truncation_cutoff=truncation_cutoff)  # [N, L, C]
            print(w_samples.shape)
            w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
            w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
            w_avg_tensor = torch.from_numpy(w_avg).to(device)
            w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

            w_opt = run(w_avg, target_images, camera_params, out_prefix=out_prefix, num_steps=num_steps)
        else:
            w_opt = run(w_opt, target_images, camera_params, out_prefix=out_prefix, num_steps=num_steps)

        torch.cuda.empty_cache()
        np.save('{}.npy'.format(out_prefix), w_opt.detach().cpu().numpy())
        generate_new_image(G, w_opt, '{}.png'.format(out_prefix))
        generate_mesh(G, w_opt, '{}.ply'.format(out_prefix))


    generate_new_image(G, w_opt, '{}/final.png'.format(out_dir))
    generate_mesh(G, w_opt, '{}/final.ply'.format(out_dir))
