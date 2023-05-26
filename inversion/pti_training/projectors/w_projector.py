# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import os
import copy
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imageio
import json
from PIL import Image
from configs import global_config
from inversion.utils import log_utils
from inversion.utils.models_utils import mask_out


def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        embedding_dir,
        target_pose,
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.025,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,  # DL: originally 1e5 from stylegan2 paper
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
        mask=None,
        writer=None,
        write_video=False,
        session_prefix: str,
):
    if mask is None:
        mask = torch.ones_like(target)

    # np.random.seed(1989)
    # torch.manual_seed(1989)

    # if w_name is None:
        # target_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=global_config.device), radius=2.7, device=global_config.device).reshape(4, 4)
        # target_pose = target_pose.cpu().numpy()
    # else:
        # import pdb;pdb.set_trace()
        # assert target.shape[0] == G.img_channels
        # assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution), \
            # 'only batch size==1  supported'
        # if os.path.basename(global_config.input_pose_path).split(".")[1] == "json":
            # f = open(global_config.input_pose_path)
            # target_pose = np.asarray(json.load(f)[global_config.input_id]['pose']).astype(np.float32)
            # f.close()
            # o = target_pose[0:3, 3]
            # print("norm of origin before normalization:", np.linalg.norm(o))
            # o = 2.7 * o / np.linalg.norm(o)
            # target_pose[0:3, 3] = o
            # target_pose = np.reshape(target_pose, -1)
        # else:
            # target_pose = np.load(global_config.input_pose_path).astype(np.float32)
            # target_pose = np.reshape(target_pose, -1)

    # intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    # target_pose = np.concatenate([target_pose, intrinsics])
    target_pose = torch.clone(target_pose).to(device)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    import dnnlib
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    if target.ndim == 4:
        batchsize = target.shape[0]
        target_images = target.to(device).float()
    else:
        batchsize = 1
        target_images = target.unsqueeze(0).to(device).to(torch.float32)

    target_images = (target_images + 1) * 255 / 2
    target_images = F.interpolate(target_images, size=(512, 512))
    real_image = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    target_images_orig = target_images.clone()
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        mask = F.interpolate(mask, size=(256, 256), mode='nearest')

    target_features = vgg16(mask*target_images, resize_images=False, return_lpips=True)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    z_samples = torch.randn([w_avg_samples, G.z_dim]).to(device)
    w_samples = G.mapping(z_samples,
                          target_pose[0:1, ...].repeat(w_avg_samples, 1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w[:, :1, ...] if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable
    w_initial = w_opt.clone().detach()
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=global_config.first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    all_w_opt = []
    num_ws = G.backbone.mapping.num_ws

    if write_video:
        vid_path = f'{embedding_dir}' + '/' + f'{session_prefix}_phase1.mp4'
        rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

    best_w_opt = None
    best_loss = 1e9
    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        # don't add noise if we're optimizing from an initial 'w'
        # we want to stay somewhat close to that
        if initial_w is not None:
            w_noise_scale = 0.

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_opt_ = w_opt.repeat(batchsize, 1, 1)
        w_noise = torch.randn_like(w_opt_) * w_noise_scale
        ws = (w_opt_ + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])
        all_w_opt.append(w_opt_.detach().repeat([1, num_ws, 1]))

        synth_images_orig = G.synthesis(ws, target_pose, noise_mode='const', force_fp32=True)['image']
        synth_images_orig = synth_images_orig.clamp(-1, 1.)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images_orig + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='nearest')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        # if global_config.use_noise_regularization:
            # reg_loss = 1.0
        # else:
            # reg_loss = 0.0

        # for v in noise_bufs.values():
            # noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            # while True:
                # reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                # reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                # if noise.shape[2] <= 8:
                    # break
                # noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist # + reg_loss * regularize_noise_weight

        if step % 5 == 0:
            synth_image = (synth_images_orig + 1) * (255/2)
            # if writer is not None:
                # log_utils.tb_log_images(torch.cat((target_images_orig / 255., (synth_images_orig+1)/2), dim=-1),
                                         # writer,  step, label='w')

            if write_video:
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                rgb_video.append_data(np.concatenate([real_image, synth_image], axis=1))

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        if loss.item() < best_loss:
            best_w_opt = w_opt.clone().detach()
            best_loss = loss.item()

    if write_video:
        rgb_video.close()

    all_w_opt = torch.cat(all_w_opt, 0)
    del G
    return best_w_opt.repeat([1, num_ws, 1]), w_initial.repeat([1, num_ws, 1]), all_w_opt


def project_get3d(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        embedding_dir,
        target_pose,
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.005,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,  # DL: originally 1e5 from stylegan2 paper
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
        mask=None,
        writer=None,
        write_video=False,
        session_prefix: str,
):
    def logprint(*args):
        if verbose:
            print(*args)

    def get_iou(mask_target, mask_pred):
        overlap_mask = mask_target * mask_pred
        union_mask = mask_target + mask_pred - overlap_mask
        iou = (overlap_mask.sum()/union_mask.sum()).item()
        return iou


    if mask is None:
        mask = torch.ones_like(target)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Features for target image.
    if target.ndim == 4:
        batchsize = target.shape[0]
        target_images = target.to(device).float()
    else:
        batchsize = 1
        target_images = target.unsqueeze(0).to(device).to(torch.float32)

    target_image_size = 1024
    target_images = F.interpolate(target_images, size=(target_image_size, target_image_size))
    mask = F.interpolate(mask, size=(target_image_size, target_image_size))

    camera = target_pose.clone().to(device)
    camera = camera[:, :16].reshape((batchsize,1,4,4)).to(device)

    # Compute w stats.
    truncation_cutoff=None
    truncation_psi=0.7
    update_geo=True
    update_emas=False
    generate_no_light=True
    generate_txture_map=True

    z_samples = torch.randn([w_avg_samples, G.z_dim]).to(device)
    geo_z_samples = torch.randn([w_avg_samples, G.z_dim]).to(device)
    c = torch.ones(1).to(device)

    if initial_w is None:
        ws = G.mapping(z_samples, c,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    update_emas=update_emas)

        ws_geo = G.mapping_geo(geo_z_samples, c,
                            truncation_psi=truncation_psi,
                            truncation_cutoff=truncation_cutoff,
                            update_emas=update_emas)
        # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        ws_avg = np.mean(ws[:, :1, :].cpu().numpy().astype(np.float32), axis=0, keepdims=True)
        #ws_avg = np.repeat(ws_avg, 9, axis=1)
        ws_geo_avg = np.mean(ws_geo[:, :1, :].cpu().numpy().astype(np.float32), axis=0, keepdims=True)
        #ws_geo_avg = np.repeat(ws_geo_avg, 22, axis=1)

        ws_avg = torch.from_numpy(ws_avg)
        ws_geo_avg = torch.from_numpy(ws_geo_avg)
    else:
        ws_avg, ws_geo_avg = initial_w[:, :1, ...], initial_w[:, 1:, ...]
    ws_avg =  torch.tensor(ws_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callabl
    ws_geo_avg = torch.tensor(ws_geo_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    optimizer = torch.optim.Adam([ws_avg, ws_geo_avg], betas=(0.9, 0.999),
                                 lr=global_config.first_inv_lr)
    # Init noise.
    #for buf in noise_bufs.values():
    #    buf[:] = torch.randn_like(buf)
    #    buf.requires_grad = True

    all_w_opt = []
    bce_loss = torch.nn.BCELoss()
    ##########
    #  train texture
    ##########
    # Load VGG16 feature detector.
    use_lpips = global_config.use_lpips
    if use_lpips:
        from lpips import LPIPS
        lpips_loss = LPIPS(net=global_config.lpips_type).to(global_config.device).eval()
    else:
        pass
        # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        # import dnnlib
        # with dnnlib.util.open_url(url) as f:
            # vgg16 = torch.jit.load(f).eval().to(device)
        # if target_images.shape[2] > 256:
            # resized_target_images = F.interpolate((target_images.to(global_config.device) + 1) * 255 / 2, size=(256, 256), mode='area')
            # resized_mask = F.interpolate(mask, size=(256, 256), mode='nearest')
        # target_features = vgg16(resized_mask*resized_target_images, resize_images=False, return_lpips=True)

    # num_ws = G.backbone.mapping.num_ws

    if write_video:
        vid_path = f'{embedding_dir}' + '/' + f'{session_prefix}_phase1.mp4'
        rgb_video = imageio.get_writer(vid_path, mode='I', fps=5, codec='libx264', bitrate='16M')

    best_w_opt = None
    best_iou = -1e9
    best_loss = 1e9
    for step in tqdm(range(num_steps)):

        # Learning rate schedule1
        t = step / num_steps
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        w_noise_scale = initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        # don't add noise if we're optimizing from an initial 'w'
        # we want to stay somewhat close to that 
        if initial_w is not None:
            w_noise_scale = 0.

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_opt = torch.cat([ws_avg, ws_geo_avg], dim=1)
        if step == 0:
            initial_w_opt = w_opt.clone().detach()
        w_opt_ = w_opt.repeat(batchsize, 1, 1)
        w_noise = torch.randn_like(w_opt_) * w_noise_scale
        ws = (w_opt_ + w_noise) #.repeat([1, G.backbone.mapping.num_ws, 1])
        all_w_opt.append(w_opt_.detach())

        # synth_images_orig, _ = G.my_forward(ws, camera)
        synth_images_orig, _ = G.forward(ws.squeeze(0), camera)
        synth_images_orig = synth_images_orig.unsqueeze(0)
        synth_images_orig, synth_masks_orig = synth_images_orig[:, :3, ...], synth_images_orig[:, 3:, ...]
        # Features for synth images.
        # synth_features = vgg16(resized_mask*resized_synth_images, resize_images=False, return_lpips=True)
        #dist = (target_features - synth_features).square().sum()
        #assert synth_masks_orig.min().item() == 0.
        #assert synth_masks_orig.max().item() == 1.
        assert mask.min().item() == 0.
        assert mask.max().item() == 1.
        assert synth_masks_orig.shape == mask.shape

        if global_config.target_instance_type == 5 or global_config.target_instance_type == 7:
            dist = bce_loss(synth_masks_orig.clamp(0., 1.), mask)
        else:
            # fg_weight = (mask != 1).sum().item()/mask.shape[-1]/mask.shape[-2]
            # bg_weight = (mask == 1).sum().item()/mask.shape[-1]/mask.shape[-2]
            # weight_mask = (mask == 1.)*fg_weight + (mask != 1.)*bg_weight
            # dist = F.binary_cross_entropy(synth_masks_orig.clamp(0., 1.), mask, weight=weight_mask)
            dist = F.binary_cross_entropy(synth_masks_orig.clamp(0., 1.), mask)


        # Features for synth images.
        if use_lpips:
            # images' range should be [-1, 1]
            loss_lpips = lpips_loss(mask_out(synth_images_orig, mask).clamp(-1, 1),
                                    mask_out(target_images, mask).clamp(-1, 1))
            # loss_lpips = lpips_loss(mask_out(synth_images_orig, synth_masks_orig).clamp(-1, 1),
                                    # mask_out(target_images, mask).clamp(-1, 1))
            text_dist = torch.mean(loss_lpips)
        else:
            text_dist = 0.
            # if synth_images.shape[2] > 256:
                # resized_synth_images = F.interpolate((synth_images + 1) * 255 / 2, size=(256, 256), mode='area')
            # synth_features = vgg16(resized_mask*resized_synth_images, resize_images=False, return_lpips=True)
            # text_dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        # if global_config.use_noise_regularization:
            # reg_loss = 1.0
        # else:
            # reg_loss = 0.0
        #for v in noise_bufs.values():
        #    noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
        #    while True:
        #        reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
        #        reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
        #        if noise.shape[2] <= 8:
        #            break
        #        noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + text_dist # + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
        except:
            print('Failed backward pass! Exit training...')
            return best_w_opt, [best_loss, best_iou], None
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} text_dist {text_dist:<4.2f} loss {float(loss):<5.2f}')

        if step % 50 == 0 or step == num_steps-1:
            synth_images = (synth_images_orig + 1) * (255 / 2)
            real_image = ((target_images + 1) * 255 / 2).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
            real_mask = mask.permute(0, 2, 3, 1).clamp(0, 1).to(torch.uint8).repeat([1,1,1,3]).cpu().numpy()

            ious = []
            final_imgs = []
            for i in range(real_image.shape[0]):
                synth_image = F.interpolate(synth_images[i:i+1, ...],
                                            size=(real_image.shape[1], real_image.shape[2]),
                                            mode='area')
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                synth_mask = F.interpolate(synth_masks_orig.repeat([1,3,1,1])[i:i+1, ...],
                                            size=(real_image.shape[1], real_image.shape[2]),
                                            mode='area')
                synth_mask = synth_mask.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()

                ious.append(get_iou(real_mask[i, ...], synth_mask))

                tmp_img1 = np.concatenate([real_image[i, ...], synth_image], axis=1)
                tmp_img2 = np.concatenate([real_mask[i, ...]*255, synth_mask*255], axis=1)
                final_imgs.append(np.concatenate([tmp_img1, tmp_img2], axis=0))

            if best_loss > loss.item():
                print('update best {} --> {}'.format(best_loss, loss.item()))
                best_w_opt = w_opt.clone().detach()
                best_iou = np.mean(ious)
                best_loss = loss.item()

            final_img = np.concatenate(final_imgs, axis=1)
            if write_video:
                rgb_video.append_data(final_img)

            if step == num_steps-1:
                Image.fromarray(final_img).save(f'{embedding_dir}/{session_prefix}_train_phase1.png')

    if write_video:
        rgb_video.close()

    all_w_opt = torch.cat(all_w_opt, 0)
    del G
    w_opt = torch.cat([ws_avg, ws_geo_avg], dim=1)
    return best_w_opt, initial_w_opt, all_w_opt
