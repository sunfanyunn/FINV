import torch
import os
import time
import copy
import json
import pickle
import numpy as np
from configs import global_config


def load_eg3d(data_camera_mode='shapenet_chair', reload_modules=True):
    import sys
    sys.path.append('./eg3d/')
    import legacy
    import dnnlib

    if data_camera_mode == 'shapenet_chair':
        assert global_config.target_instance_type == 5
        network_pkl = './pretrained_models/eg3d_shapenet_chairs.pkl'
    elif data_camera_mode == 'shapenet_table':
        assert global_config.target_instance_type == 7
        network_pkl = './pretrained_models/shapnet_table_eg3d_network-snapshot-005000.pkl'
    else:
        network_pkl = './pretrained_models/eg3d-car-network-snapshot-003600.pkl'


    if network_pkl.endswith('pt'):
        G = torch.load(network_pkl)
    else:
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(global_config.device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        from torch_utils import misc
        from training.triplane import TriPlaneGenerator
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(global_config.device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    # G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore
    return G.float()

def load_vgg():
    import sys
    sys.path.append('./eg3d')
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    import dnnlib
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    return vgg16


def load_get3d(data_camera_mode='shapenet_chair', fovyangle=45.107971604139024):
    import sys
    sys.path.append('./get3d')
    import dnnlib
    # from training.inference_utils import save_visualization, save_visualization_with_encoder, \
        # save_visualization_pc_encode, save_visualization_for_interpolation, save_textured_mesh_for_inference, \
        # save_textured_tet_mesh_for_inference
    # from training.dataset import collate_fn

    device = torch.device('cuda', 0)
    c = {'G_kwargs': {'class_name': 'training.networks_get3d.GeneratorDMTETMesh', 'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 8}, 'one_3d_generator': True, 'n_implicit_layer': 1, 'deformation_multiplier': 1.0, 'use_style_mixing': True, 'dmtet_scale': 0.8, 'feat_channel': 16, 'mlp_latent_channel': 32, 'tri_plane_resolution': 256, 'n_views': 1, 'render_type': 'neural_render', 'use_tri_plane': True, 'tet_res': 90, 'geometry_type': 'conv3d', 'data_camera_mode': 'shapenet_chair', 'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'}, 'D_kwargs': {'class_name': 'training.networks_get3d.Discriminator', 'block_kwargs': {'freeze_layers': 0}, 'mapping_kwargs': {}, 'epilogue_kwargs': {'mbstd_group_size': 4}, 'data_camera_mode': 'shapenet_chair', 'add_camera_cond': True, 'channel_base': 32768, 'channel_max': 512, 'architecture': 'skip'}, 'G_opt_kwargs': {'class_name': 'torch.optim.Adam', 'betas': [0, 0.99], 'eps': 1e-08, 'lr': 0.002}, 'D_opt_kwargs': {'class_name': 'torch.optim.Adam', 'betas': [0, 0.99], 'eps': 1e-08, 'lr': 0.002}, 'loss_kwargs': {'class_name': 'training.loss.StyleGAN2Loss', 'gamma_mask': 40.0, 'r1_gamma': 40.0, 'style_mixing_prob': 0.9, 'pl_weight': 0.0}, 'data_loader_kwargs': {'pin_memory': True, 'prefetch_factor': 2, 'num_workers': 3}, 'inference_vis': True, 'inference_to_generate_textured_mesh': False, 'inference_save_interpolation': False, 'inference_compute_fid': False, 'inference_generate_geo': False, 'training_set_kwargs': {'class_name': 'training.dataset.ImageFolderDataset', 'path': './tmp', 'use_labels': False, 'max_size': 1234, 'xflip': False, 'resolution': 1024, 'data_camera_mode': 'shapenet_chair', 'add_camera_cond': True, 'camera_path': './tmp', 'split': 'test', 'random_seed': 0}, 'resume_pretrain': '../pretrained_models/shapenet_chair.pt', 'D_reg_interval': 16, 'num_gpus': 1, 'batch_size': 4, 'batch_gpu': 4, 'metrics': ['fid50k'], 'total_kimg': 20000, 'kimg_per_tick': 1, 'image_snapshot_ticks': 50, 'network_snapshot_ticks': 200, 'random_seed': 0, 'ema_kimg': 1.25, 'G_reg_interval': 4, 'run_dir': 'save_inference_results/shapenet_chair/inference'}
    c['G_kwargs']['device'] = device

    # G_kwargs = {'class_name': 'training.networks_stylegan3d.GeneratorDMTETMesh', 'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 8},
                # 'add_camera_noise': False, 'add_img_encoder': False, 'iso_surface': 'dmtet', 'one_3d_generator_detach_tex_gradient': False,
                # 'one_3d_generator': True, 'add_sdf_normalization': False, 'inference_noise_mode': 'random', 'detach_bad_geo_gradient': True, 'detach_geo_gradient_from_tex': False,
                # 'use_pytorch_grid_sample': False, 'n_implicit_layer': 1, 'cat_ws_geo': True, 'subdivision_step': 0, 'deformation_multiplier': 1.0,
                # 'use_all_tex_gradient': False, 'use_depth_supervision': False, 'use_normal_supervision': False, 'use_style_mixing': True, 'dmtet_scale': 0.8,
                # 'align_blend_camera': True, 'feat_channel': 16, 'mlp_latent_channel': 32, 'part_xyz_plane': False,
                # 'one_3d_generator_conv3d' : False, 'use_xyz_part_tex': False, 'n_freq_posenc_tex': 10, 'add_xyz_position': False, 'tri_plane_resolution': 256,
                # 'n_views': 1, 'use_res_implicit': False, 'voxel_resolution': 32, 'use_conv3d': False, 'add_part_label': False, 'n_part': 32, 'detach_tex_gradient': False,
                 # 'n_tex_mlp': 10, 'render_type': 'neural_render', 'remove_background': True, 'use_tri_plane': True, 'seprate_back_mapping': True,
                # 'add_input_pos_encode': False, 'fix_sdf_pred': True, 'use_stylegan_grid_sample': True, 'n_freq_posenc_geo': 1, 'soft_symmetric_geo': False,
                # 'hard_symmetric_geo': True, 'hard_symmetric_tex': True, 'seprate_geo_mapping': True, 'tet_res': 90, 'geometry_type': 'conv3d',
                # 'add_part_decoder': False, 'geo_pos_enc': 'normal', 'data_camera_mode': 'shapenet_chair', 'add_mask_for_d': True, 'channel_base': 32768,
                # 'channel_max': 512, 'use_softmax_part': False, 'fused_modconv_default': 'inference_only', 'device': device}

    # D_kwargs = {'class_name': 'training.networks_stylegan3d.Discriminator', 'block_kwargs': {'freeze_layers': 0}, 'mapping_kwargs': {},
                # 'epilogue_kwargs': {'mbstd_group_size': 4}, 'randomize_background': False, 'two_d_for_part': False, 'use_depth_supervision': False,
                # 'use_normal_supervision': False, 'patchgan_part': False, 'add_part_d': False, 'add_part_label': False, 'n_part': 32,
                # 'data_camera_mode': 'shapenet_chair', 'add_camera_cond': True, 'channel_base': 32768, 'channel_max': 512,
                # 'add_mask_for_d': True, 'd_for_mask': True, 'architecture': 'skip', 'device': device}

    common_kwargs = {'c_dim': 0, 'img_resolution': 1024, 'img_channels': 3}

    G = dnnlib.util.construct_class_by_name(**c['G_kwargs'], **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**c['D_kwargs'], **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).train() # .eval()  # deepcopy can make sure they are correct.

    if data_camera_mode == 'shapenet_chair':
        # assert global_config.target_instance_type == 5
        resume_pretrain = './pretrained_models/shapenet_chair.pt'
    elif data_camera_mode == 'shapenet_table':
        # assert global_config.target_instance_type == 7
        resume_pretrain = './pretrained_models/shapenet_table.pt'
    else:
        resume_pretrain = './pretrained_models/shapenet_car.pt'

    print('==> resume from pretrained path %s, fovyangle %f' % (resume_pretrain, fovyangle))
    model_state_dict = torch.load(resume_pretrain, map_location=device)
    G.load_state_dict(model_state_dict['G'], strict=True)
    G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
    D.load_state_dict(model_state_dict['D'], strict=True)

    from uni_rep.camera.perspective_camera import PerspectiveCamera
    dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=device)
    G_ema.synthesis.dmtet_geometry.renderer.camera = dmtet_camera

    return G_ema, D
    # return G_ema
