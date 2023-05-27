import os
import abc
import pickle
from argparse import Namespace
import wandb
import os.path
import torch
from torchvision import transforms
from lpips import LPIPS
import numpy as np
import json
import torch.nn.functional as F

from configs import global_config
from inversion.criteria.localitly_regulizer import Space_Regulizer
from inversion.utils.log_utils import log_image_from_w
from inversion.utils.models_utils import toogle_grad
from inversion.pti_training.projectors import w_projector
from inversion.criteria import l2_loss
from inversion.utils.models_utils import mask_out
from utils.pose_transform import apply_random_rotation


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

class BaseCoach:
    def __init__(self, gan_type, fovyangle, use_wandb=False):

        self.gan_type = gan_type
        self.fovyangle = fovyangle
        self.use_wandb = use_wandb
        self.w_pivots = {}
        self.image_counter = 0
        self.G = self.original_G = None

        # Initialize loss
        self.lpips_loss = LPIPS(net=global_config.lpips_type).to(global_config.device).eval()
        if global_config.clip_loss:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device=global_config.device)

        # self.restart_training()

        # Initialize checkpoint dir
        # self.checkpoint_dir = global_config.checkpoints_dir
        # os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):
        if global_config.target_instance_type == 5:
            data_camera_mode = 'shapenet_chair'
        elif global_config.target_instance_type == 7:
            data_camera_mode = 'shapenet_table'
        else:
            data_camera_mode = 'shapenet_car'

        if self.gan_type == 'eg3d':
            from utils.load_models import load_eg3d
            # Initialize networks
            del self.G
            torch.cuda.empty_cache()

            self.G = load_eg3d(data_camera_mode)
            toogle_grad(self.G, True)
            # self.original_G = load_eg3d(data_camera_mode)
        elif self.gan_type == 'get3d':
            from utils.load_models import load_get3d
            self.G, _  = load_get3d(data_camera_mode, fovyangle=self.fovyangle)
            toogle_grad(self.G, True)
            # self.original_G, self.D = load_get3d(data_camera_mode, fovyangle=self.fovyangle)
        else:
            assert False

        # if global_config.use_locality_regularization:
            # self.space_regulizer = Space_Regulizer(self.original_G, self.gan_type, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{global_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if global_config.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not global_config.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if global_config.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{global_config.e4e_results_keyword}/{image_name}/model_{image_name}.pt'
        else:
            w_potential_path = f'{w_path_dir}/{global_config.pti_results_keyword}/{image_name}/model_{image_name}.pt'

        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, id_image, image_name, embedding_dir, target_pose=None, grayscale=False, mask=None,
                        initial_w=None, writer=None, num_steps=global_config.first_inv_steps, session_prefix='',
                        write_video=False):

        w_avg_samples = global_config.w_avg_samples
        if global_config.first_inv_type == 'w+':
            w = self.get_e4e_inversion(id_image)
        else:
            # id_image = torch.squeeze((id_image.to(global_config.device) + 1) / 2) * 255
            if self.gan_type == 'eg3d':
                if grayscale:
                    assert False
                    id_image = id_image.unsqueeze(0)
                    w = w_projector_grayscale.project(self.G, id_image, embedding_dir, device=torch.device(global_config.device), w_avg_samples=w_avg_samples,
                            num_steps=global_config.first_inv_steps, w_name=image_name,
                            use_wandb=self.use_wandb)
                else:
                    w = w_projector.project(self.G, id_image, embedding_dir, target_pose=target_pose,
                                            device=torch.device(global_config.device), w_avg_samples=w_avg_samples,
                                            num_steps=num_steps, w_name=image_name,
                                            use_wandb=self.use_wandb, mask=mask, initial_w=initial_w,
                                            session_prefix=session_prefix, writer=writer, write_video=write_video, verbose=True)
            elif self.gan_type == 'get3d':
                w = w_projector.project_get3d(self.G, id_image, embedding_dir, target_pose=target_pose,
                                        device=torch.device(global_config.device), w_avg_samples=w_avg_samples,
                                        num_steps=num_steps, w_name=image_name,
                                        use_wandb=self.use_wandb, mask=mask, initial_w=initial_w,
                                        session_prefix=session_prefix, writer=writer, write_video=write_video, verbose=True)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp

        if self.gan_type == 'get3d':
            if global_config.only_texture:
                texture_parameters = list(self.G.synthesis.generator.mlp_synthesis_tex.parameters())
                for res in self.G.synthesis.generator.tri_plane_synthesis.block_resolutions:
                    module = getattr(self.G.synthesis.generator.tri_plane_synthesis, f'b{res}')
                    if hasattr(module, 'totex'):
                        param = module.totex.parameters()
                        texture_parameters += list(param)
                    else:
                        assert False
                optimizer = torch.optim.Adam(texture_parameters, lr=global_config.pti_learning_rate)
            else:
                optimizer = torch.optim.Adam(self.G.parameters(), lr=global_config.pti_learning_rate)
            return optimizer
        elif self.gan_type == 'eg3d':
            optimizer = torch.optim.Adam(self.G.parameters(), lr=global_config.pti_learning_rate)
            return optimizer

    def calc_loss(self, original_images, real_images, instance_mask, log_name, new_G, use_ball_holder, w_batch,
                  gen_camera=None, target_pose=None, depth=None, temporal_mask=None):
        loss = 0.0

        original_generated_images, generated_masks = original_images[:, :3, :, :], original_images[:, 3:, :, :]
        if self.gan_type == 'get3d':
            generated_images = mask_out(original_generated_images, instance_mask)
        else:
            generated_images = original_generated_images

        l2_loss_val = loss_lpips = None
        if global_config.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * global_config.pt_l2_lambda

        if global_config.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.mean(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * global_config.pt_lpips_lambda

        # if use_ball_holder and global_config.use_locality_regularization:
            # ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch,
                                                                             # target_pose=target_pose, use_wandb=self.use_wandb)
            # loss += ball_holder_loss_val

        # if global_config.pt_temporal_photo_lambda > 0:
            # loss_tc = l2_loss.l2_loss(temporal_mask[::2] * generated_images[::2],
                                      # temporal_mask[1::2] * generated_images[1::2])

            # loss += loss_tc * global_config.pt_temporal_photo_lambda

        # if global_config.pt_temporal_depth_lambda > 0:
            # loss_depth_tc = l2_loss.l2_loss(temporal_mask[::2] * depth[::2],
                                            # temporal_mask[1::2] * depth[1::2])

            # loss += loss_depth_tc * global_config.pt_temporal_depth_lambda

        if global_config.directional_loss:
            from training.sample_camera_distribution import sample_camera
            gen_camera = sample_camera(camera_data_mode='shapenet_chair', n=1)
            camera_condition = torch.cat((gen_camera[-2], gen_camera[-1]), dim=-1)
            emb = self.D.forward(original_images, camera_condition)
            # import pdb;pdb.set_trace()

        if global_config.clip_loss:
            def preprocess(image_input):
                # [0.. 1]
                image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(global_config.device)
                image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(global_config.device)
                image_input -= image_mean[:, None, None]
                image_input /= image_std[:, None, None]
                return F.interpolate(image_input, size=(224, 224), mode='area')

            new_target_pose = apply_random_rotation(target_pose, inverse=self.gan_type=='get3d')
            new_target_pose = new_target_pose.to(global_config.device)
            new_target_pose.requires_grad = False
            # new_target_pose = target_pose
            # new_generated_images, _ = new_G.my_forward(w_batch, new_target_pose[:, :16].reshape(-1, 1, 4, 4))
            new_generated_images, _ = new_G.my_forward(w_batch, new_target_pose[:, :16].reshape(-1, 1, 4, 4))
            new_generated_images = mask_out(new_generated_images[:, :3, ...],
                                            new_generated_images[:, 3:, ...])

            new_generated_images = preprocess((new_generated_images.clamp(-1, 1) + 1) / 2)
            real_images = preprocess((real_images.clamp(-1, 1) + 1) /2)

            ori_features = self.clip_model.encode_image(real_images)
            gen_features = self.clip_model.encode_image(new_generated_images)
            clip_loss = (ori_features - gen_features).square().sum() * global_config.clip_lambda
            print('clip_loss', clip_loss.item())
            loss += clip_loss

        return loss, l2_loss_val, loss_lpips

    def forward(self, w, target_pose, target_size=512):
        if self.gan_type == 'eg3d':
            batch_size = target_pose.shape[0]
            if target_pose.shape[-1] == 25:
                target_pose = target_pose.to(global_config.device)
            else:
                # default intrinsics
                intrinsics = np.asarray([1.2038971, 0., 0.5, 0., 1.2038971, 0.5, 0., 0., 1.]).astype(np.float32)
                target_pose = np.concatenate([target_pose.cpu().numpy().flatten(), intrinsics])
                target_pose = torch.tensor(target_pose, device=global_config.device).unsqueeze(0)

            w = w.repeat((batch_size, 1, 1))
            generated_images = self.G.synthesis(w, target_pose, noise_mode='const', force_fp32=True)['image']
            return generated_images, None
        elif self.gan_type == 'get3d':
            batchsize = target_pose.shape[0]
            w = w.repeat([batchsize, 1, 1])
            camera = target_pose[:, :16].reshape((batchsize, 1, 4, 4)).to(global_config.device)
            images, gen_camera = self.G.my_forward(w, camera)
            # images, gen_camera = self.G.forward(w.squeeze(0), camera)
            # images = images.unsqueeze(0)
            return images, gen_camera
        else:
            assert False
