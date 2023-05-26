import torch
import numpy as np
import wandb
import json
from configs import global_config
from inversion.criteria import l2_loss


class Space_Regulizer:
    def __init__(self, original_G, gan_type, lpips_net):
        self.original_G = original_G
        self.gan_type = gan_type
        self.morphing_regulizer_alpha = global_config.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = global_config.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        # self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w

    # def get_image_from_ws(self, w_codes, G):
        # return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, target_pose, use_wandb=False):
        loss = 0.0

        if self.gan_type == 'eg3d':
            z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
            target_pose = torch.tensor(target_pose, device=global_config.device) #.unsqueeze(0)
            w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(global_config.device), target_pose[:1, :].repeat(num_of_sampled_latents, 1), truncation_psi=0.5)
            # territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        elif self.gan_type == 'get3d':
            truncation_cutoff=None
            truncation_psi=0.7
            update_emas=False
            device = global_config.device

            z_samples = torch.randn([num_of_sampled_latents, self.original_G.z_dim]).to(device)
            geo_z_samples = torch.randn([num_of_sampled_latents, self.original_G.z_dim]).to(device)
            c = torch.ones(1).to(device)

            ws = self.original_G.mapping(z_samples, c,
                        truncation_psi=truncation_psi,
                        truncation_cutoff=truncation_cutoff,
                        update_emas=update_emas)
            ws = ws[:, :1, :]

            ws_geo = self.original_G.mapping_geo(geo_z_samples, c,
                                truncation_psi=truncation_psi,
                                truncation_cutoff=truncation_cutoff,
                                update_emas=update_emas)
            ws_geo = ws_geo[:, :1, :]
            w_samples = torch.cat([ws, ws_geo], dim=1)
            assert w_samples.shape[1] == w_batch.shape[1]
            # territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]
        else:
            assert False

        assert w_samples.shape[0] == num_of_sampled_latents
        for i in range(w_samples.shape[0]):
            w_code = w_samples[i:i+1, ...]
            w_code = self.get_morphed_w_code(w_code, w_batch)
            camera = target_pose[i:i+1, ...]

            if self.gan_type == 'eg3d':
                new_img = new_G.synthesis(w_code, camera, noise_mode='none', force_fp32=True)['image']
                with torch.no_grad():
                    old_img = self.original_G.synthesis(w_code, camera, noise_mode='none', force_fp32=True)['image']

            elif self.gan_type == 'get3d':
                camera = camera[:, :16].reshape((-1,1,4,4)).to(device)
                new_img = new_G.my_forward(w_code, camera)[:, :3, :]
                with torch.no_grad():
                    old_img = self.original_G.my_forward(w_code, camera)[:, :3, :]

            if global_config.regulizer_l2_lambda > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)
                if use_wandb:
                    wandb.log({f'space_regulizer_l2_loss_val': l2_loss_val.detach().cpu()},
                              step=global_config.training_step)
                loss += l2_loss_val * global_config.regulizer_l2_lambda

            if global_config.regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                if use_wandb:
                    wandb.log({f'space_regulizer_lpips_loss_val': loss_lpips.detach().cpu()},
                              step=global_config.training_step)
                loss += loss_lpips * global_config.regulizer_lpips_lambda

        return loss / num_of_sampled_latents

    def space_regulizer_loss(self, new_G, w_batch, target_pose, use_wandb):
        ret_val = self.ball_holder_loss_lazy(new_G, global_config.latent_ball_num_of_samples, w_batch, target_pose, use_wandb)
        return ret_val
