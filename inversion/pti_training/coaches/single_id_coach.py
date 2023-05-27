from PIL import Image
from tqdm import tqdm
import imageio
import numpy as np
import open3d as o3d
import os
import pickle
import torch
import torch.nn.functional as F
import trimesh
import vg

from configs import global_config
from inversion.pti_training.coaches.base_coach import BaseCoach
from utils.camera_utils import get_int_ext
from utils.metrics import psnr, ssim, get_bbox
from utils.pose_transform import load_point_cloud, save_point_cloud
from inversion.utils.models_utils import mask_out
from data_preparation.scannet_preprocess.batch_load_scannet_data import get_object_mesh


def calc_3d_metric(mesh_rec, scene_id, obj_id, align=False):
    """
    3D reconstruction metric.
    """
    # gt 
    #o3d_gt_pc = o3d.geometry.PointCloud(points=mesh_gt.vertices)
    path = './data/gt_object_mesh/{}_{}.ply'.format(scene_id, obj_id)
    assert os.path.exists(path)
    xyz = load_point_cloud(path)
    # else:
        # xyz = get_object_mesh(scene_id, obj_id)
        # save_point_cloud(xyz, path)

    o3d_gt_pc = o3d.geometry.PointCloud()
    o3d_gt_pc.points = o3d.utility.Vector3dVector(xyz)
    # print([xyz[:, i].max()-xyz[:, i].min() for i in range(3)])

    longest_axis = 0
    gt_length = xyz[:, 0].max() - xyz[:, 0].min()
    for i in [1,2]:
        a_length = xyz[:, i].max() - xyz[:, i].min()
        if a_length > gt_length:
            gt_length = a_length
            longest_axis = i

    # mesh
    #o3d_rec_pc = o3d.geometry.PointCloud(points=mesh_rec.vertices)
    # mesh_rec = trimesh.load(rec_meshfile, process=False)
    # trimesh.exchange.export.export_mesh(mesh_rec, '{}_{}_rec.ply'.format(scene_id, obj_id))
    vertices, faces = np.array(mesh_rec.vertices), mesh_rec.faces
    # rotate 90 degrees
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    transform = mesh.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))

    vertices= np.dot(vertices, transform)
    for i in range(3):
        center_i = (vertices[:, i].max() + vertices[:, i].min())/2
        vertices[:, i] -= center_i

    # rescale
    rec_length = vertices[:, longest_axis].max() - vertices[:, longest_axis].min()
    vertices *= gt_length/rec_length

    # padded_obj_mesh = np.ones((vertices.shape[0], 4))
    # padded_obj_mesh[:, :3] = vertices

    # save_point_cloud(vertices, path.split('.')[0] + '{}.ply'.format(global_config.gan))

    mesh_rec = trimesh.Trimesh(vertices=vertices, faces=faces)
    o3d_rec_pc = o3d.geometry.PointCloud()
    o3d_rec_pc.points = o3d.utility.Vector3dVector(mesh_rec.vertices)

    if align:
        trans_init = np.eye(4)
        threshold = 0.1
        reg_p2p = o3d.pipelines.registration.registration_icp(
            o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformation = reg_p2p.transformation
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    from kaolin.metrics.pointcloud import chamfer_distance, f_score
    xyz = torch.FloatTensor(xyz).cuda().unsqueeze(0)
    rec_xyz = torch.FloatTensor(rec_pc_tri.vertices).cuda().unsqueeze(0)
    chamfer_distance = chamfer_distance(xyz, rec_xyz).item()
    f_score = f_score(xyz, rec_xyz, radius=0.04).item()
    return chamfer_distance, f_score

def generate_mesh(G, ws, output_path=None):
    truncation_cutoff=None
    truncation_psi=0.7
    update_geo=True
    update_emas=False
    generate_no_light=True
    generate_txture_map=True
    device = torch.device('cuda:0')
    if global_config.gan == 'get3d':
        # get3d
        G.synthesis.align_blend_camera = True
        camera_list = G.synthesis.generate_rotate_camera_list(n_batch=1)
        with torch.no_grad():
            img, (mesh_v, mesh_f) = G.my_forward(ws, camera_list[0])
        v_list, f_list = mesh_v[0].cpu().numpy(), mesh_f[0].cpu().numpy()
        # print(v_list.shape, f_list.shape)
        # ws, ws_geo = ws[:, :1, :], ws[:, 1:, :]
        # ws = ws.repeat([1, 9, 1]).to(device)
        # ws_geo = ws.repeat([1, 22, 1]).to(device)
        # ws_back = None
        # camera = torch.ones((1,1,4,4,)).to(device) # camera_params[0, :16].reshape((1,1,4,4)).to(device)

        # with torch.no_grad():
            # generated_mesh = G.synthesis.extract_3d_shape(ws, ws_geo)
        # for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
            # v_list, f_list = mesh_v.cpu().numpy(), mesh_f.cpu().numpy()
        # all_mesh = G.generate_3d_mesh(ws_geo, ws,
                                      # truncation_psi=0.7,
                                      # c=None,
                                      # use_mapping=False)
        # with torch.no_grad():
            # img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
            # sdf_reg_loss, render_return_value = G.synthesis.generate(
                            # ws, update_emas=update_emas, camera=camera,
                            # update_geo=update_geo, ws_geo=ws_geo,
                            # ws_back=ws_back,
                            # generate_no_light=generate_no_light,
                            # generate_txture_map=generate_txture_map,
                            # noise_mode='const')
        # v_list, f_list = all_mesh[0][0].detach().cpu().numpy(), all_mesh[1][0].detach().cpu().numpy()
    else:
        # eg3d
        if ws.shape[1] == 1:
            ws = ws.repeat([ws.shape[0], G.backbone.mapping.num_ws, 1])
        with torch.no_grad():
            v_list, f_list = G.my_generate_3d_mesh(ws)
            # v_list, f_list = G.generate_3d_mesh(ws)
        v_list, f_list =v_list[0].cpu().numpy(), f_list[0].cpu().numpy()

    if v_list.shape[0] == 0:
        print('failed to generate mesh')
        return None

    v_list[..., 0] -= (v_list[..., 0].max() + v_list[..., 0].min())/2
    v_list[..., 2] -= (v_list[..., 2].max() + v_list[..., 2].min())/2
    v_list[..., 1] = v_list[..., 1] - v_list[..., 1].min()

    scale_factor =  v_list[..., 1].max() - v_list[..., 1].min()
    v_list /= scale_factor

    mesh = trimesh.Trimesh(v_list, f_list)
    if output_path:
        mesh.export(output_path)
    return mesh

def get_iou(mask_target, mask_pred):
    overlap_mask = mask_target * mask_pred
    union_mask = mask_target + mask_target - overlap_mask
    iou = (overlap_mask.sum() / union_mask.sum()).item()
    return iou

def get_rotational_delta(camera_param1, camera_param2, inverse=False):
    assert camera_param1.shape == camera_param2.shape
    trans1 = camera_param1.reshape(4,4).cpu().numpy()
    trans2 = camera_param2.reshape(4,4).cpu().numpy()
    if inverse:
        trans1 = np.linalg.inv(trans1)
        trans2 = np.linalg.inv(trans2)

    return vg.angle(trans1[:3, 3], trans2[:3, 3])

    return vg.angle(trans1.cpu().numpy(), trans2.cpu().numpy())

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, fovyangle, use_wandb):
        super().__init__(data_loader, fovyangle, use_wandb)

    def generate_object_video(self, ws, vid_path):
        if self.gan_type == 'eg3d':
            if ws.shape[1] == 1:
                ws = ws.repeat([1, self.G.backbone.mapping.num_ws, 1])
            rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')
            imgs = []
            angle_p = -0.2
            for rotation in np.arange(0, 360, 15):
                camera_params = get_int_ext(rotation, 90, use_random_label=True)
                camera_params = torch.from_numpy(camera_params.reshape(1, -1)).to(global_config.device)
                img = self.G.synthesis(ws, camera_params, noise_mode='const', force_fp32=True)['image']
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                rgb_video.append_data(img[0, ...].cpu().numpy())
            rgb_video.close()

        elif self.gan_type == 'get3d':
            rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')
            self.G.synthesis.align_blend_camera = True
            # len: 24
            camera_list = self.G.synthesis.generate_rotate_camera_list(n_batch=1)
            for i_camera, camera in tqdm(enumerate(camera_list)):
                img, _ = self.G.my_forward(ws, camera)
                # img, _ = self.G.forward(ws.squeeze(0), camera)
                # img = img.unsqueeze(0)
                img, mask = img[0, :3, :], img[0, 3:, :].repeat([3,1,1])
                img = img * mask + (1. - mask)
                img = (img.permute(1,2,0) * 127.5 + 128).clamp(0, 255.).to(torch.uint8)
                rgb_video.append_data(img.cpu().numpy())
            rgb_video.close()

    def evaluate(self, ws, camera_params, real_images, instance_masks, image_names=None, align=False, crop_locations=None, output_prefix=None):
        chamfer, f_score = -1, -1
        if image_names is not None:
            scene_id = image_names[0].split(',')[0][:-3]
            if global_config.dataset == 'scannet':
                obj_id = int(image_names[0].split(',')[-1].split('_')[0])
                mesh_rec = generate_mesh(self.G, ws)
                if mesh_rec is not None:
                    chamfer, f_score = calc_3d_metric(mesh_rec, scene_id, obj_id, align=align)
            else:
                obj_id = 0
        # [batch_size, 3, 480, 480]
        real_images = real_images.type(torch.FloatTensor).to(global_config.device)
        target_image_size = real_images.shape[-1]

        assert ws.shape[0] == 1
        with torch.no_grad():
            # generated_images, _ = self.forward(ws, camera_params)
            generated_images = []
            for j in range(camera_params.shape[0]):
                generated_image, _ = self.forward(ws, camera_params[j:j+1])
                generated_images.append(generated_image[0])
            generated_images = torch.stack(generated_images)

            generated_images, generated_masks = generated_images[:, :3, :, :], generated_images[:, 3:, :, :]
            generated_images = F.interpolate(generated_images, size=(target_image_size, target_image_size))
            if self.gan_type == 'get3d':
                generated_masks = F.interpolate(generated_masks, size=(target_image_size, target_image_size))
                generated_images = mask_out(generated_images, generated_masks)

            if instance_masks is not None:
                generated_images = mask_out(generated_images, instance_masks)
                real_images = mask_out(real_images, instance_masks)

        generated_images = generated_images.clip(-1, 1.)

        if crop_locations:
            ori_generated_images = generated_images
            ori_instance_masks = instance_masks
            ori_real_images = real_images

            assert generated_images.shape[0] == real_images.shape[0]
            assert len(crop_locations) == generated_images.shape[0]
            losses_lpips, losses_psnr, losses_ssim = [], [], []

            for i in range(real_images.shape[0]):
                crop_location = crop_locations[i]
                generated_images = ori_generated_images[i:i+1, ..., crop_location[0]:crop_location[1], crop_location[2]: crop_location[3]]
                if ori_instance_masks is not None:
                    instance_masks = ori_instance_masks[i:i+1, ..., crop_location[0]:crop_location[1], crop_location[2]: crop_location[3]]
                    assert instance_masks.sum().item() == ori_instance_masks[i].sum().item()
                real_images = ori_real_images[i:i+1, ..., crop_location[0]:crop_location[1], crop_location[2]: crop_location[3]]

                with torch.no_grad():
                    loss_lpips = self.lpips_loss(generated_images, real_images)
                    loss_lpips = torch.mean(loss_lpips)

                loss_ssim = ssim((generated_images+1)/2, (real_images+1)/2)

                generated_images = 0.5 * (generated_images + 1) * 255
                generated_images = generated_images.clamp(0, 255)
                real_images = 0.5 * (real_images + 1) * 255
                real_images = real_images.clamp(0, 255)

                if instance_masks is not None:
                    instance_masks = instance_masks.to(torch.uint8)
                    loss_psnr = psnr(generated_images, real_images, valid_mask=instance_masks.repeat([1,3,1,1]))
                else:
                    loss_psnr = psnr(generated_images, real_images)

                losses_lpips.append(loss_lpips.item())
                losses_psnr.append(loss_psnr.item())
                losses_ssim.append(loss_ssim.item())

            ret_dic = {'lpips': np.mean(losses_lpips),
                       'psnr': np.mean(losses_psnr),
                       'ssim': np.mean(losses_ssim),
                       'iou': -1,
                       'chamfer': chamfer,
                       'f_score': f_score}

            if output_prefix:
                generated_images = (ori_generated_images + 1) * 255/2
                real_images = (ori_real_images + 1) * 255/2
                generated_images = generated_images.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                real_images = real_images.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

                final_imgs = []
                for i in range(generated_images.shape[0]):
                    tmp_img = np.concatenate([real_images[i, ...], generated_images[i, ...]], axis=1)
                    final_imgs.append(tmp_img)
                final_img = np.concatenate(final_imgs, axis=1)
                Image.fromarray(final_img).save(f'{output_prefix}_test.png')
            return ret_dic

        else:
            with torch.no_grad():
                loss_lpips = self.lpips_loss(generated_images, real_images)
                loss_lpips = torch.mean(loss_lpips)

            loss_ssim = ssim((generated_images+1)/2, (real_images+1)/2)

            generated_images = 0.5 * (generated_images + 1) * 255
            generated_images = generated_images.clamp(0, 255)
            real_images = 0.5 * (real_images + 1) * 255
            real_images = real_images.clamp(0, 255)

            if instance_masks is not None:
                instance_masks = instance_masks.to(torch.uint8)
                loss_psnr = psnr(generated_images, real_images, valid_mask=instance_masks.repeat([1,3,1,1]))
            else:
                loss_psnr = psnr(generated_images, real_images)

            ret_dic = {'lpips': loss_lpips.item(),
                       'psnr': loss_psnr.item(),
                       'ssim': loss_ssim.item(),
                       'chamfer': chamfer,
                       'f_score': f_score
                       }
            if self.gan_type == 'get3d' and instance_masks is not None:
                ious = []
                for i in range(generated_masks.shape[0]):
                    generated_mask = generated_masks[i, ...].to(torch.uint8)
                    instance_mask = instance_masks[i, ...].to(torch.uint8)
                    ious.append(get_iou(generated_mask, instance_mask))
                ret_dic['iou'] = np.mean(ious)
            else:
                ret_dic['iou'] = -1

            if output_prefix:
                generated_images = generated_images.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

                real_images = real_images.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                final_imgs = []
                for i in range(generated_images.shape[0]):
                    tmp_img = np.concatenate([real_images[i, ...], generated_images[i, ...]], axis=1)
                    final_imgs.append(tmp_img)
                final_img = np.concatenate(final_imgs, axis=1)
                Image.fromarray(final_img).save(f'{output_prefix}_test.png')
            return ret_dic


    def train(self, image_names, target_images, camera_params, instance_masks, crop_locations=None):

        w_path_dir = f'{global_config.embedding_base_dir}/'
        os.makedirs(w_path_dir, exist_ok=True)
        # import shutil
        # shutil.copy('inversion/configs/global_config.py', f'{w_path_dir}/global_config.py')

        use_ball_holder = True
        sequence_name = '_'.join(image_names[-1].split('_')[:-1])
        embedding_dir = f'{w_path_dir}/{global_config.pti_results_keyword}_{self.gan_type}/{sequence_name}'
        if not sequence_name.startswith('scene'):
            sequence_name += ',0'
        os.makedirs(embedding_dir, exist_ok=True)

        for img_name, img, mask in zip(image_names, target_images, instance_masks):
            img = (img + 1) * (255/2)
            img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            Image.fromarray(img, 'RGB').save(f'{embedding_dir}' + '/' + '{}.png'.format(img_name))

            mask = mask.repeat([1,3,1,1]) * 255
            mask = mask.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            Image.fromarray(mask, 'RGB').save(f'{embedding_dir}' + '/' + '{}_mask.png'.format(img_name))

        target_images = [(target_image+1) * (instance_mask==1.)-1 for target_image, instance_mask in zip(target_images, instance_masks)]

        for img_name, img, mask in zip(image_names, target_images, instance_masks):
            img = (img + 1) * (255/2)
            img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            Image.fromarray(img, 'RGB').save(f'{embedding_dir}' + '/' + '{}_masked.png'.format(img_name))

        ### fix this
        for idx, instance_mask in enumerate(instance_masks):
            instance_masks[idx] = (instance_mask == 1).type(torch.cuda.FloatTensor)

        NUM_ACTIVE_WS = global_config.num_active_ws
        KEEP_BEST = global_config.keep_best
        NUM_TRAIN = global_config.num_train
        current_ws = []

        for idx in range(0, NUM_TRAIN, 2):
            # use the first idx+1 images for training
            def evaluate_all(iteration=0):
                if idx == 0:
                    for j in range(NUM_TRAIN, len(camera_params)):
                        train_result = self.evaluate(w_pivot,
                                               torch.cat(camera_params[:idx+1], axis=0),
                                               torch.cat(target_images[:idx+1], axis=0),
                                               torch.cat(instance_masks[:idx+1], axis=0),
                                               crop_locations=crop_locations[:idx+1] if crop_locations else None,
                                               output_prefix=None)

                        unmasked_train_result = self.evaluate(w_pivot,
                                               torch.cat(camera_params[:idx+1], axis=0),
                                               torch.cat(target_images[:idx+1], axis=0),
                                               instance_masks=None,
                                               crop_locations=crop_locations[:idx+1] if crop_locations else None,
                                               output_prefix=None)

                        result = self.evaluate(w_pivot,
                                               torch.cat(camera_params[j:j+1], axis=0),
                                               torch.cat(target_images[j:j+1], axis=0),
                                               torch.cat(instance_masks[j:j+1], axis=0),
                                               align=True,
                                               image_names=image_names[j:j+1],
                                               crop_locations=crop_locations[j:j+1] if crop_locations else None,
                                               output_prefix=None)

                        unmasked_result = self.evaluate(w_pivot,
                                               torch.cat(camera_params[j:j+1], axis=0),
                                               torch.cat(target_images[j:j+1], axis=0),
                                               instance_masks=None,
                                               image_names=image_names[j:j+1],
                                               crop_locations=crop_locations[j:j+1] if crop_locations else None,
                                               output_prefix=f'{embedding_dir}/{session_prefix}_iter:{iteration}')

                        results = [train_result['iou'], train_result['lpips'], train_result['psnr'], train_result['ssim'],
                                   unmasked_train_result['iou'], unmasked_train_result['lpips'], unmasked_train_result['psnr'], unmasked_train_result['ssim'],
                                   result['iou'], result['lpips'], result['psnr'], result['ssim'],
                                   unmasked_result['iou'], unmasked_result['lpips'], unmasked_result['psnr'], unmasked_result['ssim'],
                                   unmasked_result['chamfer'], unmasked_result['f_score'], result['chamfer'], result['f_score']]
                        str_result = ','.join(['{:.4f}'.format(x) for x in results])
                        rotational_delta = get_rotational_delta(camera_params[0][0, :16], camera_params[j][0, :16], inverse=self.gan_type == 'get3d')
                        with open(f'{w_path_dir}/results.log', 'a+') as f:
                            f.write('{},{},{},{},{},{},{}\n'.format(sequence_name, idx+1, self.gan_type, tmp_idx, iteration, rotational_delta, str_result))

                train_result = self.evaluate(w_pivot,
                                       torch.cat(camera_params[:idx+1], axis=0),
                                       torch.cat(target_images[:idx+1], axis=0),
                                       torch.cat(instance_masks[:idx+1], axis=0),
                                       crop_locations=crop_locations[:idx+1] if crop_locations else None,
                                       output_prefix=None)

                unmasked_train_result = self.evaluate(w_pivot,
                                       torch.cat(camera_params[:idx+1], axis=0),
                                       torch.cat(target_images[:idx+1], axis=0),
                                       instance_masks=None,
                                       crop_locations=crop_locations[:idx+1] if crop_locations else None,
                                       output_prefix=None)

                result = self.evaluate(w_pivot,
                                       torch.cat(camera_params[NUM_TRAIN:], axis=0),
                                       torch.cat(target_images[NUM_TRAIN:], axis=0),
                                       torch.cat(instance_masks[NUM_TRAIN:], axis=0),
                                       align=True,
                                       image_names=image_names[NUM_TRAIN:],
                                       crop_locations=crop_locations[NUM_TRAIN:] if crop_locations else None,
                                       output_prefix=f'{embedding_dir}/{session_prefix}_iter:{iteration}')

                unmasked_result = self.evaluate(w_pivot,
                                       torch.cat(camera_params[NUM_TRAIN:], axis=0),
                                       torch.cat(target_images[NUM_TRAIN:], axis=0),
                                       instance_masks=None,
                                       image_names=image_names[NUM_TRAIN:],
                                       crop_locations=crop_locations[NUM_TRAIN:] if crop_locations else None,
                                       output_prefix=None)

                results = [train_result['iou'], train_result['lpips'], train_result['psnr'], train_result['ssim'],
                           unmasked_train_result['iou'], unmasked_train_result['lpips'], unmasked_train_result['psnr'], unmasked_train_result['ssim'],
                           result['iou'], result['lpips'], result['psnr'], result['ssim'],
                           unmasked_result['iou'], unmasked_result['lpips'], unmasked_result['psnr'], unmasked_result['ssim'],
                           unmasked_result['chamfer'], unmasked_result['f_score'], result['chamfer'], result['f_score']]

                str_result = ','.join(['{:.4f}'.format(x) for x in results])
                with open(f'{w_path_dir}/results.log', 'a+') as f:
                    f.write('{},{},{},{},{},{},{}\n'.format(sequence_name, idx+1, self.gan_type, tmp_idx, iteration, -1, str_result)) 

            if idx > 0:
                tmp_ws = []
                for j, ws in enumerate(current_ws):
                    val_result = self.evaluate(ws,
                                               torch.cat(camera_params[:idx+1], axis=0),
                                               torch.cat(target_images[:idx+1], axis=0),
                                               instance_masks=None,
                                               crop_locations=crop_locations[:idx+1] if crop_locations else None,
                                               output_prefix=None)
                    # if self.gan_type == 'get3d':
                        # tmp_ws.append((-val_result['iou'], j))
                    # else:
                        # tmp_ws.append((val_result['lpips'], j))
                    tmp_ws.append((val_result['lpips'], j))

                tmp_ws = sorted(tmp_ws)
                new_ws = [current_ws[tmp[1]] for tmp in tmp_ws[:KEEP_BEST]]
                current_ws = new_ws


            num_new_ws = NUM_ACTIVE_WS - len(current_ws)
            for tmp_idx in range(NUM_ACTIVE_WS):
                self.restart_training()
                session_prefix = f'obs:{idx+1}_{tmp_idx}_'
                if idx > 0 and tmp_idx < KEEP_BEST:
                    w_pivot, _, all_w_opt = self.calc_inversions(torch.cat(target_images[:idx+1], dim=0),
                                                                     f'{sequence_name}_{idx}',
                                                                     embedding_dir,
                                                                     mask=torch.cat(instance_masks[:idx+1], dim=0),
                                                                     target_pose=torch.cat(camera_params[:idx+1], dim=0),
                                                                     session_prefix=session_prefix + 'keep',
                                                                     write_video=True,
                                                                     initial_w=current_ws[tmp_idx])
                    w_pivot = w_pivot.to(global_config.device).detach().clone()
                    current_ws[tmp_idx] = w_pivot
                else:
                    w_pivot, w_initial, all_w_opt = self.calc_inversions(torch.cat(target_images[:idx+1], dim=0),
                                                                     f'{sequence_name}_{idx}',
                                                                     embedding_dir,
                                                                     mask=torch.cat(instance_masks[:idx+1], dim=0),
                                                                     target_pose=torch.cat(camera_params[:idx+1], dim=0),
                                                                     session_prefix=session_prefix,
                                                                     write_video=True)
                    w_pivot = w_pivot.to(global_config.device).detach().clone()
                    self.generate_object_video(w_initial.to(global_config.device), f'{embedding_dir}/{session_prefix}_initial_obj.mp4')
                    current_ws.append(w_pivot)
                    # evaluate(iteration=0)
                    # self.generate_object_video(w_pivot, f'{embedding_dir}/{session_prefix}_pivot_obj.mp4')

                self.generate_object_video(w_pivot, f'{embedding_dir}/{session_prefix}_pivot_obj.mp4')
                # torch.save(w_pivot, f'{embedding_dir}/{session_prefix}_w_pivot.pt')
                # w_pivot = w_pivot.to(global_config.device)

            for tmp_idx, w_pivot in enumerate(current_ws):
                assert w_pivot.requires_grad is False
                self.restart_training()
                session_prefix = f'obs:{idx+1}_{tmp_idx}_'

                log_images_counter = 0

                vid_path = f'{embedding_dir}/{session_prefix}_phase2.mp4'
                rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

                for i in tqdm(range(global_config.max_pti_steps)):
                    if i % 100 == 0:
                        evaluate_all(iteration=i)
                    # sample_i = np.random.randint(0, idx+1)
                    image_name= image_names[idx]
                    real_images = torch.cat(target_images[:idx+1], dim=0)
                    camera_param = torch.cat(camera_params[:idx+1], dim=0)
                    instance_mask = torch.cat(instance_masks[:idx+1], dim=0)

                    real_images = real_images.type(torch.FloatTensor).to(global_config.device)

                    generated_images, _ = self.forward(w_pivot, camera_param)
                    generated_masks = generated_images[:, 3:, :, :]

                    if self.gan_type == 'get3d':
                        target_image_size = 1024
                        real_images = F.interpolate(real_images, size=(target_image_size, target_image_size))
                        instance_mask = F.interpolate(instance_mask, size=(target_image_size, target_image_size))
                    else:
                        target_image_size = 512
                        real_images = F.interpolate(real_images, size=(target_image_size, target_image_size))

                    loss, _, _ = self.calc_loss(generated_images, real_images, instance_mask,
                                                image_name, self.G, use_ball_holder, w_pivot, target_pose=camera_param)

                    if i < global_config.pt_mask_steps and self.gan_type == 'get3d':
                        bbox = get_bbox(instance_mask).to(global_config.device)
                        # train, (1) region outside of the bbox,
                        # (2) region inside the bbox and part of gt
                        # (3) region inside the bbox not part of gt and not part of generated_mask
                        weight_mask = (((1. - bbox) + instance_mask + bbox * (1 - instance_mask) * (1 - generated_masks.detach()))).clamp(0., 1.)
                        mask_loss = F.binary_cross_entropy(generated_masks.clamp(0., 1.), instance_mask, weight=weight_mask)
                        loss += mask_loss
                        print('mask_loss', mask_loss.item())
                    print('total loss', loss.item())
                    self.optimizer.zero_grad()
                    try:
                        loss.backward()
                    except:
                        print('Failed backward pass! Exiciting training ...')
                        break
                    self.optimizer.step()

                    use_ball_holder = global_config.training_step % global_config.locality_regularization_interval == 0

                    if i % 5 == 0 or i == global_config.max_pti_steps - 1:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.995
                        real_images = (real_images + 1) * (255/2)
                        synth_images = (generated_images[:, :3, :, :] + 1) * (255/2)
                        final_imgs = []
                        if self.gan_type == 'eg3d':
                            generated_masks = torch.FloatTensor(generated_images.shape[0], 1, generated_images.shape[-2], generated_images.shape[-1])
                        for i in range(real_images.shape[0]):
                            real_image = real_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[i].cpu().numpy()
                            synth_image = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[i].cpu().numpy()
                            tmp_img1 = np.concatenate([real_image, synth_image], axis=1)

                            if self.gan_type == 'get3d':
                                real_mask = instance_mask.permute(0, 2, 3, 1).clamp(0, 1).to(torch.uint8)[i].repeat([1,1,3]).cpu().numpy() * 255
                                synth_mask = generated_masks.permute(0, 2, 3, 1).clamp(0, 1).to(torch.uint8)[i].repeat([1,1,3]).cpu().numpy() * 255
                                tmp_img2 = np.concatenate([real_mask, synth_mask], axis=1)
                                final_imgs.append(np.concatenate([tmp_img1, tmp_img2], axis=0))
                            else:
                                final_imgs.append(np.concatenate([tmp_img1], axis=0))
                        final_img = np.concatenate(final_imgs, axis=1)
                        rgb_video.append_data(final_img)
                        if i == global_config.max_pti_steps - 1:
                            evaluate_all(iteration=i)
                            Image.fromarray(final_img, 'RGB').save(f'{embedding_dir}/{session_prefix}_train_final.png')
                    global_config.training_step += 1
                    log_images_counter += 1

                # self.image_counter += 1
                rgb_video.close()
                self.generate_object_video(w_pivot, f'{embedding_dir}/{session_prefix}_final_obj.mp4')
                # torch.save(self.G.state_dict(), f'{embedding_dir}/model_{idx}.pt')
        return w_pivot
