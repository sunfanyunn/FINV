from omegaconf import OmegaConf
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pickle
import shutil
import sys
import torch
import torch.nn.functional as F
import wandb
from datasets.scannet.generic_dataset import GenericDataset
from configs import global_config
from inversion.pti_training.coaches.single_id_coach import SingleIDCoach
from utils.pose_transform import get_image_camera_params
from datasets.shapenet.dataset import ImageFolderDataset


def run_all(run_name='', use_wandb=False):
    ### configs for the inversion
    ### eg3d or get3d
    conf_cli = OmegaConf.from_cli()
    if hasattr(conf_cli, 'gan'):
        global_config.gan = conf_cli.gan
    sys.path.append(f'./{global_config.gan}')

    if global_config.dataset == 'scannet':
        conf_dataset = OmegaConf.load("./datasets/scannet/scannet_base.yaml")
        conf_default = OmegaConf.load("./datasets/scannet/default_conf.yaml")
        conf = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

        scene_id = conf_cli.scene_id
        ## dirty, overwrite with arguments in config.py
        conf.dataset_extra["root_dir"] = global_config.root_dir.replace('SCENE_ID', scene_id)
        conf.dataset_extra["bbox_dir"] = global_config.bbox_dir.replace('SCENE_ID', scene_id)
        conf.dataset_extra["scans_dir"] = global_config.scans_dir
        conf.dataset_extra["scene_id"] = scene_id
        frame_reader = GenericDataset(split="full",
                                 img_wh=tuple(conf.img_wh),
                                 dataset_extra=conf.dataset_extra)
        print('train', len(frame_reader.meta["frames"]))

        # this is how frame_reader.focal is calculated
        # w = 640
        # frame_reader.focal =  self.focal =  0.5 * w / np.tan(0.5 * self.meta["camera_angle_x"]) 
        h = 480
        fovyangle = np.rad2deg(2 * np.arctan2(h, 2*frame_reader.focal))
        coach = SingleIDCoach(global_config.gan, fovyangle, use_wandb=use_wandb)

        freq = global_config.freq
        scene_id = frame_reader.scene_id
        for obj_id in range(1, 1 + frame_reader.instance_bboxes.shape[0]):
            if not frame_reader.instance_bboxes[obj_id-1, -1] == global_config.target_instance_type:
                continue
            print('scene_id', scene_id, freq, 'obj_id', obj_id)
            identifier = f'{scene_id}_{freq},{obj_id}'

            if use_wandb:
                run = wandb.init(project='{}-inversion'.format(global_config.gan), reinit=True)
                run.name = '{}_{},{}'.format(scene_id, freq, obj_id)

            pickled_path = './data/{}_{}_{}_{}.pkl'.format(scene_id, freq, obj_id, global_config.gan)
            if os.path.exists(pickled_path):
                print('load data from', pickled_path)
                with open(pickled_path, 'rb') as f:
                    image_names, target_images, camera_params, instance_masks = pickle.load(f)
            else:
                target_images, image_names, camera_params, instance_masks = [], [], [], []

                device = global_config.device
                frame_idxs = range(0, len(frame_reader.meta['frames']), freq)
                for obs_id, frame_idx in enumerate(frame_idxs):
                    if global_config.gan == 'eg3d':
                        instance_mask, target_image, camera_param, _ = \
                                get_image_camera_params(frame_reader, scene_id=f'{scene_id}', obj_id=obj_id, frame_idx=frame_idx,
                                                        custom_rotation=[0,-90,0], adjust_origin=1.0, inverse=False)
                    elif global_config.gan == 'get3d':
                        instance_mask, target_image, camera_param, _ = \
                                get_image_camera_params(frame_reader, scene_id=f'{scene_id}', obj_id=obj_id, frame_idx=frame_idx,
                                                        custom_rotation=[0,-90,0], adjust_origin=1.0, inverse=True)
                    else:
                        assert False

                    # check validity
                    def invalid(instance_mask):
                        # M = cam_param[0, :16].cpu().numpy().reshape(4,4)
                        # if ((M[:3, 3]**2).sum()) > 6: return True
                        return  instance_mask.sum().item() <= instance_mask.shape[0]*instance_mask.shape[1]/40
 
                    if invalid(instance_mask):
                        print('skip {}'.format(frame_idx))
                        continue
                    else:
                        camera_param = camera_param.to(device)
                        target_image = torch.from_numpy(target_image*2-1).to(device).permute(2,0,1)
                        target_image = target_image.unsqueeze(0)
                        # target_image = F.interpolate(target_image.unsqueeze(0),
                                                     # size=(target_image_size, target_image_size), mode='nearest')
                        assert camera_param.shape[-1] == 25
                        assert target_image.shape[1] == 3
                        instance_mask = torch.from_numpy(instance_mask).to(device)
                        instance_mask = instance_mask.type(torch.cuda.FloatTensor)
                        instance_mask = instance_mask.unsqueeze(0).unsqueeze(0)
                        # instance_mask = F.interpolate(instance_mask.unsqueeze(0).unsqueeze(0),
                                                      # size=(target_image_size, target_image_size), mode='nearest')

                        image_names.append('{}_{},{}_{}'.format(scene_id, freq, obj_id, frame_idx))
                        target_images.append(target_image)
                        camera_params.append(camera_param)
                        instance_masks.append(instance_mask)
                        # just for debugging
                        # coach.train(image_names, target_images, camera_params, instance_masks)
                with open(pickled_path, 'wb') as f:
                    pickle.dump([image_names, target_images, camera_params, instance_masks], f)
            if len(image_names) > 5:
                coach.train(image_names, target_images, camera_params, instance_masks)

    elif global_config.dataset == 'shapenet':
        camera_angle_x = 0.8575560450553894
        coach = SingleIDCoach(global_config.gan, camera_angle_x*180/np.pi, use_wandb=use_wandb)
        data_camera_mode = 'shapenet_chair'
        frame_reader = ImageFolderDataset(path='./data/shapenet-get3d-chair/img/03001627',
                                          camera_path='./data/shapenet_get3d-chair/camera/03001627',
                                          data_camera_mode=data_camera_mode,
                                          resolution=1024,
                                          split='test')
        for image_names, target_images, camera_params, instance_masks in frame_reader.read_sequences():
            if len(image_names) > 5:
                coach.train(image_names, target_images, camera_params, instance_masks)

    return global_config.run_name


if __name__ == '__main__':
    global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    run_all(global_config.run_name, use_wandb=False)
