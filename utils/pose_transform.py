from IPython.display import HTML
from base64 import b64encode
from data_preparation.scannet_preprocess.batch_load_scannet_data import get_object_mesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from utils.camera_utils import convert_extrinsic_toeg3d
from utils.camera_utils import convert_intrinsic_toeg3d
import PIL
import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import sys
import torch
import trimesh
from configs import global_config


def save_point_cloud(xyz, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(path, pcd)

def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def get_pure_rotation(progress_11: float, max_angle: float = 180):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "y", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose

def create_my_world2cam_matrix(direction, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    direction = normalize_vecs(direction)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(direction)

    left_vector = normalize_vecs(torch.cross(up_vector, direction, dim=-1))

    up_vector = normalize_vecs(torch.cross(direction, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(direction.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(direction.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), direction.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam

def get_scan2cad_poses(scene_id='scene0113_00'):
    import json
    with open(global_config.annotation_file_path, 'r') as f:
        all_data = json.load(f)
    poses = []
    for data in all_data:
        if data['id_scan'] == scene_id:
            scene_trs = data['trs']
            for obj in data['aligned_models']:
                poses.append(obj)
            break
    return scene_trs, poses

def get_info(T, unit_axis=[0,0,1]):
     origin = -T[:3, :3].T @ T[:3, 3]
     direction = T[:3, :3].T @ np.array(unit_axis)
     return origin, direction

def get_matrix_from(translation,  rotation):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = np.eye(4)
    R[:3, :3] = mesh.get_rotation_matrix_from_quaternion(rotation)
    tmp = np.eye(4)
    tmp[:3, 3] = translation
    return tmp @ R

def get_frame_reader(scene_id):
    from omegaconf import OmegaConf
    from datasets.generic_dataset import GenericDataset
    conf_cli = OmegaConf.from_cli()
    conf_cli.scene_id = scene_id
    conf_dataset = OmegaConf.load("/data/sunfanyun/ActiveNeRF/config/scannet_base.yaml")
    conf_default = OmegaConf.load("/data/sunfanyun/ActiveNeRF/config/default_conf.yaml")
    conf = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    scene_id = conf_cli.scene_id
    conf.dataset_extra["root_dir"] = conf.dataset_extra["root_dir"].replace('SCENE_ID', scene_id)
    conf.dataset_extra["bbox_dir"] = conf.dataset_extra["bbox_dir"].replace('SCENE_ID', scene_id)
    conf.dataset_extra["split"] = conf.dataset_extra["split"].replace('SCENE_ID', scene_id)
    conf.dataset_extra["scene_id"] = conf.dataset_extra["scene_id"].replace('SCENE_ID', scene_id)
    train_dataset = GenericDataset(split="full",
                                   img_wh=tuple(conf.img_wh),
                                   dataset_extra=conf.dataset_extra)
    return train_dataset

def get_image_camera_params(frame_reader=None, scene_id='scene0113_00', frame_idx=0, obj_id=3, seed=1, normalize=True,
                            G=None, inverse=False, custom_rotation=[0,0,0], adjust_origin=1.):

    if frame_reader is None:
        frame_reader = get_frame_reader(scene_id)

    W, H = 640, 480
    scene_trs, objs = get_scan2cad_poses(scene_id)
    assert np.all(np.isclose(scene_trs['scale'], 1.))

    sample = frame_reader.read_frame_data(
        frame=frame_reader.meta['frames'][frame_idx],
        instance_id=obj_id,
        read_instance_only=False
    )

    c2w = np.array(frame_reader.meta['frames'][frame_idx]['transform_matrix'])
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    frame = {'transform_matrix': c2w}

    color_data = sample['rgbs']
    _depth_data = sample['depths']
    camera_pose = sample['c2w']
    instance_mask = sample['instance_mask']

    # index, color_data, depth_data, camera_pose, instance_mask = frame_reader[frame_idx]
    instance_mask = instance_mask.cpu().numpy().flatten()

    image_origin = color_data.view(-1, 3).detach().cpu().numpy()
    if G is not None:
        plt.figure()
        plt.imshow(image_origin.reshape(H,W,3))

    # image_origin[~instance_mask, :] = [0,0,0]
    image_origin = image_origin.reshape(H, W, 3)
    image_origin = image_origin[:, 80:-80:, :]
    if G is not None:
        plt.figure()
        plt.imshow(image_origin)

    # Tow_orig = np.eye(4)
    # Tow_orig[:3, 3] = -frame_reader.instance_bboxes[obj_id-1, :3]
    # Tow_orig = Tow_orig @ frame_reader.axis_align_mat
    
    # Toc = Tow_orig @ camera_pose_Twc
    # Tco = np.linalg.inv(Toc)
    
    # 5 is chair
    # assert frame_reader.instance_bboxes[obj_id-1, -1] == 5
    
    tmp = np.zeros(4)
    tmp[3] = 1
    tmp[:3] = frame_reader.instance_bboxes[obj_id-1, :3]
    pos = np.linalg.inv(frame_reader.axis_align_mat) @ tmp

    Tws = get_matrix_from(scene_trs['translation'], scene_trs['rotation'])
    min_dist = 1e9
    mn_idx = 100
    for obj_idx, obj in enumerate(objs):
        pose = obj['trs']
        Twcad = get_matrix_from(obj['trs']['translation'], obj['trs']['rotation'])

        tmp = np.zeros(4)
        tmp[-1] = 1.
        tmp[:3] = obj['center']
        cad_pos = (np.linalg.inv(Tws) @ Twcad @ tmp)

        dist = ((cad_pos[:3] - pos[:3])**2).sum()
        if dist < min_dist:
            min_dist = dist
            mn_idx = obj_idx
            tmp = np.eye(4)
            tmp[:3, 3] = np.array(obj['center'])
            # cad 2 scan
            T_scad = (np.linalg.inv(Tws) @ Twcad) @ tmp

    T_sc = np.array(frame['transform_matrix'])
    T_cadc =  np.linalg.inv(T_scad) @ T_sc

    trans_obj_mesh = None
    if normalize:
        obj_mesh = get_object_mesh(scene_id, obj_id=obj_id)
        padded_obj_mesh = np.ones((obj_mesh.shape[0], 4))
        padded_obj_mesh[:, :3] = obj_mesh
        trans_obj_mesh = np.dot(padded_obj_mesh, np.linalg.inv(T_scad).transpose())
        trans_obj_mesh = trans_obj_mesh[:, :3]
        for i in range(3):
            center_i = (trans_obj_mesh[:, i].max() + trans_obj_mesh[:, i].min())/2
            trans_obj_mesh[:, i] -= center_i


        obj_bbox = [trans_obj_mesh[:, i].max() - trans_obj_mesh[:, i].min() for i in range(3)]
        # print(frame_reader.instance_bboxes[obj_id-1, 3:6])
        # print(obj_bbox)
        path = './data/gt_object_mesh/{}_{}.ply'.format(scene_id, obj_id)
        if not os.path.exists(path):
            save_point_cloud(trans_obj_mesh, path)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(trans_obj_mesh)
        # o3d.io.write_point_cloud("./data.ply", pcd)
        # import pdb;pdb.set_trace()

        # obj_bbox = frame_reader.instance_bboxes[obj_id-1, 3:6]
        scale = 0.7
        factor = np.array(obj_bbox).max() / scale
        trans = np.eye(4)
        trans[0,0] = trans[1,1] = trans[2,2] = 1/factor
        T_cadc = trans @ T_cadc
        # import pdb;pdb.set_trace()
        # the following is equivalent
        # T_cadc[:3, 3] /= factor

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T_ocad = np.eye(4)
    custom_rotation = (np.array(custom_rotation) / 360) * 2 * np.pi
    T_ocad[:3, :3] = mesh.get_rotation_matrix_from_xyz((custom_rotation[0], custom_rotation[1], custom_rotation[2]))
    new_Toc =  T_ocad @ T_cadc

    if inverse:
        new_Toc = np.linalg.inv(new_Toc)

    w, h = image_origin.shape[0], image_origin.shape[1]
    fx = fy = frame_reader.focal
    fovyangle = np.rad2deg(2 * np.arctan2(h, 2 * fy))
    intrinsics = convert_intrinsic_toeg3d(fovyangle_y=fovyangle, ratio=1)
    camera_params = np.concatenate([new_Toc.reshape(-1), intrinsics.reshape(-1)]).astype(np.float32)
    camera_params = torch.from_numpy(camera_params).reshape(1, 25)

    if G is not None:
        device = torch.device('cuda')
        for seed in range(5):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            ws = G.mapping(z, torch.zeros_like(camera_params).to(device))
            img = G.synthesis(ws, camera_params.to(device))['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            plt.figure()
            plt.imshow(PIL.Image.fromarray(img[0, ...].cpu().numpy(), 'RGB'))

    return instance_mask.reshape(H, W)[:, 80:-80], image_origin, camera_params, frame_reader

def apply_random_rotation(camera, inverse=False):
    batch_size = camera.shape[0]
    intrinsics = camera[:, 16:].cpu().numpy()
    camera = camera[:, :16].reshape(batch_size, 4, 4).cpu().numpy()
    if inverse:
        camera = np.linalg.inv(camera)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    transform = np.eye(4)
    custom_rotation = np.random.rand(1) * 360
    custom_rotation = (np.array(custom_rotation) / 360) * 2 * np.pi
    transform[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, custom_rotation[0], 0))
    camera = transform @ camera
    if inverse:
        camera = np.linalg.inv(camera)
    return torch.FloatTensor(np.concatenate([camera.reshape(batch_size, -1), intrinsics], axis=1))


if __name__ == '__main__':
    cfg, args = get_arguments()
    frame_reader = get_dataset(cfg, args, cfg['scale'])
    # scene0113
    instance_mask, target_image, camera_params = \
        get_image_camera_params(frame_reader, obj_id=3, frame_idx=40, G=G)
    instance_mask, target_image, camera_params = \
        get_image_camera_params(frame_reader, obj_id=4, frame_idx=200, G=G)
    instance_mask, target_image, camera_params = \
        get_image_camera_params(frame_reader, obj_id=5, frame_idx=500, G=G)
