import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_my_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam

def create_camera_matrix(rotation_angle, elevation_angle, camera_radius):
    device = 'cpu'
    n = rotation_angle.shape[0]
    phi = elevation_angle
    theta = rotation_angle
    sample_r = camera_radius
    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = sample_r * torch.cos(phi)
    camera_origin = output_points

    forward_vector = normalize_vecs(camera_origin)
    world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device=device)
    
    return world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle

def create_condition_from_camera_angle( rotation, elevation, radius):

    fovy = np.arctan(32 / 2 / 35) * 2
    fovyangle = fovy / np.pi * 180.0
    ######################
    # # fov = fovyangle
    # focal = np.tan(fovyangle / 180.0 * np.pi * 0.5)
    # proj_mtx = projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)
    elevation = torch.zeros(1) + elevation
    rotation = torch.zeros(1) + rotation
    radius = torch.zeros(1) + radius
    world2cam, _, _, _, _ = create_camera_matrix(rotation_angle=rotation, elevation_angle=elevation, camera_radius=radius)
    intrinsics = convert_intrinsic_toeg3d(fovyangle_y = fovyangle, ratio = 1.0)
    cam2world_mat = convert_extrinsic_toeg3d(world2cam.data.cpu().numpy()[0])
    # camera = viewpoint_estimator
    condition = np.concatenate([cam2world_mat.reshape(-1), intrinsics.reshape(-1)]).astype(
        np.float32)
    return condition



######################

def perspectiveprojectionnp_wz(fovy, ratio=1.0, near=0.01, far=10.0):
    tanfov = np.tan(fovy / 2.0)
    mtx = [[1.0 / (ratio * tanfov), 0, 0, 0], \
           [0, 1.0 / tanfov, 0, 0], \
           [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], \
           [0, 0, -1.0, 0]]
    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=np.float32)


def convert_intrinsic_toeg3d(fovyangle_y, ratio):
    # ratio:  w/h
    fovy_y = fovyangle_y / 180.0 * np.pi
    mtx = perspectiveprojectionnp_wz(fovy_y, ratio)

    fx, fy = mtx[:2]
    fx = fx / 2
    fy = fy / 2
    cx = 0.5
    cy = 0.5

    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)

    return mtx


def convert_extrinsic_toeg3d(pos_world2cam_4x4):
    P_c2w_4x4 = np.linalg.inv(pos_world2cam_4x4)

    return P_c2w_4x4


def create_condition_from_camera_angle( rotation, elevation, radius):

    fovy = np.arctan(32 / 2 / 35) * 2
    fovyangle = fovy / np.pi * 180.0
    ######################
    # # fov = fovyangle
    # focal = np.tan(fovyangle / 180.0 * np.pi * 0.5)
    # proj_mtx = projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)
    elevation = torch.zeros(1) + elevation
    rotation = torch.zeros(1) + rotation
    radius = torch.zeros(1) + radius
    world2cam, _, _, _, _ = create_camera_matrix(rotation_angle=rotation, elevation_angle=elevation, camera_radius=radius)
    intrinsics = convert_intrinsic_toeg3d(fovyangle_y = fovyangle, ratio = 1.0)
    cam2world_mat = convert_extrinsic_toeg3d(world2cam.data.cpu().numpy()[0])
    # camera = viewpoint_estimator
    condition = np.concatenate([cam2world_mat.reshape(-1), intrinsics.reshape(-1)]).astype(
        np.float32)
    return condition

def get_int_ext(rotation, elevation, radius=1.2, use_random_label=False, random_elevation_max=30):
    #############
    # Rotation angle is 0~360
    #if use_random_label:
    #    rotation_camera = np.random.rand(rotation_camera.shape[0]) * 360  # 0 ~ 360
    #    elevation_camera = np.random.rand(elevation_camera.shape[0]) * self.random_elevation_max  # ~ 0~30
    # rotation = (-rotation_camera[img_idx] - 90) / 180 * np.pi
    # elevation = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
    rotation = (rotation - 90) / 180 * np.pi
    elevation = (elevation - 0) / 180.0 * np.pi
    results = create_condition_from_camera_angle(rotation, elevation, radius)
    return results

