# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
import pdb
import open3d as o3d
from configs import global_config
from data_preparation.scannet_preprocess import scannet_utils
from data_preparation.scannet_preprocess.load_scannet_data import read_aggregation, read_segmentation, export


def get_object_mesh(scan_name, obj_id):

    SCANNET_DIR = global_config.scans_dir
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + "_vh_clean_2.ply")
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + ".aggregation.json")
    seg_file = os.path.join(
        SCANNET_DIR, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json"
    )
    meta_file = os.path.join(
        SCANNET_DIR, scan_name, scan_name + ".txt"
    )  # includes axisAlignment info for the train set scans.
    # (
        # mesh_vertices,
        # semantic_labels,
        # instance_labels,
        # instance_bboxes,
        # instance2semantic,
    # ) = export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)
    label_map = scannet_utils.read_label_mapping(
        global_config.label_map_file, label_from="raw_category", label_to="nyu40id"
    )
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    # axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    # pts = np.ones((mesh_vertices.shape[0], 4))
    # pts[:, 0:3] = mesh_vertices[:, 0:3]
    # pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    # mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 8))

    # for obj_id in object_id_to_segs:
        # label_id = object_id_to_label_id[obj_id]
    obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]

    # transform = mesh.get_rotation_matrix_from_xyz((0, 90/360*2*np.pi, 0))
    # obj_pc = transform @ obj_pc
    return obj_pc

def export_one_scan(scan_name, output_filename_prefix):
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + "_vh_clean_2.ply")
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + ".aggregation.json")
    seg_file = os.path.join(
        SCANNET_DIR, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json"
    )
    meta_file = os.path.join(
        SCANNET_DIR, scan_name, scan_name + ".txt"
    )  # includes axisAlignment info for the train set scans.
    (
        mesh_vertices,
        semantic_labels,
        instance_labels,
        instance_bboxes,
        instance2semantic,
    ) = export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask, :]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print("Num of instances: ", num_instances)

    # bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    # instance_bboxes = instance_bboxes[bbox_mask,:]
    # print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix + "_vert.npy", mesh_vertices)
    np.save(output_filename_prefix + "_sem_label.npy", semantic_labels)
    np.save(output_filename_prefix + "_ins_label.npy", instance_labels)
    np.save(output_filename_prefix + "_bbox.npy", instance_bboxes)

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print("Creating new data folder: {}".format(OUTPUT_FOLDER))
        os.mkdir(OUTPUT_FOLDER)

    for scan_name in TRAIN_SCAN_NAMES:
        print("-" * 20 + "begin")
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        # if os.path.isfile(output_filename_prefix + "_vert.npy"):
            # print("File already exists. skipping.")
            # print("-" * 20 + "done")
            # continue
        export_one_scan(scan_name, output_filename_prefix)
        try:
            export_one_scan(scan_name, output_filename_prefix)
        except:
            print("Failed export scan: %s" % (scan_name))
        print("-" * 20 + "done")


if __name__ == "__main__":
    SCANNET_DIR = sys.argv[1]
    LABEL_MAP_FILE = sys.argv[2]
    OUTPUT_FOLDER = sys.argv[3] #"./scannet_train_detection_data"
    TRAIN_SCAN_NAMES = [sys.argv[4]]
    #TRAIN_SCAN_NAMES = sorted(
    #    [line.rstrip() for line in open("meta_data/scannet_train.txt")]
    #)
    DONOTCARE_CLASS_IDS = np.array([])
    OBJ_CLASS_IDS = np.array(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    )
    # MAX_NUM_POINT = 50000
    MAX_NUM_POINT = sys.maxsize
    batch_export()

    # xyz = get_object_mesh('scene0113_00', 3)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("./data.ply", pcd)
