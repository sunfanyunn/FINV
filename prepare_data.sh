#!/bin/bash -ex

root_dir=$pwd
scene_id=$1
prefix=./data/scannet/scans/
mkdir $prefix -p

# download data
python ./data_preparation/download-scannet.py --out_dir ./data/scannet --id ${scene_id}

# first preprocessing
python ./data_preparation/scannet_sens_reader/reader.py --filename $prefix/$1/$1.sens --output_path $prefix/$1 --export_color_images --export_poses --export_intrinsics  --export_depth_images

unzip data/scannet/scans/$scene_id/${scene_id}_2d-instance-filt.zip
mv instance-filt data/scannet/scans/$scene_id

python ./data_preparation/scannet_sens_reader/convert_to_nerf_style_data.py \
  --input data/scannet/scans/${scene_id}/ \
  --output data/scannet/processed_scannet/${scene_id} \
  --instance_filt_dir data/scannet/scans/${scene_id}/instance-filt

python -m data_preparation.scannet_preprocess.batch_load_scannet_data \
       $root_dir/data/scannet/scans \
       $root_dir/data_preparation/scannet_preprocess/meta_data/scannetv2-labels.combined.tsv \
       $root_dir/data/scannet/processed_scannet/$scene_id \
       $scene_id \
	 
