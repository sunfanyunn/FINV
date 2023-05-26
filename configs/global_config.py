## Device
device = 'cuda:0'

## Data
dataset = 'scannet'
dataset = 'shapenet'
# sampling frequency
freq = 50
# 5 chair, 7 table, and 8 car
target_instance_type = 5
## Dirs for output files
embedding_base_dir = './output'
DEBUG = False

## Paths
root_dir = './data/scannet/processed_scannet/SCENE_ID'
bbox_dir =  './data/scannet/processed_scannet/SCENE_ID'
scans_dir = './data/scannet/scans'
annotation_file_path = './data/scannet/scan2cad/full_annotations.json'
label_map_file = './data/scannet/processed_scannet/meta_data/scannetv2-labels.combined.tsv'

## FINV parameters
if dataset == 'nuscenes':
    first_inv_steps = 350
    max_pti_steps = 1001
    pt_mask_steps = 50
elif dataset == 'scannet':
    first_inv_steps = 350
    max_pti_steps = 1001
    pt_mask_steps = 350
elif dataset == 'shapenet':
    first_inv_steps = 350
    max_pti_steps = 1001
    pt_mask_steps = 50
else:
    assert False

gan = 'eg3d'
directional_loss = False
clip_loss = False
only_texture = False
# whether to use lpips loss in the first phase or not
use_lpips = True

num_active_ws = 10
keep_best = 3
num_train = 5
clip_lambda = 1e-3
w_avg_samples = 10

if DEBUG:
    embedding_base_dir += '_DEBUG'
    first_inv_steps = 10
    max_pti_steps = 1
    num_active_ws = 1
    keep_best = 1
    num_train = 1


#########################################################
## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
jg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Architechture
lpips_type = 'alex'
first_inv_type = 'w'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
use_noise_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1
pt_temporal_photo_lambda = 0
pt_temporal_depth_lambda = 0
temporal_consistency_loss = False

## Steps
LPIPS_value_threshold = 0.06
max_images_to_invert = 30

## Optimization
pti_learning_rate = 3e-4
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False

## Logs
training_step = 1
image_rec_result_log_snapshot = 100
pivotal_training_steps = 1
model_snapshot_interval = 400
