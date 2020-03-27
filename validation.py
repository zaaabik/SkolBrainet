import os

import numpy as np
import torch
from medpy.metric import dc
from torch import nn, optim

from Net import Net
from helpers import predict_full, get_gt_filename, loader

gt_base_path = 'domain_gt'
img_base_path = 'domain_img'
model_filename = os.path.join('models', 'model_epoch_060')

######### PARAMETERS

train = True

raw_img_filenames = os.listdir(img_base_path)

max_idx = 169
min_idx = 120
index_reg_exp = 'CC(.*?)_'

img_filenames = []
for file_name in raw_img_filenames:
    img_filenames.append(file_name)

gt_filenames = []
for img_filename in img_filenames:
    gt_filename = get_gt_filename(img_filename)
    gt_filenames.append(gt_filename)

imgs = []
gts = []
for img_filename, gt_filename in zip(img_filenames, gt_filenames):
    gt_path = os.path.join(gt_base_path, gt_filename)
    img_path = os.path.join(img_base_path, img_filename)

    gt = loader(gt_path)
    img = loader(img_path)

    imgs.append(img)
    gts.append(gt)

print('Files count', len(img_filenames))

batch_size = 4
epochs = 100000000
crops_per_image = 500
lr = 1e-5
epochs_per_save = 1

crop_size = 65
mini_crop_size = 7

device = torch.device('cpu')
if torch.cuda.is_available():
    print('GPU !!!')
    device = torch.device('cuda:0')

net = Net().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

net.load_state_dict(torch.load(model_filename, map_location=device))
net.to(device)
print('Loaded!')

padding = crop_size // 2
pad = ((padding, padding), (padding, padding), (padding, padding))
padded_img = np.pad(imgs[0], pad)
padded_gt = np.pad(gts[0], pad)

full_predict = predict_full(net, padded_img, crop_size=crop_size, mini_crop_size=mini_crop_size,
                            thr=0.5,
                            step_size=7,
                            device=device)


np.save('/predictions/test_predict.npy', full_predict)
print(dc(full_predict, padded_gt))
