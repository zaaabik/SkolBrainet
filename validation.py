import os
import nibabel as nib
import numpy as np
import torch
from medpy.metric import dc
from torch import nn, optim

from Net import Net
from helpers import predict_full, get_gt_filename, loader

gt_base_path = '/nmnt/media/home/kechua/CC-359-dataset/Silver-standard-ML'
img_base_path = '/nmnt/media/home/kechua/CC-359-dataset/Original'
model_filename = os.path.join('models/v2', 'model_epoch_093')

######### PARAMETERS

train = True

raw_img_filenames = os.listdir(img_base_path)

max_idx = 169
min_idx = 120
index_reg_exp = 'CC(.*?)_'

img_filenames = []
for file_name in raw_img_filenames:

    # here I am testing that step along each axis (x,y,z) is nearly equal to 1. Otherwise
    # prediction is not viable unless the scan was preliminary rescaled (!!!To Be Done!!!)
    img_path = os.path.join(img_base_path, file_name)
    img = nib.load(img_path)
    tolL = 0.999
    tolH = 1.001
    x_step = img.header['pixdim'][1]
    y_step = img.header['pixdim'][2]
    z_step = img.header['pixdim'][3]
    if tolL < x < tolH and tolL < y < tolH and tolL < z < tolH:
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

for i in range(len(imgs)):
    print('Working with scan' + img_filenames[i])

    padded_img = np.pad(imgs[i], pad)
    padded_gt = np.pad(gts[i], pad)

    full_predict = predict_full(net, padded_img, crop_size=crop_size, mini_crop_size=mini_crop_size,
                                thr=0.5,
                                step_size=7,
                                device=device)


    np.save('predictions/test_predict_' + img_filenames[i][:-7] + '.npy', full_predict)
    print(dc(full_predict, padded_gt))
