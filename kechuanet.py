# -*- coding: utf-8 -*-
"""kechuaNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K-fmbQrNh7G7zjdd2w5Q7UERxpVSDsWz
"""

# !unzip kichua.zip
# !unzip -o kichua_validation.zip;
# !pip install medpy

import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils import data
from tqdm import tqdm

######### PARAMETERS 

gt_base_path = 'Silver-standard-ML'
img_base_path = 'Original'

######### PARAMETERS

train = True

raw_img_filenames = os.listdir(img_base_path)

max_idx = 169
min_idx = 120
index_reg_exp = 'CC(.*?)_'


def filter_function(file_name):
    index = re.findall(index_reg_exp, file_name)[0]
    index = int(index)
    if min_idx <= index <= max_idx:
        return True
    return False


img_filenames = []
for file_name in raw_img_filenames:
    if filter_function(file_name):
        img_filenames.append(file_name)

gt_filenames = []
for img_filename in img_filenames:
    comma_idx = img_filename.find('.')
    gt_file_name = img_filename[:comma_idx] + '_ss' + img_filename[comma_idx:]
    gt_filenames.append(gt_file_name)

print('Files count', len(img_filenames))


def loader(path):
    data = nib.load(path)
    img = data.get_fdata()
    return img


imgs = []
gts = []

for img_filename, gt_filename in zip(img_filenames, gt_filenames):
    gt_path = os.path.join(gt_base_path, gt_filename)
    img_path = os.path.join(img_base_path, img_filename)

    gt = loader(gt_path)
    img = loader(img_path)

    imgs.append(img)
    gts.append(gt)


def crop(img, gt, voxes_size, mini_voxel_size, start_coordinates):
    img_size = np.array(img.shape)
    gt_size = np.array(gt.shape)
    start_coordinates = np.array(start_coordinates)

    end_coordinates = start_coordinates + voxes_size
    if not np.all(end_coordinates < img_size):
        raise AttributeError('Crop is outsize of image')

    cropped_img = img[
                  start_coordinates[0]:end_coordinates[0],
                  start_coordinates[1]:end_coordinates[1],
                  start_coordinates[2]:end_coordinates[2]
                  ]

    assert np.all(np.array(cropped_img.shape) == voxes_size)

    cropped_gt_start = (end_coordinates + start_coordinates) // 2 - mini_voxel_size // 2

    cropped_gt_end = cropped_gt_start + mini_voxel_size

    cropped_gt = gt[
                 cropped_gt_start[0]:cropped_gt_end[0],
                 cropped_gt_start[1]:cropped_gt_end[1],
                 cropped_gt_start[2]:cropped_gt_end[2]
                 ]

    assert np.all(np.array(cropped_gt.shape) == mini_voxel_size)

    return cropped_img, cropped_gt


def augmentation(imgs, gts):
    augmentation_imgs = []
    augmentation_gts = []

    for idx in range(len(imgs)):
        augmentation_imgs.append(
            imgs[idx][:, :, ::-1].copy()
        )

        augmentation_gts.append(
            gts[idx][:, :, ::-1].copy()
        )

    return imgs + augmentation_imgs, gts + augmentation_gts


class MriDataset(data.Dataset):
    def __init__(self, X, y, crop_size, mini_crop_size, crops_per_image, crop_function):
        super(MriDataset)
        self.X = [x / x.max() for x in X]
        self.y = y
        self.crop_size = crop_size
        self.mini_crop_size = mini_crop_size
        self.crops_per_image = crops_per_image
        self.crops_per_image = crops_per_image
        self.crop_function = crop_function
        self.crops, self.img_idxs = self.create_crops()

        permutation = np.random.permutation(len(self.crops))
        self.crops = np.array(self.crops)[permutation]
        self.img_idxs = np.array(self.img_idxs)[permutation]

    def create_crops(self):
        crops = []
        img_idxs = []

        for i in range(len(self.X)):
            x_shape = np.array(self.X[i].shape)
            y_shape = np.array(self.y[i].shape)

            sub_crops = []
            for j in range(self.crops_per_image):
                x_max = x_shape[0] - self.crop_size
                y_max = x_shape[1] - self.crop_size
                z_max = x_shape[2] - self.crop_size

                x = np.random.randint(0, x_max)
                y = np.random.randint(0, y_max)
                z = np.random.randint(0, z_max)

                crops.append([x, y, z])
                img_idxs.append(i)

        return crops, img_idxs

    def __len__(self):
        return len(self.X) * self.crops_per_image

    def __getitem__(self, idx):
        crop = self.crops[idx]
        img = self.X[self.img_idxs[idx]]
        gt = self.y[self.img_idxs[idx]]
        x, y = self.crop_function(img, gt, self.crop_size, self.mini_crop_size, crop)

        return torch.Tensor(x[None, :, :, :]), torch.Tensor(y[None, :, :, :])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = self._make_conv_layer(out_channels=16, k_size=4)
        self.conv_layer2 = self._make_conv_layer(in_channels=16, out_channels=24)
        self.conv_layer3 = self._make_conv_layer(in_channels=24, out_channels=28)
        self.conv_layer4 = self._make_conv_layer(in_channels=28, out_channels=34)
        self.conv_layer5 = self._make_conv_layer(in_channels=34, out_channels=42)
        self.conv_layer6 = self._make_conv_layer(in_channels=42, out_channels=50)
        self.conv_layer7 = self._make_conv_layer(in_channels=50, out_channels=50)
        self.final_layer = self._make_conv_layer(in_channels=50, out_channels=1, activation=False)

    def forward(self, x):

        x = self.conv_layer1(x)
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(x, (2, 2, 2))
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)

        return x

    @staticmethod
    def _make_conv_layer(out_channels, in_channels=1, k_size=5, activation=True):
        if activation:
            conv_layer = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k_size, k_size, k_size),
                          padding=0),
                nn.LeakyReLU(),  # (!)Think of padding(!)
            )
        else:
            conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(k_size, k_size, k_size), padding=2)

        return conv_layer


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

if not os.path.exists('models'):
    os.mkdir('models')

net = Net().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)  # lr = 1e-5 in the original paper

augmentation_imgs, augmentation_gts = augmentation(imgs, gts)
mri_dataset = MriDataset(augmentation_imgs, augmentation_gts, crop_size, mini_crop_size, crops_per_image,
                         crop_function=crop)
mri_dataloader = data.DataLoader(mri_dataset, batch_size)
assert len(mri_dataset) == len(augmentation_imgs) * crops_per_image
print(len(mri_dataset))

if train:
    total_losses = []

    for epoch in range(epochs):
        losses = []
        for x, y in tqdm(mri_dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            losses.append(loss.detach().cpu().item())

            loss.backward()
            optimizer.step()
            del x
            del y
        if epoch % epochs_per_save == 0:
            print(epoch)
            torch.save(net.state_dict(), os.path.join('models', f'model_epoch_{epoch:03}'))

        mean_loss = np.mean(losses)
        total_losses.append(mean_loss)

        pd.DataFrame(total_losses).to_csv('loss.csv', index='Epoch')
        print('iter', epoch, ':', mean_loss)
else:
    net.load_state_dict(torch.load("model_epoch_100"))
    net.to(device)
    print('Loaded!')

# n = 1
# img = augmentation_imgs[n]
# gt = augmentation_gts[n]
# img = img / img.max()
# diff = crop_size // 2 - mini_crop_size // 2
# layer = 72
# pred_layer = layer - 32 + mini_crop_size // 2
# _, max_y, max_z = img.shape
# preds = np.zeros_like(img[0])
# coef = np.zeros_like(img[0])
# kernel = gkern(mini_crop_size, 5)
# batch = []
# for i in tqdm(range(0, max_z - crop_size, 3)):
# for j in range(0, max_y - crop_size, 3):
# pred_j = j + diff
# pred_i = i + diff
# vis_x, vis_y = crop(img, gt, crop_size, mini_crop_size, (pred_layer, j, i))
# pred_y = vis_y
# pred_y = net(torch.Tensor(vis_x[None,None,:,:,:]).to(device))[0,0].data.cpu().numpy()
# preds[pred_j:pred_j + mini_crop_size, pred_i: pred_i + mini_crop_size] += pred_y[0] * 1

#       coef[pred_j:pred_j + mini_crop_size, pred_i: pred_i + mini_crop_size] += 1

# coef[coef == 0] = 1
# preds = np.array(preds)

# fig, (pred_axis, gt_axis) = plt.subplots(1, 2, figsize=(15,7))
# pred_img = (preds / coef) > 0.5
# pred_axis.imshow(pred_img)
# pred_axis.set_title('Prediction')

# gt_axis.imshow(gt[layer])
# gt_axis.set_title('Target');
# (preds != gt[layer]).sum()
# dc(pred_img, gt[layer]), np.allclose(pred_img, gt[layer])
