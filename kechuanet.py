import os
import re

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from tqdm import tqdm

from Dataset import MriDataset
from Net import Net
from helpers import loader, augmentation, crop

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

imgs = []
gts = []

for img_filename, gt_filename in zip(img_filenames, gt_filenames):
    gt_path = os.path.join(gt_base_path, gt_filename)
    img_path = os.path.join(img_base_path, img_filename)

    gt = loader(gt_path)
    img = loader(img_path)

    imgs.append(img)
    gts.append(gt)

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
