import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from tqdm import tqdm

from Dataset import MriDatasetWithDomain
from Net import DANet
from helpers import crop, augmentation, loader_np, loader, get_gt_filename

######### PARAMETERS

# gt_base_path = 'da_gt/'
# img_base_path = 'da_img/'

gt_base_path = '/nmnt/media/home/kechua/CC-359-dataset/Silver-standard-ML/'
img_base_path = '/nmnt/media/home/kechua/CC-359-dataset/originalScaled/'
models_save_path = 'da_v1/models'

labels_path = 'labels.csv'
labeled_domain = 'siemens_15'

# for debugging !
max_files_count = 9999999

alpha = 1
batch_size = 4
epochs = 100000000
crops_per_image = 250
lr = 1e-5
lmbd = 0.1
epochs_per_save = 1

categorical_batch_size = 4 * 5

crop_size = 65
mini_crop_size = 7

######### PARAMETERS
train = True

labels_df = pd.read_csv(labels_path)
labels_df.Domain = pd.Categorical(labels_df.Domain)
labels_df.Domain = labels_df.Domain.cat.codes
labeled_df = labels_df[labels_df['Labeled'] == True]
unlabeled_df = labels_df[labels_df['Labeled'] == False]

labeled_img_filenames = labeled_df.Filename.values[:max_files_count]
labeled_img_cat = labeled_df.Domain.values[:max_files_count]

unlabeled_img_filenames = unlabeled_df.Filename.values[:max_files_count]
unlabeled_img_cat = unlabeled_df.Domain.values[:max_files_count]

labeled_gt_filenames = []
for img_filename in labeled_img_filenames:
    gt_file_name = get_gt_filename(img_filename)
    labeled_gt_filenames.append(gt_file_name)

unlabeled_gt_filenames = []
for img_filename in unlabeled_img_filenames:
    gt_file_name = get_gt_filename(img_filename)
    unlabeled_gt_filenames.append(gt_file_name)

print(f'Labled count {len(labeled_img_filenames)} Unlabeled {len(unlabeled_img_filenames)}', flush=True)

labeled_imgs = []
labeled_gts = []

print('loading labeled images', flush=True)
for img_filename, gt_filename in tqdm(zip(labeled_img_filenames, labeled_gt_filenames)):
    gt_path = os.path.join(gt_base_path, gt_filename)
    img_path = os.path.join(img_base_path, img_filename)

    gt = loader(gt_path)
    img = loader_np(img_path)

    labeled_imgs.append(img)
    labeled_gts.append(gt)

unlabled_imgs = []
unlabled_gts = []

print('loading unlabeled images', flush=True)
for img_filename in tqdm(unlabeled_img_filenames):
    img_path = os.path.join(img_base_path, img_filename)

    img = loader_np(img_path)
    # zeros!
    gt = np.zeros_like(img)

    unlabled_imgs.append(img)
    unlabled_gts.append(gt)


# validation_imgs = []
# validation_gts = []
# for img_filename, gt_filename in zip(unlabled_img_filenames, unlabled_gt_filenames):
#     img_path = os.path.join(img_base_path, img_filename)
#     gt_path = os.path.join(gt_base_path, gt_filename)
#
#     img = loader_np(img_path)
#     gt = loader(gt_path)
#
#     validation_imgs.append(img)
#     validation_gts.append(gt)


def validate(model, dataloader):
    criterion = nn.BCELoss()
    model.eval()
    losses = []
    for (x, y, c) in tqdm(dataloader, desc='Validation'):
        x, y = x.to(device), y.to(device)
        segmentation, _ = model(x)
        loss = criterion(segmentation, y)
        losses.append(
            loss.detach().cpu().item()
        )
    return np.mean(losses)


device = torch.device('cpu')
if torch.cuda.is_available():
    print('GPU !!!', flush=True)
    device = torch.device('cuda:0')

if not os.path.exists(models_save_path):
    os.makedirs(models_save_path)

net = DANet(alpha).to(device)

segmentation_criterion = nn.BCELoss(reduction='none')
classification_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=lr)  # lr = 1e-5 in the original paper

labeled_augmentation_imgs, labeled_augmentation_gts = augmentation(labeled_imgs, labeled_gts)
labeled_augmentation_img_cat = np.repeat(labeled_img_cat, 2)

unlabeled_augmentation_imgs, unlabeled_augmentation_gts = augmentation(unlabled_imgs, unlabled_gts)
unlabeled_augmentation_img_cat = np.repeat(unlabeled_img_cat, 2)

labeled_mri_dataset = MriDatasetWithDomain(labeled_augmentation_imgs, labeled_augmentation_gts,
                                           labeled_augmentation_img_cat, crop_size, mini_crop_size,
                                           crops_per_image, crop_function=crop)

labeled_mri_dataloader = data.DataLoader(labeled_mri_dataset, batch_size=batch_size, shuffle=True)

unlabeled_mri_dataset = MriDatasetWithDomain(unlabeled_augmentation_imgs, unlabeled_augmentation_gts,
                                             unlabeled_augmentation_img_cat, crop_size, mini_crop_size,
                                             crops_per_image, crop_function=crop)

unlabeled_mri_dataloader = data.DataLoader(unlabeled_mri_dataset, batch_size=categorical_batch_size, shuffle=True)

# validate_dataset = MriDatasetWithDomain(validation_imgs, validation_gts,
#                               np.zeros(len(validation_imgs)), crop_size, mini_crop_size,
#                               1000, crop_function=crop)
#
# validate_dataloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

total_segmentation_losses = []
total_classification_losses = []
total_validation_losses = []

net.train()
for epoch in range(epochs):
    segmentation_losses = []
    classification_losses = []
    for (x, y, c), (unl_x, unl_y, unl_c) in tqdm(
            zip(labeled_mri_dataloader, unlabeled_mri_dataloader), total=len(labeled_mri_dataloader),
            desc=f'Epoch {epoch:03}'):
        optimizer.zero_grad()
        labeled_data_size = len(x)

        full_x = torch.cat((x, unl_x)).to(device)
        full_y = torch.cat((y, unl_y)).to(device)
        full_c = torch.cat((c, unl_c)).to(device)

        segmentation, classification = net(full_x)

        segmentation_loss = segmentation_criterion(segmentation, full_y)

        # set loss to there for unlabled_data
        segmentation_loss = segmentation_loss[labeled_data_size:].mean()

        classification_loss = classification_criterion(classification, full_c)

        total_loss = segmentation_loss + classification_loss

        segmentation_losses.append(
            segmentation_loss.detach().cpu().item()
        )

        classification_losses.append(
            classification_loss.detach().cpu().item()
        )

        total_loss.backward()
        optimizer.step()
        del full_x
        del full_y
        del full_c
    if epoch % epochs_per_save == 0:
        torch.save(net.state_dict(), os.path.join(models_save_path, f'model_epoch_{epoch:03}'))

    mean_segmentation_loss = np.mean(segmentation_losses)
    mean_classification_losses = np.mean(classification_losses)

    total_segmentation_losses.append(mean_segmentation_loss)
    total_classification_losses.append(mean_classification_losses)

    # val_loss = validate(net, validate_dataloader)
    # total_validation_losses.append(val_loss)
    # net.eval()
    print(
        f'Epoch {epoch}: Segmetation loss {mean_segmentation_loss:.5f} Class loss {mean_classification_losses}',
        flush=True)
    loss_df = pd.DataFrame({
        'Segmentation loss': total_segmentation_losses,
        'Classification loss': total_classification_losses
    })
    loss_df.to_csv(os.path.join(models_save_path, 'loss_da.csv'), index='Epoch')
