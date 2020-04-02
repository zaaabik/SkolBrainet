import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from tqdm import tqdm

######### PARAMETERS
from Net import DANet

gt_base_path = 'da_gt/'
img_base_path = 'da_img/'
labels_path = 'labels.csv'
labled_domain = 'siemens_15'

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

lables_df = pd.read_csv(labels_path)
lables_df.Domain = pd.Categorical(lables_df.Domain)
lables_df.Domain = lables_df.Domain.cat.codes
labled_df = lables_df[lables_df['Labeled'] == True]
unlabled_df = lables_df[lables_df['Labeled'] == False]

labled_img_filenames = labled_df.Filename.values
labled_img_cat = labled_df.Domain.values

unlabled_img_filenames = unlabled_df.Filename.values
unlabled_img_cat = unlabled_df.Domain.values

labled_gt_filenames = []
for img_filename in labled_img_filenames:
    comma_idx = img_filename.find('.')
    gt_file_name = img_filename[:comma_idx] + '_ss' + img_filename[comma_idx:]
    labled_gt_filenames.append(gt_file_name)

unlabled_gt_filenames = []
for img_filename in unlabled_img_filenames:
    comma_idx = img_filename.find('.')
    gt_file_name = img_filename[:comma_idx] + '_ss' + img_filename[comma_idx:]
    unlabled_gt_filenames.append(gt_file_name)

print(f'Labled count {len(labled_img_filenames)} Unlabled {len(unlabled_img_filenames)}', flush=True)


def loader(path):
    data = nib.load(path)
    img = data.get_fdata()
    return img


labled_imgs = []
labled_gts = []

for img_filename, gt_filename in zip(labled_img_filenames, labled_gt_filenames):
    gt_path = os.path.join(gt_base_path, gt_filename)
    img_path = os.path.join(img_base_path, img_filename)

    gt = loader(gt_path)
    img = loader(img_path)

    labled_imgs.append(img)
    labled_gts.append(gt)

unlabled_imgs = []
unlabled_gts = []

for img_filename in unlabled_img_filenames:
    img_path = os.path.join(img_base_path, img_filename)

    img = loader(img_path)
    # zeros!
    gt = np.zeros_like(img)

    unlabled_imgs.append(img)
    unlabled_gts.append(gt)


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

    cropped_gt_start = (end_coordinates + start_coordinates) // 2  # - mini_voxel_size // 2

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
    def __init__(self, X, y, cat, crop_size, mini_crop_size, crops_per_image, crop_function):
        super(MriDataset)
        self.X = [x / x.max() for x in X]
        self.y = y
        self.cat = cat
        self.crop_size = crop_size
        self.mini_crop_size = mini_crop_size
        self.crops_per_image = crops_per_image
        self.crops_per_image = crops_per_image
        self.crop_function = crop_function
        self.crops, self.img_idxs, self.img_cats = self.create_crops()


def create_crops(self):
    crops = []
    img_idxs = []
    img_cats = []

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
            img_cats.append(self.cat[i])

    return crops, img_idxs, img_cats


def __len__(self):
    return len(self.X) * self.crops_per_image


def __getitem__(self, idx):
    crop = self.crops[idx]
    img = self.X[self.img_idxs[idx]]
    gt = self.y[self.img_idxs[idx]]
    cat = self.img_cats[idx]
    x, y = self.crop_function(img, gt, self.crop_size, self.mini_crop_size, crop)

    return torch.Tensor(x[None, :, :, :]), torch.Tensor(y[None, :, :, :]), np.array(cat, dtype=np.long)


validation_imgs = []
validation_gts = []
for img_filename, gt_filename in zip(unlabled_img_filenames, unlabled_gt_filenames):
    img_path = os.path.join(img_base_path, img_filename)
    gt_path = os.path.join(gt_base_path, gt_filename)

    img = loader(img_path)
    gt = loader(gt_path)

    validation_imgs.append(img)
    validation_gts.append(gt)


def validate(model, dataloader):
    criterion = nn.BCELoss()
    model.eval()
    losses = []
    metrics = []
    for (x, y, c) in tqdm(dataloader, desc='Validation'):
        x, y = x.to(device), y.to(device)
        segmentation, _ = model(x)
        loss = criterion(segmentation, y)
        losses.append(
            loss.detach().cpu().item()
        )
    return np.mean(losses)


def learn_nn ():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU !!!')
        device = torch.device('cuda:0')

    if not os.path.exists('models'):
        os.mkdir('models')

    net = DANet(alpha).to(device)

    segmentation_criterion = nn.BCELoss(reduction='none')
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)  # lr = 1e-5 in the original paper

    labled_augmentation_imgs, labled_augmentation_gts = augmentation(labled_imgs, labled_gts)
    labled_augmentation_img_cat = np.repeat(labled_img_cat, 2)

    unlabled_augmentation_imgs, unlabled_augmentation_gts = augmentation(unlabled_imgs, unlabled_gts)
    unlabled_augmentation_img_cat = np.repeat(unlabled_img_cat, 2)

    labled_mri_dataset = MriDataset(labled_augmentation_imgs, labled_augmentation_gts,
                                    labled_augmentation_img_cat, crop_size, mini_crop_size,
                                    crops_per_image, crop_function=crop)

    labled_mri_dataloader = data.DataLoader(labled_mri_dataset, batch_size=batch_size, shuffle=True)

    unlabeled_mri_dataset = MriDataset(unlabled_augmentation_imgs, unlabled_augmentation_gts,
                                       unlabled_augmentation_img_cat, crop_size, mini_crop_size,
                                       crops_per_image, crop_function=crop)

    unlabeled_mri_dataloader = data.DataLoader(unlabeled_mri_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = MriDataset(validation_imgs, validation_gts,
                                  np.zeros(len(validation_imgs)), crop_size, mini_crop_size,
                                  1000, crop_function=crop)

    validate_dataloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    total_segmentation_losses = []
    total_classification_losses = []
    total_validation_losses = []
    for epoch in range(epochs):
        segmentation_losses = []
        classification_losses = []
        net.train()
        for (x, y, c), (unl_x, unl_y, unl_c) in tqdm(
                zip(labled_mri_dataloader, unlabeled_mri_dataloader), total=len(labled_mri_dataloader),
                desc=f'Epoch {epoch:03}'):
            optimizer.zero_grad()
            labled_data_size = len(x)

            full_x = torch.cat((x, unl_x)).to(device)
            full_y = torch.cat((y, unl_y)).to(device)
            full_c = torch.cat((c, unl_c)).to(device)

            segmentation, classification = net(full_x)

            segmentation_loss = segmentation_criterion(segmentation, full_y)

            # set loss to there for unlabled_data
            segmentation_loss[labled_data_size:] = 0
            segmentation_loss = segmentation_loss.mean()

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
            torch.save(net.state_dict(), os.path.join('models', f'model_epoch_{epoch:03}'))

        mean_segmentation_loss = np.mean(segmentation_losses)
        mean_classification_losses = np.mean(classification_losses)

        total_segmentation_losses.append(mean_segmentation_loss)
        total_classification_losses.append(mean_classification_losses)

        vall_loss = validate(net, validate_dataloader)
        total_validation_losses.append(vall_loss)
        net.eval()
        print(
            f'Epoch {epoch}: Segmetation loss {mean_segmentation_loss:.5f} Class loss {mean_classification_losses} Val loss : {vall_loss}',
            flush=True)
        loss_df = pd.DataFrame({
            'Segmetation loss': total_segmentation_losses,
            'Classification loss': total_classification_losses,
            'Val loss': total_validation_losses
        })
        loss_df.to_csv('loss.csv', index='Epoch')
