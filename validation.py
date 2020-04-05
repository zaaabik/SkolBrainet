import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import MriDataset
from Net import Net, DANet
from helpers import get_gt_filename, loader, crop

gt_base_path = '/nmnt/media/home/kechua/CC-359-dataset/Silver-standard-ML'
img_base_path = '/nmnt/media/home/kechua/CC-359-dataset/Original'


def get_loss_da(net, dataloader, criterion, device):
    net.eval()
    losses = []
    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        segmentation, _ = net(x)
        loss = criterion(segmentation, y)
        losses.append(
            loss.detach().cpu().item()
        )
    return np.mean(losses)


def get_loss(net, dataloader, criterion, device):
    net.eval()
    losses = []
    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        segmentation = net(x)
        loss = criterion(segmentation, y)
        losses.append(
            loss.detach().cpu().item()
        )
    return np.mean(losses)


def get_model_filename(epoch):
    return f'model_epoch_{epoch:03}'


def validate(model_path, labels_df, da, out, epochs):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU !!!')
        device = torch.device('cuda:0')
    if da:
        print('DA on')
        full_path = os.path.join('validations', 'da', out)
        net = DANet(1).to(device)
    else:
        print('DA off')
        full_path = os.path.join('validations', 'without_da', out)
        net = Net().to(device)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    labels_df = pd.read_csv(labels_df)
    img_filenames = labels_df.Filename
    print(img_filenames)

    ##

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

    print('Files count', len(img_filenames), flush=True)
    ##

    batch_size = 25
    crops_per_image = 500

    crop_size = 65
    mini_crop_size = 7

    print('Loaded!', flush=True)

    mri_dataset = MriDataset(imgs, gts, crop_size, mini_crop_size, crops_per_image,
                             crop_function=crop)
    mri_dataloader = DataLoader(mri_dataset, batch_size)
    criterion = nn.BCELoss()

    total_epochs = []
    total_loss = []
    for epoch in tqdm(epochs, desc='Epoch'):
        model_filename = get_model_filename(epoch)
        net_path = os.path.join(model_path, model_filename)
        net.load_state_dict(torch.load(net_path, map_location=device))
        cur_loss = get_loss(net, mri_dataloader, criterion, device)

        total_loss.append(cur_loss)
        total_epochs.append(epoch)

        loss_df = pd.DataFrame({
            'Epoch': total_epochs,
            'Loss': total_loss
        })
        save_loss_path = os.path.join(full_path, 'loss.csv')
        loss_df.to_csv(save_loss_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for create prediction')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--out', type=str, help='Path to model')
    parser.add_argument('--labels', type=str, help='Path to csv')
    parser.add_argument('--da',
                        action='store_true',
                        help='This is a boolean flag.',
                        default=False)
    parser.add_argument('--epochs', nargs='+', type=int)

    args = parser.parse_args()
    da = args.da
    model = args.model
    labels = args.labels
    out = args.out
    epochs = args.epochs
    print(epochs, flush=True)
    print(f'Da {da}')

    validate(model, labels, da, out, epochs)
