import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Net import Net, DANet
from helpers import loader, predict_full, predict_full_da

base_dir = './'

batch_size = 4
epochs = 100000000
crops_per_image = 500
lr = 1e-5
epochs_per_save = 1

crop_size = 65
mini_crop_size = 7


def get_files(file_names):
    imgs = []
    for img_filename in file_names:
        img_path = os.path.join(base_dir, img_filename)
        img = loader(img_path)
        imgs.append(img)
    return imgs


def predict(model_path, df_path, da, out):
    print(f'Labels {df_path}', flush=True)
    print(f'Model {model_path}', flush=True)
    print(f'Domain adaptation {da}', flush=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU !!!')
        device = torch.device('cuda:0')

    model_filename = os.path.basename(model_path)
    if da:
        full_path = out
        net = DANet(1).to(device)
    else:
        full_path = out
        net = Net().to(device)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    files_df = pd.read_csv(df_path)
    files_names = files_df.Filename
    imgs = get_files(files_names)

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    print('Loaded!', flush=True)

    padding = crop_size // 2
    pad = ((padding, padding), (padding, padding), (padding, padding))

    for img, img_filename in tqdm(zip(imgs, files_names)):
        padded_img = np.pad(img, pad)

        if da:
            full_predict = predict_full_da(net, padded_img, crop_size=crop_size, mini_crop_size=mini_crop_size,
                                           step_size=7,
                                           device=device)
        else:
            full_predict = predict_full(net, padded_img, crop_size=crop_size, mini_crop_size=mini_crop_size,
                                        step_size=7,
                                        device=device)
        save_path = os.path.join(full_path, img_filename)
        np.save(save_path, full_predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for create prediction')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--labels', type=str, help='Path to csv')
    parser.add_argument('--out', type=str, help='Out path')
    parser.add_argument('--da',
                        action='store_true',
                        help='This is a boolean flag.',
                        default=False)

    args = parser.parse_args()
    da = args.da
    model = args.model
    labels = args.labels
    out = args.out

    predict(model_path=model, df_path=labels, da=da)
