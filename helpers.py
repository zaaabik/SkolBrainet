import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def loader(path):
    img = np.load(path)
    return img


def loader_np(path):
    img = np.load(path)
    return img


def get_gt_filename(img_filename):
    comma_idx = img_filename.find('.')
    gt_file_name = img_filename[:comma_idx] + '_ss' + img_filename[comma_idx:]
    return gt_file_name


def crop(img, gt, voxes_size, mini_voxel_size, start_coordinates, fake=False):
    img_size = np.array(img.shape)
    start_coordinates = np.array(start_coordinates)

    end_coordinates = start_coordinates + voxes_size
    if not np.all(end_coordinates < img_size):
        raise AttributeError('Crop is outsize of image')

    cropped_img = img[
                  start_coordinates[0]:end_coordinates[0],
                  start_coordinates[1]:end_coordinates[1],
                  start_coordinates[2]:end_coordinates[2]
                  ]
    if fake:
        return cropped_img, 9

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


def augmentation(imgs, gts, fake=False):
    augmentation_imgs = []
    augmentation_gts = []

    for idx in range(len(imgs)):
        augmentation_imgs.append(
            imgs[idx][:, :, ::-1].copy()
        )
        if fake:
            augmentation_gts.append(
                gts[idx]
            )
        else:
            augmentation_gts.append(
                gts[idx][:, :, ::-1].copy()
            )

    return imgs + augmentation_imgs, gts + augmentation_gts


def predict_full(net, img, crop_size=65, mini_crop_size=7,
                 device=torch.device('cpu'), step_size=7, batch_size=25):
    img = img / img.max()
    diff = crop_size // 2 - mini_crop_size // 2

    max_x, max_y, max_z = img.shape
    preds = np.zeros_like(img)
    coef = np.zeros_like(img)

    z_range = range(0, max_z - crop_size, step_size)
    y_range = range(0, max_y - crop_size, step_size)
    x_range = tqdm(range(0, max_x - crop_size, step_size))

    for x in x_range:
        pred_x = x + diff
        crops = []
        for y in y_range:
            pred_y = y + diff
            for z in z_range:
                pred_z = z + diff
                vis_x, vis_y = crop(img, img, crop_size, mini_crop_size, (x, y, z))
                crops.append(vis_x)

        crops = np.array(crops)
        crops = crops[:, None, :, :, :]
        crops = torch.FloatTensor(crops)
        ds = TensorDataset(crops)
        dl = DataLoader(ds, batch_size=batch_size)
        outputs = []
        for x in dl:
            x = x[0]
            x = x.to(device)
            output = net(x)
            del x
            output = output[:, 0].data.detach().cpu().numpy()
            outputs.append(output)
        del dl
        del ds

        outputs = np.concatenate(outputs, axis=0)
        assert outputs[0].shape == (mini_crop_size, mini_crop_size, mini_crop_size)

        i = 0
        for y in y_range:
            pred_y = y + diff
            for z in z_range:
                pred_z = z + diff
                preds[pred_x:pred_x + mini_crop_size, pred_y:pred_y + mini_crop_size,
                pred_z:pred_z + mini_crop_size] += outputs[i]
                coef[pred_x:pred_x + mini_crop_size, pred_y:pred_y + mini_crop_size,
                pred_z: pred_z + mini_crop_size] += 1
                i += 1

    coef[coef == 0] = 1
    res = (preds / coef)
    return res


def predict_full_da(net, img, crop_size=65, mini_crop_size=7,
                    device=torch.device('cpu'), step_size=7, batch_size=25):
    img = img / img.max()
    diff = crop_size // 2 - mini_crop_size // 2

    max_x, max_y, max_z = img.shape
    preds = np.zeros_like(img)
    coef = np.zeros_like(img)

    z_range = range(0, max_z - crop_size, step_size)
    y_range = range(0, max_y - crop_size, step_size)
    x_range = tqdm(range(0, max_x - crop_size, step_size))

    for x in x_range:
        pred_x = x + diff
        crops = []
        for y in y_range:
            for z in z_range:
                vis_x, vis_y = crop(img, img, crop_size, mini_crop_size, (x, y, z))
                crops.append(vis_x)

        crops = np.array(crops)
        crops = crops[:, None, :, :, :]
        crops = torch.FloatTensor(crops)
        ds = TensorDataset(crops)
        dl = DataLoader(ds, batch_size=batch_size)
        outputs = []
        for x in dl:
            x = x[0]
            x = x.to(device)
            output, _ = net(x)
            del x
            output = output[:, 0].data.detach().cpu().numpy()
            outputs.append(output)
        del dl
        del ds

        outputs = np.concatenate(outputs, axis=0)
        assert outputs[0].shape == (mini_crop_size, mini_crop_size, mini_crop_size)

        i = 0
        for y in y_range:
            pred_y = y + diff
            for z in z_range:
                pred_z = z + diff
                preds[pred_x:pred_x + mini_crop_size, pred_y:pred_y + mini_crop_size,
                pred_z:pred_z + mini_crop_size] += outputs[i]
                coef[pred_x:pred_x + mini_crop_size, pred_y:pred_y + mini_crop_size,
                pred_z: pred_z + mini_crop_size] += 1
                i += 1

    coef[coef == 0] = 1
    res = (preds / coef)
    return res
