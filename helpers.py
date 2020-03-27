import nibabel as nib
import numpy as np


def loader(path):
    data = nib.load(path)
    img = data.get_fdata()
    return img


def crop(img, gt, voxes_size, mini_voxel_size, start_coordinates):
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
