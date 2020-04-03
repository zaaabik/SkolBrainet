import numpy as np
import torch
from torch.utils import data


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


class MriDatasetWithDomain(data.Dataset):
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

        return torch.Tensor(x[None, :, :, :]), torch.Tensor(y[None, :, :, :]), np.array(cat, dtype=np.int64)
