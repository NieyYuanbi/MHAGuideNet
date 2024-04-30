import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
import nibabel
import pandas as pd
import itertools

class AdniDataSet(Dataset):

    def __init__(self, data_path, img_path, sets):
        self.data_path = data_path
        self.img_path = img_path
        self.subjects = pd.read_csv(data_path)
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, x, y, z])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.subjects['index'])

    def __getitem__(self, idx):

        img = nibabel.load(self.img_path + str(self.subjects['id'][
                                                   idx]))

        assert img is not None

        group = self.subjects['class'][idx]
        assert group is not None
        if group == 'CN':
            label = 0
        else:
            label = 1

        img_array = self.__training_data_process__(img)
        img_array = self.__nii2tensorarray__(img_array)
        # print(img_array.shape)

        patches1 = np.zeros((40, 1, img_array.shape[2], img_array.shape[3]),
                            dtype='float64')
        patches2 = np.zeros((40, 1, img_array.shape[2], img_array.shape[3]),
                            dtype='float64')
        patches3 = np.zeros((40, 1, img_array.shape[2], img_array.shape[3]),
                            dtype='float64')

        for i in range(0, 40):
            patches1[i, ...] = img_array[:, i + 25, :, :]
            patches2[i, ...] = img_array[:, :, i + 25, :]
            patches3[i, ...] = img_array[:, :, :, i + 25]


        return img_array, patches1, patches2, patches3, label

    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()
        data = self.__resize_data__(data)

        return data

    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=2)

        return data


