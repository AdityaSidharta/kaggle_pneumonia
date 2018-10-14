import os

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage, Grayscale
from utils.transform import normalize_bb


class LabelDataset(Dataset):
    def __init__(self, train_path, device):
        self.train_path = train_path


class Single_Pneumonia_Dataset_Test(Dataset):
    def __init__(self, dcm_path, device):
        self.dcm_path = dcm_path
        self.patientId = self.get_patientid(dcm_path)
        self.n_obs = len(self.patientId)
        self.device = device
        self.tsfm = Compose([ToPILImage(), Resize(224), Grayscale(3), ToTensor()])

    @staticmethod
    def get_patientid(dcm_path):
        return [
            str(x.split(".")[0]) for x in os.listdir(dcm_path) if x.endswith(".dcm")
        ]

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        filename = self.patientId[idx]
        file_path = os.path.join(self.dcm_path, filename + ".dcm")
        img_array = pydicom.read_file(file_path).pixel_array
        img_array = np.expand_dims(img_array, -1)
        return self.tsfm(img_array).to(self.device)


class Single_Pneumonia_Dataset(Dataset):
    def __init__(self, single_label_df, dcm_path, device):
        self.single_label_df = single_label_df
        self.patientId = single_label_df.patientId.values
        self.x = single_label_df.x.values
        self.y = single_label_df.y.values
        self.width = single_label_df.width.values
        self.height = single_label_df.height.values
        self.Target = single_label_df.Target.values
        self.area = single_label_df.area.values
        self.dcm_path = dcm_path
        self.n_obs = len(single_label_df)
        self.device = device
        self.tsfm = Compose([ToPILImage(), Resize(224), Grayscale(3), ToTensor()])

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        # TODO what if the bounding box exceeds the image. assert x_min + height smaller than 1, y_min + width smaller than 1
        filename = self.patientId[idx]
        file_path = os.path.join(self.dcm_path, filename + ".dcm")
        img_array = pydicom.read_file(file_path).pixel_array
        img_h, img_w = img_array.shape[0], img_array.shape[1]
        target = self.Target[idx]
        x_min, y_min, width, height = (
            self.x[idx],
            self.y[idx],
            self.width[idx],
            self.height[idx],
        )
        x_min, y_min, width, height = normalize_bb(
            img_w, img_h, x_min, y_min, width, height
        )
        img_array = np.expand_dims(img_array, -1)
        return (
            self.tsfm(img_array).to(self.device),
            torch.from_numpy(np.array([target, x_min, y_min, width, height]))
            .float()
            .to(self.device),
        )
