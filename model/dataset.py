import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose

from utils.envs import train_images_path, test_images_path


class LabelDataset(Dataset):
    def __init__(self, is_train, device, label_df=None):
        self.is_train = is_train
        if self.is_train:
            self.path = train_images_path
            self.label_df = label_df
            self.patientId = self.patientid_from_df(self.label_df)
            self.id2label = self.get_id2label(self.label_df)
        else:
            self.path = test_images_path
            self.patientId = self.patientid_from_path(self.path)
        self.n_patientId = len(self.patientId)
        self.device = device
        self.tsfm = Compose([ToTensor()])

    @staticmethod
    def patientid_from_path(path):
        return [str(x.split(".")[0]) for x in os.listdir(path) if x.endswith(".png")]

    @staticmethod
    def patientid_from_df(label_df):
        return label_df["name"].values.tolist()

    @staticmethod
    def get_id2label(label_df):
        return label_df.set_index("name")["label"].to_dict()

    def __len__(self):
        return self.n_patientId

    def __getitem__(self, idx):
        filename = self.patientId[idx]
        filepath = os.path.join(self.path, filename + ".png")
        pil_image = Image.open(filepath)
        if self.is_train:
            label = self.id2label[filename]
            return (
                self.tsfm(pil_image).to(self.device),
                torch.Tensor([float(label)]).to(self.device),
            )
        else:
            return self.tsfm(pil_image).to(self.device)


class BBDataset(Dataset):
    def __init__(self, is_train, device, bb_df=None):
        self.image_size = 1024.
        self.is_train = is_train
        if self.is_train:
            self.path = train_images_path
            self.bb_df = self.preprocess_bb(bb_df)
            self.patientId = self.patientid_from_df(self.bb_df)
            self.id2label = self.get_id2label(self.bb_df)
        else:
            self.path = test_images_path
            self.patientId = self.patientid_from_path(self.path)
        self.n_patientId = len(self.patientId)
        self.device = device
        self.tsfm = Compose([ToTensor()])

    def preprocess_bb(self, bb_df):
        bb_df["list_label"] = bb_df["label"].apply(lambda x: x.split(" "))
        bb_df["list_label"] = bb_df["list_label"].apply(lambda x: [float(y) for y in x])
        bb_df["ratio_label"] = bb_df["list_label"].apply(
            lambda x: [y / float(self.image_size) for y in x]
        )
        return bb_df

    @staticmethod
    def patientid_from_path(path):
        return [str(x.split(".")[0]) for x in os.listdir(path) if x.endswith(".png")]

    @staticmethod
    def patientid_from_df(df):
        return df["name"].values.tolist()

    @staticmethod
    def get_id2label(bb_df):
        return bb_df.set_index("name")["ratio_label"].to_dict()

    def __len__(self):
        return self.n_patientId

    def __getitem__(self, idx):
        filename = self.patientId[idx]
        filepath = os.path.join(self.path, filename + ".png")
        pil_image = Image.open(filepath)
        if self.is_train:
            label = self.id2label[filename]
            return (
                self.tsfm(pil_image).to(self.device),
                torch.Tensor(label).to(self.device),
            )
        else:
            return self.tsfm(pil_image).to(self.device)
