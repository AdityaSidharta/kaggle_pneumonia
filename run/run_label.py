import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.arch.header import Res50ClassHead
from model.arch.respneunet import ResPneuNet
from model.dataset import LabelDataset
from model.optim import CLR
from model.test import predict_model
from model.train import fit_model
from utils.common import get_batch_info
from utils.data_load import *


def loss_fn(model, criterion, data):
    img, target = data
    prediction = model(img)
    loss = criterion(prediction, target)
    return loss


def metric_fn(model, data):
    img, target = data
    prediction = model(img)
    metric = F.binary_cross_entropy_with_logits(prediction, target)
    return metric


def pred_fn(model, data):
    img = data
    prediction = model(img)
    true_prediction = F.sigmoid(prediction)
    return true_prediction.data.cpu().numpy().reshape(-1).tolist()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_df = pd.read_csv(label_repo)
    train_idx = np.arange(len(label_df))
    dev_idx, val_idx = train_test_split(train_idx, test_size=0.20)
    dev_df = label_df.iloc[dev_idx, :].reset_index(drop=True)
    val_df = label_df.iloc[val_idx, :].reset_index(drop=True)

    label_train_dataset = LabelDataset(True, device, label_df)
    label_dev_dataset = LabelDataset(True, device, dev_df)
    label_val_dataset = LabelDataset(True, device, val_df)
    label_test_dataset = LabelDataset(False, device)
    train_dataloader = DataLoader(label_train_dataset, batch_size=32)
    dev_dataloader = DataLoader(label_dev_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(label_val_dataset, batch_size=32)
    test_dataloader = DataLoader(label_test_dataset, batch_size=32)

    preload_model = torchvision.models.resnet50(pretrained=True).to(device)
    header_model = Res50ClassHead([1000], 0.5).to(device)
    model = ResPneuNet(preload_model, header_model)

    n_epoch = 5
    optimizer = optim.Adam(
        [
            {"params": model.preload_backbone.parameters(), "lr": 0.0001},
            {"params": model.header.parameters(), "lr": 0.001},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )
    criterion = nn.BCEWithLogitsLoss().to(device)
    n_obs, batch_size, n_batch_per_epoch = get_batch_info(dev_dataloader)
    clr = CLR(optimizer, n_epoch, n_batch_per_epoch, 0.1, 1., 0.95, 0.85, 2)
    callbacks = [clr]

    model = fit_model(
        model,
        n_epoch,
        dev_dataloader,
        optimizer,
        criterion,
        loss_fn,
        metric_fn,
        val_dataloader,
        checkpoint=True,
        model_fn="label",
    )

    prediction = predict_model(model, test_dataloader, pred_fn)
    patientid = test_dataloader.dataset.patientId
    pneu_prob = prediction
    label_df = pd.DataFrame({"name": patientid, "prob": pneu_prob})
    label_df.to_csv(label_predict_repo, index=False)


if __name__ == "__main__":
    main()
