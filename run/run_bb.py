import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.arch.header import Res50BBHead
from model.arch.respneunet import ResPneuNet
from model.dataset import BBDataset
from callbacks.optim import CLR
from model.test import predict_model
from model.train import fit_model
from utils.checkpoint import save_checkpoint
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
    metric = F.l1_loss(prediction, target)
    return metric


def pred_fn(model, data):
    img = data
    prediction = model(img)
    prediction_array = prediction.data.cpu().numpy() * 1024.
    return prediction_array.tolist()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bb_df = pd.read_csv(bb_repo)
    train_idx = np.arange(len(bb_df))
    dev_idx, val_idx = train_test_split(train_idx, test_size=0.20)
    dev_df = bb_df.iloc[dev_idx, :].reset_index(drop=True)
    val_df = bb_df.iloc[val_idx, :].reset_index(drop=True)

    bb_train_dataset = BBDataset(True, device, dev_df)
    bb_dev_dataset = BBDataset(True, device, dev_df)
    bb_val_dataset = BBDataset(True, device, val_df)
    bb_test_dataset = BBDataset(False, device)
    train_dataloader = DataLoader(bb_train_dataset, batch_size=32)
    dev_dataloader = DataLoader(bb_dev_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(bb_val_dataset, batch_size=32)
    test_dataloader = DataLoader(bb_test_dataset, batch_size=32)

    preload_model = torchvision.models.resnet50(pretrained=True).to(device)
    header_model = Res50BBHead([1000], 0.5).to(device)
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
    criterion = nn.L1Loss().to(device)

    n_obs, batch_size, n_batch_per_epoch = get_batch_info(dev_dataloader)
    clr = CLR(n_epoch, n_batch_per_epoch, 0.1, 1., 0.95, 0.85, 2)
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
        model_fn="bb",
    )

    prediction = predict_model(model, test_dataloader, pred_fn)
    string_prediction = [
        "{} {} {} {}".format(x[0], x[1], x[2], x[3]) for x in prediction
    ]
    patientid = test_dataloader.dataset.patientId
    pneu_bb = string_prediction
    bb_pred_df = pd.DataFrame({"name": patientid, "label": pneu_bb})
    bb_pred_df.to_csv(bb_predict_repo, index=False)
    save_checkpoint(model, optimizer, fname="bb")


if __name__ == "__main__":
    main()
