import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import torch
from utils.callback import CallBacks


class LossRecorder(CallBacks):
    def __init__(self, n_epoch, n_batch_per_epoch, loss_momentum=0.8):
        self.train_loss = []
        self.smooth_train_loss = []
        self.epoch_train_loss = []

        self.val_loss = []

        self.smooth_loss = 0
        self.n_epoch = n_epoch
        self.n_batch_per_epoch = n_batch_per_epoch
        self.loss_momentum = loss_momentum

    def calc_loss(self, cur_loss, new_loss):
        n_loss = len(self.train_loss)
        mom_loss = self.loss_momentum * cur_loss + (1 - self.loss_momentum) * new_loss
        smooth_loss = mom_loss / float(1 - self.loss_momentum ** n_loss)
        return smooth_loss

    def record_train_loss(self, new_loss, return_loss=True):
        self.train_loss.append(new_loss)
        self.smooth_loss = self.calc_loss(self.smooth_loss, new_loss)
        self.smooth_train_loss.append(self.smooth_loss)
        if return_loss:
            return self.smooth_loss

    def record_val_loss(self, new_loss, return_loss=False):
        self.val_loss.append(new_loss)
        if return_loss:
            return new_loss

    def plot_batch_loss(self, smooth=False):
        n_batches = len(self.train_loss)
        if smooth:
            sns.lineplot(x=range(n_batches), y=self.smooth_train_loss)
        else:
            sns.lineplot(x=range(n_batches), y=self.train_loss)

    def plot_epoch_loss(self, val=False):
        n_epoch = len(self.epoch_train_loss)
        if val:
            sns.lineplot(x=range(n_epoch), y=self.epoch_train_loss)
            sns.lineplot(x=range(n_epoch), y=self.val_loss)
        else:
            sns.lineplot(x=range(n_epoch), y=self.epoch_train_loss)

    def on_epoch_end(self, epoch_idx):
        epoch_mean = np.mean(self.train_loss[-self.n_batch_per_epoch :])
        self.epoch_train_loss.append(epoch_mean)
        assert len(self.epoch_train_loss) == epoch_idx + 1


class BoundBoxCriterion(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        assert (self.alpha >= 0.0) and (self.alpha < 1.0)

    def forward(self, pred, target):
        pred_label, target_label = pred[:, 0], target[:, 0]
        pred_bb, target_bb = pred[:, 1:], target[:, 1:]
        pos_pred_bb = pred_bb[target_label == 1.]
        pos_target_bb = target_bb[target_label == 1.]
        loss_label = F.binary_cross_entropy_with_logits(pred_label, target_label)
        if pos_target_bb.shape[0] > 0:
            loss_bb = F.smooth_l1_loss(F.sigmoid(pos_pred_bb), pos_target_bb)
            loss_total = (self.alpha * loss_label) + ((1 - self.alpha) * loss_bb)
        else:
            loss_bb = torch.zeros(1)
            loss_total = loss_label
        return loss_total, loss_label, loss_bb


def calc_loss(model, criterion, data):
    img, target = data
    prediction = model(img)
    loss, loss_label, loss_bb, = criterion(prediction, target)
    return loss, loss_label, loss_bb


def record_loss(lossr_list, loss_list, train=True):
    loss, loss_label, loss_bb = loss_list
    total_lossr, label_lossr, bb_lossr = lossr_list
    if train:
        smooth_loss = total_lossr.record_train_loss(loss, True)
        smooth_label_loss = label_lossr.record_train_loss(loss_label, True)
        smooth_bb_loss = label_lossr.record_train_loss(loss_bb, True)
        return smooth_loss, smooth_label_loss, smooth_bb_loss
    else:
        total_lossr.record_val_loss(loss)
        label_lossr.record_val_loss(loss_label)
        bb_lossr.record_val_loss(loss_bb)
