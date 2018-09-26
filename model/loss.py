import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns

from model.callback import CallBacks


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

    def record_val_loss(self, new_loss):
        self.val_loss.append(new_loss)

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

    def on_batch_end(self, batch_idx):
        assert len(self.train_loss) == batch_idx + 1, "Don't forget to record loss"
        assert len(self.smooth_train_loss) == batch_idx + 1

    def on_epoch_end(self, epoch_idx):
        epoch_mean = np.mean(self.train_loss[-self.n_batch_per_epoch :])
        self.epoch_train_loss.append(epoch_mean)
        assert len(self.epoch_train_loss) == epoch_idx + 1


class BoundBoxCriterion(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_label, target_label = pred[:, 0], target[:, 0]
        pred_bb, target_bb = pred[:, 1:], target[:, 1:]
        pos_pred_bb = pred_bb[target_label == 1.]
        pos_target_bb = target_bb[target_label == 1.]
        loss_label = F.binary_cross_entropy_with_logits(pred_label, target_label)
        loss_bb = F.smooth_l1_loss(F.sigmoid(pos_pred_bb), pos_target_bb)
        loss_total = loss_label + (self.alpha * loss_bb)
        return loss_total, loss_label, loss_bb
