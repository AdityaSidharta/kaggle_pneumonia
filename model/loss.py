import numpy as np
import seaborn as sns

from utils.logger import logger
from utils.callback import CallBacks

# TODO perform action on_batch_end, on_epoch_end, on_train_end. Extend this on metric
class LossRecorder(CallBacks):
    def __init__(self, n_epoch, n_batch_per_epoch, loss_momentum=0.8, is_val=True):
        self.batch_list = []
        self.smooth_batch_list = []

        self.epoch_train_list = []
        self.epoch_val_list = []
        self.is_val = is_val
        self.smooth_loss = 0
        self.n_epoch = n_epoch
        self.n_batch_per_epoch = n_batch_per_epoch
        self.loss_momentum = loss_momentum

    def calc_loss(self, cur_loss, new_loss):
        n_loss = len(self.batch_list)
        mom_loss = self.loss_momentum * cur_loss + (1 - self.loss_momentum) * new_loss
        smooth_loss = mom_loss / float(1 - self.loss_momentum ** n_loss)
        return smooth_loss

    def record_train_loss(self, new_loss):
        self.batch_list.append(new_loss)
        self.smooth_loss = self.calc_loss(self.smooth_loss, new_loss)
        self.smooth_batch_list.append(self.smooth_loss)

    def plot_batch_loss(self, smooth=False):
        n_batches = len(self.batch_list)
        if smooth:
            sns.lineplot(x=range(n_batches), y=self.smooth_batch_list)
        else:
            sns.lineplot(x=range(n_batches), y=self.batch_list)

    def plot_epoch_loss(self, val=False):
        n_epoch = len(self.epoch_train_list)
        if val:
            sns.lineplot(x=range(n_epoch), y=self.epoch_train_list)
            sns.lineplot(x=range(n_epoch), y=self.epoch_val_list)
        else:
            sns.lineplot(x=range(n_epoch), y=self.epoch_train_list)

    def get_epoch_train_mean(self):
        epoch_mean = np.mean(self.batch_list[-self.n_batch_per_epoch :])
        return epoch_mean

    def on_train_end(self, model, optimizer):
        self.plot_batch_loss(smooth=True)
        self.plot_epoch_loss(val=self.is_val)

    def on_epoch_end(self, epoch_idx, model, optimizer, val_loss=None, val_metric=None):
        train_epoch = self.get_epoch_train_mean()
        self.epoch_train_list.append(train_epoch)
        if (val_loss is None) or (val_metric is None):
            pass
        # TODO continue from here

    def on_epoch_end(self, epoch_idx):
        epoch_mean = np.mean(self.batch_list[-self.n_batch_per_epoch :])
        self.epoch_train_list.append(epoch_mean)
        assert len(self.epoch_train_list) == epoch_idx + 1
