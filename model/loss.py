import numpy as np
import seaborn as sns
from utils.callback import CallBacks
from utils.logger import debug_pred_target


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


def calc_loss(model, criterion, data):
    img, target = data
    prediction = model(img)
    debug_pred_target(prediction, target)
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
