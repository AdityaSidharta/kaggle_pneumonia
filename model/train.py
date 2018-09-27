import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from model.loss import LossRecorder
from utils.common import get_batch_info


def train_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# TODO remove all the debugging state
def calc_loss(model, criterion, data):
    img, target = data
    prediction = model(img)
    loss, loss_label, loss_bb, pos_pred_bb, pred_label, pos_target_bb, target_label = criterion(prediction, target)
    return loss, loss_label, loss_bb, pos_pred_bb, pred_label, pos_target_bb, target_label


def calc_validation_metric(model, criterion, val_dataloader):
    n_val_obs, val_batch_size, val_batch_per_epoch = get_batch_info(val_dataloader)
    total_val_loss, total_val_loss_label, total_val_loss_bb = np.zeros(val_batch_per_epoch),\
                                                              np.zeros(val_batch_per_epoch),\
                                                              np.zeros(val_batch_per_epoch)
    model = model.eval()
    t = tqdm(enumerate(val_dataloader), total = val_batch_per_epoch)
    for idx, data in t:
        val_loss, val_loss_label, val_loss_bb, pos_pred_bb, pred_label, pos_target_bb, target_label = calc_loss(model, criterion, data)
        val_loss, val_loss_label, val_loss_bb = (
            val_loss.item(),
            val_loss_label.item(),
            val_loss_bb.item(),
        )
        total_val_loss[idx], total_val_loss_label[idx], total_val_loss_bb[idx] = val_loss, val_loss_label, val_loss_bb
        t.set_postfix({
            'loss': val_loss,
            'loss_label': val_loss_label,
            'loss_bb': val_loss_bb
        })
    return total_val_loss.mean(), total_val_loss_label.mean(), total_val_loss_bb.mean()


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


def fit_model(model, n_epoch, dev_dataloader, optimizer, criterion, callbacks = None, val_dataloader = None):
    n_dev_obs, dev_batch_size, dev_batch_per_epoch = get_batch_info(dev_dataloader)
    total_lossr, label_lossr, bb_lossr = (
        LossRecorder(n_epoch, dev_batch_per_epoch),
        LossRecorder(n_epoch, dev_batch_per_epoch),
        LossRecorder(n_epoch, dev_batch_per_epoch),
    )
    lossr_list = [total_lossr, label_lossr, bb_lossr]
    callbacks.extend(lossr_list)
    for cb in callbacks:
        cb.on_train_begin()

    for idx_epoch in tqdm(range(n_epoch), total=n_epoch):

        model = model.train()

        for cb in callbacks:
            cb.on_epoch_begin(idx_epoch)

        t = tqdm(enumerate(dev_dataloader), total=dev_batch_per_epoch)
        for idx_batch, data in t:
            for cb in callbacks:
                cb.on_batch_begin(idx_batch)

            loss, label_loss, bb_loss, pos_pred_bb, pred_label, pos_target_bb, target_label =\
                calc_loss(model, criterion, data)
            train_step(optimizer, loss)
            smooth_loss, smooth_label_loss, smooth_bb_loss = record_loss(
                lossr_list, [loss.item(), label_loss.item(), bb_loss.item()], train=True
            )
            t.set_postfix({'loss': smooth_loss,
                           'label_loss': smooth_label_loss,
                           'bb_loss': smooth_bb_loss})

            for cb in callbacks:
                cb.on_batch_end(idx_batch)
        if val_dataloader is not None:
            val_loss, val_loss_label, val_loss_bb = calc_validation_metric(
                model, criterion, val_dataloader
            )
            record_loss(lossr_list, [val_loss, val_loss_label, val_loss_bb])

        for cb in callbacks:
            cb.on_epoch_end(idx_epoch)

    for cb in callbacks:
        cb.on_train_end()

    return model, callbacks
