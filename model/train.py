from tqdm import tqdm_notebook as tqdm
from model.loss import LossRecorder, calc_loss, record_loss
from model.validation import validate_model
from utils.common import get_batch_info
from utils.logger import logger


def train_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def fit_model(
    model,
    n_epoch,
    dev_dataloader,
    optimizer,
    criterion,
    loss_fn,
    metric_fn,
    val_dataloader=None,
):
    n_dev_obs, dev_batch_size, dev_batch_per_epoch = get_batch_info(dev_dataloader)
    for idx_epoch in tqdm(range(n_epoch), total=n_epoch):
        model = model.train()
        t = tqdm(enumerate(dev_dataloader), total=dev_batch_per_epoch)
        for idx_batch, data in t:
            loss = loss_fn(model, criterion, data)
            train_step(optimizer, loss)
        train_metric = validate_model(model, metric_fn, dev_dataloader)
        print("train_loss : {}".format(train_metric))
        if val_dataloader is not None:
            val_metric = validate_model(model, metric_fn, val_dataloader)
            print("val_loss : {}".format(val_metric))
    return model


def fit_model_full(
    model,
    n_epoch,
    dev_dataloader,
    optimizer,
    criterion,
    callbacks=[],
    val_dataloader=None,
):
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

            loss, label_loss, bb_loss = calc_loss(model, criterion, data)
            train_step(optimizer, loss)
            smooth_loss, smooth_label_loss, smooth_bb_loss = record_loss(
                lossr_list, [loss.item(), label_loss.item(), bb_loss.item()], train=True
            )
            t.set_postfix(
                {
                    "loss": smooth_loss,
                    "label_loss": smooth_label_loss,
                    "bb_loss": smooth_bb_loss,
                }
            )

            for cb in callbacks:
                cb.on_batch_end(idx_batch)
        if val_dataloader is not None:
            val_loss, val_loss_label, val_loss_bb = validate_model(
                model, criterion, val_dataloader
            )
            record_loss(lossr_list, [val_loss, val_loss_label, val_loss_bb])

        for cb in callbacks:
            cb.on_epoch_end(idx_epoch)

    for cb in callbacks:
        cb.on_train_end()

    return model, callbacks
