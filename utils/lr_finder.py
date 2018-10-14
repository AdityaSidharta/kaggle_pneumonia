import copy

import seaborn as sns
from torch import optim as optim

from dev.train import fit_model_full
from model.optim import LR_Finder
from utils.common import get_batch_info


def lr_plot(lr_finder_cb, loss_rec_cb):
    sns.lineplot(x=lr_finder_cb.lr_schedule, y=loss_rec_cb.smooth_train_loss)


def lr_find(model, dataloader, criterion, min_lr=1e-8, max_lr=10.0):
    clone_model = copy.deepcopy(model)
    optimizer = optim.SGD(clone_model.parameters(), lr=min_lr)
    n_epoch = 1
    n_obs, batch_size, batch_per_epoch = get_batch_info(dataloader)
    lr_finder = LR_Finder(optimizer, n_epoch, batch_per_epoch, min_lr, max_lr)
    model, callbacks = fit_model_full(
        model=clone_model,
        n_epoch=n_epoch,
        dev_dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[lr_finder],
    )
    return model, callbacks
