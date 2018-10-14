import numpy as np
from tqdm import tqdm_notebook as tqdm

from dev.loss import calc_loss
from utils.common import get_batch_info


def validate_model_full(model, criterion, val_dataloader):
    n_val_obs, val_batch_size, val_batch_per_epoch = get_batch_info(val_dataloader)
    total_val_loss, total_val_loss_label, total_val_loss_bb = (
        np.zeros(val_batch_per_epoch),
        np.zeros(val_batch_per_epoch),
        np.zeros(val_batch_per_epoch),
    )
    model = model.eval()
    t = tqdm(enumerate(val_dataloader), total=val_batch_per_epoch)
    for idx, data in t:
        val_loss, val_loss_label, val_loss_bb = calc_loss(model, criterion, data)
        val_loss, val_loss_label, val_loss_bb = (
            val_loss.item(),
            val_loss_label.item(),
            val_loss_bb.item(),
        )
        total_val_loss[idx], total_val_loss_label[idx], total_val_loss_bb[idx] = (
            val_loss,
            val_loss_label,
            val_loss_bb,
        )
        t.set_postfix(
            {"loss": val_loss, "loss_label": val_loss_label, "loss_bb": val_loss_bb}
        )
    return total_val_loss.mean(), total_val_loss_label.mean(), total_val_loss_bb.mean()
