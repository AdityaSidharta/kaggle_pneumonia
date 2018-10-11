import torch
from torch import nn as nn
from torch.nn import functional as F


class LabelCriterion(nn.Module):
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)


class LabelBoundBoxCriterion(nn.Module):
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
