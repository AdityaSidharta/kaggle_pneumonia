import torch
import torch.nn as nn
import torch.nn.functional as F


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


def train(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        .unsqueeze(0)
        .expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def class_metric(pred_label, target_label):
    TP = torch.sum((pred_label >= 0.5) & (target_label == 1.))
    FP = torch.sum((pred_label >= 0.5) & (target_label == 0.))
    TN = torch.sum((pred_label < 0.5) & (target_label == 0.))
    FN = torch.sum((pred_label < 0.5) & (target_label == 1.))
    acc = (TP + TN) / (TP + FP + TN + FN)
    prec = (TP) / (TP + FP)
    rec = (TP) / (TP + FN)
    return acc, prec, rec


# TODO IoU
def IoU(pred_bb, target_bb):
    pred_bb[:2] = pred_bb[:1] + pred_bb[:2]
    pred_bb[:3] = pred_bb[:0] + pred_bb[:3]
    target_bb[:2] = target_bb[:1] + target_bb[:2]
    target_bb[:3] = target_bb[:0] + target_bb[:3]
