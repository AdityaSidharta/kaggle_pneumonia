import torch.nn.functional as F


def accuracy(prediction, target):
    return (target.long() == (F.sigmoid(prediction) > 0.5).long()).sum().item() / float(
        len(target)
    )
