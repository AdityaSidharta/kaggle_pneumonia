from torch import nn as nn


def linear_block(
    input_tensor, output_tensor, dropout=False, activation=None, batchnorm=True
):
    layers = nn.ModuleList()
    layers.append(nn.Linear(input_tensor, output_tensor))
    if activation:
        layers.append(nn.Tanh())
    if batchnorm:
        layers.append(nn.BatchNorm1d(output_tensor))
    if dropout:
        layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)
