from torch import nn as nn

from model.arch.common import linear_block
from utils.common import to_list, pairwise


class Header(nn.Module):
    def __init__(self, input_tensor, output_tensor, n_hidden_list, dropout):
        super().__init__()
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.n_hidden_list = to_list(n_hidden_list)
        self.n_hidden = len(self.n_hidden_list)
        self.dropout = dropout

        self.layers = []
        for before_hidden, after_hidden in pairwise(
            [self.input_tensor] + self.n_hidden_list
        ):
            self.layers.append(
                linear_block(
                    before_hidden, after_hidden, activation=True, dropout=dropout
                )
            )
        self.layers.append(
            linear_block(
                self.n_hidden_list[-1],
                self.output_tensor,
                dropout=False,
                activation=False,
                batchnorm=False,
            )
        )
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class DensePneuHead(Header):
    def __init__(self, n_hidden_list, dropout):
        input_tensor = 1024
        output_tensor = 5
        super(DensePneuHead, self).__init__(input_tensor, output_tensor, n_hidden_list, dropout)


class ResPneuHead(Header):
    def __init__(self, n_hidden_list, dropout):
        input_tensor = 512
        output_tensor = 5
        super(ResPneuHead, self).__init__(input_tensor, output_tensor, n_hidden_list, dropout)


class DenseClassHead(Header):
    def __init__(self, n_hidden_list, dropout):
        input_tensor = 1024
        output_tensor = 1
        super(DenseClassHead, self).__init__(input_tensor, output_tensor, n_hidden_list, dropout)


class ResClassHead(Header):
    def __init__(self, n_hidden_list, dropout):
        input_tensor = 512
        output_tensor = 1
        super(ResClassHead, self).__init__(input_tensor, output_tensor, n_hidden_list, dropout)