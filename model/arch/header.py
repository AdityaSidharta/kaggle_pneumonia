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

class Res50BBHead(Header):
    def __init__(self, n_hidden_list, dropout):
        input_tensor = 2048
        output_tensor = 4
        super(Res50BBHead, self).__init__(
            input_tensor, output_tensor, n_hidden_list, dropout
        )

class Res50ClassHead(Header):
    def __init__(self, n_hidden_list, dropout):
        input_tensor = 2048
        output_tensor = 1
        super(Res50ClassHead, self).__init__(
            input_tensor, output_tensor, n_hidden_list, dropout
        )
