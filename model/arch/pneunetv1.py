from torch import nn as nn

from model.arch.common import linear_block
from utils.common import to_list, pairwise


class PneuNetHeader(nn.Module):
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


class PneuNetv1(nn.Module):
    def __init__(self, preload_model, header_model):
        super().__init__()
        self.preload_model = preload_model
        self.preload_backbone, self.preload_header = self.dissect_model(
            self.preload_model
        )
        self.preload_backbone_output_tensor = self.preload_header[0].in_features

        self.backbone = self.preload_backbone
        self.freeze(self.backbone)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.header = header_model

        self.backbone_children = list(self.backbone.children())
        self.header_children = list(self.header.children())
        self.backbone_layers = len(self.backbone_children)
        self.header_layers = len(self.header_children)

    @staticmethod
    def dissect_model(model):
        return [nn.Sequential(*to_list(x)) for x in list(model.children())]

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model, n_layers, sequential=True):
        if sequential:
            model = model[0]
        model_children = list(model.children())
        model_layers = len(model_children)
        n_layers = model_layers if n_layers > model_layers else n_layers
        for children in model_children[(model_layers - n_layers) :]:
            for param in children.parameters():
                param.requires_grad = True

    def forward(self, x):
        bs = x.shape[0]
        x = self.preload_backbone(x)
        x = self.pooling(x).view(bs, -1)
        x = self.header(x)
        return x