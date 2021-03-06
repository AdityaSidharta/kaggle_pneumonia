from torch import nn as nn

from utils.common import to_list


# densenet
class DensePneuNet(nn.Module):
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
        n_layers = (
            model_layers if (n_layers > model_layers) or (n_layers == -1) else n_layers
        )
        for children in model_children[(model_layers - n_layers) :]:
            for param in children.parameters():
                param.requires_grad = True

    def forward(self, x):
        bs = x.shape[0]
        x = self.preload_backbone(x)
        x = self.pooling(x).view(bs, -1)
        x = self.header(x)
        return x
